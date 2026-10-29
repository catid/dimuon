#!/usr/bin/env python3
"""Strict, matched REINFORCE harness for FFN-only LoRA.

This file is deliberately separate from ``dimuon_lora_rl.py``.  The original
pilot used a permissive numeric fallback and inherited generation processors
from the checkpoint.  v2 samples from the raw temperature-1 policy, requires a
complete boxed answer, and verifies answers with ``math_verify``.

The training loop is synchronous data-parallel REINFORCE.  Every rank samples
the same deterministic row schedule with different rank seeds, DDP wraps the
gradient-bearing forward, and the baseline is the previous global step reward
mean.  This is a correctness/protocol harness, not GRPO/PPO.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete env
    raise ImportError(
        "dimuon_lora_rl_v2.py requires the `math-verify` package in the training environment"
    ) from exc

from dimuon import Dimuon, GroupedDimuon
from dimuon_lora_benchmark import inject_ffn_lora


ANSWER_CONFIG = (
    LatexExtractionConfig(try_extract_without_anchor=True, boxed_match_priority=100),
    ExprExtractionConfig(try_extract_without_anchor=True),
)
BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def question_key(row: dict) -> str:
    """Return a stable identity even for manifests without question_hash."""
    if row.get("question_hash"):
        return str(row["question_hash"])
    return hashlib.sha256(str(row["question"]).encode()).hexdigest()


def load_rows(path: str, max_rows: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError(f"manifest produced no rows: {path}")
    return rows


def assert_disjoint(train_rows: list[dict], eval_rows: list[dict]) -> None:
    train = {question_key(r) for r in train_rows}
    evaluation = {question_key(r) for r in eval_rows}
    overlap = train & evaluation
    if overlap:
        raise ValueError(f"train/eval question overlap ({len(overlap)} rows), first={next(iter(overlap))}")


def prompt_for(row: dict, tokenizer) -> str:
    """Use the model's chat template and leave an assistant generation prompt."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("v2 requires a tokenizer chat_template")
    content = (
        "Solve the following problem step by step. End with your final answer "
        "in \\boxed{...}.\n\nProblem:\n" + row["question"]
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def boxed_values(text: str) -> tuple[list[str], bool]:
    """Return balanced boxed contents and whether any boxed construct is malformed."""
    values: list[str] = []
    malformed = False
    for match in BOXED_OPEN.finditer(text):
        start, depth = match.end(), 1
        found = False
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[start:i])
                    found = True
                    break
        if not found:
            malformed = True
    return values, malformed


def last_box_end(text: str) -> int | None:
    """Return the end offset of the final balanced boxed expression."""
    end = None
    for match in BOXED_OPEN.finditer(text):
        depth = 1
        for i in range(match.end(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    return end


def _parse_boxed(expr: str):
    """Parse one expression using both LaTeX and symbolic extraction."""
    try:
        parsed = math_parse(
            r"\boxed{" + expr + "}",
            extraction_config=ANSWER_CONFIG,
            fallback_mode="first_match",
            extraction_mode="any_match",
            raise_on_error=False,
        )
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def score_answer(text: str, gold: str, completed: bool) -> dict:
    """Strict answer score.  Incomplete, malformed, or unboxed output earns zero."""
    boxes, malformed = boxed_values(text)
    result = {
        "reward": 0.0,
        "correct": False,
        "status": "",
        "boxed": bool(boxes),
    }
    if not completed:
        result["status"] = "truncated"
        return result
    if malformed:
        result["status"] = "malformed_box"
        return result
    if not boxes:
        result["status"] = "no_boxed_answer"
        return result
    end = last_box_end(text)
    tail = text[end:].strip() if end is not None else ""
    if tail and not re.fullmatch(r"[\s.,;:!?)]*", tail):
        result["status"] = "trailing_text"
        return result
    pred = _parse_boxed(boxes[-1])
    # Dataset gold is usually an unboxed expression; wrapping it prevents the
    # extractor from selecting a stray coefficient in a symbolic expression.
    gold_expr = str(gold)
    gold_boxes, gold_malformed = boxed_values(gold_expr)
    gold_expr = gold_boxes[-1] if gold_boxes and not gold_malformed else gold_expr
    target = _parse_boxed(gold_expr)
    if not pred or not target:
        result["status"] = "parse_error"
        return result
    try:
        ok = bool(math_verify(target, pred, strict=True, raise_on_error=False))
    except Exception:
        ok = False
    result["correct"] = ok
    result["reward"] = 1.0 if ok else 0.0
    result["status"] = "correct" if ok else "wrong"
    return result


def generation_config(tokenizer, response_tokens: int, sample: bool = True) -> GenerationConfig:
    """Explicit raw-policy config; no checkpoint sampling processors are inherited."""
    eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos_ids:
        # Qwen checkpoints use both the assistant-turn terminator and the
        # end-of-text token as stops.  Keep those explicit stop IDs while
        # overriding all sampling processors.
        eos_ids.append(tokenizer.pad_token_id)
    return GenerationConfig(
        do_sample=sample,
        temperature=1.0,
        # `top_k=0` explicitly disables truncation for sampling.  Leave it
        # unset for greedy evaluation because Transformers warns that top-k is
        # sampling-only in that mode.
        # Greedy mode treats the default top-k=50 as neutral and avoids a
        # sampling-only validation warning; it is never applied when
        # `do_sample=False`.
        top_k=0 if sample else 50,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=response_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        use_cache=True,
    )


def tokenize_prompts(rows: list[dict], tokenizer, max_prompt_tokens: int, device):
    texts = [prompt_for(row, tokenizer) for row in rows]
    # Keep the end of a long prompt (the generation instruction and answer
    # context) while reporting the number of truncated prompts.
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    raw_lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    tokenizer.truncation_side = old_side
    return (
        enc.input_ids.to(device, non_blocking=True),
        enc.attention_mask.to(device, non_blocking=True),
        [n > max_prompt_tokens for n in raw_lengths],
    )


@torch.inference_mode()
def sample_batch(model, rows, tokenizer, args, device, seed: int):
    input_ids, prompt_mask, prompt_truncated = tokenize_prompts(
        rows, tokenizer, args.prompt_tokens, device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    cfg = generation_config(tokenizer, args.response_tokens, sample=True)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=prompt_mask,
        generation_config=cfg,
    )
    prompt_width = input_ids.shape[1]
    response = out[:, prompt_width:]
    eos_ids = {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        eos_ids.add(tokenizer.pad_token_id)
    lengths, completed, masks = [], [], []
    for row in response:
        hits = torch.stack([(row == token) for token in eos_ids]).any(dim=0).nonzero(as_tuple=False).flatten()
        if len(hits):
            n = int(hits[0].item()) + 1
            lengths.append(n)
            completed.append(True)
        else:
            n = int(row.numel())
            lengths.append(n)
            completed.append(False)
        mask = torch.zeros(row.numel(), dtype=torch.bool, device=device)
        mask[:n] = True
        masks.append(mask)
    response_mask = torch.stack(masks)
    texts = [
        tokenizer.decode(response[i, : lengths[i]].tolist(), skip_special_tokens=True)
        for i in range(len(rows))
    ]
    return {
        "input_ids": input_ids,
        "prompt_mask": prompt_mask,
        "response": response,
        "response_mask": response_mask,
        "lengths": lengths,
        "completed": completed,
        "texts": texts,
        "prompt_truncated": prompt_truncated,
    }


def sequence_logprobs(model, batch: dict) -> torch.Tensor:
    """Return summed log p(response | prompt), with pad-after-EOS masked out."""
    input_ids = batch["input_ids"]
    response = batch["response"]
    prompt_mask = batch["prompt_mask"]
    seq = torch.cat((input_ids, response), dim=1)
    full_mask = torch.cat(
        (prompt_mask, torch.ones_like(response, dtype=prompt_mask.dtype)), dim=1
    )
    logits = model(
        input_ids=seq[:, :-1],
        attention_mask=full_mask[:, :-1],
        use_cache=False,
    ).logits
    token_lp = F.log_softmax(logits.float(), dim=-1).gather(
        -1, seq[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    start = input_ids.shape[1] - 1
    response_lp = token_lp[:, start : start + response.shape[1]]
    return (response_lp * batch["response_mask"].to(response_lp.dtype)).sum(dim=1)


def allreduce_sum(value, device, world):
    t = torch.as_tensor(value, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def finite_or_raise(tensor, name: str, step: int, device, world):
    local_bad = torch.tensor(0 if torch.isfinite(tensor).all() else 1, device=device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
    if int(local_bad.item()):
        raise FloatingPointError(f"non-finite {name} at step {step}")


def finite_params_or_raise(params, name: str, step: int, device, world):
    """Check a parameter/gradient set with one collective, not one per tensor."""
    local_bad = torch.zeros((), device=device, dtype=torch.int32)
    for p in params:
        # Keep the reduction on-device; converting every parameter check to a
        # Python bool would serialize hundreds of CUDA streams per step.
        local_bad = torch.maximum(
            local_bad,
            (~torch.isfinite(p)).any().to(torch.int32),
        )
    if world > 1:
        dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
    if int(local_bad.item()):
        raise FloatingPointError(f"non-finite {name} at step {step}")


@dataclass
class Stats:
    count: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    completed: int = 0
    truncated: int = 0
    boxed: int = 0
    correct: int = 0
    malformed: int = 0
    no_boxed: int = 0
    parse_error: int = 0
    wrong: int = 0
    prompt_truncated: int = 0
    response_tokens: int = 0

    def add(self, score: dict, completed: bool, length: int, prompt_truncated: int = 0):
        reward = float(score["reward"])
        self.count += 1
        self.reward_sum += reward
        self.reward_sq_sum += reward * reward
        self.completed += int(completed)
        self.truncated += int(not completed)
        self.boxed += int(score["boxed"])
        self.correct += int(score["correct"])
        self.malformed += int(score["status"] == "malformed_box")
        self.no_boxed += int(score["status"] == "no_boxed_answer")
        self.parse_error += int(score["status"] == "parse_error")
        self.wrong += int(score["status"] == "wrong")
        self.prompt_truncated += int(prompt_truncated)
        self.response_tokens += int(length)

    def reduced(self, device, world) -> dict:
        vals = torch.tensor(
            [self.count, self.reward_sum, self.reward_sq_sum, self.completed, self.truncated,
             self.boxed, self.correct, self.malformed, self.no_boxed, self.parse_error,
             self.wrong, self.prompt_truncated, self.response_tokens],
            device=device, dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        n = max(1.0, float(vals[0]))
        mean = float(vals[1] / n)
        var = max(0.0, float(vals[2] / n) - mean * mean)
        return {
            "count": int(vals[0].item()),
            "reward_mean": mean,
            "reward_sd": math.sqrt(var),
            "completed": int(vals[3].item()),
            "truncated": int(vals[4].item()),
            "boxed": int(vals[5].item()),
            "correct": int(vals[6].item()),
            "malformed": int(vals[7].item()),
            "no_boxed": int(vals[8].item()),
            "parse_error": int(vals[9].item()),
            "wrong": int(vals[10].item()),
            "prompt_truncated": int(vals[11].item()),
            "response_tokens": int(vals[12].item()),
            "completion_rate": float(vals[3] / n),
            "boxed_rate": float(vals[5] / n),
            "accuracy": float(vals[6] / n),
        }


def evaluate(model, rows, tokenizer, args, rank, world, device, seed: int, sample: bool):
    local = [(i, row) for i, row in enumerate(rows) if i % world == rank]
    stats = Stats()
    cfg = generation_config(tokenizer, args.response_tokens, sample=sample)
    # Process a rank's shard in batches.  The generation seed is indexed by
    # global shard position so every optimizer sees the same eval questions.
    for start in range(0, len(local), args.batch):
        indexed = local[start : start + args.batch]
        batch_rows = [r for _, r in indexed]
        batch_seed = seed + 200003 * (start // max(1, args.batch)) + rank
        if sample:
            batch = sample_batch(model, batch_rows, tokenizer, args, device, batch_seed)
        else:
            input_ids, prompt_mask, prompt_truncated = tokenize_prompts(
                batch_rows, tokenizer, args.prompt_tokens, device
            )
            if device.type == "cuda":
                torch.cuda.manual_seed(batch_seed)
            else:
                torch.manual_seed(batch_seed)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                generation_config=cfg,
            )
            width = input_ids.shape[1]
            response = out[:, width:]
            lengths, completed, masks = [], [], []
            eos_ids = {tokenizer.eos_token_id}
            if tokenizer.pad_token_id is not None:
                eos_ids.add(tokenizer.pad_token_id)
            for row in response:
                hits = torch.stack([(row == token) for token in eos_ids]).any(dim=0).nonzero(as_tuple=False).flatten()
                if len(hits):
                    n = int(hits[0].item()) + 1
                    completed.append(True)
                else:
                    n = int(row.numel())
                    completed.append(False)
                lengths.append(n)
                mask = torch.zeros(row.numel(), dtype=torch.bool, device=device)
                mask[:n] = True
                masks.append(mask)
            batch = {
                "response": response,
                "lengths": lengths,
                "completed": completed,
                "texts": [tokenizer.decode(response[i, : lengths[i]].tolist(), skip_special_tokens=True)
                          for i in range(len(batch_rows))],
                "prompt_truncated": prompt_truncated,
            }
        for i, row in enumerate(batch_rows):
            score = score_answer(batch["texts"][i], str(row.get("gold", "")), batch["completed"][i])
            stats.add(score, batch["completed"][i], batch["lengths"][i],
                      int(batch["prompt_truncated"][i]))
    return stats.reduced(device, world)


def adapter_digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        if not p.requires_grad or not (name.endswith(".A") or name.endswith(".B")):
            continue
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        cpu = p.detach().contiguous().cpu().view(torch.uint8)
        for chunk in cpu.split(8 * 1024 * 1024):
            h.update(chunk.numpy().tobytes())
    return h.hexdigest()


def verify_adapter_sync(module: nn.Module, rank: int, world: int) -> str:
    digest = adapter_digest(module)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, digest)
        if any(x != gathered[0] for x in gathered):
            raise RuntimeError(f"adapter digest mismatch across ranks: {gathered}")
    return digest


def optimizer_for(name, params, args):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=args.adamw_lr, betas=(0.9, 0.95),
                                 weight_decay=0.0, fused=(args.device_type == "cuda"))
    if name == "muon":
        return torch.optim.Muon(params, lr=args.muon_lr, momentum=0.95,
                                nesterov=True, ns_steps=5, weight_decay=0.0)
    if name == "dimuon":
        cls = GroupedDimuon if args.grouped_dimuon else Dimuon
        return cls(params, lr=args.dimuon_lr, beta1=0.9, grad_clip_norm=0.0,
                   precision="fp16", damping=1e-4, state_dtype=torch.float32,
                   momentum_dtype=torch.bfloat16)
    raise ValueError(name)


def run(method, args, train_rows, eval_rows, tokenizer, rank, world, device):
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.config.use_cache = True
    _, adapter_count = inject_ffn_lora(model, args.rank, args.alpha, args.last_n, args.seed,
                                        init=args.init)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=False,
            find_unused_parameters=False,
        )
        module = model.module
    else:
        module = model
    # Eval mode is intentional: there must be no dropout in either sampling or
    # the gradient-bearing policy-logprob forward.
    model.eval()
    params = [p for p in module.parameters() if p.requires_grad]
    opt = optimizer_for(method, params, args)
    pretrain = evaluate(module, eval_rows, tokenizer, args, rank, world, device,
                        args.seed + 71, sample=False)
    baseline = float(args.reward_baseline)
    train_stats = Stats()
    step_rewards = []
    synchronize(device)
    start_time = time.perf_counter()
    for step in range(args.steps):
        batch_rows = [
            train_rows[(step * world * args.batch + rank * args.batch + j) % len(train_rows)]
            for j in range(args.batch)
        ]
        batch = sample_batch(module, batch_rows, tokenizer, args, device,
                             args.seed + 100003 * step + rank)
        scores = [score_answer(batch["texts"][i], str(row.get("gold", "")),
                               batch["completed"][i]) for i, row in enumerate(batch_rows)]
        rewards = torch.tensor([s["reward"] for s in scores], device=device, dtype=torch.float32)
        finite_or_raise(rewards, "reward", step, device, world)
        global_sum = allreduce_sum(rewards.sum(), device, world)
        global_n = allreduce_sum(rewards.numel(), device, world)
        global_mean = float((global_sum / global_n).item())
        # The baseline is the previous *global* step mean, not a rank-local EMA.
        advantage = rewards - baseline
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            # DDP wrapper is mandatory here; calling module would skip reduction.
            seq_lp = sequence_logprobs(model if world > 1 else module, batch)
            loss = -(advantage.detach() * seq_lp).mean()
        finite_or_raise(seq_lp, "sequence log-probability", step, device, world)
        finite_or_raise(loss, "loss", step, device, world)
        loss.backward()
        finite_params_or_raise([p.grad for p in params if p.grad is not None],
                               "gradient", step, device, world)
        opt.step()
        finite_params_or_raise(params, "adapter", step, device, world)
        synchronize(device)
        baseline = global_mean
        step_rewards.append(global_mean)
        for i, score in enumerate(scores):
            train_stats.add(score, batch["completed"][i], batch["lengths"][i],
                             int(batch["prompt_truncated"][i]))
    train_elapsed = time.perf_counter() - start_time
    posttrain = evaluate(module, eval_rows, tokenizer, args, rank, world, device,
                         args.seed + 107, sample=False)
    total_elapsed = time.perf_counter() - start_time
    digest = verify_adapter_sync(module, rank, world)
    reduced_train = train_stats.reduced(device, world)
    reduced_train["steps"] = args.steps
    reduced_train["batch_per_gpu"] = args.batch
    reduced_train["global_sample_count"] = args.steps * args.batch * world
    reduced_train["step_reward_means"] = step_rewards if rank == 0 else []
    result = {
        "method": method,
        "world": world,
        "adapter_matrices": adapter_count,
        "trainable_params": sum(p.numel() for p in params),
        "pretrain_eval": pretrain,
        "train": reduced_train,
        "posttrain_eval": posttrain,
        "train_elapsed_s": train_elapsed,
        "total_elapsed_s": total_elapsed,
        "global_sample_tokens_per_s": reduced_train["response_tokens"] / max(train_elapsed, 1e-9),
        "optimizer_state_bytes": sum(v.numel() * v.element_size()
                                      for st in opt.state.values() for v in st.values()
                                      if torch.is_tensor(v)),
        "adapter_sha256": digest,
        "lr": opt.param_groups[0]["lr"],
    }
    del opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--method", choices=["adamw", "muon", "dimuon", "all"], default="all")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--eval-rows", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=768)
    ap.add_argument("--response-tokens", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--last-n", type=int, default=64)
    ap.add_argument("--init", choices=["lora", "nora", "inilora_alpha"], default="lora")
    ap.add_argument("--max-train-rows", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reward-baseline", type=float, default=0.0)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--muon-lr", type=float, default=1e-3)
    ap.add_argument("--dimuon-lr", type=float, default=1.2e-2)
    ap.add_argument("--grouped-dimuon", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rank, world, device = init_dist()
    args.device_type = device.type
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Check split separation against the complete manifests before applying
    # runtime caps.  Checking only the sampled prefixes could hide leakage in
    # rows omitted by --max-train-rows/--eval-rows.
    all_train_rows = load_rows(args.train_manifest)
    all_eval_rows = load_rows(args.eval_manifest)
    assert_disjoint(all_train_rows, all_eval_rows)
    train_rows = all_train_rows[:args.max_train_rows]
    eval_rows = all_eval_rows[:args.eval_rows]
    dataset_counts = {
        "train_manifest_rows_total": len(all_train_rows),
        "eval_manifest_rows_total": len(all_eval_rows),
        "train_manifest_rows_loaded": len(train_rows),
        "eval_manifest_rows_loaded": len(eval_rows),
        "train_question_hashes": len({question_key(r) for r in train_rows}),
        "eval_question_hashes": len({question_key(r) for r in eval_rows}),
    }
    methods = [args.method] if args.method != "all" else ["adamw", "muon", "dimuon"]
    results = []
    for method in methods:
        if rank == 0:
            print(f"=== {method} (strict RL v2) ===", flush=True)
        results.append(run(method, args, train_rows, eval_rows, tokenizer, rank, world, device))
        if world > 1:
            dist.barrier()
    if rank == 0:
        payload = {"protocol": "strict_rl_v2", "args": vars(args),
                   "dataset_counts": dataset_counts, "results": results}
        print("RESULTS " + json.dumps(payload), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
