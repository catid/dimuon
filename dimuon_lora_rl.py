#!/usr/bin/env python3
"""Small, matched REINFORCE benchmark for FFN-only LoRA.

This is intentionally a real policy-gradient loop (sample -> answer reward
-> sequence log-probability -> optimizer step), rather than the SFT
teacher-forced loss in ``dimuon_lora_benchmark.py``.  By default the reward
keeps exact answers at 1 and adds bounded numeric distance shaping; pass
``--no-dense-reward`` for binary rewards.  It is useful for a sanity check and
optimizer comparison; it is not a production GRPO trainer.

Example (8-way data parallel):
  torchrun --standalone --nproc_per_node=8 dimuon_lora_rl.py \
    --model /home/catid/models/Qwen2.5-32B-Instruct --steps 20 \
    --manifest /path/to/gsm8k_train.jsonl --method all
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dimuon_lora_benchmark import inject_ffn_lora
from dimuon import Dimuon, GroupedDimuon


def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_sum(x, device, world):
    t = torch.tensor(float(x), device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    return float(t.item())


def optimizer_for(name, params, args):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=args.adamw_lr, betas=(0.9, 0.95),
                                 weight_decay=0.0, fused=True)
    if name == "muon":
        return torch.optim.Muon(params, lr=args.muon_lr, momentum=0.95,
                                nesterov=True, ns_steps=5, weight_decay=0.0)
    if name == "dimuon":
        cls = GroupedDimuon if args.grouped_dimuon else Dimuon
        return cls(params, lr=args.dimuon_lr, beta1=0.9, grad_clip_norm=0.0,
                   precision="fp16", damping=1e-4, state_dtype=torch.float32,
                   momentum_dtype=torch.bfloat16)
    raise ValueError(name)


def boxed_values(text: str):
    """Extract balanced ``\\boxed{...}`` contents, including nested TeX braces."""
    out = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:i])
                    break
    return out


def numeric_answer(text: str):
    """Extract a final numeric answer, including simple TeX fractions."""
    boxes = boxed_values(text)
    s = boxes[-1] if boxes else text
    if boxes and re.search(r"[A-Za-z]", re.sub(r"\\frac", "", s)):
        # A boxed symbolic expression (e.g. ``2/x+4x^2`` or ``193\\pi``)
        # must not be misread as the first numeric coefficient.
        return None
    fracs = re.findall(r"\\frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}", s)
    # Some MATH answers use TeX's compact ``\\frac12`` spelling.
    fracs += re.findall(r"\\frac\s*(-?\d)\s*(-?\d)(?!\d)", s)
    fracs += re.findall(r"(?<![\w])(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)(?![\w])", s)
    if fracs:
        n, d = fracs[-1]
        try:
            return str(float(Fraction(n) / Fraction(d)))
        except (ValueError, ZeroDivisionError):
            pass
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def normalized_answer(text: str) -> str:
    """Normalize boxed or final-line answers, including symbolic MATH answers."""
    boxes = boxed_values(text)
    value = boxes[-1] if boxes else (text.strip().splitlines()[-1] if text.strip() else "")
    value = value.lower().replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "").replace(" ", "").replace("\\,", "")
    return value.strip(".。,:;\\")


def reward_value(text: str, gold: str, dense: bool = True) -> float:
    """Exact reward plus bounded numeric distance shaping for sparse rewards."""
    pred = numeric_answer(text)
    target = numeric_answer(str(gold))
    if pred is not None and target is not None:
        try:
            p, t = float(pred), float(target)
            exact = float(abs(p - t) < 1e-6)
            if exact or not dense:
                return exact
            # Bounded [0, .25] shaping signal based on log-distance.
            closeness = math.exp(-abs(math.log1p(abs(p)) - math.log1p(abs(t))))
            if p * t < 0:
                closeness *= 0.25
            return 0.25 * closeness
        except ValueError:
            pass
    return float(normalized_answer(text) == normalized_answer(str(gold)))


def exact_reward(text: str, gold: str) -> float:
    """Binary exact reward retained for held-out reporting."""
    return reward_value(text, gold, dense=False)


def load_rows(path, max_rows):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            if len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError("manifest produced no examples")
    return rows


def prompt_for(row):
    return ("Solve the following problem step by step. End with your final answer "
            "in \\boxed{...}.\n\nProblem:\n" + row["question"] +
            "\n\nSolution:\n")


def run(method, args, rows, eval_rows, tokenizer, rank, world, device):
    # Construct each arm independently to keep initialization and base weights
    # identical.  LoRA factors are seeded identically across methods/ranks.
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                  dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = True
    _, _ = inject_ffn_lora(model, args.rank, args.alpha, args.last_n, args.seed,
                           init=args.init)
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], broadcast_buffers=False,
            find_unused_parameters=False)
        module = model.module
        params = [p for p in module.parameters() if p.requires_grad]
    else:
        module = model
    opt = optimizer_for(method, params, args)
    module.train()
    # Batch one per rank keeps variable-length generation simple and makes the
    # exact reward unambiguous.  DDP still averages adapter gradients globally.
    baseline = args.reward_baseline
    rewards, losses, times, lengths = [], [], [], []
    for step in range(args.steps):
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        step_t0 = time.perf_counter()
        row = rows[(step * world + rank) % len(rows)]
        enc = tokenizer(prompt_for(row), return_tensors="pt", truncation=True,
                        max_length=args.prompt_tokens)
        prompt_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        # Sampling is outside autograd.  A per-step/rank seed makes the run
        # deterministic while retaining independent trajectories across ranks.
        with torch.no_grad():
            torch.manual_seed(args.seed + 100003 * step + rank)
            seq = module.generate(input_ids=prompt_ids, attention_mask=attn,
                                  do_sample=True, temperature=args.temperature,
                                  top_p=args.top_p,
                                  max_new_tokens=args.response_tokens,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  use_cache=True)
        p_len = prompt_ids.shape[1]
        response = seq[:, p_len:]
        text = tokenizer.decode(response[0], skip_special_tokens=True)
        reward = reward_value(text, row.get("gold", ""), dense=args.dense_reward)
        # Recompute log-probabilities with grad.  The sampled sequence itself
        # is detached; only adapter weights receive gradients.
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            # Use the DDP wrapper for the gradient-bearing forward.  Calling
            # module (the underlying model) here would bypass DDP's reducer
            # and silently train eight rank-local policies.
            train_model = model if world > 1 else module
            logits = train_model(input_ids=seq[:, :-1],
                            attention_mask=torch.ones_like(seq[:, :-1]),
                            use_cache=False).logits
        lp = F.log_softmax(logits.float(), dim=-1).gather(
            -1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        # Include all generated tokens; EOS is present in the fixed response
        # when generation terminates early.  At least one token is guaranteed.
        response_lp = lp[:, p_len - 1:]
        sequence_logp = response_lp.mean()
        # A moving baseline lowers variance.  The initial baseline can be set
        # to zero for pure REINFORCE or a small prior for sparse exact reward.
        advantage = reward - baseline
        loss = -float(advantage) * sequence_logp
        loss.backward()
        opt.step()
        if world > 1:
            # DDP should leave replicated LoRA masters equal after every step.
            # Fail loudly if a future refactor accidentally bypasses its
            # reducer again.
            check = params[0].detach().norm().to(torch.float64)
            lo, hi = check.clone(), check.clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            if float(hi - lo) > 1e-5:
                raise RuntimeError(f"DDP adapter divergence after step {step}: {float(hi-lo)}")
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * reward
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        # Generation and backward are both part of RL step time.
        elapsed = time.perf_counter() - step_t0
        rewards.append(reward)
        losses.append(float(loss.detach()))
        lengths.append(int(response.shape[1]))
        times.append(elapsed)
        if rank == 0 and (step < 3 or step == args.steps - 1):
            print(f"{method} step={step} reward={reward:.0f} loss={losses[-1]:.6f} "
                  f"answer={numeric_answer(text)}", flush=True)
    # Held-out reward uses deterministic greedy generation and no update.
    module.eval()
    eval_rows = eval_rows[:min(args.eval_rows, len(eval_rows))]
    hit = 0.0
    for row in eval_rows:
        enc = tokenizer(prompt_for(row), return_tensors="pt", truncation=True,
                        max_length=args.prompt_tokens)
        with torch.no_grad():
            seq = module.generate(input_ids=enc.input_ids.to(device),
                                  attention_mask=enc.attention_mask.to(device),
                                  do_sample=False, max_new_tokens=args.response_tokens,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(seq[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        hit += exact_reward(text, row.get("gold", ""))
    # Aggregate metrics over ranks.  Evaluation rows are repeated by each
    # rank, so only rank zero's value is used for the final report.
    if world > 1:
        total_reward = reduce_sum(sum(rewards), device, world)
        total_steps = reduce_sum(len(rewards), device, world)
        total_len = reduce_sum(sum(lengths), device, world)
        if rank == 0:
            eval_hit = hit / max(1, len(eval_rows))
        else:
            eval_hit = 0.0
    else:
        total_reward, total_steps, total_len = sum(rewards), len(rewards), sum(lengths)
        eval_hit = hit / max(1, len(eval_rows))
    result = {
        "method": method, "steps": args.steps, "world": world,
        "model": args.model, "rank": args.rank, "alpha": args.alpha,
        "last_n": args.last_n, "mean_reward": total_reward / max(1, total_steps),
        "mean_response_tokens": total_len / max(1, total_steps),
        # total_len is already summed over ranks above; do not multiply by
        # world a second time.  Each rank's elapsed time covers the same
        # synchronized DDP interval, so this is the true global rate.
        "global_rollout_tokens_per_s": total_len / max(1e-9, sum(times)),
        "mean_step_s": sum(times) / max(1, len(times)),
        "eval_exact_match": eval_hit,
        "dense_reward": args.dense_reward,
        "optimizer_state_bytes": sum(v.numel() * v.element_size() for st in opt.state.values()
                                      for v in st.values() if torch.is_tensor(v)),
        "lr": opt.param_groups[0]["lr"], "losses": losses,
        "note": "Synchronous REINFORCE with exact numeric reward; not GRPO/PPO.",
    }
    del opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--eval-manifest", required=True,
                    help="held-out manifest; do not reuse training rows")
    ap.add_argument("--method", choices=["adamw", "muon", "dimuon", "all"], default="all")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eval-rows", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=768)
    ap.add_argument("--response-tokens", type=int, default=64)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--last-n", type=int, default=64)
    ap.add_argument("--init", choices=["lora", "nora", "inilora_alpha"], default="lora")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reward-baseline", type=float, default=0.0)
    ap.add_argument("--baseline-decay", type=float, default=0.9)
    ap.add_argument("--dense-reward", action=argparse.BooleanOptionalAction, default=True,
                    help="bounded numeric distance shaping for better RL signal")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--muon-lr", type=float, default=1e-3)
    ap.add_argument("--dimuon-lr", type=float, default=1.2e-2)
    ap.add_argument("--grouped-dimuon", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rank, world, device = init_dist()
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = load_rows(args.manifest, args.max_rows)
    eval_rows = load_rows(args.eval_manifest, args.eval_rows)
    methods = [args.method] if args.method != "all" else ["adamw", "muon", "dimuon"]
    results = []
    for method in methods:
        if rank == 0:
            print(f"=== {method} ===", flush=True)
        results.append(run(method, args, rows, eval_rows, tokenizer, rank, world, device))
        if world > 1:
            dist.barrier()
    if rank == 0:
        payload = {"args": vars(args), "results": results}
        print("RESULTS " + json.dumps(payload), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
