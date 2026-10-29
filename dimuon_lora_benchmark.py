#!/usr/bin/env python3
"""Matched LoRA SFT benchmark for AdamW, Muon, and the recovered Dimuon.

The base model is frozen.  Only FFN LoRA factors are trainable, with identical
initial factors, token order, microbatch shape, precision, and schedule for all
optimizers.  The script is intentionally small enough to run on one GPU, and
also supports torchrun for an 8-GPU data-parallel measurement.

Example (single GPU smoke test):
  /home/catid/experiments/nora-math-dust/.venv/bin/python dimuon_lora_benchmark.py \
      --model /home/catid/experiments/nora-math-dust/data/math_rl/raw/model \
      --method all --steps 20 --batch 1 --seq 512 --last-n 8

Example (8 GPU throughput):
  torchrun --standalone --nproc_per_node=8 dimuon_lora_benchmark.py \
      --method dimuon --steps 100 --batch 1 --seq 512
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dimuon import Dimuon, GroupedDimuon


DEFAULT_MODEL = "/home/catid/experiments/nora-math-dust/data/math_rl/raw/model"
DEFAULT_MANIFEST = "/home/catid/experiments/nora-math-dust/data/math_rl/manifests/math_train.jsonl"


class LoRALinear(nn.Module):
    """Frozen Linear plus FP32 LoRA masters.

    ``nora`` uses the paper's parameterization: A is column-normalized on
    every forward. ``nora_init`` retains the one-time normalized-A
    initialization used by the earlier runs.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, seed: int,
                 init: str = "lora"):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        gen = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        b = torch.empty(base.out_features, rank, dtype=torch.float32)
        self.normalize_forward = init == "nora"
        if init == "lora":
            # Conventional LoRA: Kaiming A, zero B, so the initial effective
            # weight is exactly the frozen checkpoint.
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            b.zero_()
        elif init in ("nora", "nora_init"):
            # NoRA initialization: random A with unit-norm columns and zero B.
            # The full NoRA path additionally repeats this normalization in
            # every forward (see self.normalize_forward).
            a.normal_(0, 1, generator=gen)
            a.div_(a.norm(dim=0, keepdim=True).clamp_min(1e-12))
            b.zero_()
        elif init == "inilora_alpha":
            # IniLoRA-alpha as implemented by the authors: both factors are
            # N(0, .5), and the frozen base is replaced by W - s B A so the
            # complete initialized adapter still represents W.
            a.normal_(0, 0.5, generator=gen)
            b.normal_(0, 0.5, generator=gen)
            with torch.no_grad():
                delta = (b @ a).to(base.weight.device)
                base.weight.copy_((base.weight.float() - (alpha / rank) * delta)
                                  .to(base.weight.dtype))
        else:
            raise ValueError(f"unknown adapter initialization {init!r}")
        self.A = nn.Parameter(a.to(base.weight.device))
        self.B = nn.Parameter(b.to(base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        a = self.A
        if self.normalize_forward:
            a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
        h = F.linear(x, a.to(dtype=x.dtype))
        return y + F.linear(h, self.B.to(dtype=x.dtype)) * self.scaling


def inject_ffn_lora(model, rank: int, alpha: float, last_n: int, seed: int,
                    init: str = "lora"):
    """Inject adapters into gate/up/down projections in the selected suffix."""
    model.requires_grad_(False)
    layers = model.model.layers
    first = max(0, len(layers) - last_n)
    names = []
    counter = 0
    for i in range(first, len(layers)):
        mlp = layers[i].mlp
        for attr in ("gate_proj", "up_proj", "down_proj"):
            base = getattr(mlp, attr)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"layer {i} {attr} is {type(base).__name__}, expected Linear")
            wrapper = LoRALinear(base, rank, alpha, seed + counter, init=init)
            counter += 1
            setattr(mlp, attr, wrapper)
            names.extend([f"model.layers.{i}.mlp.{attr}.A", f"model.layers.{i}.mlp.{attr}.B"])
    trainable = dict(model.named_parameters())
    actual = sorted(n for n, p in trainable.items() if p.requires_grad)
    if actual != sorted(names):
        raise AssertionError(f"unexpected trainable parameters; missing/extra={set(actual)^set(names)}")
    return [p for p in model.parameters() if p.requires_grad], len(names) // 2


def load_records(path: str, tokenizer, max_rows: int = 4096):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # The pinned manifests contain questions and verified short answers.
            prompt = "Solve the following problem step by step. End with your final answer in \\boxed{...}.\\n\\nProblem:\\n" + r["question"] + "\\n\\nSolution:\\n"
            text = prompt + r.get("gold", "")
            ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=1024)["input_ids"]
            if len(ids) >= 8:
                rows.append(torch.tensor(ids, dtype=torch.long))
            if len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError("manifest produced no tokenized examples")
    # Pack examples into one repeatable stream.  This avoids measuring EOS
    # padding as a training target and makes every step have the same number of
    # real tokens.
    return torch.cat([torch.cat((r, torch.tensor([tokenizer.eos_token_id]))) for r in rows])


def make_batch(rows, step: int, rank: int, world: int, batch: int, seq: int, device):
    # Every method sees the same global index sequence.  Short examples are
    # packed with EOS; long examples are deterministically cropped.
    out = torch.full((batch, seq + 1), 151643, dtype=torch.long)
    for j in range(batch):
        start = ((step * world * batch + rank * batch + j) * seq) % max(1, rows.numel() - 1)
        if start + seq + 1 <= rows.numel():
            out[j] = rows[start:start + seq + 1]
        else:
            n = rows.numel() - start
            out[j, :n] = rows[start:]
            out[j, n:] = rows[:seq + 1 - n]
    return out.to(device, non_blocking=True)


def optimizer_for(method, params, args):
    if method == "adamw":
        return torch.optim.AdamW(params, lr=args.adamw_lr, betas=(0.9, 0.95),
                                 weight_decay=0.0, fused=True)
    if method == "muon":
        return torch.optim.Muon(params, lr=args.muon_lr, momentum=0.95,
                                nesterov=True, ns_steps=5, weight_decay=0.0)
    if method == "dimuon":
        opt_cls = GroupedDimuon if args.grouped_dimuon else Dimuon
        return opt_cls(params, lr=args.dimuon_lr, beta1=0.9, grad_clip_norm=0.0,
                      precision="fp16", damping=1e-4, state_dtype=torch.float32,
                      momentum_dtype=torch.bfloat16)
    raise ValueError(method)


def state_bytes(opt):
    return sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values()
               if torch.is_tensor(v))


def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_scalar(x, device, world):
    t = torch.tensor(float(x), device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(t)
    return float(t.item() / world)


def run(method, args, rows, eval_rows, rank, world, device, initial_state):
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                  dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False
    params, adapter_count = inject_ffn_lora(model, args.rank, args.alpha, args.last_n, args.seed,
                                            init=args.init)
    model.load_state_dict(initial_state, strict=False)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index],
                                                     broadcast_buffers=False, find_unused_parameters=False)
        module = model.module
    else:
        module = model
    # Re-select parameters after DDP wrapping so optimizer references are valid.
    params = [p for p in module.parameters() if p.requires_grad]
    opt = optimizer_for(method, params, args)
    model.train()
    times, losses, nonfinite = [], [], []
    # Warmup is included in the same fixed workload but excluded from timing.
    for step in range(args.steps):
        tok = make_batch(rows, step, rank, world, args.batch, args.seq, device)
        inp, labels = tok[:, :-1], tok[:, 1:]
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=inp, use_cache=False).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        opt.step()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        elapsed = time.perf_counter() - t0
        lv = reduce_scalar(loss.detach(), device, world)
        losses.append(lv)
        if not math.isfinite(lv):
            nonfinite.append(step)
        if step >= args.warmup:
            times.append(elapsed)
        if rank == 0 and (step < 3 or step == args.steps - 1):
            print(f"{method} step={step} loss={lv:.6f} time_ms={elapsed*1e3:.1f}", flush=True)
        if len(nonfinite) >= 2:
            raise RuntimeError(f"{method}: nonfinite loss")
    # Held-out teacher-forced loss is measured after training with the same
    # fixed sequence length and no gradient.  It is deliberately separate from
    # the timed training interval.
    model.eval()
    eval_sum = 0.0
    for step in range(args.eval_steps):
        tok = make_batch(eval_rows, step, rank, world, args.batch, args.seq, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            logits = model(input_ids=tok[:, :-1], use_cache=False).logits
            ev = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tok[:, 1:].reshape(-1))
        eval_sum += reduce_scalar(ev, device, world)
    eval_loss = eval_sum / max(1, args.eval_steps)
    mean = sum(times) / len(times) if times else float("nan")
    result = {
        "method": method, "steps": args.steps, "warmup": args.warmup,
        "world": world, "model": args.model, "rank": args.rank, "alpha": args.alpha,
        "last_n": args.last_n, "adapter_matrices": adapter_count,
        "trainable_params": sum(p.numel() for p in params),
        "mean_step_ms": mean * 1000.0,
        "tokens_per_s": world * args.batch * args.seq / mean if mean == mean else None,
        "first_loss": losses[0], "final_loss": losses[-1],
        "eval_loss": eval_loss,
        "losses": losses, "optimizer_state_bytes": state_bytes(opt),
        "nonfinite_steps": nonfinite,
        "lr": opt.param_groups[0]["lr"],
    }
    del opt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--eval-manifest", default="/home/catid/experiments/nora-math-dust/data/math_rl/manifests/math_val.jsonl")
    ap.add_argument("--method", choices=["adamw", "muon", "dimuon", "all"], default="all")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--last-n", type=int, default=8)
    ap.add_argument("--init", choices=["lora", "nora_init", "nora", "inilora_alpha"], default="lora",
                    help="adapter parameterization; nora normalizes A every forward, nora_init only at initialization")
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--eval-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--muon-lr", type=float, default=1e-3)
    ap.add_argument("--dimuon-lr", type=float, default=1.5e-3)
    ap.add_argument("--grouped-dimuon", action=argparse.BooleanOptionalAction, default=True,
                    help="batch equal-shaped factors in Dimuon (same math, much lower overhead)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rank, world, device = init_dist()
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_records(args.manifest, tokenizer, args.max_rows)
    eval_rows = load_records(args.eval_manifest, tokenizer, max(args.batch * args.eval_steps * world, 64))
    # Capture one common initialization.  The state dict contains frozen base
    # tensors, but that is the only reliable way to guarantee identical LoRA
    # factors after independent model construction on every rank.
    torch.manual_seed(args.seed)
    ref = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True)
    ref.to(device)
    _, _ = inject_ffn_lora(ref, args.rank, args.alpha, args.last_n, args.seed, init=args.init)
    # The base checkpoint is immutable and is reloaded from the same local
    # files for each arm.  Carry only adapter tensors between arms; retaining a
    # second 14B copy would waste both host and device memory.
    initial_state = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()
                     if k.endswith(".A") or k.endswith(".B")}
    del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()
    methods = [args.method] if args.method != "all" else ["adamw", "muon", "dimuon"]
    results = []
    for method in methods:
        if rank == 0:
            print(f"=== {method} ===", flush=True)
        results.append(run(method, args, rows, eval_rows, rank, world, device, initial_state))
        if world > 1:
            dist.barrier()
    if rank == 0:
        payload = {"args": vars(args), "packed_tokens": int(rows.numel()), "results": results}
        print("RESULTS " + json.dumps(payload), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
