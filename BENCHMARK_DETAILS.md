# Detailed benchmark report

## Hardware and software

- 8× NVIDIA B300 SXM6, 275 GB each.
- Python 3.12, torch `2.16.0.dev20261007+cu134`, CUDA 13.4, Transformers 5.19.
- Model: local Qwen2.5-32B-Instruct, Qwen2 architecture, 64 layers, hidden size 5120, FFN intermediate size 27648.
- Base weights are frozen and replicated on every GPU. DDP synchronizes the 201,326,592 trainable FP32 LoRA parameters.
- Each layer's `gate_proj`, `up_proj`, and `down_proj` receives rank 32, alpha 32 adapters: 192 matrices total.
- Per-GPU microbatch 2, sequence length 512, BF16 model compute, FP32 cross-entropy.
- Optimizers: fused AdamW, PyTorch Muon, and grouped Dimuon. Grouped Dimuon batches equal-shaped factors into one batched Polar Express update; random-matrix tests show grouped and ungrouped updates match in tested cases.

## Prospective equal-effort confirmation

The completed study ran all 54 confirmation jobs after all 54 tuning jobs.
Each of the 18 cells (2 datasets × 3 parameterizations × 3 optimizers) used
three fresh seeds (`2345`, `3456`, `4567`) at the learning rate selected from
three predeclared candidates on validation with seed `1234`. No test metric was
used for selection. Each job used 200 updates including 10 warmups, per-GPU
batch 2, sequence length 512, and 32 test evaluation batches on all 8 B300s.
The full per-seed tables and compact JSON/CSV are in
[`EQUAL_EFFORT_CONFIRMATION.md`](EQUAL_EFFORT_CONFIRMATION.md) and
`results/equal_effort_confirmation.{json,csv}`.

Held-out test CE (mean ± sample SD; lower is better):

| Parameterization / split | AdamW | Muon | Dimuon |
| --- | ---: | ---: | ---: |
| Standard LoRA / GSM8K | 1.5293 ± 0.0012 | **1.4990 ± 0.0032** | 1.5314 ± 0.0019 |
| Standard LoRA / MATH | **0.4905 ± 0.0005** | 0.4946 ± 0.0007 | 0.5078 ± 0.0018 |
| Full NoRA / GSM8K | 1.4472 ± 0.0042 | 1.4242 ± 0.0084 | **1.3893 ± 0.0013** |
| Full NoRA / MATH | 0.5279 ± 0.0031 | 0.7021 ± 0.0192 | **0.5056 ± 0.0031** |
| IniLoRA-alpha / GSM8K | **1.4939 ± 0.0149** | 2.1378 ± 0.6797 | 1.5060 ± 0.0137 |
| IniLoRA-alpha / MATH | **0.7227 ± 0.0065** | 1.5610 ± 0.5389 | 0.7409 ± 0.0044 |

This test panel gives Dimuon the lowest CE in both Full NoRA cells, Muon the
lowest CE in GSM8K Standard LoRA, and AdamW the lowest CE in MATH Standard
LoRA and both IniLoRA-alpha cells. It does not support a blanket quality win
for any optimizer.

Mean global throughput for Standard LoRA was 44.5k tok/s for AdamW, 26.9k
for Muon, and 38.3k for Dimuon. The optimizer state stayed at 1.61 GB/rank
(AdamW), 0.81 GB/rank (Muon), and 0.40 GB/rank (Dimuon). Throughput and state
are systems outcomes; the CE rows are the fixed short test slice described
above, not full-dataset accuracy.

## SFT results

Throughput is global tokens/s. Optimizer state is measured on one replicated rank.

### Standard LoRA, 50 updates

| Dataset | Optimizer | LR | Train CE | Held-out CE | Global tok/s | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 1e-4 | 0.1901 | 0.1859 | 43,325 | 1.61 GB |
| GSM8K | Muon | 1e-3 | 0.1879 | 0.1874 | 29,681 | 0.81 GB |
| GSM8K | Dimuon | 1.2e-2 | 0.1866 | **0.1812** | 37,212 | **0.40 GB** |
| MATH | AdamW | 1e-4 | 0.1967 | 0.2200 | 41,825 | 1.61 GB |
| MATH | Muon | 1e-3 | 0.2000 | 0.2236 | 28,267 | 0.81 GB |
| MATH | Dimuon | 1.2e-2 | 0.1891 | **0.2185** | 39,569 | **0.40 GB** |

A 20-update GSM8K LR sensitivity check gave held-out CE `(0.3597, 0.3360, 0.4028)` at `(3e-5, 3e-4, 3e-3)` and `(0.2155, 0.2261, 0.2116)` at `(1e-4, 1e-3, 1.2e-2)` for AdamW/Muon/Dimuon. The second panel overlapped a separate GPU job, so its throughput is excluded from speed claims.

### Full paper NoRA, 20 updates

Full NoRA normalizes A's columns on every forward. It is distinct from NoRA-init, which normalizes only once at initialization.

| Dataset | Optimizer | LR | Train CE | Held-out CE | Global tok/s | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 1e-4 | 0.2341 | 0.2080 | 32,324 | 1.61 GB |
| GSM8K | Muon | 3e-4 | 0.2413 | 0.2175 | 22,098 | 0.81 GB |
| GSM8K | Dimuon | 3e-3 | 0.2258 | **0.2054** | 27,861 | **0.40 GB** |
| MATH | AdamW | 1e-4 | 0.2204 | 0.2473 | 32,306 | 1.61 GB |
| MATH | Muon | 3e-4 | 0.2250 | 0.2712 | 23,363 | 0.81 GB |
| MATH | Dimuon | 3e-3 | 0.2192 | **0.2347** | 29,935 | **0.40 GB** |

### NoRA-init, 50 updates

These historical rows use one-time normalized A and zero B. They should be labeled NoRA-init, not full NoRA.

| Dataset | Optimizer | LR | Train CE | Held-out CE | Global tok/s | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 1e-4 | 0.2018 | 0.1958 | 43,630 | 1.61 GB |
| GSM8K | Muon | 3e-4 | 0.2758 | 0.2572 | 25,670 | 0.81 GB |
| GSM8K | Dimuon | 3e-3 | 0.2060 | **0.1980** | 36,529 | **0.40 GB** |
| MATH | AdamW | 1e-4 | 0.2029 | 0.2455 | 41,077 | 1.61 GB |
| MATH | Muon | 3e-4 | 0.2380 | 0.3201 | 25,750 | 0.81 GB |
| MATH | Dimuon | 3e-3 | **0.2016** | **0.2394** | 36,459 | **0.40 GB** |

### IniLoRA-alpha

IniLoRA-alpha samples both factors from N(0, 0.5) and subtracts the low-rank product from the frozen projection. In this BF16 implementation, that subtraction is rounded into the base weights, so the effective starting model differs from the original checkpoint.

| Dataset | Optimizer | LR | Updates | Train CE | Held-out CE | Global tok/s | State |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 1e-4 | 20 | 0.8248 | 0.8925 | 40,902 | 1.61 GB |
| GSM8K | Muon | 1e-3 | 20 | 1.3694 | 1.4832 | 25,502 | 0.81 GB |
| GSM8K | Dimuon | 3e-3 | 20 | **0.7843** | **0.8447** | 39,096 | **0.40 GB** |

A lower-LR panel (AdamW 3e-5, Muon 3e-4, Dimuon 1e-3) reached held-out CE 0.7194, 0.8380, and 0.7165, respectively. This lower-LR panel was better for all three methods than the mid-LR panel above; it is evidence that IniLoRA-alpha needs its own tuning, not evidence of a final optimizer ranking. The initialization itself changes the model under BF16 rounding.

## RL diagnostics

`dimuon_lora_rl.py` is retained as a DDP-plumbing diagnostic. It calls the DDP wrapper for the gradient-bearing forward and checks adapter replication after each update. Its quality artifacts must not be used for optimizer ranking: the checkpoint's inherited `top_k=20` and `repetition_penalty=1.05` remained active during generation while the backward pass used raw-logit log-probabilities, and its symbolic-answer fallback can score an incomplete expression by its last numeric coefficient. A replacement RL harness is required for valid quality evidence.

| Task | Optimizer | Mean shaped reward | Greedy held-out exact | Rollout tok/s | State |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 0.0549 | 0/16 | 98 | 1.61 GB |
| GSM8K | Muon | 0.0806 | 0/16 | 121 | 0.81 GB |
| GSM8K | Dimuon | 0.0588 | 1/16 | 125 | 0.40 GB |
| MATH | AdamW | 0.0929 | 2/16 | 103 | 1.61 GB |
| MATH | Muon | 0.0869 | 1/16 | 122 | 0.81 GB |
| MATH | Dimuon | 0.0850 | 1/16 | 130 | 0.40 GB |

These are synchronized DDP plumbing diagnostics, not an RL quality result. Numeric answers receive bounded log-distance shaping; the symbolic fallback is known to be unsound for some MATH expressions. The reward baseline is rank-local, and the sampling/log-probability mismatch invalidates quality ranking. Do not cite these rewards or exact-match counts as evidence for any optimizer.

## Post-study optimizer speed fix

The benchmark always sets `grad_clip_norm=0`, but both Dimuon implementations
were still converting every trainable gradient to the state dtype and reducing
its full norm before discarding the result. The patch now performs that
reduction only when clipping is enabled; clipping behavior is unchanged.

A matched 30-update A/B on 8 B300s (same Qwen2.5-32B checkpoint, seed 7777,
rank 32, batch 2, sequence 512, five warmups) measured 260.4 ms → 215.2 ms
per timed step and 31,455 → 38,064 global tokens/s: a 17.4% step-time
reduction and 21.0% throughput increase. Before/after losses remained finite
and held-out CE was 0.30647/0.30681. This is a microbenchmark of the optimizer
patch, separate from the preregistered confirmation numbers above.

## Interpretation

The strongest reproducible systems result is a 32B, eight-B300 study in which
Dimuon uses one quarter of AdamW's optimizer state and runs materially faster
than Muon. The completed equal-effort test panel favors Dimuon on both Full
NoRA cells, while Muon or AdamW is lower on the other parameterization/task
cells; it does not justify a blanket optimizer quality claim. AdamW remains
the fastest optimizer in raw Standard-LoRA throughput. The IniLoRA-alpha
residual subtraction changes the effective BF16 starting model, so its rows
are an optimizer comparison within that parameterization. The legacy RL
artifacts validate some DDP plumbing but are invalid for quality ranking
because of their sampling and answer-verification protocol. The strict v2
harness has passed CPU tests and an 8-GPU smoke, but no v2 RL quality result is
claimed. The exact local DeepSeek-V4.1-Flash checkpoint could not be used
because its available implementation is inference-only.
