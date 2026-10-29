# Detailed benchmark report

## Hardware and software

- 8× NVIDIA B300 SXM6, 275 GB each.
- Python 3.12, torch `2.16.0.dev20261007+cu134`, CUDA 13.4, Transformers 5.19.
- Model: local Qwen2.5-32B-Instruct, Qwen2 architecture, 64 layers, hidden size 5120, FFN intermediate size 27648.
- Base weights are frozen and replicated on every GPU. DDP synchronizes the 201,326,592 trainable FP32 LoRA parameters.
- Each layer's `gate_proj`, `up_proj`, and `down_proj` receives rank 32, alpha 32 adapters: 192 matrices total.
- Per-GPU microbatch 2, sequence length 512, BF16 model compute, FP32 cross-entropy.
- Optimizers: fused AdamW, PyTorch Muon, and grouped Dimuon. Grouped Dimuon batches equal-shaped factors into one batched Polar Express update; random-matrix tests show grouped and ungrouped updates match in tested cases.

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

## Interpretation

The defensible result is a 32B, eight-B300 SFT/NoRA engineering pilot with a
second-seed replication check: Dimuon reaches the lowest held-out CE in the
standard LoRA and full NoRA panels while using one quarter of AdamW's optimizer
state and running materially faster than Muon. AdamW remains faster in raw
standard-LoRA throughput. The IniLoRA-alpha two-seed panel also favors Dimuon
at its separately tuned lower learning rates, but its BF16 residual
subtraction changes the starting model. The legacy RL artifacts validate some
DDP plumbing but are invalid for quality ranking because of their sampling and
answer-verification protocol. The strict v2 harness has passed CPU tests and
an 8-GPU smoke, but no v2 RL quality result is claimed. The exact local
DeepSeek-V4.1-Flash checkpoint could not be used because its available
implementation is inference-only. A full equal-effort study is specified in
`STUDY_PROTOCOL.md`; the 20-update panels are not that study.
