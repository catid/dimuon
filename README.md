# Dimuon for FFN LoRA fine-tuning

A compact reproduction of the Dimuon optimizer comparison on a **32B** language model using **all 8 B300 GPUs**. The base model is frozen; only rank-32 LoRA factors on the `gate_proj`, `up_proj`, and `down_proj` FFN projections in all 64 Qwen2.5-32B layers are trained.

> The headline result is a promising SFT/NoRA pilot. The RL artifacts are retained as debugging provenance only: an audit found that the legacy sampler's effective distribution differed from the gradient log-probability and that symbolic-answer scoring could accept incomplete answers. No RL quality ranking is claimed.

## Headline results

Held-out teacher-forced CE, lower is better. Each row uses the same model, token stream, seed, batch shape, and update budget for all three optimizers. Standard LoRA uses 50 updates; full NoRA and IniLoRA-alpha use 20. Learning rates were selected post-hoc from short sweeps; search effort was not yet balanced across all method/parameterization pairs.

| Parameterization / task | AdamW | Muon | Dimuon |
| --- | ---: | ---: | ---: |
| Standard LoRA / GSM8K | 0.1859 | 0.1874 | **0.1812** |
| Standard LoRA / MATH | 0.2200 | 0.2236 | **0.2185** |
| Full NoRA / GSM8K | 0.2080 | 0.2175 | **0.2054** |
| Full NoRA / MATH | 0.2473 | 0.2712 | **0.2347** |
| IniLoRA-alpha / GSM8K* | 0.8925 | 1.4832 | **0.8447** |

`*`This is the mid-LR IniLoRA-alpha diagnostic panel. A lower-LR panel performed better for every optimizer (AdamW/Muon/Dimuon CE: 0.7194/0.8380/0.7165). BF16 residual subtraction changes the effective starting model; the displayed panel is not a tuned comparison to standard LoRA.

![Held-out quality](figures/quality.png)

For standard LoRA, the optimizer state per replicated rank was 1.61 GB for AdamW, 0.81 GB for Muon, and 0.40 GB for Dimuon. Standard-LoRA throughput averaged across GSM8K/MATH was 42.6k, 29.0k, and 38.4k global tokens/s respectively. The GSM8K Dimuon row comes from a separate matched run at the selected LR (0.012); the earlier raw all-method file includes an untuned LR (0.003) arm and is retained as an ablation.

![Systems comparison](figures/systems.png)

| Optimizer | State / rank | Standard-LoRA global tok/s* |
| --- | ---: | ---: |
| AdamW | 1.61 GB | 42.6k |
| Muon | 0.81 GB | 29.0k |
| Dimuon | 0.40 GB | 38.4k |

`*`Mean of the clean 50-update GSM8K and MATH SFT runs; throughput is workload-specific, not a universal hardware peak.

## Replication checks

The additional matched two-seed 20-update panels are in
[`MULTI_SEED_PILOT.md`](MULTI_SEED_PILOT.md). At the fixed learning rates used
for each parameterization, Dimuon has the lowest mean held-out CE in both
standard LoRA and full NoRA on GSM8K and MATH. The IniLoRA-alpha panel also
favors Dimuon at its separately tuned lower learning rates. These are short
variance checks, not long-run convergence or a claim that one optimizer wins
for every learning-rate budget.

## RL status

The repository contains a real synchronous REINFORCE loop: sample completions,
score answers, recompute sampled sequence log-probabilities, and apply the
optimizer update. The current harness calls the DDP wrapper for the
gradient-bearing forward and checks adapter replication after every update.
The legacy pilots are not valid quality measurements. They are retained only
to document the DDP plumbing fix:

| Task | AdamW reward / exact | Muon reward / exact | Dimuon reward / exact |
| --- | --- | --- | --- |
| GSM8K | 0.0549 / 0% | 0.0806 / 0% | 0.0588 / 6.25% |
| MATH | 0.0929 / 12.5% | 0.0869 / 6.25% | 0.0850 / 6.25% |

These artifacts use 128 rollouts per optimizer, a 16-question held-out set,
and bounded numeric shaping. They must not be used for optimizer ranking. An
audit found that the model's inherited `top_k=20` and `repetition_penalty=1.05`
were active during sampling even though the gradient recomputation assumed
raw logits, and the symbolic-answer scorer could reduce an unboxed expression
to its last coefficient. A replacement protocol is required before reporting
RL quality.

A separate 32-update diagnostic seed had these raw outputs before that audit:

| Task | AdamW reward / exact | Muon reward / exact | Dimuon reward / exact |
| --- | --- | --- | --- |
| GSM8K | 0.0660 / 0% | 0.0690 / 0% | 0.0468 / 12.5% |
| MATH | 0.0720 / 0% | 0.0905 / 12.5% | 0.0796 / 12.5% |

Dimuon was faster than Muon in these runs, but the metrics are invalid for
quality comparison for the same reason. They are retained as provenance only.

`dimuon_lora_rl_v2.py` is the replacement strict harness. It uses an explicit
raw-policy generation configuration, the tokenizer chat template, complete
boxed answers, `math_verify` for symbolic/numeric equivalence, summed response
log-probabilities, a global previous-step baseline, full split-disjointness
checks, and a full adapter digest across ranks. Its six CPU protocol tests
pass; the 8-GPU smoke artifacts are protocol checks only and are not used as
quality evidence in the tables.

## Reproduce

The scripts expect a local model and the pinned question-only manifests. The commands below are the exact SFT workload shape used for the headline tables:

```bash
MODEL=/path/to/Qwen2.5-32B-Instruct
DATA=/path/to/manifests

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 dimuon_lora_benchmark.py \
  --model "$MODEL" \
  --manifest "$DATA/gsm8k_train.jsonl" \
  --eval-manifest "$DATA/gsm8k_val.jsonl" \
  --method all --init lora --steps 50 --warmup 10 \
  --batch 2 --seq 512 --rank 32 --last-n 64 \
  --adamw-lr 1e-4 --muon-lr 1e-3 --dimuon-lr 1.2e-2 \
  --grouped-dimuon --out results/my_gsm8k.json
```

Use `--manifest "$DATA/math_train.jsonl"` and `--eval-manifest "$DATA/math_val.jsonl"` for MATH. Use `--init nora` for full NoRA (continuous A normalization), `--init nora_init` for one-time NoRA-init, and `--init inilora_alpha` for IniLoRA-alpha.

Generate the README plots from committed JSON files with:

```bash
python make_figures.py
```

For a preregistered equal-effort comparison with validation-only LR selection
and held-out confirmation seeds, use [`run_study.py`](run_study.py) and read
[`STUDY_PROTOCOL.md`](STUDY_PROTOCOL.md). It plans by default; pass
`--execute` only after reviewing the generated plan.

The optimizer implementation is in [`dimuon.py`](dimuon.py); the SFT harness is [`dimuon_lora_benchmark.py`](dimuon_lora_benchmark.py); the legacy policy-gradient diagnostic is [`dimuon_lora_rl.py`](dimuon_lora_rl.py); the replacement protocol is [`dimuon_lora_rl_v2.py`](dimuon_lora_rl_v2.py), with CPU checks in [`test_rl_v2.py`](test_rl_v2.py).

## Scope and limitations

- The measured checkpoint is Qwen2.5-32B-Instruct, not DeepSeek-V4.1. The available local DeepSeek-V4.1-Flash implementation is inference-only and cannot provide a valid autograd training run.
- These SFT runs are short (50 updates for the headline standard-LoRA panel, 20 for the added replication checks and for NoRA/IniLoRA-alpha). The two-seed checks do not replace a longer equal-effort hyperparameter study. The CE metric includes the prompt/question text and short final answers, rather than measuring exact-answer accuracy or response-only reasoning quality.
- The RL comparison is currently invalid for quality ranking: the legacy artifacts have both a sampling-policy/log-probability mismatch and an incomplete symbolic-answer scorer. The DDP-forward synchronization fix itself passed, but a new sampler, verifier, and matched multi-seed protocol are required before RL claims.
- No model weights, datasets, checkpoints, or large logs are committed. See [`BENCHMARK_DETAILS.md`](BENCHMARK_DETAILS.md) for the complete protocol, tables, learning rates, and caveats.
