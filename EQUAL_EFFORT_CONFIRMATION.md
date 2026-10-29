# Equal-effort confirmation study

This report records the completed confirmation phase of the preregistered study in [`STUDY_PROTOCOL.md`](STUDY_PROTOCOL.md). It is separate from the short historical pilot in `README.md` and does not tune on the test split.

## Protocol

- Local checkpoint: `Qwen2.5-32B-Instruct`; frozen base, rank-32 adapters on `gate_proj`, `up_proj`, and `down_proj` in the last 64 layers.
- 8× B300, BF16 model compute, per-GPU batch 2, sequence length 512, 200 updates (10 warmups), 32 held-out test batches.
- 54 tuning jobs (3 predeclared LR candidates × 18 cells), one tuning seed (`1234`), then 54 confirmation jobs at the frozen validation winner and three fresh seeds (`2345, 3456, 4567`).
- CE is teacher-forced CE over the harness's packed question/prompt and answer tokens. It is not exact-match accuracy or response-only CE.
- Throughput is global tokens/s, averaged over the three confirmation seeds. State is optimizer state on one replicated rank.
- Frozen plan hash: `80a382776c2f8f2e9eb9150ce36738705b4e8fd65b5d55c5ce795fd07ce2e913`; completion timestamp: `2026-10-29T14:04:47Z`.

## Held-out test CE (mean ± sample SD; lower is better)

| Initialization / split | AdamW | Muon | Dimuon |
| --- | ---: | ---: | ---: |
| Standard LoRA / GSM8K | 1.5293 ± 0.0012 | 1.4990 ± 0.0032 | 1.5314 ± 0.0019 |
| Standard LoRA / MATH | 0.4905 ± 0.0005 | 0.4946 ± 0.0007 | 0.5078 ± 0.0018 |
| Full NoRA / GSM8K | 1.4472 ± 0.0042 | 1.4242 ± 0.0084 | 1.3893 ± 0.0013 |
| Full NoRA / MATH | 0.5279 ± 0.0031 | 0.7021 ± 0.0192 | 0.5056 ± 0.0031 |
| IniLoRA-alpha / GSM8K | 1.4939 ± 0.0149 | 2.1378 ± 0.6797 | 1.5060 ± 0.0137 |
| IniLoRA-alpha / MATH | 0.7227 ± 0.0065 | 1.5610 ± 0.5389 | 0.7409 ± 0.0044 |

Dimuon is lowest in both Full NoRA cells. AdamW is lowest in the Standard LoRA and IniLoRA-alpha cells on this held-out test panel; the earlier short pilot used different validation-oriented artifacts and should not be conflated with this confirmation.

## Throughput and state

| Initialization / split | Optimizer | LR | Global tok/s (mean ± SD) | State / rank |
| --- | --- | ---: | ---: | ---: |
| Standard LoRA / GSM8K | AdamW | 3e-05 | 45.0k ± 2.1k | 1.61 GB |
| Standard LoRA / GSM8K | Muon | 0.0003 | 26.7k ± 2.1k | 0.81 GB |
| Standard LoRA / GSM8K | Dimuon | 0.003 | 37.2k ± 1.2k | 0.40 GB |
| Standard LoRA / MATH | AdamW | 3e-05 | 44.1k ± 1.0k | 1.61 GB |
| Standard LoRA / MATH | Muon | 0.0003 | 27.1k ± 2.1k | 0.81 GB |
| Standard LoRA / MATH | Dimuon | 0.003 | 39.4k ± 0.8k | 0.40 GB |
| Full NoRA / GSM8K | AdamW | 3e-05 | 35.7k ± 0.6k | 1.61 GB |
| Full NoRA / GSM8K | Muon | 0.0003 | 23.8k ± 0.9k | 0.81 GB |
| Full NoRA / GSM8K | Dimuon | 0.001 | 32.2k ± 2.9k | 0.40 GB |
| Full NoRA / MATH | AdamW | 3e-05 | 36.6k ± 2.0k | 1.61 GB |
| Full NoRA / MATH | Muon | 0.0003 | 24.0k ± 1.0k | 0.81 GB |
| Full NoRA / MATH | Dimuon | 0.001 | 31.2k ± 1.7k | 0.40 GB |
| IniLoRA-alpha / GSM8K | AdamW | 3e-05 | 42.5k ± 2.7k | 1.61 GB |
| IniLoRA-alpha / GSM8K | Muon | 0.0003 | 27.6k ± 1.1k | 0.81 GB |
| IniLoRA-alpha / GSM8K | Dimuon | 0.001 | 37.8k ± 1.0k | 0.40 GB |
| IniLoRA-alpha / MATH | AdamW | 3e-05 | 44.2k ± 1.3k | 1.61 GB |
| IniLoRA-alpha / MATH | Muon | 0.0003 | 27.4k ± 0.3k | 0.81 GB |
| IniLoRA-alpha / MATH | Dimuon | 0.001 | 38.2k ± 1.4k | 0.40 GB |

## Per-seed test CE

| Initialization / split | Optimizer | LR | 2345 | 3456 | 4567 |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard LoRA / GSM8K | AdamW | 3e-05 | 1.5300 | 1.5280 | 1.5299 |
| Standard LoRA / GSM8K | Muon | 0.0003 | 1.5016 | 1.5000 | 1.4954 |
| Standard LoRA / GSM8K | Dimuon | 0.003 | 1.5325 | 1.5292 | 1.5325 |
| Standard LoRA / MATH | AdamW | 3e-05 | 0.4900 | 0.4911 | 0.4905 |
| Standard LoRA / MATH | Muon | 0.0003 | 0.4937 | 0.4951 | 0.4949 |
| Standard LoRA / MATH | Dimuon | 0.003 | 0.5063 | 0.5074 | 0.5098 |
| Full NoRA / GSM8K | AdamW | 3e-05 | 1.4427 | 1.4477 | 1.4511 |
| Full NoRA / GSM8K | Muon | 0.0003 | 1.4145 | 1.4290 | 1.4291 |
| Full NoRA / GSM8K | Dimuon | 0.001 | 1.3881 | 1.3907 | 1.3891 |
| Full NoRA / MATH | AdamW | 3e-05 | 0.5260 | 0.5262 | 0.5315 |
| Full NoRA / MATH | Muon | 0.0003 | 0.6906 | 0.7243 | 0.6914 |
| Full NoRA / MATH | Dimuon | 0.001 | 0.5092 | 0.5039 | 0.5037 |
| IniLoRA-alpha / GSM8K | AdamW | 3e-05 | 1.4947 | 1.5084 | 1.4786 |
| IniLoRA-alpha / GSM8K | Muon | 0.0003 | 1.7744 | 1.7171 | 2.9219 |
| IniLoRA-alpha / GSM8K | Dimuon | 0.001 | 1.4926 | 1.5200 | 1.5053 |
| IniLoRA-alpha / MATH | AdamW | 3e-05 | 0.7210 | 0.7172 | 0.7298 |
| IniLoRA-alpha / MATH | Muon | 0.0003 | 2.1694 | 1.3700 | 1.1438 |
| IniLoRA-alpha / MATH | Dimuon | 0.001 | 0.7446 | 0.7360 | 0.7421 |

## Interpretation and limits

- These are descriptive three-seed results on a fixed local checkpoint and a fixed 32-batch test slice. They establish a reproducible comparison under this protocol, not a universal optimizer ranking.
- Test rows use frozen validation-selected learning rates. No test loss was used to choose a learning rate, seed, parameterization, or stopping point.
- The model is Qwen2.5-32B-Instruct; DeepSeek-V4.1 was unavailable for autograd training in this environment.
- `results/equal_effort_confirmation.json` and `.csv` contain the compact machine-readable rows. Raw job receipts and full loss histories remain outside the repository.
