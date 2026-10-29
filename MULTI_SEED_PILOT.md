# Two-seed standard-LoRA pilot

This note records the additional matched 20-update runs completed after the
headline 50-update benchmark. It is a small variance check, not the full
equal-effort study: both seeds use the same learning rates (`AdamW 1e-4`,
`Muon 1e-3`, `Dimuon 1.2e-2`), rank-32 FFN adapters in all 64 layers, per-GPU
batch 2, sequence length 512, BF16 model compute, and all 8 B300 GPUs.

Held-out teacher-forced cross-entropy (lower is better):

| Task | AdamW mean ± SD | Muon mean ± SD | Dimuon mean ± SD |
| --- | ---: | ---: | ---: |
| GSM8K (seeds 0, 5678) | 0.21500 ± 0.00032 | 0.22612 ± 0.00022 | **0.21171 ± 0.00027** |
| MATH (seeds 0, 5678) | 0.24445 ± 0.00001 | 0.25733 ± 0.00104 | **0.24414 ± 0.00013** |

Mean global training throughput, in tokens/s, was:

| Task | AdamW | Muon | Dimuon |
| --- | ---: | ---: | ---: |
| GSM8K | 39,905 | 28,377 | 37,625 |
| MATH | 41,694 | 28,593 | 38,362 |

The raw records are `results/gsm8k_standard_seed0_20.json`,
`results/gsm8k_standard_seed5678_20.json`,
`results/math_standard_seed0_20.json`, and
`results/math_standard_seed5678_20.json`. The seed-0/5678 runs are each only
20 updates and therefore should be treated as a reproducibility check. They do
not establish long-run convergence, broad hyperparameter optimality, or an RL
quality result.

An additional matched 50-update seed (5678) reproduced the headline result:

| Task | AdamW CE | Muon CE | Dimuon CE |
| --- | ---: | ---: | ---: |
| GSM8K | 0.185599 | 0.187262 | **0.180645** |
| MATH | 0.220302 | 0.223807 | **0.218737** |

The corresponding global throughput was 43.4k/29.4k/39.6k tokens/s on GSM8K
and 42.2k/26.8k/35.3k on MATH (AdamW/Muon/Dimuon). This is a second seed at
the headline budget, but it is still not a substitute for the preregistered
multi-candidate, three-confirmation-seed study in `STUDY_PROTOCOL.md`.

Additional seed-0 parameterization checks used the LRs selected for the
corresponding earlier panels. Full NoRA reached held-out CE 0.207997, 0.216084,
and 0.204235 on GSM8K (AdamW/Muon/Dimuon), and 0.246120, 0.269842, and
0.235631 on MATH. The lower-LR IniLoRA-alpha panel reached 0.718389,
0.896113, and 0.714262 on GSM8K. These are single-seed checks for those
parameterizations; IniLoRA changes the effective BF16 starting model and should
not be compared directly with standard LoRA.

A second GSM8K NoRA seed (5678) reached 0.206272, 0.218253, and 0.205329,
respectively. Across these two NoRA seeds, the means are 0.20713 (AdamW),
0.21717 (Muon), and **0.20478 (Dimuon)**. This remains a 20-update pilot with
only two seeds and is not a long-run NoRA result.

The second MATH NoRA seed (5678) reached 0.244564, 0.268502, and 0.234305;
the two-seed means are 0.24534, 0.26917, and **0.23501**. Dimuon is lower in
both seeds on both datasets at these fixed NoRA learning rates.

The lower-LR IniLoRA-alpha panel also has a second seed (5678): held-out GSM8K
CE was 0.730016, 0.876015, and 0.719011. Across seeds 0 and 5678 the means
are 0.72420, 0.88606, and **0.71664**. This is an internally matched
IniLoRA-alpha comparison; its BF16 residual-subtraction initialization changes
the starting model relative to standard LoRA.

Corrected DDP RL follow-ups also completed at seed 5678 with 32 updates and 16
held-out questions. Raw shaped reward / exact-match outputs are retained in
`results/gsm8k_rl32_seed5678.json` and `results/math_rl32_seed5678.json` as
provenance. They must not be used as optimizer-quality evidence: an audit found
that generation inherited `top_k=20` and `repetition_penalty=1.05` while the
backward pass used raw-logit log-probabilities, and symbolic MATH scoring could
accept an incomplete expression by its last coefficient. The DDP adapter
replication check itself passed. A replacement sampler/verifier is required
before a valid RL comparison.

The replacement strict harness was smoke-tested on all 8 B300s for 16 updates,
256 training samples, and 16 held-out questions per task. It uses raw-policy
sampling and `math_verify`; these are protocol and plumbing checks, not a tuned
RL comparison:

| Task | AdamW post exact | Muon post exact | Dimuon post exact | Generated tok/s (A/M/D) |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 16/16 | 15/16 | 16/16 | 178 / 173 / 194 |
| MATH | 16/16 | 16/16 | 16/16 | 154 / 178 / 174 |

All methods had complete answers on every held-out row after training. The
small evaluation set and 16-update budget make these unsuitable for ranking;
the artifacts are `results/gsm8k_rlv2_smoke16.json` and
`results/math_rlv2_smoke16.json`.
