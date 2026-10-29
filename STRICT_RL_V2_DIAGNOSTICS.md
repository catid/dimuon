# Strict RL v2 diagnostics

These are matched, single-seed protocol diagnostics for the corrected
REINFORCE harness. They are separate from the invalid legacy RL artifacts and
are not a multi-seed optimizer quality claim.

## Fixed workload

- Qwen2.5-32B-Instruct, frozen base, rank-32 FFN LoRA on the last 64 layers.
- 8× B300, BF16 model compute, per-GPU batch 2, 32 updates, 512 global
  samples, 64 held-out validation questions.
- Prompt cap 768 tokens, response cap 256 tokens, seed 2345.
- Raw temperature-1 sampling (`top_k=0`, `top_p=1`, repetition penalty 1),
  complete boxed answers, strict `math_verify`, summed response
  log-probabilities, and a global previous-step reward baseline.
- Learning rates: AdamW `3e-5`, Muon `3e-4`, Dimuon `3e-3`.

## Results

Accuracy is strict verified held-out exact match over 64 questions. The train
reward is the mean binary verifier reward over the 512 sampled training
completions. RL sample throughput includes generation, verification, and
gradient updates; it is not comparable to teacher-forced SFT tok/s.

| Task | Optimizer | Pre accuracy | Post accuracy | Train reward | RL sample tok/s | State/rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | AdamW | 28.1% | 84.4% (54/64) | 0.475 | 249 | 1.61 GB |
| GSM8K | Muon | 28.1% | 98.4% (63/64) | 0.613 | 241 | 0.81 GB |
| GSM8K | Dimuon | 28.1% | 93.8% (60/64) | 0.533 | 253 | 0.40 GB |
| MATH | AdamW | 15.6% | 82.8% (53/64) | 0.549 | 205 | 1.61 GB |
| MATH | Muon | 15.6% | 93.8% (60/64) | 0.611 | 220 | 0.81 GB |
| MATH | Dimuon | 15.6% | 89.1% (57/64) | 0.428 | 239 | 0.40 GB |

Both runs passed adapter SHA synchronization across all eight ranks and had
no nonfinite checks. These results show that the corrected RL path runs on
both tasks and preserves Dimuon's memory advantage; one seed and 64 evaluation
questions are insufficient to rank optimizer quality. The raw small JSON
artifacts are [`results/gsm8k_rlv2_32.json`](results/gsm8k_rlv2_32.json) and
[`results/math_rlv2_32.json`](results/math_rlv2_32.json).
