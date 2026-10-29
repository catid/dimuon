# Committed result artifacts

The JSON files here contain compact rank-0 outputs from the 8×B300 Qwen2.5-32B runs. They contain no model weights, checkpoints, token streams, or rollout transcripts.

- `*_standard_lora_tuned_50.json`: rank-32 FFN LoRA, all 64 layers, 50 SFT updates; GSM8K combines the matched AdamW/Muon arms with the separately tuned Dimuon LR arm and records that selection note.
- `math_standard_lora_50.json`: rank-32 FFN LoRA, all 64 layers, 50 SFT updates.
- `*_nora_20.json`: paper-style NoRA, normalizing A's columns on every forward, 20 SFT updates.
- `*_inilora_alpha_20.json`: IniLoRA-alpha, 20 GSM8K SFT updates.
- `*_standard_seed{0,5678}_20.json`: two-seed 20-update standard-LoRA replication checks.
- `*_standard_seed5678_50.json`: second 50-update standard-LoRA seed at the headline workload.
- `*_nora_seed{0,5678}_20.json`: two-seed 20-update full-NoRA checks at the fixed NoRA learning rates.
- `gsm8k_inilora_alpha_seed{0,5678}_20.json`: two-seed lower-learning-rate IniLoRA-alpha checks.
- `*_rl16_ddpfix.json`: corrected 16-update REINFORCE plumbing pilots with bounded numeric shaping and an adapter-replication check after every DDP update. **Do not use their reward/exact metrics for quality ranking**: generation and gradient policies differ, and symbolic scoring is incomplete.
- `*_rl32_seed5678.json`: corrected 32-update second-seed provenance pilots. They have the same sampling/scoring invalidity and must not be used for optimizer ranking.
- `*_rl16_diagnostic.json`: legacy pre-fix pilots retained as provenance only; do not use them for optimizer ranking.

The replacement strict RL protocol is `dimuon_lora_rl_v2.py`; its CPU protocol
tests are in `test_rl_v2.py`. No v2 quality artifact is committed yet.

`gsm8k_rlv2_smoke16.json` and `math_rlv2_smoke16.json` are 8-GPU protocol
smokes (16 updates, 16 held-out rows), not optimizer-quality benchmarks.

All losses are teacher-forced held-out cross-entropy except the RL files, which report shaped training reward and binary greedy exact match.
