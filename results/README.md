# Committed result artifacts

The JSON files here contain compact rank-0 outputs from the 8×B300 Qwen2.5-32B runs. They contain no model weights, checkpoints, token streams, or rollout transcripts.

- `*_standard_lora_tuned_50.json`: rank-32 FFN LoRA, all 64 layers, 50 SFT updates; GSM8K combines the matched AdamW/Muon arms with the separately tuned Dimuon LR arm and records that selection note.
- `math_standard_lora_50.json`: rank-32 FFN LoRA, all 64 layers, 50 SFT updates.
- `*_nora_20.json`: paper-style NoRA, normalizing A's columns on every forward, 20 SFT updates.
- `*_inilora_alpha_20.json`: IniLoRA-alpha, 20 GSM8K SFT updates.
- `*_rl16_ddpfix.json`: corrected 16-update REINFORCE pilots with bounded numeric shaping and an adapter-replication check after every DDP update. They remain diagnostic, not a quality ranking.
- `*_rl16_diagnostic.json`: legacy pre-fix pilots retained as provenance only; do not use them for optimizer ranking.

All losses are teacher-forced held-out cross-entropy except the RL files, which report shaped training reward and binary greedy exact match.
