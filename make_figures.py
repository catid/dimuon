#!/usr/bin/env python3
"""Generate the compact figures used by README.md from committed JSON results."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load(name):
    with (RESULTS / name).open() as f:
        return json.load(f)


def by_method(name):
    return {row["method"]: row for row in load(name)["results"]}

std_gsm = by_method("gsm8k_standard_lora_50.json")
std_math = by_method("math_standard_lora_50.json")
nora_gsm = by_method("gsm8k_nora_20.json")
nora_math = by_method("math_nora_20.json")
ini_gsm = by_method("gsm8k_inilora_alpha_20.json")

labels = ["adamw", "muon", "dimuon"]
display_labels = ["AdamW", "Muon", "Dimuon"]
colors = ["#64748b", "#f59e0b", "#2563eb"]

# Headline quality figure: each method is always compared in the same order.
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), dpi=160)
fig.patch.set_facecolor("white")
quality = [
    ("Standard LoRA", std_gsm, std_math),
    ("Full NoRA", nora_gsm, nora_math),
    ("IniLoRA-alpha", ini_gsm, None),
]
for ax, (title, gsm, math_run) in zip(axes, quality):
    x = np.arange(3)
    width = 0.36
    gsm_vals = [gsm[m]["eval_loss"] for m in labels]
    ax.bar(x - (width / 2 if math_run else 0), gsm_vals, width if math_run else 0.58,
           label="GSM8K", color=colors, alpha=0.9)
    if math_run:
        math_vals = [math_run[m]["eval_loss"] for m in labels]
        ax.bar(x + width / 2, math_vals, width, label="MATH", color=colors, alpha=0.42,
               hatch="//", edgecolor="#334155")
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xticks(x, display_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("held-out CE" if ax is axes[0] else "")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    vals = gsm_vals + ([math_run[m]["eval_loss"] for m in labels] if math_run else [])
    ax.set_ylim(0, max(vals) * 1.22)
    for i, v in enumerate(gsm_vals):
        xpos = i - (width / 2 if math_run else 0)
        ax.text(xpos, v + max(vals) * .025, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    if math_run:
        for i, v in enumerate(math_vals):
            ax.text(i + width / 2, v + max(vals) * .025, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Qwen2.5-32B FFN LoRA: lower is better", fontsize=13, weight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "quality.png", bbox_inches="tight")
plt.close(fig)

# Headline systems figure: state is per replicated rank; throughput is global.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=160)
fig.patch.set_facecolor("white")
x = np.arange(3)
state_gb = [std_gsm[m]["optimizer_state_bytes"] / 1e9 for m in labels]
axes[0].bar(x, state_gb, color=colors)
axes[0].set_title("Optimizer state / rank", fontsize=10, weight="bold")
axes[0].set_ylabel("GB")
axes[0].set_xticks(x, display_labels)
axes[0].grid(axis="y", alpha=.22); axes[0].set_axisbelow(True)
for i, v in enumerate(state_gb): axes[0].text(i, v + .04, f"{v:.2f}", ha="center", fontsize=8)
# Standard LoRA throughput, averaged across the two datasets for readability.
tok = [np.mean([std_gsm[m]["tokens_per_s"], std_math[m]["tokens_per_s"]]) for m in labels]
axes[1].bar(x, tok, color=colors)
axes[1].set_title("Standard LoRA throughput", fontsize=10, weight="bold")
axes[1].set_ylabel("global tokens/s")
axes[1].set_xticks(x, display_labels)
axes[1].grid(axis="y", alpha=.22); axes[1].set_axisbelow(True)
for i, v in enumerate(tok): axes[1].text(i, v + max(tok)*.025, f"{v/1000:.1f}k", ha="center", fontsize=8)
fig.suptitle("8× B300 · 64 layers · rank-32 FFN adapters", fontsize=12, weight="bold", y=1.03)
fig.tight_layout()
fig.savefig(FIGURES / "systems.png", bbox_inches="tight")
plt.close(fig)
print("wrote", FIGURES / "quality.png", FIGURES / "systems.png")
