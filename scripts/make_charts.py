"""Generate the two supplementary charts attached to the LinkedIn post / model card.

1. training_loss.png   — eval_loss vs step across the 12 mid-training checkpoints.
2. eval_comparison.png — base vs tuned across ROUGE-L, chrF, sacreBLEU, pairwise win-rate.

Inputs are taken directly from eval/results/*.json + the eval_loss table in the
HF model card. Outputs go to eval/charts/.

Run:  py -3.12 scripts/make_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
CHARTS = ROOT / "eval" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

EVAL_LOSS = [
    (200, 0.7670), (400, 0.7019), (600, 0.6689), (800, 0.6447),
    (1000, 0.6324), (1200, 0.6178), (1400, 0.6081), (1600, 0.5982),
    (1800, 0.5947), (2000, 0.5913), (2200, 0.5885), (2400, 0.5874),
    (2532, 0.5871),
]


def chart_training_loss() -> Path:
    steps = [s for s, _ in EVAL_LOSS]
    losses = [l for _, l in EVAL_LOSS]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.plot(steps, losses, marker="o", linewidth=2, color="#1f4e79", markerfacecolor="#1f4e79")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("eval_loss (held-out val, n=1500)", fontsize=11)
    ax.set_title(
        "Training: eval_loss 0.767 → 0.587 (−23.5%), monotonic across 3 epochs",
        fontsize=12, pad=12,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 2700)

    ax.annotate(
        "start: 0.767",
        xy=(200, 0.7670),
        xytext=(360, 0.760),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#777", lw=0.8),
    )
    ax.annotate(
        "best: 0.587\n(step 2532, end of 3 epochs)",
        xy=(2532, 0.5871),
        xytext=(1700, 0.620),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#777", lw=0.8),
    )

    fig.tight_layout()
    out = CHARTS / "training_loss.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024:.1f} KB)")
    return out


def chart_eval_comparison() -> Path:
    tb = json.loads((RESULTS / "test_base.json").read_text(encoding="utf-8"))["metrics"]
    tt = json.loads((RESULTS / "test_tuned.json").read_text(encoding="utf-8"))["metrics"]
    pw = json.loads((RESULTS / "compare_test_base_vs_test_tuned.json").read_text(encoding="utf-8"))

    metrics = ["ROUGE-L F1", "chrF", "sacreBLEU"]
    base_vals = [tb["rougeL_f1"], tb["chrF"], tb["bleu"]]
    tuned_vals = [tt["rougeL_f1"], tt["chrF"], tt["bleu"]]

    # Each metric has a different scale — show three small grouped-bar panels
    # plus a fourth panel for the pairwise outcome split.
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5), dpi=140)

    for ax, name, b, t in zip(axes[:3], metrics, base_vals, tuned_vals):
        bars = ax.bar(["base", "tuned"], [b, t],
                      color=["#999999", "#1f4e79"], edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        for rect, val in zip(bars, [b, t]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{val:.2f}" if val > 1 else f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9)
        rel = (t - b) / b * 100 if b else 0
        ax.text(0.5, -0.18, f"Δ {rel:+.0f}%",
                ha="center", transform=ax.transAxes, fontsize=10, color="#1f4e79", weight="bold")

    ax = axes[3]
    outcomes = ["tuned wins", "ties", "base wins"]
    counts = [pw["wins_b"], pw["ties"], pw["wins_a"]]
    colors = ["#1f4e79", "#aaaaaa", "#999999"]
    bars = ax.bar(outcomes, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("GPT-4o pairwise (n=100)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    for rect, c in zip(bars, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{c}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.text(0.5, -0.18, "tuned preferred 3.7× over base",
            ha="center", transform=ax.transAxes, fontsize=10, color="#1f4e79", weight="bold")

    fig.suptitle(
        "Held-out test split (n=500): tuned beats base across every automatic metric and a head-to-head LLM judge",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = CHARTS / "eval_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024:.1f} KB)")
    return out


if __name__ == "__main__":
    chart_training_loss()
    chart_eval_comparison()
