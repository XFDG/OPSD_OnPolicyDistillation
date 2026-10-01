"""Generate anonymous, relative timing figures from the adjacent public data.

Run: python3 plot_normalized_timings.py
"""
import json
import os
from pathlib import Path

directory = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads((directory / "normalized_timings.json").read_text())
fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), layout="constrained")
for ax, metric in zip(axes, data["metrics"], strict=True):
    values = [metric["baseline"], metric["optimized"]]
    ax.barh(["Baseline", "Optimized"], values, color=["#73859b", "#14877b"], height=0.52)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative time (baseline = 100)")
    ax.set_title(metric["label"], fontsize=11, loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(value + 2, i, f"{value:.2f}", va="center", fontsize=10)
    ax.text(0.98, 0.22, f"Time reduction: {metric['time_reduction_percent']:.2f}%\n"
            f"{metric['samples_per_variant']} samples per variant", transform=ax.transAxes,
            ha="right", va="center", fontsize=9, color="#334155")
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.suptitle("On-policy distillation: controlled short-run comparison", fontsize=13)
fig.savefig(directory / "normalized_timings.png", dpi=180)
svg = directory / "normalized_timings.svg"
fig.savefig(svg, metadata={"Date": None})
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
