#!/usr/bin/env python
"""Training dynamics from train_log.jsonl files (aggregates only) -> docs/figures/fig_training.png|svg.

Panels (shared x = GRPO step): in-set deferral rate, dev accuracy on committed cases, mean
reward, OOD deferral rate. Batch-level series are smoothed with a centred moving average;
dev-set points (every eval_every steps) are drawn as markers. Used to show, run against
run, whether deferral collapses (Che et al. 2026) or stays bounded.

    python scripts/plot_training.py --runs deferdx=outputs/runs/deferdx abl_std_norm=outputs/runs/abl_std_norm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LABELS = {"deferdx": "DEFER-Dx", "nodefer": "GRPO control (no DEFER)", "abl_std_norm": "+ group std normalisation",
          "abl_cdm_only": "CDM-only training", "abl_no_constraint": "no coverage constraint",
          "abl_forced_loo": "leave-one-out p-hat", "deferdx_seed1": "DEFER-Dx, seed 1"}


def smooth(y, k=9):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")


def load(path: Path):
    rows = [json.loads(line) for line in (path / "train_log.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if "reward_mean" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "axes.titlesize": 10, "axes.titlecolor": INK})
    panels = [("inset_defer_rate", "In-set deferral rate (batch)", True), ("dev_selective_acc", "Dev accuracy on committed cases", False),
              ("reward_mean", "Mean reward (batch)", True), ("ood_defer_rate", "Out-of-set deferral rate (batch)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True)
    for i, item in enumerate(args.runs):
        name, path = item.split("=", 1)
        rows = load(Path(path))
        if not rows:
            continue
        color = SLOTS[i % len(SLOTS)]
        for ax, (key, title, batch) in zip(axes.flat, panels):
            pts = [(r["step"], r[key]) for r in rows if key in r and r[key] == r[key]]
            if not pts:
                continue
            x, y = zip(*pts)
            if batch:
                ax.plot(x, np.asarray(y) * (100 if "rate" in key else 1), color=color, lw=0.6, alpha=0.25)
                ax.plot(x, smooth(y) * (100 if "rate" in key else 1), color=color, lw=1.8, label=LABELS.get(name, name))
            else:
                ax.plot(x, np.asarray(y) * 100, color=color, lw=1.8, marker="o", ms=5, mec=SURFACE, mew=1.2,
                        label=LABELS.get(name, name))
            ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("GRPO step")
    axes[0, 0].legend(fontsize=8.5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"fig_training.{ext}", dpi=220, bbox_inches="tight")
    print(f"-> {out / 'fig_training.png'}")


if __name__ == "__main__":
    main()
