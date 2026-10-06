#!/usr/bin/env python
"""Paper figures from evaluation-suite outputs (aggregates only) -> docs/figures/*.png|svg.

  fig_risk_coverage   selective accuracy vs coverage on CDM val + test (the headline figure)
  fig_open_world      outcomes on OTHER cases from time-critical groups never seen in training
  fig_reliability     stated probability vs observed accuracy on committed cases (one panel per system)
  fig_per_class       accuracy at full coverage by condition

Palette: the validated four-slot categorical order (blue, orange, aqua, yellow; adjacent
CVD dE >= 9.1). Aqua and yellow sit below 3:1 on the light surface, so every series also
carries a direct label and a distinct marker, and docs/RESULTS.md is the table view.

    python scripts/make_figures.py --eval-dir outputs/eval --out docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deferdx.eval.evaluate import selective_score  # noqa: E402
from deferdx.eval.report import cdm_pooled, load_suite  # noqa: E402
from deferdx.labels import IN_SET_LABELS, OTHER  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order, never cycled
MARKERS = ["o", "s", "^", "D"]
# fixed system -> slot assignment (colour follows the entity, never its rank)
SYSTEMS = [("zs_nodefer", "Zero-shot (forced choice)"), ("grpo_nodefer", "GRPO control (no DEFER)"),
           ("deferdx", "DEFER-Dx"), ("zs_defer", "Zero-shot, prompted DEFER")]
STYLE = {name: (SLOTS[i], MARKERS[i]) for i, (name, _) in enumerate(SYSTEMS)}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titleweight": "regular", "axes.titlesize": 11, "axes.titlecolor": INK, "axes.axisbelow": True,
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rc_curve(results):
    """(coverage, selective accuracy) along the stated-probability ranking; deferrals rank last
    and never count as covered."""
    ins = [r for r in results if r.label != OTHER]
    s = np.array([selective_score(r) for r in ins])
    c = np.array([r.correct for r in ins], dtype=float)  # only commits can be correct
    order = np.argsort(-s, kind="stable")
    committed = np.array([r.terminal == "commit" for r in ins])[order]
    k = np.cumsum(committed)
    cov = k / len(ins)
    acc = np.cumsum(c[order]) / np.maximum(k, 1)
    keep = committed & (k >= max(5, int(0.05 * len(ins))))
    return cov[keep], acc[keep]


def fig_risk_coverage(suites, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, label in SYSTEMS:
        if name not in suites or name == "zs_defer":
            continue
        color, marker = STYLE[name]
        cov, acc = rc_curve(cdm_pooled(suites[name]))
        ax.plot(cov * 100, acc * 100, color=color, lw=1.8, solid_capstyle="round", solid_joinstyle="round",
                label=label)
        ax.plot(cov[-1] * 100, acc[-1] * 100, marker=marker, ms=7, color=color, mec=SURFACE, mew=1.5, ls="none")
        ax.annotate(label, (cov[-1] * 100, acc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    if "zs_defer" in suites:  # prompted deferral is a single operating point
        color, marker = STYLE["zs_defer"]
        pooled = [r for r in cdm_pooled(suites["zs_defer"])]
        cov = np.mean([r.terminal == "commit" for r in pooled])
        acc = np.mean([r.correct for r in pooled if r.terminal == "commit"])
        ax.plot(cov * 100, acc * 100, marker=marker, ms=7, color=color, mec=SURFACE, mew=1.5, ls="none",
                label=dict(SYSTEMS)["zs_defer"])
        ax.annotate(dict(SYSTEMS)["zs_defer"], (cov * 100, acc * 100), xytext=(-8, -2), textcoords="offset points",
                    color=INK2, fontsize=8.5, ha="right", va="top")
    ax.set_xlabel("Coverage: share of in-set cases the agent commits on (%)")
    ax.set_ylabel("Accuracy on committed cases (%)")
    ax.set_title("Risk-coverage on MIMIC-IV-Ext-CDM (val + test, 480 cases)", loc="left")
    ax.set_xlim(40, 101)
    ax.legend(loc="lower left", fontsize=8.5)
    save(fig, out, "fig_risk_coverage")


def fig_open_world(suites, out, set_name="eval_other_unseen", title="Time-critical presentations never seen in training"):
    # colour follows the outcome's identity (slot order), stacking order follows the story
    cats = [("false commit", 1, lambda r: r.terminal == "commit" and r.diagnosis != OTHER),
            ("defer to clinician", 0, lambda r: r.terminal == "defer"),
            ("commit OTHER", 2, lambda r: r.terminal == "commit" and r.diagnosis == OTHER),
            ("invalid", 3, lambda r: r.terminal in ("invalid", "timeout"))]
    rows = [(n, l) for n, l in SYSTEMS if n in suites and suites[n].get(set_name)]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(rows) + 1.3))
    for y, (name, label) in enumerate(rows):
        rs = suites[name][set_name]
        left = 0.0
        for cname, slot, f in cats:
            v = 100 * np.mean([f(r) for r in rs])
            ax.barh(y, v, left=left, height=0.5, color=SLOTS[slot], edgecolor=SURFACE, linewidth=1.5,
                    label=cname if y == 0 else None)
            if v >= 7:  # light fills (aqua, yellow) take dark ink; dark fills take the surface colour
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=INK if slot >= 2 else SURFACE)
            left += v
    ax.set_yticks(range(len(rows)), [l for _, l in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of out-of-set cases (%)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.28), fontsize=8.5)
    save(fig, out, f"fig_open_world_{set_name.replace('eval_', '')}")


def fig_reliability(suites, out, bins=10):
    rows = [(n, l) for n, l in SYSTEMS if n in suites and n != "zs_defer"]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, label) in zip(axes, rows):
        c = [r for r in cdm_pooled(suites[name]) if r.terminal == "commit" and r.probability is not None]
        p = np.array([r.probability for r in c])
        y = np.array([r.correct for r in c], dtype=float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        mids, accs, ns = [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() >= 10:  # bins with fewer than 10 episodes are not drawn
                mids.append(p[m].mean() * 100)
                accs.append(y[m].mean() * 100)
                ns.append(m.sum())
        ax.plot([0, 100], [0, 100], color=AXIS, lw=1.0)
        color, marker = STYLE[name]
        ax.plot(mids, accs, color=color, lw=1.8, marker=marker, ms=6, mec=SURFACE, mew=1.2)
        ece = np.sum([n * abs(a - m) for n, a, m in zip(ns, accs, mids)]) / max(1, sum(ns)) / 100
        ax.set_title(f"{label}\nECE {ece:.3f}", loc="left", fontsize=9.5)
        ax.set_xlabel("Stated probability (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Observed accuracy (%)")
    save(fig, out, "fig_reliability")


def fig_per_class(suites, out):
    rows = [(n, l) for n, l in SYSTEMS if n in suites]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    width = 0.8 / len(rows)
    x = np.arange(len(IN_SET_LABELS))
    for i, (name, label) in enumerate(rows):
        pooled = cdm_pooled(suites[name])
        vals = [100 * np.mean([r.forced_prediction == r.label for r in pooled if r.label == lab]) for lab in IN_SET_LABELS]
        color, _ = STYLE[name]
        ax.bar(x + (i - (len(rows) - 1) / 2) * width, vals, width * 0.9, color=color, edgecolor=SURFACE,
               linewidth=1.5, label=label)
    ax.set_xticks(x, [lab.capitalize() for lab in IN_SET_LABELS])
    ax.set_ylabel("Accuracy at full coverage (%)")
    ax.set_ylim(40, 100)
    ax.grid(axis="x", visible=False)
    ax.set_title("Accuracy by condition (CDM val + test; forced prediction for deferrals)", loc="left")
    ax.legend(ncol=2, fontsize=8.5, loc="lower right")
    save(fig, out, "fig_per_class")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    setup()
    ev = Path(args.eval_dir)
    suites = {n: load_suite(ev / n) for n, _ in SYSTEMS if (ev / n).exists()}
    out = Path(args.out)
    fig_risk_coverage(suites, out)
    fig_open_world(suites, out)
    fig_open_world(suites, out, "eval_other_seen", "Out-of-set presentations from groups seen in training")
    fig_reliability(suites, out)
    fig_per_class(suites, out)
    print(f"figures -> {out} ({', '.join(suites) or 'no systems yet'})")


if __name__ == "__main__":
    main()
