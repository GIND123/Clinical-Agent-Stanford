#!/usr/bin/env python
"""Publication figures for DEFER-Dx -> docs/figures/<name>.{pdf,png,svg}. Aggregates only.

Data figures (this script; make_report.py and the GPU queue rerun it as results arrive):
  fig_risk_coverage        a: accuracy on answered cases vs coverage, CDM val + test; b: AURC, 95% CI
  fig_paired_differences   paired bootstrap differences, the reference system minus each comparator
  fig_clinical_safety      unflagged, confident and severity-weighted errors; tests per case
  fig_open_world           OTHER cases from seen groups and from unseen time-critical groups
  fig_training_dynamics    coverage multiplier, deferral rates and dev coverage during training
  fig_reliability          stated probability vs observed accuracy on committed cases
  fig_per_class            accuracy at full coverage by condition
  fig_ablations            DEFER-Dx at step 100 against its two ablations
  fig_unseen_groups        false commits and deferral by unseen time-critical diagnosis group
Schematics (TikZ sources in docs/figures/tikz/, built by scripts/build_tikz.sh): fig_overview,
fig_cev, fig_data, fig_training.

A figure whose inputs do not exist yet is written as a labelled placeholder, so links stay valid.
Colour follows the system in every figure (the validated categorical order of the dataviz method;
DEFER-Dx is always blue). Every series is also named by a row label, legend or direct label and has
its own marker, so colour never carries identity alone; docs/RESULTS.md is the table view. PDFs embed
TrueType (Type 42) fonts, never Type 3. Self-consistency and SGR rows stay in the tables only.

    python scripts/make_figures.py [--eval-dir outputs/eval] [--results docs/results.json] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.evaluate import selective_score  # noqa: E402
from deferdx.eval.report import cdm_pooled, load_groups, load_suite, suite_complete  # noqa: E402
from deferdx.labels import IN_SET_LABELS, OTHER  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FULL = 7.0  # inches: double-column width

SYSTEMS = {  # eval-suite name: (label, colour slot, marker). Colour follows the system in every figure.
    "cev": ("DEFER-Dx (ours)", 0, "o"),
    "deferdx": ("Group-consensus reward", 1, "s"),
    "grpo_nodefer": ("GRPO control (no DEFER)", 2, "^"),
    "zs_nodefer": ("Zero-shot, forced choice", 3, "D"),
    "zs_defer": ("Zero-shot, prompted DEFER", 4, "v"),
    "abl_cev_no_handoff": ("No scored handoff", 5, "P"),
    "cev_dualascent": ("DEFER-Dx, dual-ascent cap", 6, "X"),
    "abl_constant_defer": ("Constant deferral reward", 7, "h"),
    "cev_step100": ("DEFER-Dx, step 100", 0, "o"),
}
SHORT = {"grpo_nodefer+thr": "GRPO control\n+ threshold", "zs_nodefer+thr": "zero-shot\n+ threshold",
         "deferdx": "group-consensus\nreward", "zs_defer": "zero-shot,\nprompted DEFER", "grpo_nodefer": "GRPO control",
         "cev_dualascent": "dual-ascent\ncoverage cap"}  # paired-difference panel titles
MAIN_ROWS = ["cev", "deferdx", "grpo_nodefer+thr", "grpo_nodefer", "zs_nodefer+thr", "zs_nodefer", "zs_defer"]
CURVE_SYSTEMS = ["cev", "deferdx", "grpo_nodefer", "zs_nodefer", "zs_defer"]
GROUP_NAMES = {"mesenteric_ischemia": "Mesenteric ischaemia", "peptic_ulcer_perforation_or_bleed": "Perforated or bleeding ulcer",
               "abdominal_aortic_aneurysm": "Abdominal aortic aneurysm", "diabetic_ketoacidosis": "Diabetic ketoacidosis",
               "ectopic_pregnancy": "Ectopic pregnancy"}


def style(name: str) -> tuple[str, str, str, bool]:
    """(label, colour, marker, filled). '<base>+thr' is the base model thresholded at matched coverage:
    same colour and marker, drawn hollow."""
    base, _, suffix = name.partition("+")
    label, slot, marker = SYSTEMS[base]
    if suffix == "thr":
        return label + " + threshold", SLOTS[slot], marker, False
    return label, SLOTS[slot], marker, True


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "svg.hashsalt": "deferdx",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlepad": 5,
        "lines.linewidth": 1.3, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans:italic",
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})
    fig.savefig(out / f"{name}.png", metadata={"Software": None})
    plt.close(fig)


def placeholder(out: Path, name: str, what: str):
    """Keeps the README and paper links valid until the inputs of a figure exist."""
    fig = plt.figure(figsize=(FULL, 1.1))
    fig.patches.append(plt.Rectangle((0.005, 0.03), 0.99, 0.94, transform=fig.transFigure, fill=False,
                                     edgecolor=AXIS, linewidth=0.6))
    fig.text(0.5, 0.6, f"{name}: pending", ha="center", va="center", fontsize=8.5, color=INK, fontweight="bold")
    fig.text(0.5, 0.33, what, ha="center", va="center", fontsize=7.5, color=INK2)
    save(fig, out, name)


def letters(fig, axes_and_letters):
    """Bold panel letters 3.5 pt left of each panel's title (or of its top-left corner without one).
    Anchored to the title artist itself, so the position is resolved at draw time, after layout."""
    for ax, s in axes_and_letters:
        title = getattr(ax, "_left_title", None)
        anchor, xy, dy = (title, (0, 0), 0) if title is not None and title.get_text() else (ax, (0, 1), 2)
        t = ax.annotate(s, xy=xy, xycoords=anchor, xytext=(-3.5, dy), textcoords="offset points", ha="right",
                        va="bottom", fontsize=9, fontweight="bold", color=INK, annotation_clip=False)
        t.set_in_layout(False)


def marker_kw(color: str, filled: bool, ms: float = 5.0, marker: str | None = None) -> dict:
    """Filled marks carry a surface ring; hollow marks (thresholded variants) a coloured edge. Triangles
    are drawn slightly larger to match the visual size of the other shapes, and thin shapes (P, X) skip
    the ring, which would eat their arms."""
    if marker in ("^", "v"):
        ms += 0.8
    if filled:
        ring = color if marker in ("P", "X") else SURFACE
        return dict(ms=ms + (0.2 if marker in ("P", "X") else 0), color=color, mfc=color, mec=ring, mew=0.8)
    return dict(ms=ms - 0.4, color=color, mfc=SURFACE, mec=color, mew=1.1)


def legend_kw(color: str, marker: str, ms: float = 5.0) -> dict:
    return {k: v for k, v in marker_kw(color, True, ms, marker).items() if k != "color"}


def ticks(ax, axis="x", n=4):
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(MaxNLocator(nbins=n, steps=[1, 2, 5, 10]))


def dots(ax, rows, get, scale=100.0, labels=True, zero=True):
    """Dot-and-whisker rows: point estimate and 95% bootstrap interval, one system per row. With
    sharey, only the panel that shows labels sets the (shared) ticks; `zero` keeps 0 in view."""
    for i, name in enumerate(rows):
        d = get(name)
        if not d or d.get("value") is None or not np.isfinite(d["value"]):
            continue
        _, color, marker, filled = style(name)
        if d.get("lo") is not None and np.isfinite(d["lo"]):
            ax.plot([d["lo"] * scale, d["hi"] * scale], [i, i], color=color, lw=1.2, zorder=2)
        ax.plot(d["value"] * scale, i, marker=marker, ls="none", zorder=3, **marker_kw(color, filled, marker=marker))
    if labels:
        ax.set_yticks(range(len(rows)), [style(n)[0] for n in rows])
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ticks(ax)
    if zero:
        lo, hi = ax.get_xlim()
        ax.set_xlim(-0.04 * max(hi, 1e-9), hi)


def table(tables, name, section, key):
    return ((tables.get(name) or {}).get(section) or {}).get(key)


# ---------------------------------------------------------------------------------------------------
def rc_curve(results):
    """(coverage, accuracy on answered cases) along the stated-probability ranking. Deferrals rank last
    and never count as answered, so a deferring system's curve ends at its own coverage."""
    ins = [r for r in results if r.label != OTHER]
    s = np.array([selective_score(r) for r in ins])
    c = np.array([r.correct for r in ins], dtype=float)
    order = np.argsort(-s, kind="stable")
    committed = np.array([r.terminal == "commit" for r in ins])[order]
    k = np.cumsum(committed)
    cov = k / len(ins)
    acc = np.cumsum(c[order]) / np.maximum(k, 1)
    keep = committed & (k >= max(5, int(0.05 * len(ins))))
    return cov[keep], acc[keep]


def fig_risk_coverage(suites, tables, out):
    names = [n for n in CURVE_SYSTEMS if n in suites and cdm_pooled(suites[n])]
    if not names:
        return placeholder(out, "fig_risk_coverage", "Needs at least one evaluated system (outputs/eval/<system>).")
    fig = plt.figure(figsize=(FULL, 2.75), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1])
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lo_y = 100.0
    handles = []
    for n in names:
        label, color, marker, _ = style(n)
        cov, acc = rc_curve(cdm_pooled(suites[n]))
        if not len(cov):
            continue
        ax.plot(cov * 100, acc * 100, color=color, zorder=3)
        ax.plot(cov[-1] * 100, acc[-1] * 100, marker=marker, ls="none", zorder=4, **marker_kw(color, True, marker=marker))
        lo_y = min(lo_y, float(np.min(acc[cov >= 0.2])) * 100 if np.any(cov >= 0.2) else lo_y)
        handles.append(Line2D([], [], color=color, marker=marker, label=label, **legend_kw(color, marker)))
    thr = [n for n in ("grpo_nodefer+thr", "zs_nodefer+thr") if table(tables, n, "ml", "coverage")]
    for n in thr:
        _, color, marker, _ = style(n)
        x, y = table(tables, n, "ml", "coverage")["value"] * 100, table(tables, n, "ml", "selective_acc")["value"] * 100
        ax.plot(x, y, marker=marker, ls="none", zorder=5, **marker_kw(color, False, 5.6, marker))
        lo_y = min(lo_y, y)
    if thr:
        handles.append(Line2D([], [], ls="none", marker="o", ms=5, mfc=SURFACE, mec=MUTED, mew=1.1,
                              label="Forced choice + threshold at matched coverage (hollow)"))
    ref = next((n for n in ("cev", "deferdx") if table(tables, n, "ml", "coverage")), None)
    if ref:
        x = table(tables, ref, "ml", "coverage")["value"] * 100
        ax.axvline(x, color=AXIS, lw=0.6, zorder=1)
        ax.annotate(f"{style(ref)[0]}\ncoverage {x:.0f}%", (x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-3, -2), textcoords="offset points", ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlim(20, 101)
    ax.set_ylim(max(0, 5 * np.floor((lo_y - 2) / 5)), 100.5)
    ax.set_xlabel("Coverage: in-set cases answered (%)")
    ax.set_ylabel("Accuracy on answered cases (%)")
    ax.set_title("Risk–coverage on CDM val + test (480 cases, 3 seeds)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, handlelength=1.6, borderaxespad=0.2)
    rows = [n for n in names if table(tables, n, "ml", "aurc")]
    dots(bx, rows, lambda n: table(tables, n, "ml", "aurc"), scale=1.0, zero=False)
    bx.set_xlabel("AURC (lower is better)")
    bx.set_title("Area under the risk–coverage curve ↓")
    bx.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    letters(fig, [(ax, "a"), (bx, "b")])
    save(fig, out, "fig_risk_coverage")


PAIRED_METRICS = [  # key, label, higher is better
    ("selective_acc", "Accuracy on answered cases", True), ("coverage", "Coverage", True),
    ("acc_full", "Accuracy, full coverage", True), ("unsafe_errors", "Unflagged errors ↓", False),
    ("confident_errors", "Confident errors ↓", False), ("aurc", "AURC (×100) ↓", False), ("ece", "ECE (×100) ↓", False),
    ("handoff_quality", "Handoff quality", True), ("false_commit:eval_other_seen", "False commits, OTHER seen ↓", False),
    ("false_commit:eval_other_unseen", "False commits, OTHER unseen ↓", False)]
COMPARATORS = ["grpo_nodefer+thr", "deferdx", "zs_defer", "cev_dualascent", "zs_nodefer+thr", "grpo_nodefer"]


def fig_paired(results, out):
    paired = results.get("paired") or {}
    ref = next((n for n in ("cev", "deferdx") if any(k.startswith(f"{n} vs ") for k in paired)), None)
    comps = [c for c in COMPARATORS if ref and c != ref and f"{ref} vs {c}" in paired][:4]
    if not comps:
        return placeholder(out, "fig_paired_differences", "Needs the paired comparisons in docs/results.json.")
    fig, axes = plt.subplots(1, len(comps), figsize=(FULL, 2.7), sharey=True, layout="constrained", squeeze=False)
    axes = axes[0]
    _, color, _, _ = style(ref)
    rows = [m for m in PAIRED_METRICS if any(m[0] in paired[f"{ref} vs {c}"] for c in comps)]
    for ax, c in zip(axes, comps):
        d = paired[f"{ref} vs {c}"]
        for i, (key, _, up) in enumerate(rows):
            v = d.get(key)
            if not v or v.get("diff") is None or not np.isfinite(v["diff"]):
                continue
            sgn = 1 if up else -1  # oriented so that positive favours the reference system
            x, lo, hi = sgn * v["diff"] * 100, sgn * v["lo"] * 100, sgn * v["hi"] * 100
            lo, hi = min(lo, hi), max(lo, hi)
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [i, i], color=color, lw=1.2, zorder=2)
            ax.plot(x, i, marker="o", ls="none", zorder=3, **marker_kw(color, sig))
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_yticks(range(len(rows)), [r[1] for r in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("vs " + SHORT.get(c, style(c)[0]), fontsize=7)
        ax.set_xlabel("Difference (points)")
        ticks(ax, n=5)
    fig.suptitle(f"{style(ref)[0]} minus each comparator on the same cases; positive favours {style(ref)[0]}; "
                 "filled = 95% interval excludes 0", x=0.0, ha="left", fontsize=7.5, color=INK)
    save(fig, out, "fig_paired_differences")


def fig_clinical_safety(tables, out):
    rows = [n for n in MAIN_ROWS if table(tables, n, "clinical", "unsafe_errors")]
    if not rows:
        return placeholder(out, "fig_clinical_safety", "Needs evaluated systems in docs/results.json.")
    panels = [("unsafe_errors", "Unflagged errors (%) ↓", 100.0),
              ("confident_errors", "Confident errors (%) ↓", 100.0),
              ("severity_error", "Severity-weighted error ↓", 1.0),
              ("mean_tests", "Tests per case", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 0.28 * len(rows) + 0.85), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    for j, (ax, (key, title, scale)) in enumerate(zip(axes, panels)):
        dots(ax, rows, lambda n, k=key: table(tables, n, "clinical", k), scale=scale, labels=j == 0,
             zero=key != "mean_tests")
        ax.set_title(title, fontsize=7)
    letters(fig, [(ax, s) for ax, s in zip(axes, "abcd")])
    save(fig, out, "fig_clinical_safety")


def fig_open_world(tables, out):
    rows = [n for n in MAIN_ROWS if table(tables, n, "eval_other_unseen", "false_commit")]
    if not rows:
        return placeholder(out, "fig_open_world", "Needs evaluated systems in docs/results.json.")
    cols = [("false_commit", "False commit (%) ↓"), ("confident_false_commit", "Confident false commit (%) ↓"),
            ("defer", "Deferred to a clinician (%)"), ("commit_other", "Named OTHER (%)")]
    sets = [("eval_other_seen", "a", "OTHER from diagnosis groups seen in training (206 cases)"),
            ("eval_other_unseen", "b", "OTHER from time-critical groups never seen in training (187 cases)")]
    fig = plt.figure(figsize=(FULL, 2 * (0.27 * len(rows) + 0.95)), layout="constrained")
    for sub, (set_name, letter, heading) in zip(fig.subfigures(2, 1, hspace=0.04), sets):
        axes = sub.subplots(1, 4, sharey=True)
        for j, (ax, (key, title)) in enumerate(zip(axes, cols)):
            dots(ax, rows, lambda n, s=set_name, k=key: table(tables, n, s, k), labels=j == 0)
            ax.set_title(title, fontsize=7)
        sub.suptitle(f"{letter}   {heading}", x=0.0, ha="left", fontsize=8, fontweight="bold", color=INK)
    save(fig, out, "fig_open_world")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["step"]] = {k: v for k, v in r.items() if isinstance(v, (int, float)) or v is None}
    return [rows[k] for k in sorted(rows)]


def moving(xs, ys, w=5):
    """Trailing w-step mean that skips missing values (steps without OTHER cases have no OOD rate)."""
    y = np.array([np.nan if v is None else v for v in ys], dtype=float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        win = y[max(0, i - w + 1): i + 1]
        if np.isfinite(win).any():
            out[i] = np.nanmean(win)
    return np.array(xs), out


def fig_training_dynamics(runs_dir: Path, queue_dir: Path, out):
    # Until queue_v3 moves the first CEV run aside, outputs/runs/cev is that (dual-ascent) run.
    archived = (queue_dir / "archive_cev_dualascent.done").exists()
    paths = {"cev": runs_dir / "cev", "cev_dualascent": runs_dir / "cev_dualascent"} if archived else \
            {"cev_dualascent": runs_dir / "cev"}
    paths["deferdx"] = runs_dir / "deferdx"
    logs = {n: read_log(p / "train_log.jsonl") for n, p in paths.items()}
    logs = {n: rs for n, rs in logs.items() if rs}
    order = [n for n in ("cev", "cev_dualascent", "deferdx") if n in logs]
    if not order:
        return placeholder(out, "fig_training_dynamics", "Needs outputs/runs/<run>/train_log.jsonl.")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.9), sharex=True, layout="constrained")
    (a, b), (c, d) = axes
    cap = 30.0
    for n in order:
        label, color, marker, _ = style(n)
        rs = logs[n]
        steps = [r["step"] for r in rs]
        z = 4 if n == "cev" else 3  # the method draws on top
        a.plot(steps, [r.get("nu") or 0.0 for r in rs], color=color, lw=1.1, label=label, zorder=z)
        x, y = moving(steps, [r.get("inset_defer_rate") for r in rs])
        b.plot(x, y * 100, color=color, lw=1.1, zorder=z)
        x, y = moving(steps, [r.get("ood_defer_rate") for r in rs])
        c.plot(x, y * 100, color=color, lw=1.1, zorder=z)
        ev = [r for r in rs if r.get("dev_coverage") is not None]
        if ev:
            d.plot([r["step"] for r in ev], [r["dev_coverage"] * 100 for r in ev], color=color, lw=1.1, zorder=z,
                   marker=marker, **legend_kw(color, marker, 4.6))
    b.axhline(cap, color=INK2, lw=0.7, zorder=1)
    b.annotate(r"cap $\rho_\mathrm{max}$" + "\n30%", (1, cap), xycoords=("axes fraction", "data"), xytext=(3, 0),
               textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK2, annotation_clip=False)
    d.axhline(100 - cap, color=INK2, lw=0.7, zorder=1)
    d.annotate("floor\n70%", (1, 100 - cap), xycoords=("axes fraction", "data"), xytext=(3, 0),
               textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK2, annotation_clip=False)
    a.set_title("Coverage multiplier ν")
    b.set_title("In-set deferral rate, training batches (%; 5-step mean)")
    c.set_title("Out-of-set deferral rate, training batches (%; 5-step mean)")
    d.set_title("Dev set: in-set coverage at each evaluation (%)")
    for ax in (c, d):
        ax.set_xlabel("Training step")
    for ax in (b, c):
        ax.set_ylim(0, 100)
    d.set_ylim(40, 100)
    a.set_ylim(bottom=0)
    for ax in (a, b, c, d):
        ticks(ax, "y", 5)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=len(order), fontsize=7)
    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d")])
    save(fig, out, "fig_training_dynamics")


def fig_reliability(suites, tables, out, bins=10):
    names = [n for n in CURVE_SYSTEMS if n in suites and cdm_pooled(suites[n])]
    if not names:
        return placeholder(out, "fig_reliability", "Needs at least one evaluated system.")
    fig, axes = plt.subplots(1, len(names), figsize=(min(FULL, 1.45 * len(names) + 0.45), 2.05), sharey=True,
                             layout="constrained", squeeze=False)
    axes = axes[0]
    for ax, n in zip(axes, names):
        label, color, marker, _ = style(n)
        cs = [r for r in cdm_pooled(suites[n]) if r.terminal == "commit" and r.probability is not None]
        p = np.array([r.probability for r in cs])
        y = np.array([r.correct for r in cs], dtype=float)
        idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
        pts = [(p[idx == k].mean() * 100, y[idx == k].mean() * 100) for k in range(bins) if (idx == k).sum() >= 10]
        ax.plot([30, 100], [30, 100], color=AXIS, lw=0.7, zorder=1)
        if pts:
            ax.plot(*zip(*pts), color=color, marker=marker, zorder=3, **legend_kw(color, marker, 4.6))
        ece = table(tables, n, "ml", "ece")
        sub = f"ECE {ece['value']:.3f} ({ece['lo']:.3f}–{ece['hi']:.3f})" if ece else ""
        ax.set_title(f"{label}\n{sub}", fontsize=6.8)
        ax.set_xlim(30, 100)
        ax.set_ylim(30, 100)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_yticks([40, 60, 80, 100])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Stated probability (%)")
    axes[0].set_ylabel("Observed accuracy (%)")
    save(fig, out, "fig_reliability")


def fig_per_class(tables, out):
    rows = [n for n in CURVE_SYSTEMS if table(tables, n, "ml", "acc_appendicitis")]
    if not rows:
        return placeholder(out, "fig_per_class", "Needs evaluated systems in docs/results.json.")
    fig, axes = plt.subplots(1, len(IN_SET_LABELS), figsize=(FULL, 0.28 * len(rows) + 0.8), sharey=True,
                             sharex=True, layout="constrained")
    for j, (ax, lab) in enumerate(zip(axes, IN_SET_LABELS)):
        dots(ax, rows, lambda n, k=f"acc_{lab}": table(tables, n, "ml", k), labels=j == 0, zero=False)
        ax.set_title(lab.capitalize(), fontsize=7)
        ax.set_xlabel("Accuracy, full coverage (%)")
    save(fig, out, "fig_per_class")


def fig_ablations(tables, out):
    rows = [n for n in ("cev_step100", "abl_cev_no_handoff", "abl_constant_defer") if table(tables, n, "ml", "coverage")]
    if len(rows) < 2:
        return placeholder(out, "fig_ablations", "DEFER-Dx at step 100 against no scored handoff and a constant "
                           "deferral reward: written when the ablations are evaluated (queue_v3).")
    panels = [("ml", "selective_acc", "Accuracy on answered cases (%)", 100.0),
              ("ml", "coverage", "Coverage (%)", 100.0),
              ("ml", "aurc", "AURC ↓", 1.0),
              ("ml", "handoff_quality", "Handoff quality (%)", 100.0),
              ("eval_other_unseen", "false_commit", "False commit, unseen OTHER (%) ↓", 100.0),
              ("eval_other_unseen", "defer", "Deferred, unseen OTHER (%)", 100.0)]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 2 * (0.3 * len(rows) + 0.7)), sharey=True, layout="constrained")
    for j, (ax, (sec, key, title, scale)) in enumerate(zip(axes.flat, panels)):
        dots(ax, rows, lambda n, s=sec, k=key: table(tables, n, s, k), scale=scale, labels=j % 3 == 0,
             zero=key != "aurc")
        ax.set_title(title, fontsize=7)
    save(fig, out, "fig_ablations")


def fig_unseen_groups(suites, ev: Path, out, n_boot=1000):
    names = [n for n in CURVE_SYSTEMS if n in suites and suites[n].get("eval_other_unseen")]
    data = {}
    for n in names:
        groups = load_groups(ev / n)
        by: dict[str, list] = {}
        for r in suites[n]["eval_other_unseen"]:
            by.setdefault(groups.get(r.case_id, "unknown"), []).append(r)
        for g, rs in by.items():
            if len({r.case_id for r in rs}) >= 10:  # patient-derived cells under 10 cases are suppressed
                data[(n, g)] = (len({r.case_id for r in rs}), S.bootstrap(rs, S.false_commit, n_boot),
                                S.bootstrap(rs, S.defer_rate, n_boot))
    groups = sorted({g for _, g in data}, key=lambda g: -max(v[0] for (n, gg), v in data.items() if gg == g))
    if not groups:
        return placeholder(out, "fig_unseen_groups", "Needs the unseen time-critical OTHER evaluation.")
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 0.42 * len(groups) * max(1, len(names)) / 2 + 1.0), sharey=True,
                             layout="constrained")
    off = np.linspace(-0.28, 0.28, len(names)) if len(names) > 1 else np.zeros(1)
    for ax, k, title in ((axes[0], 1, "False commit (%) ↓"), (axes[1], 2, "Deferred to a clinician (%)")):
        for gi, g in enumerate(groups):
            for oi, n in enumerate(names):
                if (n, g) not in data:
                    continue
                _, color, marker, _ = style(n)
                d = data[(n, g)][k]
                y = gi + off[oi]
                ax.plot([d["lo"] * 100, d["hi"] * 100], [y, y], color=color, lw=1.1, zorder=2)
                ax.plot(d["value"] * 100, y, marker=marker, ls="none", zorder=3, **marker_kw(color, True, 4.6, marker))
        ax.set_yticks(range(len(groups)), [f"{GROUP_NAMES.get(g, g.replace('_', ' '))} (n={max(data[(n, g)][0] for n in names if (n, g) in data)})"
                                           for g in groups])
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-3, 100)
        ax.set_title(title, fontsize=7)
    handles = [Line2D([], [], color=style(n)[1], marker=style(n)[2], label=style(n)[0],
                      **legend_kw(style(n)[1], style(n)[2], 4.6)) for n in names]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(5, len(names)), fontsize=6.8)
    save(fig, out, "fig_unseen_groups")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--results", default=str(ROOT / "docs" / "results.json"))
    ap.add_argument("--runs-dir", default=str(ROOT / "outputs" / "runs"))
    ap.add_argument("--queue-dir", default=str(ROOT / "outputs" / "queue"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args(argv)
    setup()
    ev, out = Path(args.eval_dir), Path(args.out)
    results = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
    tables = results.get("tables") or {}
    suites = {n: load_suite(ev / n) for n in CURVE_SYSTEMS if suite_complete(ev / n)}
    fig_risk_coverage(suites, tables, out)
    fig_paired(results, out)
    fig_clinical_safety(tables, out)
    fig_open_world(tables, out)
    fig_training_dynamics(Path(args.runs_dir), Path(args.queue_dir), out)
    fig_reliability(suites, tables, out)
    fig_per_class(tables, out)
    fig_ablations(tables, out)
    fig_unseen_groups(suites, ev, out)
    print(f"figures -> {out} (systems with suites: {', '.join(suites) or 'none'}; tables: {len(tables)})")


if __name__ == "__main__":
    main()
