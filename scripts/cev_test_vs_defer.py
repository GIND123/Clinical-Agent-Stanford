#!/usr/bin/env python
"""Does the counterfactual escalation value separate "test more" from "defer"? Evidence from training logs.

CEV's claim: escalation is credited only where continuing (deciding now or ordering more tests) would
do worse than a scored handoff. The trainer logs, per step, what the forced continuations branched from
each deferral state did. If the claim holds, then over training:
  * cev_cont_acc (accuracy of continuing from the states where the agent deferred) falls: the agent
    stops deferring on cases it could have solved;
  * cev_A_pos (share of deferrals whose escalation advantage is positive) rises: deferrals become
    justified against their own counterfactual;
  * in-set deferral stays bounded (the coverage cap) while accuracy on committed cases holds or rises.
cev_cont_tests is how many further tests the continuations ordered: reducible uncertainty shows up as
continuations that test and then succeed.

Reads outputs/runs/<run>/train_log.jsonl (aggregates only) and prints a table per window of steps;
with --figure, also draws the four series.

    python scripts/cev_test_vs_defer.py --runs cev=outputs/runs/cev cev_dualascent=outputs/runs/cev_dualascent \
        --window 10 --figure docs/figures/fig_cev_test_vs_defer.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SERIES = [("inset_defer_rate", "in-set defer"), ("inset_commit_acc", "acc. committed"),
          ("cev_cont_acc", "continuation acc."), ("cev_A_pos", "escalations credited"),
          ("cev_cont_tests", "continuation tests"), ("cev_unscored", "unscored defers")]


def load_log(path: Path) -> list[dict]:
    f = path / "train_log.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [r for r in rows if "step" in r and "cev_roots" in r]  # training steps (not eval rows)


def windows(rows: list[dict], width: int) -> list[dict]:
    """Mean of each series over consecutive windows of `width` steps; CEV ratios are weighted by the
    number of branched deferrals in each step."""
    out = []
    rows = sorted(rows, key=lambda r: r["step"])
    for i in range(0, len(rows), width):
        chunk = rows[i:i + width]
        w = [max(0, r.get("cev_roots", 0)) for r in chunk]
        row = {"steps": f"{chunk[0]['step']}–{chunk[-1]['step']}"}
        for key, _ in SERIES:
            vals = [(r.get(key), wi) for r, wi in zip(chunk, w) if isinstance(r.get(key), (int, float))
                    and not math.isnan(r.get(key))]
            if not vals:
                row[key] = math.nan
            elif key.startswith("cev_") and key not in ("cev_unscored",) and sum(wi for _, wi in vals) > 0:
                row[key] = sum(v * wi for v, wi in vals) / sum(wi for _, wi in vals)
            else:
                row[key] = sum(v for v, _ in vals) / len(vals)
        out.append(row)
    return out


def table(name: str, rows: list[dict]) -> str:
    head = "| " + name + " steps | " + " | ".join(label for _, label in SERIES) + " |"
    lines = [head, "|" + "---|" * (len(SERIES) + 1)]
    for r in rows:
        cells = []
        for key, _ in SERIES:
            v = r[key]
            cells.append("–" if math.isnan(v) else (f"{v:.2f}" if key in ("cev_cont_tests", "cev_unscored") else f"{100 * v:.1f}"))
        lines.append(f"| {r['steps']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir (or name=train_log.jsonl)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--figure", help="write the series as a figure (pdf/png)")
    args = ap.parse_args()
    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = load_log(Path(path))
        print(table(name, windows(runs[name], args.window)) + "\n")
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        for ax, (key, label) in zip(axes.flat, [SERIES[0], SERIES[2], SERIES[3], SERIES[4]]):
            for name, rows in runs.items():
                pts = [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
                if pts:
                    ax.plot(*zip(*pts), label=name, lw=1)
            ax.set_title(label)
        axes[1, 0].set_xlabel("GRPO step")
        axes[1, 1].set_xlabel("GRPO step")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure)
        print(f"figure -> {args.figure}")


if __name__ == "__main__":
    main()
