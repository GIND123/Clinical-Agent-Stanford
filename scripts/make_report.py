#!/usr/bin/env python
"""Build docs/RESULTS.md and docs/results.json from evaluation-suite outputs.

Aggregates only: rates over evaluation sets (each >= 25 cases), bootstrap intervals and
paired differences. No rows, identifiers or text. Systems whose output directory does not
exist yet are skipped, so the report can be regenerated as runs finish.

Tables
  1. ML benchmark (CDM val + test pooled): accuracy, macro-F1, risk-coverage, calibration.
  2. Clinical safety: minority class, errors that reach the patient, severity, time-critical
     out-of-set presentations, deferral quality, cost.
  3. Open world: OTHER seen / unseen groups and same-pipeline controls.
  4. Context: published numbers next to ours on the exact LA-CDM test split.
  5. Paired DEFER-Dx-minus-comparator differences on the same cases.

    python scripts/make_report.py --eval-dir outputs/eval --out docs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from deferdx.eval import metrics as M  # noqa: E402
from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.evaluate import ood_score  # noqa: E402
from deferdx.eval.report import cdm_pooled, crossfit_posthoc, fmt_ci, load_suite, self_consistency  # noqa: E402
from deferdx.labels import OTHER  # noqa: E402
from deferdx.rewards.scoring import SeverityMatrix  # noqa: E402

SYSTEMS = [  # eval-suite directory name, label
    ("zs_nodefer", "Qwen3-8B zero-shot, forced choice"),
    ("zs_defer", "Qwen3-8B zero-shot, prompted DEFER"),
    ("grpo_nodefer", "GRPO control (no DEFER)"),
    ("deferdx", "DEFER-Dx (learned DEFER)"),
]
PCT = {"acc_full", "mean_class_acc", "macro_f1", "acc_diverticulitis", "acc_appendicitis", "acc_cholecystitis",
       "acc_pancreatitis", "coverage", "selective_acc", "acc@70", "acc@80", "acc@90", "confident_errors",
       "unsafe_errors", "false_commit", "commit_other", "defer", "invalid", "handoff_contains_truth",
       "confident_false_commit", "deferral_precision", "deferral_recall", "ood_auroc"}
ML_COLS = [("acc_full", "Acc, full cov."), ("mean_class_acc", "Mean-class acc"), ("macro_f1", "Macro-F1"),
           ("coverage", "Coverage"), ("selective_acc", "Selective acc"), ("acc@80", "Acc@80% cov."),
           ("aurc", "AURC ↓"), ("ece", "ECE ↓"), ("brier", "Brier ↓"), ("ood_auroc", "OOD AUROC")]
CLIN_COLS = [("acc_diverticulitis", "Diverticulitis acc"), ("unsafe_errors", "Unflagged errors ↓"),
             ("confident_errors", "Confident errors ↓"), ("severity_error", "Severity-wtd error ↓"),
             ("unseen_false_commit", "Time-critical OTHER missed ↓"),
             ("deferral_precision", "Deferral precision"), ("deferral_recall", "Errors caught"),
             ("mean_tests", "Tests/case"), ("mean_cost_usd", "$/case")]
OW_COLS = [("false_commit", "False commit ↓"), ("confident_false_commit", "Confident false commit ↓"),
           ("commit_other", "Commit OTHER"), ("defer", "Defer"), ("handoff_contains_truth", "Handoff has OTHER")]


def ood_auroc(rs) -> float:
    ins = [r for r in rs if r.label != OTHER]
    ood = [r for r in rs if r.label == OTHER]
    if not ins or not ood:
        return float("nan")
    s = np.array([ood_score(r) for r in ins + ood])
    return M.auroc(s, np.array([False] * len(ins) + [True] * len(ood)))


def cell(t: dict, key: str) -> str:
    if key not in t:
        return "–"
    return fmt_ci(t[key], pct=key in PCT, digits=1 if key in PCT else 3)


def tables_for(suite: dict, sev: SeverityMatrix, n_boot: int) -> dict:
    pooled = cdm_pooled(suite)
    out: dict = {}
    if not pooled:
        return out
    ml = {k: f for k, f in S.CLOSED_WORLD.items()}
    out["ml"] = S.table(pooled, ml, n_boot)
    ow_all = pooled + suite.get("eval_other_seen", []) + suite.get("eval_other_unseen", [])
    out["ml"]["ood_auroc"] = S.bootstrap(ow_all, ood_auroc, n_boot)
    clin = S.table(pooled, {"acc_diverticulitis": S.class_acc("diverticulitis"),
                            "unsafe_errors": S.unsafe_error_rate, "confident_errors": S.confident_error_rate(0.8),
                            "severity_error": S.severity_weighted_error(sev),
                            "deferral_precision": S.deferral_precision, "deferral_recall": S.deferral_recall,
                            "mean_tests": S.mean_tests, "mean_cost_usd": S.mean_cost}, n_boot)
    if suite.get("eval_other_unseen"):
        clin["unseen_false_commit"] = S.bootstrap(suite["eval_other_unseen"], S.false_commit, n_boot)
    out["clinical"] = clin
    if suite.get("eval_cdm_test"):
        out["lacdm_test"] = S.table(suite["eval_cdm_test"], {"mean_class_acc": S.mean_class_acc,
                                                              "acc_full": S.acc_full, "macro_f1": S.macro_f1,
                                                              "acc_diverticulitis": S.class_acc("diverticulitis")},
                                    n_boot)
    for s in ("eval_other_seen", "eval_other_unseen"):
        if suite.get(s):
            out[s] = S.table(suite[s], S.OPEN_WORLD, n_boot)
    if suite.get("eval_controls"):
        out["eval_controls"] = S.table(suite["eval_controls"], {
            "acc_full": S.acc_full, "coverage": S.coverage, "selective_acc": S.selective_acc,
            "commit_other": S.commit_other_rate, "defer": S.defer_rate}, n_boot)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    ev = Path(args.eval_dir)
    sev = SeverityMatrix.from_yaml(ROOT / "configs" / "severity_matrix.yaml")

    suites, labels = {}, {}
    for name, label in SYSTEMS:
        if (ev / name).exists():
            suites[name], labels[name] = load_suite(ev / name), label
            n_seeds = len(list((ev / name).glob("s*.jsonl")))
            if n_seeds >= 3:
                suites[name + "@sc"] = {k: self_consistency(v) for k, v in suites[name].items()}
                labels[name + "@sc"] = f"{label}, self-consistency over {n_seeds} samples"
    # post-hoc thresholds at DEFER-Dx's own coverage, cross-fitted on val/test
    thresholds = {}
    for ref in ("deferdx", "deferdx@sc"):
        if ref not in suites:
            continue
        target = S.coverage(cdm_pooled(suites[ref]))
        suffix = "@sc" if ref.endswith("@sc") else ""
        for base in ("grpo_nodefer", "zs_nodefer"):
            if base + suffix in suites:
                key = f"{base}{suffix}+thr"
                suites[key], thresholds[key] = crossfit_posthoc(suites[base + suffix], target)
                labels[key] = f"{labels[base + suffix]} + threshold at matched coverage ({target:.0%})"

    # SGR (Geifman & El-Yaniv 2017; plan #7): the lowest threshold whose binomial upper bound on
    # selective risk is <= 5% with probability >= 95%, cross-fitted the same way
    from deferdx.baselines.selective import sgr_threshold

    for base in ("grpo_nodefer", "zs_nodefer"):
        if base in suites:
            key = f"{base}+sgr"
            suites[key], thresholds[key] = crossfit_posthoc(suites[base], fit=lambda rs: sgr_threshold(rs, 0.05, 0.05))
            labels[key] = f"{labels[base]} + SGR (5% risk, delta 0.05)"

    tables = {k: tables_for(v, sev, args.n_boot) for k, v in suites.items()}
    paired: dict = {}
    for ref, comps in (("deferdx", ("grpo_nodefer+thr", "zs_nodefer+thr", "zs_defer", "grpo_nodefer")),
                       ("deferdx@sc", ("grpo_nodefer@sc+thr", "zs_nodefer@sc+thr", "zs_defer@sc"))):
        if ref not in suites:
            continue
        for other in comps:
            if other not in suites:
                continue
            a, b = cdm_pooled(suites[ref]), cdm_pooled(suites[other])
            d = {k: S.paired_bootstrap(a, b, f, args.n_boot) for k, f in {
                "selective_acc": S.selective_acc, "coverage": S.coverage, "acc_full": S.acc_full,
                "unsafe_errors": S.unsafe_error_rate, "confident_errors": S.confident_error_rate(0.8),
                "acc_diverticulitis": S.class_acc("diverticulitis"), "aurc": S.aurc, "ece": S.ece}.items()}
            for s in ("eval_other_seen", "eval_other_unseen"):
                if s in suites[ref] and s in suites[other]:
                    d[f"false_commit:{s}"] = S.paired_bootstrap(suites[ref][s], suites[other][s], S.false_commit,
                                                                args.n_boot)
            paired[f"{ref} vs {other}"] = d

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"tables": tables, "paired": paired, "labels": labels,
                                                  "posthoc_thresholds": thresholds}, indent=2, default=str),
                                      encoding="utf-8")

    def block(title, section, cols, note=""):
        L = ["", f"## {title}", ""] + ([note, ""] if note else [])
        L += ["| System | " + " | ".join(h for _, h in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for k, t in tables.items():
            if section in t:
                L.append(f"| {labels[k]} | " + " | ".join(cell(t[section], c) for c, _ in cols) + " |")
        return L

    L = ["# Results", "",
         "Generated by `scripts/make_report.py` from `outputs/eval/` (aggregates only). Point estimates pool all",
         "evaluation seeds; brackets are 95% case-level bootstrap intervals (2,000 resamples). CDM = LA-CDM val +",
         "test (480 cases, 51 diverticulitis), never seen by any trained model. Post-hoc thresholds are",
         "cross-fitted (fit on val, applied to test and vice versa). ↓ = lower is better."]
    L += block("1. Machine-learning benchmark (CDM, val + test)", "ml", ML_COLS)
    L += block("2. Clinical safety (CDM, val + test; time-critical column from the unseen OTHER groups)",
               "clinical", CLIN_COLS,
               "Unflagged errors: wrong diagnoses committed without escalation, per case. Severity-weighted error "
               "uses the proxy matrix in configs/severity_matrix.yaml. Deferral precision: share of deferrals whose "
               "best guess was wrong; errors caught: share of would-be errors that were deferred.")
    L += block("3a. Open world: OTHER, diagnosis groups seen in training", "eval_other_seen", OW_COLS)
    L += block("3b. Open world: OTHER, time-critical groups never seen in training", "eval_other_unseen", OW_COLS)
    L += block("3c. Same-pipeline in-set controls (source-shortcut check)", "eval_controls",
               [("acc_full", "Acc, full cov."), ("coverage", "Coverage"), ("selective_acc", "Selective acc"),
                ("commit_other", "Commit OTHER"), ("defer", "Defer")])
    L += ["", "## 4. Context: published systems (as reported) and ours on the exact LA-CDM test split", "",
          "| System | Split | Mean-class acc | Case-weighted acc | Diverticulitis | Source |", "|---|---|---|---|---|---|",
          "| LA-CDM, trained (ICLR 2026) | LA-CDM test | 81.3 | – | 75.0 | paper |",
          "| ReAct (from LA-CDM) | LA-CDM test | 74.9 | – | 66.7 | paper |",
          "| LDTL (arXiv 2604.05116) | own 70/10/20 | – | 93.4 | 78.8 | paper |",
          "| Random planner (from LDTL) | own 70/10/20 | – | 84.8 | 90.4 | paper |",
          "| DiagAgent-14B (our run, adapter) | LA-CDM test | 71.9 | 77.9 | 56.0 | docs/BASELINES.md |"]
    for k, t in tables.items():
        if "lacdm_test" in t and "+thr" not in k:
            x = t["lacdm_test"]
            L.append(f"| {labels[k]} | LA-CDM test | {cell(x, 'mean_class_acc')} | {cell(x, 'acc_full')} | "
                     f"{cell(x, 'acc_diverticulitis')} | this report |")
    L += ["", "Environments differ across rows (history summary vs full history, 12 vs 22 tests, different base "
              "models); LDTL's split is unpublished. Read rows from other papers as context, not as head-to-head."]
    if paired:
        L += ["", "## 5. Paired differences, DEFER-Dx minus comparator (same cases)", "",
              "| Comparison | Metric | Difference (95% CI) | p |", "|---|---|---|---|"]
        for comp, d in paired.items():
            for m, v in d.items():
                pct = m.split(":")[0] in PCT or m.startswith("false_commit")
                L.append(f"| {comp} | {m} | {fmt_ci(v, pct=pct, digits=1 if pct else 3)} | {v['p']:.3g} |")
    (out / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'RESULTS.md'} and {out / 'results.json'} ({len(tables)} systems)")


if __name__ == "__main__":
    main()
