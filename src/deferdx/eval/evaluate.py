"""Turn episode results into the stanford_idea §6.2 metric report."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..env.environment import EpisodeResult
from ..labels import IN_SET_LABELS, OTHER
from ..rewards.scoring import SeverityMatrix
from . import metrics as M

COVERAGES = (0.7, 0.8, 0.9)


def selective_score(r: EpisodeResult) -> float:
    """Confidence used to rank cases on the risk-coverage curve: the stated probability
    for commits; deferred / failed episodes rank below every commit."""
    if r.terminal == "commit" and r.probability is not None:
        return float(r.probability)
    return -1.0


def ood_score(r: EpisodeResult) -> float:
    """Higher = more 'this is not one of the four'. commit in-set: 1-p (< 1);
    defer / failed: 1.5; commit OTHER: 2 + p."""
    if r.terminal == "commit":
        p = r.probability or 0.0
        return 2.0 + p if r.diagnosis == OTHER else 1.0 - p
    return 1.5


def summarize(
    results: list[EpisodeResult],
    severity: SeverityMatrix | None = None,
    counterfactual_correct: dict[str, bool] | None = None,
    confident_threshold: float = 0.8,
    coverages: tuple[float, ...] = COVERAGES,
) -> dict[str, Any]:
    severity = severity or SeverityMatrix.uniform()
    rep: dict[str, Any] = {"n_episodes": len(results)}
    if not results:
        return rep

    term = Counter(r.terminal for r in results)
    rep["terminal_rates"] = {k: term[k] / len(results) for k in ("commit", "defer", "timeout", "invalid")}

    ins = [r for r in results if r.label != OTHER]
    ood = [r for r in results if r.label == OTHER]

    # ---- closed world (leaderboard-comparable on in-set cases) -----------------
    if ins:
        y = [r.label for r in ins]
        forced = [r.forced_prediction for r in ins]
        committed = [r for r in ins if r.terminal == "commit"]
        pcs = M.per_class_accuracy(y, forced, IN_SET_LABELS)
        rep["closed_world"] = {
            "n": len(ins),
            # "Mean Acc" on the MIMIC-CDM leaderboard is overall (case-weighted) accuracy.
            "accuracy_full_coverage": float(np.mean([f == t for f, t in zip(forced, y)])),
            "balanced_accuracy": float(np.mean(list(pcs.values()))),
            "macro_f1": M.macro_f1(y, forced, IN_SET_LABELS),
            "per_class_accuracy": pcs,
            "coverage": len(committed) / len(ins),
            "selective_accuracy": float(np.mean([r.correct for r in committed])) if committed else float("nan"),
            "per_class_coverage": {
                lab: float(np.mean([r.terminal == "commit" for r in ins if r.label == lab]))
                for lab in IN_SET_LABELS if any(r.label == lab for r in ins)
            },
        }

        # ---- selective prediction ------------------------------------------------
        scores = np.array([selective_score(r) for r in ins])
        correct = np.array([f == t for f, t in zip(forced, y)], dtype=float)
        cov, risk = M.risk_coverage_curve(scores, correct)
        rep["selective"] = {
            "aurc": M.aurc(scores, correct),
            **{f"accuracy@{int(c * 100)}": M.accuracy_at_coverage(scores, correct, c) for c in coverages},
            "curve": {"coverage": cov.round(4).tolist(), "risk": risk.round(4).tolist()},
        }

        # ---- deferral quality vs the counterfactual "would have been wrong" -------
        if counterfactual_correct is not None:
            would_be_wrong = np.array([not counterfactual_correct.get(r.case_id, False) for r in ins])
            cf_source = "no-defer run"
        else:
            would_be_wrong = 1.0 - correct
            cf_source = "forced prediction (differential top-1)"
        deferred = np.array([r.terminal == "defer" for r in ins])
        prec, rec = M.precision_recall(deferred, would_be_wrong.astype(bool))
        rep["deferral"] = {"precision": prec, "recall": rec, "counterfactual": cf_source,
                           "defer_rate": float(deferred.mean())}

    # ---- calibration + safety on committed cases ---------------------------------
    commits = [r for r in results if r.terminal == "commit" and r.probability is not None]
    if commits:
        p = np.array([r.probability for r in commits])
        c = np.array([r.correct for r in commits], dtype=float)
        confident = p > confident_threshold
        rep["calibration"] = {"ece": M.expected_calibration_error(p, c), "brier": M.brier_score(p, c),
                              "mean_probability": float(p.mean()), "commit_accuracy": float(c.mean())}
        n_conf_err = int((confident & (c == 0)).sum())
        rep["safety"] = {
            f"confident_errors_per_case@{confident_threshold}": n_conf_err / len(results),
            f"error_rate_among_confident@{confident_threshold}": n_conf_err / confident.sum() if confident.any() else float("nan"),
            "severity_weighted_error_per_case": sum(
                severity(r.label, r.diagnosis) for r in commits if not r.correct) / len(results),
        }

    # ---- open world ---------------------------------------------------------------
    if ood:
        rep["open_world"] = {
            "n": len(ood),
            "false_commit_rate": float(np.mean([r.terminal == "commit" and r.diagnosis != OTHER for r in ood])),
            "commit_other_rate": float(np.mean([r.terminal == "commit" and r.diagnosis == OTHER for r in ood])),
            "defer_rate": float(np.mean([r.terminal == "defer" for r in ood])),
        }
        if ins:
            s = np.array([ood_score(r) for r in ins + ood])
            is_ood = np.array([False] * len(ins) + [True] * len(ood))
            rep["open_world"]["ood_auroc"] = M.auroc(s, is_ood)

    # ---- cost ---------------------------------------------------------------------
    n_inv = [r.n_tests + r.n_asks for r in results]
    rep["cost"] = {
        "mean_tests": float(np.mean([r.n_tests for r in results])),
        "mean_asks": float(np.mean([r.n_asks for r in results])),
        "mean_cost_usd": float(np.mean([r.total_cost for r in results])),
        # only tests that returned a result (LA-CDM's "avg. test cost" convention)
        "mean_cost_performed_usd": float(np.mean([
            sum(s.cost for s in r.steps if s.available and s.action.get("type") == "TEST") for r in results])),
        "investigations_histogram": dict(sorted(Counter(n_inv).items())),
        "one_step_terminations": int(sum(n == 1 for n in n_inv)),
    }
    return rep


def format_report(rep: dict[str, Any]) -> str:
    """Compact human-readable summary (curve omitted)."""
    lines = [f"episodes: {rep.get('n_episodes', 0)}"]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) + "}"
        return str(v)

    for section, body in rep.items():
        if not isinstance(body, dict):
            continue
        lines.append(f"[{section}]")
        for k, v in body.items():
            if k == "curve":
                continue
            lines.append(f"  {k}: {fmt(v)}")
    return "\n".join(lines)
