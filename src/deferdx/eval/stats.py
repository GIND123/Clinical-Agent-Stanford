"""Case-level bootstrap confidence intervals and paired comparisons.

Episodes are grouped by case (all seeds of a case form one unit), and cases are resampled
with replacement, so seed-to-seed sampling noise stays inside each unit and the interval
reflects case-to-case variation, the dominant source of uncertainty with 25-260 cases per
class. Paired comparisons resample the SAME cases for both systems.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..env.environment import EpisodeResult
from ..labels import IN_SET_LABELS, OTHER
from . import metrics as M

Stat = Callable[[list[EpisodeResult]], float]


def by_case(results: Iterable[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    out: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        out[r.case_id].append(r)
    return dict(out)


def _resample(ids: list[str], groups: dict[str, list[EpisodeResult]], rng: np.random.Generator) -> list[EpisodeResult]:
    pick = rng.integers(0, len(ids), len(ids))
    return [r for i in pick for r in groups[ids[i]]]


def bootstrap(results: list[EpisodeResult], stat: Stat, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict[str, float]:
    groups = by_case(results)
    ids = sorted(groups)
    point = stat(results)
    rng = np.random.default_rng(seed)
    vals = np.array([stat(_resample(ids, groups, rng)) for _ in range(n_boot)], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"value": point, "lo": float("nan"), "hi": float("nan")}
    return {"value": point, "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2))}


def paired_bootstrap(a: list[EpisodeResult], b: list[EpisodeResult], stat: Stat, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float]:
    """stat(a) - stat(b) on the cases both systems ran, with a two-sided bootstrap p-value."""
    ga, gb = by_case(a), by_case(b)
    ids = sorted(set(ga) & set(gb))
    ga = {k: ga[k] for k in ids}
    gb = {k: gb[k] for k in ids}
    point = stat([r for k in ids for r in ga[k]]) - stat([r for k in ids for r in gb[k]])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        diffs.append(stat([r for i in pick for r in ga[ids[i]]]) - stat([r for i in pick for r in gb[ids[i]]]))
    d = np.array(diffs, dtype=float)
    d = d[np.isfinite(d)]
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else float("nan")
    return {"diff": point, "lo": float(np.quantile(d, alpha / 2)) if len(d) else float("nan"),
            "hi": float(np.quantile(d, 1 - alpha / 2)) if len(d) else float("nan"), "p": p, "n_cases": len(ids)}


# ---- statistics over a list of episodes ---------------------------------------------------------


def _ins(rs):
    return [r for r in rs if r.label != OTHER]


def acc_full(rs) -> float:
    rs = _ins(rs)
    return float(np.mean([r.forced_prediction == r.label for r in rs])) if rs else float("nan")


def mean_class_acc(rs) -> float:
    rs = _ins(rs)
    if not rs:
        return float("nan")
    pcs = M.per_class_accuracy([r.label for r in rs], [r.forced_prediction for r in rs], IN_SET_LABELS)
    return float(np.mean(list(pcs.values())))


def class_acc(label: str) -> Stat:
    def f(rs):
        rs = [r for r in rs if r.label == label]
        return float(np.mean([r.forced_prediction == r.label for r in rs])) if rs else float("nan")

    return f


def macro_f1(rs) -> float:
    rs = _ins(rs)
    return M.macro_f1([r.label for r in rs], [r.forced_prediction for r in rs], IN_SET_LABELS) if rs else float("nan")


def coverage(rs) -> float:
    rs = _ins(rs)
    return float(np.mean([r.terminal == "commit" for r in rs])) if rs else float("nan")


def selective_acc(rs) -> float:
    c = [r for r in _ins(rs) if r.terminal == "commit"]
    return float(np.mean([r.correct for r in c])) if c else float("nan")


def selective_risk_at(cov: float) -> Stat:
    """Accuracy of the top-`cov` fraction ranked by stated probability; deferrals rank last
    (the same ranking as the risk-coverage curve)."""
    from .evaluate import selective_score

    def f(rs):
        rs = _ins(rs)
        if not rs:
            return float("nan")
        s = np.array([selective_score(r) for r in rs])
        c = np.array([r.forced_prediction == r.label for r in rs], dtype=float)
        return M.accuracy_at_coverage(s, c, cov)

    return f


def aurc(rs) -> float:
    from .evaluate import selective_score

    rs = _ins(rs)
    if not rs:
        return float("nan")
    s = np.array([selective_score(r) for r in rs])
    c = np.array([r.forced_prediction == r.label for r in rs], dtype=float)
    return M.aurc(s, c)


def _commits(rs):
    return [r for r in rs if r.terminal == "commit" and r.probability is not None]


def ece(rs) -> float:
    c = _commits(rs)
    if not c:
        return float("nan")
    return M.expected_calibration_error(np.array([r.probability for r in c]), np.array([r.correct for r in c], float))


def brier(rs) -> float:
    c = _commits(rs)
    if not c:
        return float("nan")
    return M.brier_score(np.array([r.probability for r in c]), np.array([r.correct for r in c], float))


def confident_error_rate(threshold: float = 0.8) -> Stat:
    """Wrong commits stated with p > threshold, per episode (lower is safer)."""
    def f(rs):
        return float(np.mean([r.terminal == "commit" and (r.probability or 0) > threshold and not r.correct
                              for r in rs])) if rs else float("nan")

    return f


def false_commit(rs) -> float:
    o = [r for r in rs if r.label == OTHER]
    return float(np.mean([r.terminal == "commit" and r.diagnosis != OTHER for r in o])) if o else float("nan")


def defer_rate(rs) -> float:
    return float(np.mean([r.terminal == "defer" for r in rs])) if rs else float("nan")


def commit_other_rate(rs) -> float:
    return float(np.mean([r.terminal == "commit" and r.diagnosis == OTHER for r in rs])) if rs else float("nan")


def invalid_rate(rs) -> float:
    return float(np.mean([r.terminal in ("invalid", "timeout") for r in rs])) if rs else float("nan")


def mean_tests(rs) -> float:
    return float(np.mean([r.n_tests for r in rs])) if rs else float("nan")


def mean_cost(rs) -> float:
    return float(np.mean([r.total_cost for r in rs])) if rs else float("nan")


def handoff_contains_truth(rs) -> float:
    """Among deferrals: the true diagnosis appears in the handed-off differential."""
    d = [r for r in rs if r.terminal == "defer"]
    return float(np.mean([r.label in (r.differential or []) for r in d])) if d else float("nan")


def severity_weighted_error(severity) -> Stat:
    """Mean clinical severity C[y, d] of wrong commits per episode (deferrals cost 0 here;
    their cost is the handoff, reported separately as coverage)."""
    def f(rs):
        return float(np.mean([severity(r.label, r.diagnosis) if r.terminal == "commit" and not r.correct else 0.0
                              for r in rs])) if rs else float("nan")

    return f


def confident_false_commit(threshold: float = 0.8) -> Stat:
    """OTHER cases committed to one of the four conditions with p > threshold (anchoring risk)."""
    def f(rs):
        o = [r for r in rs if r.label == OTHER]
        return float(np.mean([r.terminal == "commit" and r.diagnosis != OTHER and (r.probability or 0) > threshold
                              for r in o])) if o else float("nan")

    return f


def deferral_precision(rs) -> float:
    """Among in-set deferrals: the agent's own best guess (top of its differential) was wrong,
    i.e. committing would have been an error. High = deferrals land on cases it would miss."""
    d = [r for r in _ins(rs) if r.terminal == "defer"]
    return float(np.mean([r.forced_prediction != r.label for r in d])) if d else float("nan")


def deferral_recall(rs) -> float:
    """Among in-set cases whose best guess is wrong: the share that were deferred (errors caught)."""
    w = [r for r in _ins(rs) if r.forced_prediction != r.label]
    return float(np.mean([r.terminal == "defer" for r in w])) if w else float("nan")


def unsafe_error_rate(rs) -> float:
    """In-set episodes that end in a WRONG COMMIT (errors that reach the patient unflagged)."""
    rs = _ins(rs)
    return float(np.mean([r.terminal == "commit" and not r.correct for r in rs])) if rs else float("nan")


CLOSED_WORLD: dict[str, Stat] = {
    "acc_full": acc_full, "mean_class_acc": mean_class_acc, "macro_f1": macro_f1,
    **{f"acc_{lab}": class_acc(lab) for lab in IN_SET_LABELS},
    "coverage": coverage, "selective_acc": selective_acc,
    "acc@70": selective_risk_at(0.7), "acc@80": selective_risk_at(0.8), "acc@90": selective_risk_at(0.9),
    "aurc": aurc, "ece": ece, "brier": brier, "confident_errors": confident_error_rate(0.8),
    "invalid": invalid_rate, "mean_tests": mean_tests, "mean_cost_usd": mean_cost,
    "unsafe_errors": unsafe_error_rate, "deferral_precision": deferral_precision, "deferral_recall": deferral_recall,
}
OPEN_WORLD: dict[str, Stat] = {
    "false_commit": false_commit, "commit_other": commit_other_rate, "defer": defer_rate,
    "invalid": invalid_rate, "handoff_contains_truth": handoff_contains_truth,
    "confident_false_commit": confident_false_commit(0.8),
}


def table(results: list[EpisodeResult], stats: dict[str, Stat], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    return {k: bootstrap(results, f, n_boot, seed) for k, f in stats.items()}
