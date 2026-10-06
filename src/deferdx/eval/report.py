"""Load evaluation-suite outputs and build the comparison tables.

Post-hoc deferral baselines are CROSS-FITTED on the CDM evaluation halves: the threshold
applied to val cases is fit on test cases and vice versa, so every evaluated case is
scored by a threshold that never saw it, and val + test (480 cases, 51 diverticulitis)
can be pooled. This is legitimate because no trained model ever uses val or test. For the
open-world sets (disjoint from CDM) the threshold is fit on all CDM evaluation cases.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..baselines.selective import apply_threshold, threshold_for_coverage
from ..data.io import iter_jsonl
from ..env.environment import EpisodeResult
from . import stats as S

CDM_SETS = ("eval_cdm_val", "eval_cdm_test")
OW_SETS = ("eval_other_seen", "eval_other_unseen", "eval_controls")


def self_consistency(results: list[EpisodeResult]) -> list[EpisodeResult]:
    """SC@k: one aggregated episode per case from its k samples (e.g. the evaluation seeds).

    Majority vote over forced predictions (commit, or top of a deferral differential); the
    stated probability becomes the vote share of the winner, so ranking and calibration use
    agreement (ties broken by the winner's mean stated probability). A case whose samples
    mostly deferred stays a deferral. This is the inference-time counterpart of the
    group-consensus reward, and a standard non-RL abstention baseline.
    """
    from collections import Counter

    import numpy as np

    out = []
    for cid, rs in S.by_case(results).items():
        k = len(rs)
        r0 = rs[0]
        agg = EpisodeResult(case_id=cid, label=r0.label, terminal="invalid", source=r0.source)
        agg.n_tests = float(np.mean([r.n_tests for r in rs]))
        agg.n_asks = float(np.mean([r.n_asks for r in rs]))
        agg.total_cost = float(np.mean([r.total_cost for r in rs]))
        votes = Counter(r.forced_prediction for r in rs if r.forced_prediction)
        if votes:
            ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
            winner, n = ranked[0]
            if 2 * sum(r.terminal == "defer" for r in rs) > k:
                agg.terminal, agg.differential = "defer", [lab for lab, _ in ranked]
            else:
                ps = [r.probability for r in rs if r.terminal == "commit" and r.diagnosis == winner
                      and r.probability is not None]
                mean_p = float(np.mean(ps)) if ps else 0.5
                agg.terminal, agg.diagnosis = "commit", winner
                agg.probability = max(0.0, n / k - 1e-3 * (1.0 - mean_p))
        out.append(agg)
    return out


def load_suite(path: str | Path) -> dict[str, list[EpisodeResult]]:
    """{set name: episodes over all seeds} from an eval-suite directory."""
    out: dict[str, list[EpisodeResult]] = defaultdict(list)
    for f in sorted(Path(path).glob("s*.jsonl")):
        for row in iter_jsonl(f):
            out[row.get("set", "all")].append(EpisodeResult.from_dict(row["result"]))
    return dict(out)


def cdm_pooled(suite: dict[str, list[EpisodeResult]]) -> list[EpisodeResult]:
    return [r for s in CDM_SETS for r in suite.get(s, [])]


def crossfit_posthoc(suite: dict[str, list[EpisodeResult]], target_coverage: float) -> tuple[dict[str, list[EpisodeResult]], dict[str, float]]:
    """Apply a coverage-targeted threshold on stated probability, cross-fitted on val/test."""
    val, test = suite.get("eval_cdm_val", []), suite.get("eval_cdm_test", [])
    thr = {"fit_on_val": threshold_for_coverage(val, target_coverage),
           "fit_on_test": threshold_for_coverage(test, target_coverage),
           "fit_on_cdm_eval": threshold_for_coverage(val + test, target_coverage)}
    out = {"eval_cdm_val": apply_threshold(val, thr["fit_on_test"]),
           "eval_cdm_test": apply_threshold(test, thr["fit_on_val"])}
    for s in OW_SETS:
        if s in suite:
            out[s] = apply_threshold(suite[s], thr["fit_on_cdm_eval"])
    return out, thr


def system_tables(suite: dict[str, list[EpisodeResult]], n_boot: int = 2000) -> dict[str, Any]:
    t: dict[str, Any] = {}
    pooled = cdm_pooled(suite)
    if pooled:
        t["cdm_pooled"] = S.table(pooled, S.CLOSED_WORLD, n_boot)
    if suite.get("eval_cdm_test"):
        t["cdm_test"] = S.table(suite["eval_cdm_test"], S.CLOSED_WORLD, n_boot)
    for s in ("eval_other_seen", "eval_other_unseen"):
        if suite.get(s):
            t[s] = S.table(suite[s], S.OPEN_WORLD, n_boot)
    if suite.get("eval_controls"):
        t["eval_controls"] = S.table(suite["eval_controls"], {
            "acc_full": S.acc_full, "coverage": S.coverage, "selective_acc": S.selective_acc,
            "commit_other": S.commit_other_rate, "defer": S.defer_rate, "invalid": S.invalid_rate}, n_boot)
    return t


def fmt_ci(d: dict[str, float], pct: bool = True, digits: int = 1) -> str:
    if d is None:
        return "–"
    k = 100.0 if pct else 1.0
    v = d.get("value", d.get("diff"))
    if v is None or v != v:
        return "–"
    lo, hi = d.get("lo"), d.get("hi")
    if lo is None or lo != lo:
        return f"{v * k:.{digits}f}"
    return f"{v * k:.{digits}f} ({lo * k:.{digits}f}–{hi * k:.{digits}f})"
