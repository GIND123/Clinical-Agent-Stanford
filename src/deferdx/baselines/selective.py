"""Non-RL abstention baselines applied on top of a no-defer policy's episodes.

* Post-hoc confidence threshold (stanford_idea §6.1 #6, the critical ablation):
  commits with p < t become deferrals; t tuned on validation for a target coverage or
  a target selective risk.
* Selection with Guaranteed Risk (SGR, Geifman & El-Yaniv 2017) — a conformal-style
  comparator (§6.1 #7): the lowest threshold whose binomial upper confidence bound on
  selective risk is <= target with probability >= 1 - delta.
"""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.stats import beta

from ..env.environment import EpisodeResult


def _arrays(results: list[EpisodeResult]) -> tuple[np.ndarray, np.ndarray]:
    commits = [r for r in results if r.terminal == "commit" and r.probability is not None]
    return (np.array([r.probability for r in commits], dtype=float),
            np.array([r.correct for r in commits], dtype=float))


def threshold_for_coverage(results: list[EpisodeResult], target_coverage: float) -> float:
    """Threshold keeping ~target_coverage of ALL episodes (non-commits never count)."""
    p, _ = _arrays(results)
    k = int(math.ceil(target_coverage * len(results)))
    if k <= 0 or len(p) == 0:
        return math.inf
    if k >= len(p):
        return float(p.min())
    return float(np.sort(p)[::-1][k - 1])


def threshold_for_risk(results: list[EpisodeResult], target_risk: float) -> float:
    """Lowest threshold whose empirical selective risk on `results` is <= target_risk."""
    p, c = _arrays(results)
    if len(p) == 0:
        return math.inf
    order = np.argsort(-p, kind="stable")
    risk = np.cumsum(1 - c[order]) / np.arange(1, len(p) + 1)
    ok = np.where(risk <= target_risk)[0]
    return float(p[order][ok.max()]) if len(ok) else math.inf


def binomial_ucb(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson one-sided upper bound on a binomial rate."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, n - errors))


def sgr_threshold(results: list[EpisodeResult], target_risk: float, delta: float = 0.05) -> float:
    """SGR: binary search over sorted confidences with a union bound over log2(n) tests."""
    p, c = _arrays(results)
    if len(p) == 0:
        return math.inf
    order = np.argsort(-p, kind="stable")
    ps, cs = p[order], c[order]
    n = len(ps)
    steps = max(1, math.ceil(math.log2(n)))
    d = delta / steps
    lo, hi = 0, n - 1  # index into the sorted list: accept ps[:k+1]
    best = math.inf
    for _ in range(steps + 1):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        k = mid + 1
        ucb = binomial_ucb(int((1 - cs[:k]).sum()), k, d)
        if ucb <= target_risk:
            best = float(ps[mid])
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def apply_threshold(results: list[EpisodeResult], threshold: float) -> list[EpisodeResult]:
    """Commits with p < threshold become DEFERs whose differential is the original commit."""
    out = []
    for r in results:
        r2 = copy.deepcopy(r)
        if r.terminal == "commit" and (r.probability is None or r.probability < threshold):
            r2.terminal = "defer"
            r2.differential = [r.diagnosis] if r.diagnosis else []
            r2.reason = f"post-hoc threshold {threshold:.3f}"
            r2.diagnosis, r2.probability = None, None
        out.append(r2)
    return out
