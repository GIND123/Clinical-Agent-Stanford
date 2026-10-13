"""Non-RL abstention baselines applied on top of a no-defer policy's episodes.

* Post-hoc confidence threshold (stanford_idea §6.1 #6, the critical ablation):
  commits with p < t become deferrals; t tuned on validation for a target coverage or
  a target selective risk. Stated probabilities are coarse (0.80, 0.85, ...), so a plain cut can
  overshoot the target coverage; `coverage_rule` returns the randomized rule that hits it, keeping
  each commit tied at the cut with a fitted probability (a fixed hash of the case, so reproducible).
* Selection with Guaranteed Risk (SGR, Geifman & El-Yaniv 2017) — a conformal-style
  comparator (§6.1 #7): the lowest threshold whose binomial upper confidence bound on
  selective risk is <= target with probability >= 1 - delta.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter

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


def coverage_rule(results: list[EpisodeResult], target_coverage: float, tol: float = 1e-9) -> tuple[float, float]:
    """(threshold, keep): commits above the threshold are kept and commits tied at it are kept with
    probability `keep`, so that exactly ~target_coverage of ALL episodes in `results` commit."""
    p, _ = _arrays(results)
    thr = threshold_for_coverage(results, target_coverage)
    if not math.isfinite(thr):
        return thr, 1.0
    above, tied = int((p > thr + tol).sum()), int((np.abs(p - thr) <= tol).sum())
    keep = (target_coverage * len(results) - above) / tied if tied else 1.0
    return thr, float(min(1.0, max(0.0, keep)))


def _uniform(case_id: str, occurrence: int, salt: int) -> float:
    """A fixed pseudo-random number in [0, 1) per (case, evaluation seed, salt)."""
    h = hashlib.sha256(f"{salt}:{case_id}:{occurrence}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


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


def apply_threshold(results: list[EpisodeResult], threshold: float, keep: float = 1.0, salt: int = 0,
                    tol: float = 1e-9) -> list[EpisodeResult]:
    """Commits with p < threshold become DEFERs whose differential is the original commit. With keep < 1,
    a commit tied at the threshold also defers unless its fixed hash (case, occurrence, salt) is below keep;
    the occurrence index separates evaluation seeds of the same case."""
    out, seen = [], Counter()
    for r in results:
        r2 = copy.deepcopy(r)
        occurrence = seen[r.case_id]
        seen[r.case_id] += 1
        tied_out = (keep < 1.0 and r.probability is not None and abs(r.probability - threshold) <= tol
                    and _uniform(r.case_id, occurrence, salt) >= keep)
        if r.terminal == "commit" and (r.probability is None or r.probability < threshold - tol or tied_out):
            r2.terminal = "defer"
            r2.differential = [r.diagnosis] if r.diagnosis else []
            r2.reason = f"post-hoc threshold {threshold:.3f}"
            r2.diagnosis, r2.probability = None, None
        out.append(r2)
    return out
