import math

import numpy as np

from deferdx.baselines import (apply_threshold, binomial_ucb, coverage_rule, sgr_threshold, threshold_for_coverage,
                               threshold_for_risk)
from deferdx.env.environment import EpisodeResult


def results(n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = float(rng.uniform(0.3, 1.0))
        correct = rng.random() < p  # calibrated
        out.append(EpisodeResult(str(i), "appendicitis", "commit",
                                 diagnosis="appendicitis" if correct else "cholecystitis", probability=p))
    return out


def test_threshold_for_coverage():
    R = results()
    thr = threshold_for_coverage(R, 0.8)
    kept = sum(r.probability >= thr for r in R)
    assert kept == 320


def test_coverage_rule_with_ties():
    R = [EpisodeResult(str(i), "appendicitis", "commit", diagnosis="appendicitis", probability=0.8) for i in range(1000)]
    thr, keep = coverage_rule(R, 0.25)
    assert thr == 0.8 and keep == 0.25
    kept = sum(r.terminal == "commit" for r in apply_threshold(R, thr, keep))
    assert 200 <= kept <= 300
    other = apply_threshold(R, thr, keep, salt=1)  # another draw keeps a different set of tied cases
    assert [r.terminal for r in other] != [r.terminal for r in apply_threshold(R, thr, keep)]
    assert sum(r.terminal == "commit" for r in apply_threshold(R, thr)) == 1000  # keep=1: ties kept, as before


def test_threshold_for_risk_and_apply():
    R = results()
    thr = threshold_for_risk(R, 0.1)
    applied = apply_threshold(R, thr)
    committed = [r for r in applied if r.terminal == "commit"]
    risk = 1 - np.mean([r.correct for r in committed])
    assert risk <= 0.1 + 1e-9
    deferred = [r for r in applied if r.terminal == "defer"]
    assert deferred and all(r.forced_prediction in ("appendicitis", "cholecystitis") for r in deferred)


def test_sgr_is_more_conservative_than_empirical():
    R = results(2000, seed=1)
    emp = threshold_for_risk(R, 0.1)
    sgr = sgr_threshold(R, 0.1, delta=0.05)
    assert math.isfinite(sgr) and sgr >= emp


def test_binomial_ucb():
    assert binomial_ucb(0, 100, 0.05) < 0.05
    assert binomial_ucb(10, 100, 0.05) > 0.1
    assert binomial_ucb(5, 5, 0.05) == 1.0
