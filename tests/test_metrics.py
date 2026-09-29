import numpy as np
import pytest

from deferdx.env.environment import EpisodeResult
from deferdx.eval import metrics as M
from deferdx.eval import summarize


def test_risk_coverage_and_aurc():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    cov, risk = M.risk_coverage_curve(scores, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.5])
    assert M.aurc(scores, correct) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert M.accuracy_at_coverage(scores, correct, 0.5) == 1.0
    # reversed ranking is worse
    assert M.aurc(-scores, correct) > M.aurc(scores, correct)


def test_ece_and_brier():
    p = np.array([0.8] * 10)
    c = np.array([1] * 8 + [0] * 2)
    assert M.expected_calibration_error(p, c) == pytest.approx(0.0)
    assert M.expected_calibration_error(np.array([1.0, 1.0]), np.array([0, 0])) == pytest.approx(1.0)
    assert M.brier_score(np.array([1.0, 0.0]), np.array([1, 1])) == pytest.approx(0.5)


def test_auroc():
    assert M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert M.auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5


def test_macro_f1():
    y = ["a", "a", "b", "b"]
    assert M.macro_f1(y, y, ("a", "b")) == 1.0
    assert M.macro_f1(y, ["a", "a", "a", "a"], ("a", "b")) == pytest.approx((2 / 3 + 0) / 2)


def test_summarize_mixed():
    R = [
        EpisodeResult("1", "appendicitis", "commit", diagnosis="appendicitis", probability=0.9, n_tests=2, total_cost=40),
        EpisodeResult("2", "diverticulitis", "commit", diagnosis="appendicitis", probability=0.95, n_tests=1),
        EpisodeResult("3", "diverticulitis", "defer", differential=["diverticulitis"], n_tests=3),
        EpisodeResult("4", "other", "defer", differential=[]),
        EpisodeResult("5", "other", "commit", diagnosis="pancreatitis", probability=0.6),
    ]
    rep = summarize(R)
    cw = rep["closed_world"]
    assert cw["n"] == 3 and cw["coverage"] == pytest.approx(2 / 3)
    assert cw["accuracy_full_coverage"] == pytest.approx(2 / 3)  # defer top-1 counts at full coverage
    assert cw["selective_accuracy"] == pytest.approx(0.5)
    assert rep["safety"]["confident_errors_per_case@0.8"] == pytest.approx(1 / 5)
    assert rep["open_world"]["false_commit_rate"] == pytest.approx(0.5)
    assert rep["deferral"]["defer_rate"] == pytest.approx(1 / 3)
    assert 0 <= rep["open_world"]["ood_auroc"] <= 1
    cf = {"1": True, "2": False, "3": False}
    rep2 = summarize(R, counterfactual_correct=cf)
    assert rep2["deferral"]["precision"] == 1.0 and rep2["deferral"]["recall"] == 0.5
