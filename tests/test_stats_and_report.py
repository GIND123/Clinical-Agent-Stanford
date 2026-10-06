"""Bootstrap statistics, clinical metrics, self-consistency and cross-fitted post-hoc thresholds."""

import pytest

from deferdx.env.environment import EpisodeResult
from deferdx.eval import stats as S
from deferdx.eval.report import crossfit_posthoc, self_consistency


def R(cid, label, terminal="commit", dx=None, p=0.9, diff=()):
    return EpisodeResult(case_id=str(cid), label=label, terminal=terminal, diagnosis=dx, probability=p,
                         differential=list(diff))


def test_bootstrap_interval_contains_point_and_shrinks_with_n():
    small = [R(i, "appendicitis", dx="appendicitis" if i % 4 else "cholecystitis") for i in range(40)]
    big = [R(i, "appendicitis", dx="appendicitis" if i % 4 else "cholecystitis") for i in range(400)]
    a, b = S.bootstrap(small, S.acc_full, 500), S.bootstrap(big, S.acc_full, 500)
    assert a["lo"] <= a["value"] <= a["hi"] and a["value"] == pytest.approx(0.75)
    assert (b["hi"] - b["lo"]) < (a["hi"] - a["lo"])


def test_bootstrap_resamples_cases_not_episodes():
    # 3 seeds of the same 30 cases: an interval over cases, not 90 independent episodes
    rs = [R(i, "appendicitis", dx="appendicitis" if i < 20 else "cholecystitis") for _ in range(3) for i in range(30)]
    one = [R(i, "appendicitis", dx="appendicitis" if i < 20 else "cholecystitis") for i in range(30)]
    a, b = S.bootstrap(rs, S.acc_full, 800, seed=1), S.bootstrap(one, S.acc_full, 800, seed=1)
    assert a["hi"] - a["lo"] == pytest.approx(b["hi"] - b["lo"], abs=0.02)


def test_paired_bootstrap_sign_and_p():
    a = [R(i, "appendicitis", dx="appendicitis") for i in range(60)]
    b = [R(i, "appendicitis", dx="appendicitis" if i % 3 else "pancreatitis") for i in range(60)]
    d = S.paired_bootstrap(a, b, S.acc_full, 500)
    assert d["diff"] == pytest.approx(1 / 3, abs=1e-9) and d["lo"] > 0 and d["p"] < 0.01 and d["n_cases"] == 60


def test_clinical_metrics(severity):
    rs = [R(1, "appendicitis", dx="appendicitis", p=0.95),
          R(2, "appendicitis", dx="cholecystitis", p=0.9),          # confident unflagged error
          R(3, "diverticulitis", "defer", diff=["appendicitis"]),   # deferral that caught an error
          R(4, "pancreatitis", "defer", diff=["pancreatitis"]),     # unnecessary deferral
          R(5, "other", dx="pancreatitis", p=0.85),                # confident false commit on OTHER
          R(6, "other", "defer", diff=["other"])]
    assert S.unsafe_error_rate(rs) == pytest.approx(1 / 4)
    assert S.deferral_precision(rs) == pytest.approx(1 / 2)
    assert S.deferral_recall(rs) == pytest.approx(1 / 2)
    assert S.confident_false_commit(0.8)(rs) == pytest.approx(1 / 2)
    assert S.false_commit(rs) == pytest.approx(1 / 2)
    assert S.severity_weighted_error(severity)(rs) == pytest.approx((0.8 + 1.0) / 6)
    assert S.handoff_contains_truth(rs) == pytest.approx(2 / 3)


def test_self_consistency_majority_and_deferral():
    rs = [R("a", "appendicitis", dx="appendicitis", p=0.9), R("a", "appendicitis", dx="cholecystitis", p=0.6),
          R("a", "appendicitis", dx="appendicitis", p=0.7),
          R("b", "other", "defer", diff=["other"]), R("b", "other", "defer", diff=["pancreatitis"]),
          R("b", "other", dx="other", p=0.8)]
    out = {r.case_id: r for r in self_consistency(rs)}
    assert out["a"].terminal == "commit" and out["a"].diagnosis == "appendicitis"
    assert out["a"].probability == pytest.approx(2 / 3, abs=1e-3)
    assert out["b"].terminal == "defer" and out["b"].differential[0] == "other"


def test_crossfit_threshold_never_uses_the_case_it_scores():
    val = [R(i, "appendicitis", dx="appendicitis", p=0.5 + i / 200) for i in range(100)]
    test = [R(1000 + i, "appendicitis", dx="appendicitis", p=0.9) for i in range(100)]
    out, thr = crossfit_posthoc({"eval_cdm_val": val, "eval_cdm_test": test}, 0.5)
    # test is thresholded with the cut fit on val (~0.75): all of test (p=0.9) survives
    assert all(r.terminal == "commit" for r in out["eval_cdm_test"])
    # val is thresholded with the cut fit on test (0.9): only val cases at or above it still commit
    for orig, r in zip(val, out["eval_cdm_val"]):
        assert r.terminal == ("commit" if orig.probability >= 0.9 else "defer")
    assert thr["fit_on_val"] < thr["fit_on_test"]
