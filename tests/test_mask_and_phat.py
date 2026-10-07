"""Mask policies (the CDM "____" label cue) and the leave-one-out p_hat estimator."""

import pytest

from deferdx.data.schema import Case, ImagingReport
from deferdx.env import DiagnosticEnv, EnvConfig
from deferdx.env.environment import EpisodeResult
from deferdx.env.prompts import apply_mask_policy
from deferdx.rewards.scoring import RewardConfig, group_rewards, loo_forced_p_hat

TEXT = "Findings consistent with acute ____. No free air. Patient ___ seen.\nSecond line ____; clause two."


def test_mask_policies():
    assert apply_mask_policy(TEXT, "keep") == TEXT
    norm = apply_mask_policy(TEXT, "normalize")
    assert "____" not in norm and norm.count("___") == 3
    drop = apply_mask_policy(TEXT, "drop_sentence")
    assert "____" not in drop and "No free air." in drop and "Patient ___ seen." in drop
    assert "consistent with" not in drop and "clause two." in drop
    assert apply_mask_policy(None, "normalize") is None
    with pytest.raises(ValueError):
        apply_mask_policy(TEXT, "bogus")


def test_env_applies_mask_policy(catalog):
    case = Case(case_id="1", label="appendicitis", hpi="Pain.", physical_exam="Tender, ____ suspected.",
                imaging=[ImagingReport("CT", "Abdomen", "CT ABD", "Dilated ____. Fat stranding.")])
    env = DiagnosticEnv(catalog, EnvConfig(mask_policy="normalize"))
    env.reset(case)
    obs = env.step_text('<action>{"type": "TEST", "test": "ct_abdomen"}</action>').observation
    assert "____" not in obs and "___" in obs
    with pytest.raises(ValueError):
        DiagnosticEnv(catalog, EnvConfig(mask_policy="bogus"))


def _res(terminal, label="appendicitis", dx=None, p=0.9, diff=()):
    return EpisodeResult(case_id="c", label=label, terminal=terminal, diagnosis=dx, probability=p,
                         differential=list(diff))


def test_loo_forced_p_hat_excludes_self_and_counts_differentials():
    cfg = RewardConfig()
    group = [_res("commit", dx="appendicitis"), _res("commit", dx="cholecystitis"),
             _res("defer", diff=["appendicitis"]), _res("defer", diff=["pancreatitis"])]
    # for rollout 2: others = [correct, wrong, wrong] -> 1/3
    assert loo_forced_p_hat(group, 2, cfg) == pytest.approx(1 / 3)
    # for rollout 3: others = [correct, wrong, correct] -> 2/3
    assert loo_forced_p_hat(group, 3, cfg) == pytest.approx(2 / 3)


def test_p_hat_mode_changes_only_deferral_rewards():
    group = [_res("commit", dx="appendicitis"), _res("commit", dx="cholecystitis"),
             _res("defer", diff=["appendicitis"]), _res("defer", diff=["pancreatitis"])]
    a = group_rewards(group, RewardConfig(p_hat_mode="commits"))
    b = group_rewards(group, RewardConfig(p_hat_mode="forced_loo"))
    assert [x.total for x in a[:2]] == [x.total for x in b[:2]]
    assert a[2].p_hat == pytest.approx(0.5) and b[2].p_hat == pytest.approx(1 / 3)
    # a deferring rollout's own differential never changes its own reward under forced_loo
    group2 = list(group)
    group2[2] = _res("defer", diff=["cholecystitis"])
    c = group_rewards(group2, RewardConfig(p_hat_mode="forced_loo"))
    assert c[2].total == pytest.approx(b[2].total)
    with pytest.raises(ValueError):
        group_rewards(group, RewardConfig(p_hat_mode="bogus"))


def test_constant_defer_mode_is_flat_and_matched(severity):
    from deferdx.rewards.scoring import crossover_p_hat, expected_commit_reward, matched_defer_constant

    cfg = RewardConfig(severity=severity, defer_mode="constant")
    easy = [_res("commit", dx="appendicitis") for _ in range(3)] + [_res("defer", diff=["appendicitis"])]
    hard = [_res("commit", dx="cholecystitis") for _ in range(3)] + [_res("defer", diff=["appendicitis"])]
    a, b = group_rewards(easy, cfg)[3], group_rewards(hard, cfg)[3]
    assert a.total == pytest.approx(b.total)  # same deferral reward whatever the group's accuracy
    c = matched_defer_constant(cfg)
    q = crossover_p_hat(RewardConfig(severity=severity))
    assert c - cfg.handoff_mu == pytest.approx(expected_commit_reward(q, cfg), abs=1e-9)
    assert cfg.defer_mode == "constant"  # matched_defer_constant restores the mode it borrowed
    fixed = RewardConfig(severity=severity, defer_mode="constant", defer_constant=0.3)
    assert group_rewards(hard, fixed)[3].consensus == pytest.approx(0.3)
    with pytest.raises(ValueError):
        group_rewards(hard, RewardConfig(severity=severity, defer_mode="bogus"))
