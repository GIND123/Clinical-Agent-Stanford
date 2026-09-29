import pytest

from deferdx.env.environment import EpisodeResult
from deferdx.rewards import CoverageConstraint, commit_reward, crossover_p_hat, group_p_hat, group_rewards


def commit(label, dx, p, cost=0.0):
    return EpisodeResult("c", label, "commit", diagnosis=dx, probability=p, total_cost=cost)


def defer(label, cost=0.0):
    return EpisodeResult("c", label, "defer", differential=[label], total_cost=cost)


def test_commit_reward_components(reward_cfg):
    rb = commit_reward(commit("appendicitis", "appendicitis", 0.9, cost=300), reward_cfg)
    assert rb.accuracy == 1.0
    assert rb.brier == pytest.approx(-0.01)
    assert rb.cost == pytest.approx(-0.15)
    assert rb.severity == 0.0
    wrong = commit_reward(commit("appendicitis", "cholecystitis", 0.9), reward_cfg)
    assert wrong.brier == pytest.approx(-0.81)
    assert wrong.severity == pytest.approx(-0.5 * 0.8)  # kappa * C[app, chole]


def test_brier_is_proper(reward_cfg):
    """Expected commit reward is maximised by reporting the true correctness rate."""
    q = 0.7
    def expected(p):
        return q * commit_reward(commit("pancreatitis", "pancreatitis", p), reward_cfg).total + \
               (1 - q) * commit_reward(commit("pancreatitis", "appendicitis", p), reward_cfg).total
    best = max((i / 100 for i in range(101)), key=expected)
    assert best == pytest.approx(q)


def test_group_consensus_defer(reward_cfg):
    hard = [commit("diverticulitis", "appendicitis", 0.9)] * 3 + [commit("diverticulitis", "diverticulitis", 0.9)] + [defer("diverticulitis")]
    assert group_p_hat(hard, reward_cfg) == pytest.approx(0.25)
    r_hard = group_rewards(hard, reward_cfg)[-1]
    assert r_hard.total == pytest.approx(2.0 * (0.85 - 0.25) - 0.1)

    easy = [commit("appendicitis", "appendicitis", 0.95)] * 4 + [defer("appendicitis")]
    r_easy = group_rewards(easy, reward_cfg)
    assert r_easy[-1].total < 0 < r_easy[0].total


def test_empty_group_phat(reward_cfg):
    grp = [defer("pancreatitis")] * 3
    assert group_p_hat(grp, reward_cfg) == reward_cfg.tau  # neutral: consensus term = 0
    assert group_rewards(grp, reward_cfg)[0].consensus == pytest.approx(0.0)


def test_open_world_defer_beats_commit_other(reward_cfg):
    rbs = group_rewards([commit("other", "other", 1.0), defer("other"), commit("other", "appendicitis", 0.9)], reward_cfg)
    assert rbs[1].total > rbs[0].total > 0 > rbs[2].total


def test_constraint_penalty_only_hits_deferrals(reward_cfg):
    grp = [commit("appendicitis", "appendicitis", 0.9), defer("appendicitis")]
    base = group_rewards(grp, reward_cfg)
    pen = group_rewards(grp, reward_cfg, defer_penalty=0.5)
    assert pen[0].total == base[0].total
    assert pen[1].total == pytest.approx(base[1].total - 0.5)


def test_constraint_dual_ascent():
    c = CoverageConstraint(rho_max=0.3, lr=1.0, patience=2)
    assert c.update(0.8) == 0.0  # first violation: patience
    assert c.update(0.8) == pytest.approx(0.5)
    assert c.update(0.8) == pytest.approx(1.0)
    assert c.update(0.1) == pytest.approx(0.8)  # satisfied -> decays
    for _ in range(10):
        c.update(0.0)
    assert c.nu == 0.0


def test_failures_penalised(reward_cfg):
    rbs = group_rewards([EpisodeResult("c", "appendicitis", "timeout"), EpisodeResult("c", "appendicitis", "invalid")], reward_cfg)
    assert all(rb.total == -1.0 for rb in rbs)


def test_action_entropy_bonus_favours_rare_terminal(reward_cfg):
    reward_cfg.action_entropy_coef = 0.1
    grp = [commit("appendicitis", "appendicitis", 0.9)] * 3 + [defer("appendicitis")]
    rbs = group_rewards(grp, reward_cfg)
    assert rbs[3].exploration > rbs[0].exploration > 0


def test_crossover_below_tau(reward_cfg):
    q = crossover_p_hat(reward_cfg)
    assert q is not None and 0.3 < q < reward_cfg.tau
    reward_cfg.tau = 0.95
    assert crossover_p_hat(reward_cfg) > q  # raising tau defers more
