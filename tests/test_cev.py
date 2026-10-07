"""Counterfactual escalation value: probabilistic handoffs, forced continuation, branching from a
deferral state, and state-level escalation advantages."""

import json

import pytest

from deferdx.data.schema import Case, ImagingReport, LabResult
from deferdx.env import DiagnosticEnv, EnvConfig
from deferdx.env.actions import Action, parse_action
from deferdx.env.environment import EpisodeResult
from deferdx.policy.base import Generation
from deferdx.rewards.scoring import (
    RewardConfig,
    continuation_value,
    escalation_value,
    expected_commit_reward,
    handoff_distribution,
    handoff_score,
)
from deferdx.rollout import branch_rollouts, group_by_case, run_episodes


def _case(label="diverticulitis"):
    return Case(case_id="c1", label=label, hpi="Left lower quadrant pain.", physical_exam="Tender LLQ.",
                labs=[LabResult(name="White Blood Cells", value="14", fluid="Blood", itemid="51301")],
                imaging=[ImagingReport("CT", "Abdomen", "CT ABD", "Sigmoid wall thickening.")])


def _act(d):
    return f"<think>ok</think><action>{json.dumps(d)}</action>"


class ScriptPolicy:
    """Turn 1: TEST cbc. Turn 2: DEFER with a probabilistic differential. Turn 3+: COMMIT."""

    def __init__(self, defer_probs=None, commit_dx="diverticulitis"):
        self.defer_probs = defer_probs or {"diverticulitis": 0.6, "appendicitis": 0.3}
        self.commit_dx = commit_dx

    def generate(self, conversations, contexts=None):
        out = []
        for conv in conversations:
            n = sum(m["role"] == "assistant" for m in conv)
            if n == 0:
                out.append(Generation(_act({"type": "TEST", "test": "cbc"})))
            elif n == 1:
                out.append(Generation(_act({"type": "DEFER", "differential": self.defer_probs, "reason": "unsure"})))
            else:
                out.append(Generation(_act({"type": "COMMIT", "diagnosis": self.commit_dx, "probability": 0.9})))
        return out


def test_probabilistic_defer_is_parsed_and_stored(catalog):
    env = DiagnosticEnv(catalog, EnvConfig(handoff_probs=True))
    env.reset(_case())
    out = env.step_text(_act({"type": "DEFER", "differential": {"appendicitis": 0.2, "diverticulitis": 0.7}}))
    assert out.done and env.result.terminal == "defer"
    assert env.result.differential == ["diverticulitis", "appendicitis"]
    assert env.result.differential_probs == {"appendicitis": 0.2, "diverticulitis": 0.7}
    assert "probability for each diagnosis" in env.system_prompt


def test_handoff_score_is_proper_and_bounded():
    def r(probs, label="diverticulitis", diff=None):
        return EpisodeResult("c", label, "defer", differential=diff or list(probs), differential_probs=probs)

    truth = handoff_score(r({"diverticulitis": 1.0}))
    assert truth == pytest.approx(1.0)
    assert handoff_score(r({"appendicitis": 1.0})) == pytest.approx(0.0)
    # honest beliefs maximise the expected score (strict propriety): belief 0.7/0.3 over two labels
    belief = {"diverticulitis": 0.7, "appendicitis": 0.3}

    def expected(report):
        return sum(belief[y] * handoff_score(r(report, y)) for y in belief)

    honest = expected(belief)
    for alt in ({"diverticulitis": 0.9, "appendicitis": 0.1}, {"diverticulitis": 0.5, "appendicitis": 0.5}):
        assert honest > expected(alt)
    q = handoff_distribution(r({"diverticulitis": 0.5}))
    assert sum(q.values()) == pytest.approx(1.0) and q["diverticulitis"] == pytest.approx(0.5)
    # a plain list is uniform over its labels
    lst = EpisodeResult("c", "appendicitis", "defer", differential=["appendicitis", "other"])
    assert handoff_distribution(lst)["appendicitis"] == pytest.approx(0.5)


def test_defer_as_commit_and_replay(catalog):
    env = DiagnosticEnv(catalog, EnvConfig(defer_as_commit=True))
    env.replay(_case(), [Action.from_json({"type": "TEST", "test": "cbc"})])
    assert env.result.n_tests == 1 and not env.done
    out = env.step_text(_act({"type": "DEFER", "differential": {"appendicitis": 0.55, "diverticulitis": 0.4}}))
    assert out.done and env.result.terminal == "commit"
    assert env.result.diagnosis == "appendicitis" and env.result.probability == pytest.approx(0.55)
    with pytest.raises(ValueError):  # replayed actions must not end the episode
        DiagnosticEnv(catalog, EnvConfig()).replay(_case(), [Action.from_json({"type": "COMMIT", "diagnosis": "other",
                                                                                "probability": 0.5})])


def test_branch_rollouts_continue_from_the_deferral_state(catalog):
    cfg = EnvConfig(max_steps=4, max_invalid=1, handoff_probs=True)
    case = _case()
    roots = run_episodes(ScriptPolicy(), catalog, cfg, [case], n_samples=2)
    assert all(r.result.terminal == "defer" for r in roots)
    branches = branch_rollouts(ScriptPolicy(), catalog, cfg, [(case, r) for r in roots], k=3)
    assert len(branches) == 2 and all(len(b) == 3 for b in branches)
    for root, conts in zip(roots, branches):
        for c in conts:
            # same conversation up to the deferral turn, same environment state (CBC already ordered)
            assert c.messages[: len(root.messages) - 1] == root.messages[:-1]
            assert c.result.n_tests >= 1 and c.result.total_cost >= root.result.total_cost
            assert c.result.terminal == "commit"
        # the script defers again at that state, so each continuation commits the differential's top
        assert all(c.result.diagnosis == "diverticulitis" and c.result.probability == pytest.approx(0.6) for c in conts)
    with pytest.raises(ValueError):  # only deferring episodes can be branched
        commits = run_episodes(ScriptPolicy(), catalog, EnvConfig(max_steps=4, allow_defer=False), [case])
        branch_rollouts(ScriptPolicy(), catalog, cfg, [(case, commits[0])], k=1)


def test_escalation_and_continuation_values(severity):
    cfg = RewardConfig(severity=severity, defer_mode="cev", cev_eta=0.3)
    good = EpisodeResult("c", "diverticulitis", "defer", differential=["diverticulitis"],
                         differential_probs={"diverticulitis": 1.0})
    assert escalation_value(good, cfg) == pytest.approx(expected_commit_reward(cfg.tau, cfg) + 0.3)
    other = EpisodeResult("c", "other", "defer", differential=["other"], differential_probs={"other": 1.0})
    assert escalation_value(other, cfg) == pytest.approx(cfg.openworld_defer_reward + 0.3)
    right = EpisodeResult("c", "diverticulitis", "commit", diagnosis="diverticulitis", probability=0.9, total_cost=500)
    wrong = EpisodeResult("c", "diverticulitis", "commit", diagnosis="appendicitis", probability=0.9, total_cost=100)
    v_right = continuation_value(right, prefix_cost=100, cfg=cfg)
    assert v_right == pytest.approx(1.0 - 0.01 - cfg.cost_scale * 400)  # only post-branch tests are charged
    assert continuation_value(wrong, 100, cfg) < 0


def test_turn_advantages_assign_state_level_escalation_advantage(catalog, severity):
    from deferdx.training.grpo_vllm import turn_advantages

    cfg = EnvConfig(max_steps=4, max_invalid=1, handoff_probs=True)
    rcfg = RewardConfig(severity=severity, defer_mode="cev")
    roots = run_episodes(ScriptPolicy(), catalog, cfg, [_case()], n_samples=4)
    vals = {id(roots[0]): 0.9, id(roots[1]): -0.5}  # continuing would be good / harmful
    assigned, st = turn_advantages(group_by_case(roots), rcfg, 0.0, True, 0.05, False, vals)
    adv = {id(ro): pt for ro, pt in assigned}
    e = escalation_value(roots[0].result, rcfg) - rcfg.handoff_mu
    assert adv[id(roots[0])][-1] == pytest.approx(e - 0.9)  # escalating where continuing succeeds: negative
    assert adv[id(roots[1])][-1] == pytest.approx(e + 0.5)  # escalating where continuing harms: positive
    assert st["cev_n"] == 2
    # identical episodes -> no reward spread: only the escalation turns carry an advantage
    assert adv[id(roots[2])] is None and adv[id(roots[0])][0] is None
