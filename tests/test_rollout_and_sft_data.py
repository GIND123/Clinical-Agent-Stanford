from collections import Counter

from deferdx.policy import OraclePolicy, RandomPolicy
from deferdx.rollout import group_by_case, run_episodes
from deferdx.training.sft_data import oracle_sft, rollouts_sft


def test_oracle_rollouts(catalog, env_cfg, synth):
    ros = run_episodes(OraclePolicy(), catalog, env_cfg, synth)
    assert all(r.result.terminal in ("commit", "defer") for r in ros)
    for r in ros:
        if r.result.label == "other":
            assert r.result.terminal == "defer"
        else:
            assert r.result.correct
        assert r.result.n_tests + r.result.n_asks <= env_cfg.max_steps
        assert r.messages[0]["role"] == "system" and r.messages[-1]["role"] == "assistant"


def test_grouping_and_random(catalog, env_cfg, synth):
    ros = run_episodes(RandomPolicy(seed=2), catalog, env_cfg, synth[:5], n_samples=4)
    groups = group_by_case(ros)
    assert len(groups) == 5 and all(len(g) == 4 for g in groups)
    assert all(r.result.terminal == "commit" for r in ros)


def test_oracle_sft(catalog, env_cfg, synth):
    rows = oracle_sft(synth, catalog, env_cfg, defer_fraction=0.2)
    kinds = Counter(r["kind"] for r in rows)
    assert kinds["oracle_defer"] >= sum(c.label == "other" for c in synth)
    assert kinds["oracle"] > 0


def test_rollouts_sft(catalog, env_cfg, synth):
    ros = run_episodes(RandomPolicy(seed=0), catalog, env_cfg, synth[:20], n_samples=6)
    # plus a perfectly solved case
    ros += run_episodes(OraclePolicy(defer_other=False), catalog, env_cfg, [c for c in synth[20:] if c.label != "other"][:1], n_samples=3)
    rows = rollouts_sft([r.to_dict() for r in ros], tau=0.85, max_defer_fraction=0.5)
    kinds = Counter(r["kind"] for r in rows)
    assert kinds["star"] >= 1
    assert kinds["defer_exemplar"] <= kinds["star"]  # capped at 50%
    for r in rows:
        assert r["messages"][-1]["role"] == "assistant"
        if r["kind"] == "defer_exemplar":
            assert '"DEFER"' in r["messages"][-1]["content"]
