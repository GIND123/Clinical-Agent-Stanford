"""Batched multi-turn rollouts: many (case, sample) episodes advanced in lock-step so
each model call is one batched generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .data.schema import Case
from .env.catalog import TestCatalog
from .env.environment import DiagnosticEnv, EnvConfig, EpisodeResult
from .policy.base import Policy


@dataclass
class Turn:
    text: str
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    logprobs: list[float] | None = None


@dataclass
class Rollout:
    case_id: str
    sample_idx: int
    messages: list[dict[str, str]]
    turns: list[Turn]
    result: EpisodeResult
    reward: dict[str, Any] | None = field(default=None)

    def to_dict(self, include_ids: bool = False) -> dict[str, Any]:
        d = {
            "case_id": self.case_id,
            "sample_idx": self.sample_idx,
            "messages": self.messages,
            "result": self.result.to_dict(),
        }
        if self.reward is not None:
            d["reward"] = self.reward
        if include_ids:
            d["turns"] = [t.__dict__ for t in self.turns]
        return d


def run_episodes(
    policy: Policy,
    catalog: TestCatalog,
    env_cfg: EnvConfig,
    cases: list[Case],
    n_samples: int = 1,
    on_turn: Callable[[int, int], None] | None = None,
) -> list[Rollout]:
    """Run `n_samples` episodes per case. Returns rollouts ordered case-major
    (all samples of case 0, then case 1, ...), which `group_by_case` relies on."""
    envs: list[DiagnosticEnv] = []
    rollouts: list[Rollout] = []
    for case in cases:
        for s in range(n_samples):
            env = DiagnosticEnv(catalog, env_cfg)
            obs = env.reset(case)
            env.sample_idx = s  # lets a policy derive per-episode sampling seeds
            envs.append(env)
            msgs = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": obs}]
            rollouts.append(Rollout(case.case_id, s, msgs, [], env.result))

    _advance(policy, envs, rollouts, env_cfg.max_steps + env_cfg.max_invalid + 2, on_turn)
    return rollouts


def _advance(policy: Policy, envs: list[DiagnosticEnv], rollouts: list[Rollout], max_turns: int,
             on_turn: Callable[[int, int], None] | None = None) -> None:
    """Advance every unfinished episode in lock-step until all are done (or max_turns)."""
    for turn in range(max_turns):
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break
        if on_turn:
            on_turn(turn, len(active))
        gens = policy.generate([rollouts[i].messages for i in active], [envs[i] for i in active])
        for i, gen in zip(active, gens):
            ro, env = rollouts[i], envs[i]
            ro.turns.append(Turn(gen.text, gen.prompt_ids, gen.completion_ids, gen.logprobs))
            ro.messages.append({"role": "assistant", "content": gen.text})
            step = env.step_text(gen.text)
            if not step.done:
                ro.messages.append({"role": "user", "content": step.observation})
    for env in envs:
        if not env.done:  # safety net; the env's own budget normally ends episodes first
            env.result.terminal = "timeout"
            env.done = True


def branch_rollouts(policy: Policy, catalog: TestCatalog, env_cfg: EnvConfig, roots: list[tuple[Case, Rollout]],
                    k: int) -> list[list[Rollout]]:
    """Counterfactual continuations of deferring episodes.

    For each (case, rollout) that ended in DEFER, rebuild the exact state just before the deferral (same
    conversation, same environment state, by deterministic replay of its earlier actions) and sample `k`
    continuations in which DEFER is not an option: a DEFER is executed as COMMIT(top of its differential,
    its stated probability), and the agent may still order tests within its remaining budget. Returns one
    list of k continuation rollouts per root, in order."""
    from dataclasses import replace

    from .env.actions import Action

    cfg = replace(env_cfg, defer_as_commit=True)
    envs: list[DiagnosticEnv] = []
    conts: list[Rollout] = []
    for case, ro in roots:
        if ro.result.terminal != "defer" or not ro.messages or ro.messages[-1]["role"] != "assistant":
            raise ValueError(f"branch root {ro.case_id} did not end in a DEFER turn")
        actions = [Action.from_json(st.action) for st in ro.result.steps[:-1]]
        prefix = [dict(m) for m in ro.messages[:-1]]
        for j in range(k):
            env = DiagnosticEnv(catalog, cfg)
            env.replay(case, actions)
            env.sample_idx = 1000 * (ro.sample_idx + 1) + j  # distinct per-request seeds when seeding is on
            envs.append(env)
            conts.append(Rollout(case.case_id, j, list(prefix), [], env.result))
    _advance(policy, envs, conts, cfg.max_steps + cfg.max_invalid + 2)
    return [conts[i * k:(i + 1) * k] for i in range(len(roots))]


def group_by_case(rollouts: list[Rollout]) -> list[list[Rollout]]:
    groups: dict[str, list[Rollout]] = {}
    for ro in rollouts:
        groups.setdefault(ro.case_id, []).append(ro)
    return list(groups.values())
