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
            envs.append(env)
            msgs = [{"role": "system", "content": env.system_prompt}, {"role": "user", "content": obs}]
            rollouts.append(Rollout(case.case_id, s, msgs, [], env.result))

    max_turns = env_cfg.max_steps + env_cfg.max_invalid + 2
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
    return rollouts


def group_by_case(rollouts: list[Rollout]) -> list[list[Rollout]]:
    groups: dict[str, list[Rollout]] = {}
    for ro in rollouts:
        groups.setdefault(ro.case_id, []).append(ro)
    return list(groups.values())
