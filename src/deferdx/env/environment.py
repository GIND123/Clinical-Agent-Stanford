"""Reveal-on-request diagnostic POMDP over a single case.

The hidden state is the true label y in {4 CDM conditions} U {OTHER}. Observations are
looked up from the case record, never generated — which is why the environment needs
no simulator model and no API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..data.schema import Case
from ..labels import OTHER
from .actions import ASK, COMMIT, DEFER, INVALID, TEST, Action, parse_action
from .catalog import TestCatalog
from .prompts import initial_observation, system_prompt, truncate


@dataclass
class EnvConfig:
    max_steps: int = 8  # ASK/TEST actions before a terminal action is required
    max_invalid: int = 2  # malformed turns tolerated before the episode ends as INVALID
    allow_defer: bool = True
    open_world: bool = True  # offer OTHER as a COMMIT option
    charge_unavailable: bool = True  # ordering a test never performed still costs
    max_obs_chars: int = 6000
    # MIMIC-CDM stores PMH/social/family history inside one history blob, and LA-CDM / LDTL
    # show it at reset; so by default the whole history is the initial observation and
    # ASK is limited to `ask_topics`. history_at_reset=false hides split-out sections
    # behind ASK (only meaningful for data whose history has line-separated headers).
    history_at_reset: bool = True
    ask_topics: list[str] | None = None  # None = every ASK topic in the catalog

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StepLog:
    action: dict[str, Any]
    cost: float
    available: bool | None = None


@dataclass
class EpisodeResult:
    case_id: str
    label: str
    terminal: str  # commit | defer | timeout | invalid
    diagnosis: str | None = None
    probability: float | None = None
    differential: list[str] = field(default_factory=list)
    reason: str = ""
    total_cost: float = 0.0
    n_tests: int = 0
    n_asks: int = 0
    n_invalid: int = 0
    steps: list[StepLog] = field(default_factory=list)
    source: str = "cdm"

    @property
    def correct(self) -> bool:
        return self.terminal == "commit" and self.diagnosis == self.label

    @property
    def forced_prediction(self) -> str | None:
        """Best single guess: the commit, or the top of a deferral differential."""
        if self.terminal == "commit":
            return self.diagnosis
        if self.terminal == "defer" and self.differential:
            return self.differential[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeResult":
        d = dict(d)
        d["steps"] = [StepLog(**s) for s in d.get("steps", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StepOutput:
    observation: str
    done: bool
    action: Action


class DiagnosticEnv:
    def __init__(self, catalog: TestCatalog, config: EnvConfig | None = None):
        self.catalog = catalog
        self.cfg = config or EnvConfig()
        self.case: Case | None = None
        self._valid_tests = set(catalog.tests)
        topics = self.cfg.ask_topics if self.cfg.ask_topics is not None else list(catalog.asks)
        unknown = set(topics) - set(catalog.asks)
        if unknown:
            raise ValueError(f"ask_topics not in catalog: {sorted(unknown)}")
        self._valid_topics = set(topics)
        self._topics = [t for t in catalog.asks if t in self._valid_topics]

    @property
    def system_prompt(self) -> str:
        return system_prompt(self.catalog, self.cfg.max_steps, self.cfg.allow_defer, self.cfg.open_world,
                             topics=self._topics)

    def reset(self, case: Case) -> str:
        self.case = case
        self.done = False
        self.revealed: set[str] = set()
        self.n_investigations = 0
        self.result = EpisodeResult(case_id=case.case_id, label=case.label, terminal="timeout", source=case.source)
        history = case.history if self.cfg.history_at_reset else {}
        return initial_observation(case.hpi, history)

    def step_text(self, text: str) -> StepOutput:
        return self.step(parse_action(text, self._valid_tests, self._valid_topics))

    def step(self, action: Action) -> StepOutput:
        assert self.case is not None and not self.done, "call reset() first / episode finished"
        res = self.result
        if action.type == DEFER and not self.cfg.allow_defer:
            action = Action(INVALID, error="DEFER is not available in this setting")
        if action.type == COMMIT and action.diagnosis == OTHER and not self.cfg.open_world:
            action = Action(INVALID, error="diagnosis must be one of the four listed conditions")

        if action.type == INVALID:
            res.n_invalid += 1
            res.steps.append(StepLog(action.to_json(), 0.0))
            if res.n_invalid > self.cfg.max_invalid:
                return self._finish("invalid", action, "Too many malformed actions. Episode ended.")
            return StepOutput(f"INVALID ACTION: {action.error}. Respond with one valid <action> block.", False, action)

        if action.type == COMMIT:
            res.diagnosis, res.probability = action.diagnosis, action.probability
            res.steps.append(StepLog(action.to_json(), 0.0))
            return self._finish("commit", action, "Diagnosis recorded.")
        if action.type == DEFER:
            res.differential, res.reason = action.differential, action.reason
            res.steps.append(StepLog(action.to_json(), 0.0))
            return self._finish("defer", action, "Case handed to clinician.")

        # ASK / TEST
        if self.n_investigations >= self.cfg.max_steps:
            res.steps.append(StepLog(action.to_json(), 0.0))
            return self._finish("timeout", action, "Investigation budget exhausted without a decision.")
        key = action.topic if action.type == ASK else action.test
        tag = f"{action.type}:{key}"
        if tag in self.revealed:
            res.steps.append(StepLog(action.to_json(), 0.0, True))
            obs = f"{key} was already provided above."
        elif action.type == ASK:
            text = self.catalog.resolve_ask(key, self.case)
            cost = self.catalog.asks[key].cost
            res.n_asks += 1
            res.total_cost += cost
            res.steps.append(StepLog(action.to_json(), cost, text is not None))
            obs = f"{self.catalog.asks[key].display.upper()}\n{text or 'Not documented.'}"
        else:
            text = self.catalog.resolve_test(key, self.case)
            spec = self.catalog.tests[key]
            cost = spec.cost if (text is not None or self.cfg.charge_unavailable) else 0.0
            res.n_tests += 1
            res.total_cost += cost
            res.steps.append(StepLog(action.to_json(), cost, text is not None))
            obs = f"{spec.display.upper()}\n{text or 'Not performed for this patient; no result available.'}"
        self.revealed.add(tag)
        self.n_investigations += 1
        remaining = self.cfg.max_steps - self.n_investigations
        if remaining <= 0:
            obs += "\n\nNo investigations remain. Your next action must be COMMIT" + (
                " or DEFER." if self.cfg.allow_defer else "."
            )
        else:
            obs += f"\n\n({remaining} investigations remaining.)"
        return StepOutput(truncate(obs, self.cfg.max_obs_chars), False, action)

    def _finish(self, terminal: str, action: Action, obs: str) -> StepOutput:
        self.result.terminal = terminal
        self.done = True
        return StepOutput(obs, True, action)
