"""Non-LLM policies: a random planner baseline and a label-aware oracle used to seed SFT.

Both emit canonical action TEXT and go through the same parser as model output, so
they exercise the full pipeline.
"""

from __future__ import annotations

import random
from typing import Any

from ..env.actions import ASK, COMMIT, DEFER, TEST, Action, render_action
from ..env.environment import DiagnosticEnv
from ..labels import APPENDICITIS, CHOLECYSTITIS, DIVERTICULITIS, IN_SET_LABELS, OTHER, PANCREATITIS
from .base import Generation

# Label-conditioned investigation plans for the oracle teacher (guideline-flavoured,
# not a claim about optimal workup). Unavailable tests are skipped.
ORACLE_PLANS: dict[str, list[tuple[str, str]]] = {
    APPENDICITIS: [(ASK, "physical_exam"), (TEST, "cbc"), (TEST, "crp"), (TEST, "ct_abdomen"), (TEST, "us_abdomen")],
    CHOLECYSTITIS: [(ASK, "physical_exam"), (TEST, "cbc"), (TEST, "liver_panel"), (TEST, "us_abdomen"), (TEST, "hida_scan")],
    DIVERTICULITIS: [(ASK, "physical_exam"), (TEST, "cbc"), (TEST, "crp"), (TEST, "ct_abdomen")],
    PANCREATITIS: [(ASK, "physical_exam"), (TEST, "lipase"), (TEST, "liver_panel"), (TEST, "triglycerides"), (TEST, "ct_abdomen")],
    OTHER: [(ASK, "physical_exam"), (TEST, "cbc"), (TEST, "bmp"), (TEST, "lipase"), (TEST, "ct_abdomen")],
}

_RATIONALE = {
    APPENDICITIS: "Right lower quadrant findings with inflammatory markers and imaging are consistent with appendicitis.",
    CHOLECYSTITIS: "Right upper quadrant findings with gallbladder imaging are consistent with cholecystitis.",
    DIVERTICULITIS: "Left lower quadrant findings with colonic imaging are consistent with diverticulitis.",
    PANCREATITIS: "Epigastric pain with enzyme elevation and pancreatic imaging are consistent with pancreatitis.",
    OTHER: "The findings do not fit appendicitis, cholecystitis, diverticulitis or pancreatitis.",
}


class OraclePolicy:
    """Knows the label. Follows ORACLE_PLANS (up to `max_tests` available ones), then
    COMMITs the true label — or DEFERs if the case id is in `defer_ids` or is OTHER and
    `defer_other` is set. Produces SFT seed trajectories, never an evaluation number."""

    def __init__(self, max_tests: int = 4, probability: float = 0.9, defer_ids: set[str] | None = None,
                 defer_other: bool = True, seed: int = 0):
        self.max_tests = max_tests
        self.probability = probability
        self.defer_ids = defer_ids or set()
        self.defer_other = defer_other
        self.rng = random.Random(seed)

    def _next(self, env: DiagnosticEnv) -> tuple[Action, str]:
        case = env.case
        done = {f"{s.action.get('type')}:{s.action.get('topic') or s.action.get('test')}" for s in env.result.steps}
        n_inv = env.result.n_tests + env.result.n_asks
        budget = min(self.max_tests, env.cfg.max_steps)
        if n_inv < budget:
            for kind, key in ORACLE_PLANS[case.label]:
                if f"{kind}:{key}" in done:
                    continue
                if kind == TEST and env.catalog.resolve_test(key, case) is None:
                    continue
                if kind == ASK and env.catalog.resolve_ask(key, case) is None:
                    continue
                target = env.catalog.asks[key].display if kind == ASK else env.catalog.tests[key].display
                act = Action(kind, topic=key) if kind == ASK else Action(kind, test=key)
                return act, f"Next I want the {target.lower()} to narrow the differential."
        wants_defer = env.cfg.allow_defer and (
            case.case_id in self.defer_ids or (case.label == OTHER and self.defer_other)
        )
        if wants_defer:
            diff = [case.label] if case.label != OTHER else []
            diff += [x for x in IN_SET_LABELS if x not in diff][: 2]
            reason = ("Presentation does not fit the listed conditions; needs clinician assessment."
                      if case.label == OTHER else
                      "Findings are not discriminative enough to commit safely; escalating.")
            return Action(DEFER, differential=diff, reason=reason), reason
        label = case.label if (case.label != OTHER or env.cfg.open_world) else IN_SET_LABELS[0]
        return Action(COMMIT, diagnosis=label, probability=self.probability), _RATIONALE[label]

    def generate(self, conversations, contexts: list[Any]) -> list[Generation]:
        out = []
        for env in contexts:
            action, thought = self._next(env)
            out.append(Generation(render_action(action, thought)))
        return out


class RandomPolicy:
    """Random planner: orders `n_tests` random available-or-not tests, then commits to a
    uniformly random diagnosis with a random probability. Pure floor baseline."""

    def __init__(self, n_tests: tuple[int, int] = (1, 3), seed: int = 0, allow_other: bool = True):
        self.n_tests = n_tests
        self.rng = random.Random(seed)
        self.allow_other = allow_other
        self._plan: dict[int, int] = {}

    def generate(self, conversations, contexts: list[Any]) -> list[Generation]:
        out = []
        for env in contexts:
            k = self._plan.setdefault(id(env.result), self.rng.randint(*self.n_tests))
            untried = [t for t in env.catalog.tests if f"{TEST}:{t}" not in env.revealed]
            if env.n_investigations < min(k, env.cfg.max_steps) and untried:
                act = Action(TEST, test=self.rng.choice(untried))
            else:
                labels = list(IN_SET_LABELS) + ([OTHER] if self.allow_other and env.cfg.open_world else [])
                act = Action(COMMIT, diagnosis=self.rng.choice(labels), probability=round(self.rng.random(), 2))
            out.append(Generation(render_action(act)))
        return out
