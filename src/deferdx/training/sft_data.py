"""Build Stage-1 SFT trajectories (stanford_idea §5.2-5.4), with no API calls.

Sources:
  * oracle   — label-aware scripted teacher (OraclePolicy). Cold-start only: its
               reasoning is templated and its investigations are label-conditioned.
  * rollouts — STaR-style rejection sampling from the model's own rollouts: keep
               correct commits on cases the model reliably solves; on cases where its
               group accuracy p_hat < tau, rewrite one trajectory's final turn into a
               DEFER exemplar. This seeds the DEFER action so RL can explore it.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Any

from ..data.schema import Case
from ..env.actions import DEFER, Action, render_action
from ..env.catalog import TestCatalog
from ..env.environment import EnvConfig, EpisodeResult
from ..labels import IN_SET_LABELS, OTHER
from ..policy.scripted import OraclePolicy
from ..rollout import run_episodes

log = logging.getLogger(__name__)


def oracle_sft(cases: list[Case], catalog: TestCatalog, env_cfg: EnvConfig, defer_fraction: float = 0.1,
               seed: int = 0, max_tests: int = 4) -> list[dict[str, Any]]:
    """Oracle trajectories. A random `defer_fraction` of in-set cases (plus OTHER cases)
    get DEFER endings so the action exists in the SFT distribution at all."""
    rng = random.Random(seed)
    in_set = [c.case_id for c in cases if c.label != OTHER]
    defer_ids = set(rng.sample(in_set, int(len(in_set) * defer_fraction))) if env_cfg.allow_defer else set()
    policy = OraclePolicy(max_tests=max_tests, defer_ids=defer_ids, seed=seed)
    rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=1)
    return [
        {"case_id": ro.case_id, "kind": "oracle_defer" if ro.result.terminal == "defer" else "oracle",
         "messages": ro.messages}
        for ro in rollouts
        if ro.result.terminal in ("commit", "defer")
    ]


def _defer_turn(votes: Counter, label: str) -> str:
    diff = [d for d, _ in votes.most_common() if d in IN_SET_LABELS]
    if label != OTHER and label not in diff:
        diff.append(label)
    diff = diff[:3]
    if label == OTHER:
        reason = "The findings do not fit any of the four listed conditions well; a clinician should evaluate."
        thought = ("The presentation and results do not match appendicitis, cholecystitis, diverticulitis or "
                   "pancreatitis convincingly. Committing to one of them could anchor care on the wrong diagnosis.")
    else:
        alt = ", ".join(diff[:2]) if len(diff) > 1 else diff[0]
        reason = f"Findings are not discriminative between {alt}; confidence is below the safety threshold."
        thought = (f"The evidence is split ({alt}). My probability for any single diagnosis is below what is "
                   "safe to act on, so a clinician should decide.")
    return render_action(Action(DEFER, differential=diff, reason=reason), thought)


def rollouts_sft(records: list[dict[str, Any]], tau: float = 0.85, max_correct_per_case: int = 2,
                 max_defer_fraction: float = 0.3, seed: int = 0) -> list[dict[str, Any]]:
    """`records` are `deferdx rollout` outputs (several samples per case)."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["case_id"]].append(rec)
    keep, defer = [], []
    for case_id, recs in groups.items():
        results = [EpisodeResult.from_dict(r["result"]) for r in recs]
        commits = [(r, res) for r, res in zip(recs, results) if res.terminal == "commit"]
        label = results[0].label
        p_hat = sum(res.correct for _, res in commits) / len(commits) if commits else 0.0
        if label != OTHER and p_hat >= tau:
            correct = [r for r, res in commits if res.correct]
            rng.shuffle(correct)
            seen = set()
            for r in correct:
                sig = tuple(m["content"] for m in r["messages"] if m["role"] == "assistant")
                if sig in seen:
                    continue
                seen.add(sig)
                keep.append({"case_id": case_id, "kind": "star", "messages": r["messages"]})
                if len(seen) >= max_correct_per_case:
                    break
        elif label == OTHER or p_hat < tau:
            votes = Counter(res.diagnosis for _, res in commits)
            # prefer a trajectory that committed wrongly: its prefix is the "tempting" path
            wrong = [r for r, res in commits if not res.correct]
            base = rng.choice(wrong or [r for r, _ in commits] or recs)
            msgs = list(base["messages"])
            while msgs and msgs[-1]["role"] != "assistant":
                msgs.pop()
            if not msgs:
                continue
            msgs[-1] = {"role": "assistant", "content": _defer_turn(votes, label)}
            defer.append({"case_id": case_id, "kind": "defer_exemplar", "messages": msgs, "p_hat": p_hat})
    # cap deferral exemplars at max_defer_fraction of the final set (over-deferral guard)
    if max_defer_fraction >= 1:
        max_defer = len(defer)
    else:
        max_defer = int(max_defer_fraction / (1 - max_defer_fraction) * len(keep))
    if len(defer) > max_defer:
        log.warning("kept %d of %d deferral exemplars (max_defer_fraction=%.2f of %d STaR trajectories)",
                    max_defer, len(defer), max_defer_fraction, len(keep))
    rng.shuffle(defer)
    return keep + defer[:max_defer]
