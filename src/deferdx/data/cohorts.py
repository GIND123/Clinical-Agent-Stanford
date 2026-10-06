"""Patient-disjoint training and evaluation cohorts for the DEFER-Dx experiments.

CDM keeps the exact LA-CDM split. Its val and test sets (480 cases) are the evaluation
set and are never used for a training-time decision: a stratified ``dev`` subset of
train (default 160 cases) is held out of RL for monitoring and checkpoint sanity checks.

The open world is split by diagnosis group:

* SEEN groups: patient-disjoint train / eval split. Training on some OTHER cases lets
  the OTHER branch of the reward fire at all.
* UNSEEN groups: evaluation only. They test whether the agent escalates presentations
  from diagnosis groups it never saw, which is the deployment question. By default the
  unseen groups are the most time-critical ones (mesenteric ischaemia, perforated or
  bleeding ulcer, ruptured AAA, DKA, ectopic pregnancy).

Same-pipeline in-set controls (MIMIC-IV admissions with a CDM condition as the principal
diagnosis, built by the open-world pipeline) are split train / eval too. Training on
them breaks the shortcut "built by the open-world pipeline => OTHER": without them every
open-world-pipeline case in training would be OTHER.

Patient rules:
  * no training case (CDM train incl. dev, OTHER train, controls train) shares a subject
    with any evaluation case;
  * CDM's CSVs carry no subject_id, so `attach_subject_ids` joins MIMIC-IV admissions.
    (The LA-CDM split itself is admission-level; fewer than 10 subjects span its splits.
    It is kept as is for comparability and the overlap is reported, not repaired.)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..labels import IN_SET_LABELS, OTHER
from .io import load_cases, save_cases
from .schema import Case

# Time-critical groups held out of training entirely (see module docstring).
DEFAULT_UNSEEN_GROUPS = (
    "mesenteric_ischemia",
    "peptic_ulcer_perforation_or_bleed",
    "abdominal_aortic_aneurysm",
    "diabetic_ketoacidosis",
    "ectopic_pregnancy",
)


def subject_map(admissions_path: str | Path) -> dict[str, str]:
    """hadm_id -> subject_id from MIMIC-IV hosp/admissions(.csv|.csv.gz)."""
    import pandas as pd

    df = pd.read_csv(admissions_path, usecols=["subject_id", "hadm_id"], dtype=str)
    return dict(zip(df["hadm_id"], df["subject_id"]))


def attach_subject_ids(cases: Iterable[Case], hadm_to_subject: dict[str, str]) -> int:
    """Fill missing subject_ids in place (case_id is the hadm_id). Returns how many were set."""
    n = 0
    for c in cases:
        if c.subject_id is None and c.case_id in hadm_to_subject:
            c.subject_id = hadm_to_subject[c.case_id]
            n += 1
    return n


def _key(c: Case) -> str:
    return c.subject_id or f"case:{c.case_id}"


def _rank(text: str, seed: int) -> str:
    """Deterministic, platform-independent shuffle key."""
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def stratified_holdout(cases: list[Case], n: int, strat: Callable[[Case], str] = lambda c: c.label,
                       seed: int = 0) -> tuple[list[Case], list[Case]]:
    """Hold out ~n cases with proportional allocation per stratum, at subject level
    (every case of a held-out subject is held out). Returns (kept, held_out)."""
    if n <= 0:
        return list(cases), []
    sizes = Counter(strat(c) for c in cases)
    total = sum(sizes.values())
    quota = {s: round(n * k / total) for s, k in sizes.items()}
    by_subject: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        by_subject[_key(c)].append(c)
    held_subjects: set[str] = set()
    taken: Counter = Counter()
    for subj in sorted(by_subject, key=lambda s: _rank(s, seed)):
        group = by_subject[subj]
        s = strat(group[0])
        if taken[s] < quota[s]:
            held_subjects.add(subj)
            for c in group:
                taken[strat(c)] += 1
    kept = [c for c in cases if _key(c) not in held_subjects]
    held = [c for c in cases if _key(c) in held_subjects]
    return kept, held


def subject_split(cases: list[Case], eval_frac: float, strat: Callable[[Case], str],
                  seed: int = 0) -> tuple[list[Case], list[Case]]:
    """Patient-level, stratum-balanced train / eval split. Returns (train, eval)."""
    n_eval = round(eval_frac * len(cases))
    return stratified_holdout(cases, n_eval, strat, seed)


@dataclass
class Cohorts:
    rl_train: list[Case] = field(default_factory=list)
    dev: list[Case] = field(default_factory=list)
    eval_cdm_val: list[Case] = field(default_factory=list)
    eval_cdm_test: list[Case] = field(default_factory=list)
    eval_other_seen: list[Case] = field(default_factory=list)
    eval_other_unseen: list[Case] = field(default_factory=list)
    eval_controls: list[Case] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def sets(self) -> dict[str, list[Case]]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, list)}


def _group(c: Case) -> str:
    return c.meta.get("group") or c.label


def build_cohorts(
    cdm_train: list[Case],
    cdm_val: list[Case],
    cdm_test: list[Case],
    other: list[Case],
    controls: list[Case],
    unseen_groups: Iterable[str] | None = None,
    n_dev: int = 160,
    other_eval_frac: float = 0.4,
    controls_eval_frac: float = 0.4,
    dev_other: int = 40,
    dev_controls: int = 20,
    seed: int = 0,
) -> Cohorts:
    unseen = set(DEFAULT_UNSEEN_GROUPS if unseen_groups is None else unseen_groups)
    rep: dict = {"seed": seed, "unseen_groups": sorted(unseen)}

    # ---- CDM: dev subset of train; val + test are evaluation ---------------------
    cdm_rl, cdm_dev = stratified_holdout(cdm_train, n_dev, seed=seed)
    eval_subjects = {c.subject_id for c in cdm_val + cdm_test if c.subject_id}

    # ---- open world: seen groups split by patient; unseen groups evaluation only --
    def drop_eval_subjects(cases: list[Case]) -> tuple[list[Case], int]:
        kept = [c for c in cases if not (c.subject_id and c.subject_id in eval_subjects)]
        return kept, len(cases) - len(kept)

    other_seen = [c for c in other if _group(c) not in unseen]
    other_unseen = [c for c in other if _group(c) in unseen]
    o_train, o_eval = subject_split(other_seen, other_eval_frac, _group, seed)
    c_train, c_eval = subject_split(controls, controls_eval_frac, lambda c: c.label, seed)
    o_train, rep["other_train_dropped_eval_subject"] = drop_eval_subjects(o_train)
    c_train, rep["controls_train_dropped_eval_subject"] = drop_eval_subjects(c_train)
    o_train, o_dev = stratified_holdout(o_train, min(dev_other, len(o_train) // 10), _group, seed)
    c_train, c_dev = stratified_holdout(c_train, min(dev_controls, len(c_train) // 10), lambda c: c.label, seed)

    train_like = cdm_rl + cdm_dev + o_train + o_dev + c_train + c_dev
    train_subjects = {c.subject_id for c in train_like if c.subject_id}

    def drop_train_subjects(cases: list[Case], name: str) -> list[Case]:
        kept = [c for c in cases if not (c.subject_id and c.subject_id in train_subjects)]
        rep[f"{name}_dropped_train_subject"] = len(cases) - len(kept)
        return kept

    co = Cohorts(
        rl_train=cdm_rl + o_train + c_train,
        dev=cdm_dev + o_dev + c_dev,
        eval_cdm_val=list(cdm_val),
        eval_cdm_test=list(cdm_test),
        eval_other_seen=drop_train_subjects(o_eval, "eval_other_seen"),
        eval_other_unseen=drop_train_subjects(other_unseen, "eval_other_unseen"),
        eval_controls=drop_train_subjects(c_eval, "eval_controls"),
    )

    # ---- checks and aggregate report ----------------------------------------------
    eval_all = co.eval_cdm_val + co.eval_cdm_test + co.eval_other_seen + co.eval_other_unseen + co.eval_controls
    train_all = co.rl_train + co.dev
    overlap_ids = {c.case_id for c in train_all} & {c.case_id for c in eval_all}
    if overlap_ids:
        raise AssertionError(f"{len(overlap_ids)} admissions are in both training and evaluation sets")
    ow_eval_subjects = {c.subject_id for c in co.eval_other_seen + co.eval_other_unseen + co.eval_controls}
    rep["open_world_eval_subjects_in_training"] = len(ow_eval_subjects & train_subjects)
    rep["cdm_eval_subjects_in_training"] = len(eval_subjects & train_subjects)  # from the LA-CDM split itself
    rep["missing_subject_id"] = sum(c.subject_id is None for c in train_all + eval_all)
    rep["counts"] = {
        name: {"n": len(cs), "by_label": dict(sorted(Counter(c.label for c in cs).items())),
               "by_group": dict(sorted(Counter(_group(c) for c in cs if c.label == OTHER).items()))}
        for name, cs in co.sets().items()
    }
    co.report = rep
    return co


def save_cohorts(co: Cohorts, out_dir: str | Path) -> dict[str, int]:
    out_dir = Path(out_dir)
    counts = {name: save_cases(out_dir / f"{name}.jsonl", cs) for name, cs in co.sets().items()}
    (out_dir / "manifest.json").write_text(json.dumps(co.report, indent=2), encoding="utf-8")
    return counts


def load_cohort_sources(cdm_dir: str | Path, openworld_dir: str | Path, admissions: str | Path | None):
    cdm_dir, ow = Path(cdm_dir), Path(openworld_dir)
    train, val, test = (load_cases(cdm_dir / f"{s}.jsonl") for s in ("train", "val", "test"))
    other = load_cases(ow / "other.jsonl")
    controls = load_cases(ow / "controls.jsonl") if (ow / "controls.jsonl").exists() else []
    attached = 0
    if admissions:
        hmap = subject_map(admissions)
        attached = attach_subject_ids(train + val + test + other + controls, hmap)
    return train, val, test, other, controls, attached


def sample_mixture(pools: dict[str, list[Case]], weights: dict[str, float], n: int,
                   rng: random.Random) -> list[Case]:
    """Draw n cases: pick a pool by weight, then a case uniformly from it."""
    names = [k for k in pools if pools[k] and weights.get(k, 0) > 0]
    w = [weights[k] for k in names]
    return [rng.choice(pools[rng.choices(names, w)[0]]) for _ in range(n)]


def in_set(c: Case) -> bool:
    return c.label in IN_SET_LABELS
