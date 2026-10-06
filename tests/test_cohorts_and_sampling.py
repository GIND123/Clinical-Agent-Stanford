"""Cohort construction (patient disjointness, unseen groups) and the trainer's case sampler."""

import random
from collections import Counter

from deferdx.data.cohorts import attach_subject_ids, build_cohorts, stratified_holdout
from deferdx.data.schema import Case
from deferdx.labels import IN_SET_LABELS, OTHER
from deferdx.training.grpo_vllm import MixtureSampler


def _case(cid, label, subject, source="cdm", group=None):
    return Case(case_id=str(cid), label=label, hpi="x", subject_id=subject, source=source,
                meta={"group": group} if group else {})


def _world(seed=0):
    rng = random.Random(seed)
    cdm = [_case(i, IN_SET_LABELS[i % 4], f"s{rng.randrange(900)}") for i in range(400)]
    train, val, test = cdm[:320], cdm[320:360], cdm[360:]
    groups = ["bowel_obstruction", "gastroenteritis_colitis", "mesenteric_ischemia", "abdominal_aortic_aneurysm"]
    other = [_case(1000 + i, OTHER, f"s{rng.randrange(900)}", "openworld", groups[i % 4]) for i in range(120)]
    controls = [_case(2000 + i, IN_SET_LABELS[i % 4], f"s{rng.randrange(900)}", "openworld_control")
                for i in range(80)]
    return train, val, test, other, controls


def test_stratified_holdout_quota_and_subject_grouping():
    cases = [_case(i, IN_SET_LABELS[i % 4], f"s{i // 2}") for i in range(200)]
    kept, held = stratified_holdout(cases, 40, seed=3)
    assert len(kept) + len(held) == 200
    assert 36 <= len(held) <= 48
    held_subjects = {c.subject_id for c in held}
    assert not held_subjects & {c.subject_id for c in kept}  # whole subjects move together
    assert set(Counter(c.label for c in held)) == set(IN_SET_LABELS)


def test_cohorts_are_patient_disjoint_and_hold_out_unseen_groups():
    train, val, test, other, controls = _world()
    co = build_cohorts(train, val, test, other, controls,
                       unseen_groups=["mesenteric_ischemia", "abdominal_aortic_aneurysm"], n_dev=32, seed=1)
    training = co.rl_train + co.dev
    evaluation = co.eval_cdm_val + co.eval_cdm_test + co.eval_other_seen + co.eval_other_unseen + co.eval_controls
    assert not {c.case_id for c in training} & {c.case_id for c in evaluation}
    ow_eval = co.eval_other_seen + co.eval_other_unseen + co.eval_controls
    assert not {c.subject_id for c in ow_eval} & {c.subject_id for c in training}
    # open-world TRAINING cases never share a subject with the CDM evaluation set
    cdm_eval_subjects = {c.subject_id for c in val + test}
    assert not {c.subject_id for c in training if c.source != "cdm"} & cdm_eval_subjects
    unseen = {"mesenteric_ischemia", "abdominal_aortic_aneurysm"}
    assert all(c.meta.get("group") not in unseen for c in training if c.label == OTHER)
    assert all(c.meta.get("group") in unseen for c in co.eval_other_unseen)
    assert co.report["open_world_eval_subjects_in_training"] == 0
    assert co.eval_cdm_val == val and co.eval_cdm_test == test  # the LA-CDM split is untouched


def test_attach_subject_ids_fills_only_missing():
    cases = [_case(1, "appendicitis", None), _case(2, "appendicitis", "keep")]
    n = attach_subject_ids(cases, {"1": "s1", "2": "s2"})
    assert n == 1 and cases[0].subject_id == "s1" and cases[1].subject_id == "keep"


def test_mixture_sampler_weights_and_no_duplicates():
    cases = ([_case(i, "appendicitis", f"a{i}") for i in range(300)]
             + [_case(1000 + i, OTHER, f"b{i}", "openworld") for i in range(300)])
    s = MixtureSampler(cases, {"cdm": 0.8, "openworld": 0.2}, random.Random(0))
    draws = [c for _ in range(200) for c in s.sample(10)]
    share = sum(c.source == "openworld" for c in draws) / len(draws)
    assert 0.15 < share < 0.25
    batch = s.sample(50)
    assert len({c.case_id for c in batch}) == len(batch)


def test_mixture_sampler_ignores_empty_pools():
    cases = [_case(i, "appendicitis", f"a{i}") for i in range(5)]
    s = MixtureSampler(cases, {"cdm": 1.0, "openworld": 0.5}, random.Random(0))
    assert set(s.weights) == {"cdm"}
    assert len(s.sample(5)) == 5
