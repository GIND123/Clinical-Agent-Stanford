"""Train/val/test split schemes for MIMIC-IV-Ext-CDM.

PhysioNet ships NO official split. Published numbers use different ones:

* ``lacdm``      — LA-CDM (ICLR 2026), scripts/split_dataset.py: 80/10/10, stratified by
                   label, sklearn ``train_test_split(random_state=269)`` over the four
                   per-pathology pickles concatenated in the order appendicitis,
                   cholecystitis, diverticulitis, pancreatitis. Reproduced exactly here
                   from pathology_ids.json (the pickles are built in that file's order by
                   Hager et al.'s ConvertPhysionet.py). LA-CDM's 81.3 and ReAct's 74.9,
                   which LDTL's table copies, are on this test set.
* ``stratified`` — label-stratified split with arbitrary ratios (e.g. 70/10/20, the ratio
                   LDTL reports; LDTL's actual assignment is unpublished).
* ``hash``       — deterministic patient-level hash split (``io.assign_splits``); use when
                   subject-level leakage matters more than comparability.
"""

from __future__ import annotations

from ..labels import IN_SET_LABELS

LACDM_ORDER = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")
LACDM_SEED = 269


def _ordered(pathology_ids: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    ids, labels = [], []
    for cond in LACDM_ORDER:
        for i in pathology_ids.get(cond, []):
            ids.append(str(i))
            labels.append(cond)
    return ids, labels


def lacdm_split(pathology_ids: dict[str, list[str]], seed: int = LACDM_SEED) -> dict[str, str]:
    """Exact reproduction of LA-CDM's 80/10/10 split. Returns {hadm_id: split}."""
    from sklearn.model_selection import train_test_split

    ids, labels = _ordered(pathology_ids)
    train, rest, _, y_rest = train_test_split(ids, labels, test_size=0.2, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return {**{i: "train" for i in train}, **{i: "val" for i in val}, **{i: "test" for i in test}}


def stratified_split(case_ids: list[str], labels: list[str], fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
                     seed: int = 0) -> dict[str, str]:
    from sklearn.model_selection import train_test_split

    tr, va, te = fractions
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    # integer sizes: float fractions like 0.1 + 0.2 make sklearn round the held-out set up
    n = len(case_ids)
    n_test = round(n * te)
    n_rest = n_test + round(n * va)
    train, rest, _, y_rest = train_test_split(case_ids, labels, test_size=n_rest, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=n_test, stratify=y_rest, random_state=seed)
    return {**{i: "train" for i in train}, **{i: "val" for i in val}, **{i: "test" for i in test}}


def subject_leakage(splits: dict[str, str], subject_of: dict[str, str | None]) -> int:
    """Number of subjects whose admissions land in more than one split."""
    seen: dict[str, set[str]] = {}
    for cid, split in splits.items():
        sub = subject_of.get(cid)
        if sub:
            seen.setdefault(sub, set()).add(split)
    return sum(len(v) > 1 for v in seen.values())


def split_counts(splits: dict[str, str], label_of: dict[str, str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for cid, split in splits.items():
        lab = label_of.get(cid)
        if lab in IN_SET_LABELS:
            out.setdefault(split, {}).setdefault(lab, 0)
            out[split][lab] += 1
    return out
