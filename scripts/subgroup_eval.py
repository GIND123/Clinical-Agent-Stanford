"""Accuracy by class, demographic group and mask stratum, with bootstrap intervals.

Complements `deferdx evaluate` with the checks scripts/audit_data.py showed are needed:
  * 95% bootstrap intervals over cases (the LA-CDM test split has 25 diverticulitis cases,
    so one case moves that class by 4 points);
  * accuracy per sex / age / race / insurance / language group, using the audit's group
    definitions (MIMIC-IV admissions + patients);
  * accuracy split by whether the case's radiology contains CDM's "____" mask, which the
    audit found in 27-51% of reports depending on the class (a label cue);
  * the demographics-only floor the audit measured, for reference.

Correctness is full-coverage (`EpisodeResult.forced_prediction == label`, as in
`accuracy_full_coverage`), averaged over samples of the same case. Output is aggregates
only; groups with fewer than --min-cell cases are not shown.

    python scripts/subgroup_eval.py --rollouts outputs/eval/zs.jsonl --cases data/cdm/all.jsonl \
        --out outputs/eval/zs.groups.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import audit_data  # noqa: E402
from audit_data import ATTRS, MASK_RE, derive_groups, md_table, suppress  # noqa: E402
from deferdx.data.io import load_cases, read_jsonl  # noqa: E402
from deferdx.env import EpisodeResult  # noqa: E402

DEMOGRAPHIC_FLOOR = {"accuracy": 0.478, "majority_accuracy": 0.399,
                     "source": "docs/DATA_AUDIT.md §7.3 (5-fold CV, all 2,400 CDM cases)"}


def per_case(rollout_path: str | Path) -> pd.DataFrame:
    """One row per case: mean full-coverage correctness over its samples."""
    rows = []
    for r in read_jsonl(rollout_path):
        res = EpisodeResult.from_dict(r["result"])
        rows.append({"case_id": str(res.case_id), "label": res.label,
                     "correct": float(res.forced_prediction == res.label)})
    df = pd.DataFrame(rows)
    return df.groupby("case_id").agg(label=("label", "first"), correct=("correct", "mean"), samples=("correct", "size"))


def case_features(case_paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in case_paths:
        for c in load_cases(p):
            rad = "\n".join(im.text for im in c.imaging)
            rows.append({"case_id": str(c.case_id), "rad_mask": bool(re.search(MASK_RE, rad))})
    return pd.DataFrame(rows).drop_duplicates("case_id").set_index("case_id")


def demographics(hosp: Path, case_ids: list[str]) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect()
    con.register("ids", pd.DataFrame({"hadm_id": case_ids}))
    src = lambda t: f"read_csv('{(hosp / f'{t}.csv.gz').as_posix()}', header=true, all_varchar=true)"  # noqa: E731
    adm = con.execute(f"SELECT a.subject_id, a.hadm_id, a.admittime, a.admission_type, a.admission_location, "
                      f"a.insurance, a.language, a.marital_status, a.race FROM {src('admissions')} a "
                      "JOIN ids USING (hadm_id)").df()
    pat = con.execute(f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group FROM {src('patients')}").df()
    return derive_groups(adm, pat).set_index("hadm_id")


def bootstrap(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile interval over cases."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def label_adjusted(sub: pd.DataFrame, weights: pd.Series) -> float:
    """Accuracy directly standardised to the cohort's class mix, so a group is not rewarded for
    having more of the easy classes (appendicitis is young and easiest). NaN if the group lacks a class."""
    acc = sub.groupby("label")["correct"].mean()
    if not set(weights.index) <= set(acc.index):
        return float("nan")
    return float(sum(weights[c] * acc[c] for c in weights.index))


def group_table(frame: pd.DataFrame, by: str, min_cell: int, n_boot: int = 2000,
                weights: pd.Series | None = None) -> pd.DataFrame:
    """Groups under min_cell are hidden, and a second group size is hidden ("·") when the
    total would otherwise reveal the first (same rule as the data audit). With `weights`
    (class shares), a label-adjusted accuracy column is added."""
    audit_data.MIN_CELL = min_cell
    sizes = frame.groupby(by, sort=True).size()
    shown = suppress(pd.DataFrame({"n": sizes}))["n"]
    rows = {}
    for g, sub in frame.groupby(by, sort=True):
        if len(sub) < min_cell:
            rows[str(g)] = {"cases": f"<{min_cell}", "accuracy": "–", "95% CI": "–"}
            if weights is not None:
                rows[str(g)]["label-adjusted"] = "–"
            continue
        m, lo, hi = bootstrap(sub["correct"].to_numpy(), n_boot)
        rows[str(g)] = {"cases": shown[g], "accuracy": f"{100 * m:.1f}", "95% CI": f"{100 * lo:.1f}–{100 * hi:.1f}"}
        if weights is not None:
            adj = label_adjusted(sub, weights)
            rows[str(g)]["label-adjusted"] = "–" if np.isnan(adj) else f"{100 * adj:.1f}"
    return pd.DataFrame(rows).T


def report(frame: pd.DataFrame, min_cell: int = 10, n_boot: int = 2000) -> dict:
    m, lo, hi = bootstrap(frame["correct"].to_numpy(), n_boot)
    out = {"cases": int(len(frame)), "accuracy": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "mean_class_accuracy": round(float(frame.groupby("label")["correct"].mean().mean()), 4),
           "demographic_floor": DEMOGRAPHIC_FLOOR, "tables": {}}
    shares = frame["label"].value_counts(normalize=True)
    for col in ["label"] + [a for a in ATTRS if a in frame] + [c for c in ("rad_mask",) if c in frame]:
        out["tables"][col] = group_table(frame, col, min_cell, n_boot, shares if col in ATTRS else None)
    if "rad_mask" in frame:  # does the model do better when the mask is present, within each class?
        out["tables"]["label x rad_mask"] = group_table(
            frame.assign(stratum=frame["label"] + " / mask=" + frame["rad_mask"].astype(str)), "stratum", min_cell, n_boot)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--cases", nargs="+", default=["data/cdm/all.jsonl"], help="case files, for the mask stratum")
    ap.add_argument("--hosp", default="data/physionet/mimiciv/2.2/hosp", help="MIMIC-IV hosp dir, for demographics")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out")
    args = ap.parse_args()

    frame = per_case(args.rollouts)
    frame = frame.join(case_features(args.cases), how="left")
    if Path(args.hosp).exists():
        frame = frame.join(demographics(Path(args.hosp), list(frame.index)), how="left")
    rep = report(frame, args.min_cell, args.bootstrap)
    print(f"cases {rep['cases']}; accuracy {100 * rep['accuracy']:.1f} (95% CI {100 * rep['ci95'][0]:.1f}–"
          f"{100 * rep['ci95'][1]:.1f}); mean class accuracy {100 * rep['mean_class_accuracy']:.1f}; "
          f"demographics-only floor {100 * DEMOGRAPHIC_FLOOR['accuracy']:.1f}")
    for name, t in rep["tables"].items():
        print(f"\n{ATTRS.get(name, name)}\n{md_table(t, name)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rep["tables"] = {k: t.to_dict("index") for k, t in rep["tables"].items()}
        Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
