#!/usr/bin/env python
"""Severity-stratified evaluation: does each system escalate complicated disease, where staging decides
management (antibiotics, drainage or surgery)?

DEFER-Dx names a diagnosis; it does not stage it. This script asks what it does with complicated
presentations. Each in-set CDM evaluation case (val + test, 480) is labelled complicated or uncomplicated
from structured data only: the admission's ICD diagnosis codes (MIMIC-IV 2.2 hosp) and the admission labs in
the case record. No note text is read. The proxies follow the guideline each surgeon would use:

  appendicitis    WSES 2020 / AAST: gangrenous, perforated, abscess or generalised peritonitis
  diverticulitis  WSES 2020: perforation, abscess, peritonitis, fistula or obstruction (stage >= 1)
  pancreatitis    Revised Atlanta 2012: necrosis or pseudocyst (local complication) or organ failure
                  (codes; or creatinine > 1.9 mg/dL, Marshall renal score >= 2): moderately severe or severe
  cholecystitis   Tokyo Guidelines 2018: grade II (WBC > 18 x10^9/L; gallbladder perforation, fistula,
                  pericholecystic or liver abscess, peritonitis) or grade III (organ dysfunction codes;
                  creatinine > 2.0 mg/dL, INR > 1.5, platelets < 100 x10^9/L)

For every system it reports, by stratum, the share of cases escalated, unflagged errors, accuracy on answered
cases and full-coverage accuracy, with 95% case-level bootstrap intervals, and the paired difference in
escalation between complicated and uncomplicated cases. Cells with fewer than 10 cases are suppressed.
Writes docs/severity.json (aggregates only) and docs/figures/fig_severity.{pdf,png,svg}.

    python scripts/severity_analysis.py
"""

from __future__ import annotations

import argparse
import gzip
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from deferdx.data.io import load_cases  # noqa: E402
from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.report import crossfit_posthoc, load_suite  # noqa: E402

CDM_SETS = ("eval_cdm_val", "eval_cdm_test")
MIN_CELL = 10

# ICD codes as stored in MIMIC-IV (no dots). A code matches if it starts with a listed prefix.
APPENDICITIS = {"10": ("K352", "K3531", "K3532", "K3533", "K35891", "K350", "K351", "K353"),
                "9": ("5400", "5401")}
# K353 (FY2016-2018: localized peritonitis, with or without perforation, incl. peritoneal abscess) is complicated;
# its FY2019 successor K3530 (localized peritonitis WITHOUT perforation or gangrene) is not, so it is excluded.
APPENDICITIS_EXCLUDE = {"10": ("K3530",), "9": ()}
DIVERTICULITIS = {"10": ("K5700", "K5701", "K5720", "K5721", "K5740", "K5741", "K5780", "K5781",
                         "K631", "K630", "K632", "K65", "K56"),
                  "9": ("56983", "5695", "567", "56981", "560")}
PANCREATITIS_LOCAL = {"10": ("K8501", "K8502", "K8511", "K8512", "K8521", "K8522", "K8531", "K8532", "K8581",
                             "K8582", "K8591", "K8592", "K863"),
                      "9": ("5772",)}
CHOLECYSTITIS_LOCAL = {"10": ("K822", "K823", "K65", "K750"), "9": ("5754", "5755", "567", "5720")}
ORGAN_FAILURE = {"10": ("N17", "J960", "J80", "R57", "R6520", "R6521"),
                 "9": ("584", "51881", "51882", "51884", "7855", "99592")}

# MIMIC-IV lab itemids in the case records (admission values)
WBC, CREATININE, INR, PLATELETS = ("51301", "51300"), ("50912",), ("51237",), ("51265",)


def has(codes: list[tuple[str, str]], table: dict, exclude: dict | None = None) -> bool:
    for code, ver in codes:
        if exclude and code.startswith(exclude.get(ver, ())):
            continue
        if code.startswith(table.get(ver, ())):
            return True
    return False


def first_value(case, itemids) -> float | None:
    for lab in case.labs:
        if lab.itemid in itemids:
            try:
                return float(str(lab.value).strip().split()[0].lstrip("<>"))
            except (ValueError, IndexError):
                continue
    return None


def grade(case, codes: list[tuple[str, str]]) -> tuple[bool, str]:
    """(complicated, criterion) for an in-set case."""
    of = has(codes, ORGAN_FAILURE)
    if case.label == "appendicitis":
        return (True, "codes") if has(codes, APPENDICITIS, APPENDICITIS_EXCLUDE) else (False, "")
    if case.label == "diverticulitis":
        return (True, "codes") if has(codes, DIVERTICULITIS) else (False, "")
    if case.label == "pancreatitis":
        cr = first_value(case, CREATININE)
        if has(codes, PANCREATITIS_LOCAL):
            return True, "local complication"
        if of or (cr is not None and cr > 1.9):
            return True, "organ failure"
        return False, ""
    if case.label == "cholecystitis":
        cr, inr, plt, wbc = (first_value(case, x) for x in (CREATININE, INR, PLATELETS, WBC))
        if of or (cr is not None and cr > 2.0) or (inr is not None and inr > 1.5) or (plt is not None and plt < 100):
            return True, "grade III"
        if has(codes, CHOLECYSTITIS_LOCAL) or (wbc is not None and wbc > 18):
            return True, "grade II"
        return False, ""
    return False, ""


def load_codes(hadm_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    path = ROOT / "data/physionet/mimiciv/2.2/hosp/diagnoses_icd.csv.gz"
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if row["hadm_id"] in hadm_ids:
                out[row["hadm_id"]].append((row["icd_code"].strip(), row["icd_version"].strip()))
    return out


SYSTEMS = [("cev", "DEFER-Dx"), ("cev_seed1", "DEFER-Dx, second seed"),
           ("grpo_nodefer+thr", "GRPO control + threshold"), ("grpo_nodefer_seed1+thr", "GRPO control (seed 2) + threshold"),
           ("deferdx", "Group-consensus reward"), ("zs_defer", "Zero-shot, prompted DEFER")]


def suites_for(ev: Path) -> dict[str, list]:
    out = {}
    for name, _ in SYSTEMS:
        base = name.split("+")[0]
        if not (ev / base).exists():
            continue
        suite = load_suite(ev / base)
        if name.endswith("+thr"):
            target = S.coverage(out["cev"]) if "cev" in out else 0.776
            suite, _ = crossfit_posthoc({k: v for k, v in suite.items() if k in CDM_SETS}, target)
        out[name] = [r for s in CDM_SETS for r in suite.get(s, [])]
    return out


def escalated(rs) -> float:
    """Share of episodes that did not commit (deferred, or for a thresholded system, held back)."""
    return float(np.mean([r.terminal != "commit" for r in rs])) if rs else float("nan")


def boot(rs, stat, n=2000, seed=0):
    groups = S.by_case(rs)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    point = stat(rs)
    vals = [stat([x for i in rng.integers(0, len(ids), len(ids)) for x in groups[ids[i]]]) for _ in range(n)]
    vals = np.array([v for v in vals if v == v])
    return {"value": point, "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)),
            "n_cases": len(ids)}


def strata_diff(rs, comp_ids, stat, n=2000, seed=0):
    """stat(complicated) - stat(uncomplicated), cases resampled within each stratum."""
    a = S.by_case([r for r in rs if r.case_id in comp_ids])
    b = S.by_case([r for r in rs if r.case_id not in comp_ids])
    ia, ib = sorted(a), sorted(b)
    rng = np.random.default_rng(seed)
    point = stat([x for k in ia for x in a[k]]) - stat([x for k in ib for x in b[k]])
    d = []
    for _ in range(n):
        pa = [x for i in rng.integers(0, len(ia), len(ia)) for x in a[ia[i]]]
        pb = [x for i in rng.integers(0, len(ib), len(ib)) for x in b[ib[i]]]
        d.append(stat(pa) - stat(pb))
    d = np.array([v for v in d if v == v])
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": point, "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)), "p": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--out", default="docs/severity.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cases = [c for s in CDM_SETS for c in load_cases(ROOT / "data/cohorts" / f"{s}.jsonl")]
    codes = load_codes({c.case_id for c in cases})
    labels, comp, why = {}, set(), Counter()
    for c in cases:
        labels[c.case_id] = c.label
        g, crit = grade(c, codes.get(c.case_id, []))
        if g:
            comp.add(c.case_id)
            why[(c.label, crit)] += 1
    prevalence = {}
    for lab in ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"):
        n = sum(1 for c in cases if c.label == lab)
        k = sum(1 for c in cases if c.label == lab and c.case_id in comp)
        prevalence[lab] = {"cases": n, "complicated": k if k >= MIN_CELL or k == 0 else None,
                           "uncomplicated": n - k if n - k >= MIN_CELL else None,
                           "criteria": {crit: (v if v >= MIN_CELL else None) for (l2, crit), v in why.items() if l2 == lab}}
    print(json.dumps(prevalence, indent=1))

    suites = suites_for(ROOT / args.eval_dir)
    metrics = {"escalated": escalated, "unflagged_errors": S.unsafe_error_rate, "selective_acc": S.selective_acc,
               "acc_full": S.acc_full}
    res = {"definition": __doc__.split("\n\n")[1].strip(), "prevalence": prevalence,
           "n_complicated": len(comp), "n_cases": len(cases), "systems": {}}
    for name, label in SYSTEMS:
        if name not in suites:
            continue
        rs = suites[name]
        entry = {"label": label, "strata": {}, "complicated_minus_uncomplicated": {}}
        for stratum, keep in (("complicated", lambda r: r.case_id in comp), ("uncomplicated", lambda r: r.case_id not in comp)):
            sub = [r for r in rs if keep(r)]
            entry["strata"][stratum] = {m: boot(sub, f, args.n_boot) for m, f in metrics.items()}
        for m, f in metrics.items():
            entry["complicated_minus_uncomplicated"][m] = strata_diff(rs, comp, f, args.n_boot)
        # per condition, escalation by stratum (cells >= MIN_CELL cases only)
        per = {}
        for lab in prevalence:
            for stratum, inside in (("complicated", True), ("uncomplicated", False)):
                ids = {c.case_id for c in cases if c.label == lab and (c.case_id in comp) == inside}
                if len(ids) >= MIN_CELL:
                    per[f"{lab}:{stratum}"] = boot([r for r in rs if r.case_id in ids], escalated, args.n_boot)
        entry["per_condition_escalated"] = per
        res["systems"][name] = entry
        c, u = entry["strata"]["complicated"]["escalated"], entry["strata"]["uncomplicated"]["escalated"]
        print(f"{label:40s} escalated: complicated {c['value']*100:5.1f} ({c['lo']*100:.1f}-{c['hi']*100:.1f})  "
              f"uncomplicated {u['value']*100:5.1f}  diff {entry['complicated_minus_uncomplicated']['escalated']['diff']*100:+.1f}")
    # paired, on the same complicated cases: DEFER-Dx against each comparator
    # (a thresholded control is re-fitted at each DEFER-Dx seed's own coverage, so every pairing is matched)
    res["paired_complicated"] = {}
    ev = ROOT / args.eval_dir
    raw = {b: load_suite(ev / b) for b in ("grpo_nodefer", "grpo_nodefer_seed1") if (ev / b).exists()}
    for ref in ("cev", "cev_seed1"):
        for other in ("grpo_nodefer+thr", "grpo_nodefer_seed1+thr", "deferdx", "zs_defer"):
            if ref in suites and (other in suites or other.split("+")[0] in raw):
                if other.endswith("+thr"):
                    fitted, _ = crossfit_posthoc({k: v for k, v in raw[other.split("+")[0]].items() if k in CDM_SETS},
                                                 S.coverage(suites[ref]))
                    pool = [r for st in CDM_SETS for r in fitted.get(st, [])]
                else:
                    pool = suites[other]
                a = [r for r in suites[ref] if r.case_id in comp]
                b = [r for r in pool if r.case_id in comp]
                res["paired_complicated"][f"{ref} vs {other}"] = {
                    m: S.paired_bootstrap(a, b, f, args.n_boot) for m, f in
                    (("escalated", escalated), ("unflagged_errors", S.unsafe_error_rate))}
    for k, d in res["paired_complicated"].items():
        e, u = d["escalated"], d["unflagged_errors"]
        print(f"{k:40s} escalated {e['diff']*100:+.1f} ({e['lo']*100:+.1f},{e['hi']*100:+.1f}) p={e['p']:.3f}  "
              f"unflagged {u['diff']*100:+.1f} ({u['lo']*100:+.1f},{u['hi']*100:+.1f}) p={u['p']:.3f}")
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
