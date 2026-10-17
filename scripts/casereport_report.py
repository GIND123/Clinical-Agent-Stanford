#!/usr/bin/env python
"""Results on MIMIC-CDM-CR, the published-case benchmark (benchmarks/casereports/): every model, 3 seeds.

Per system:
  out of scope (19 cases)   wrong commitment to one of the four conditions (false commit), escalation (DEFER, or
                            COMMIT of OTHER: "none of the four"), and handoffs whose top suggestion is one of the
                            four (would steer a clinician towards a wrong in-scope diagnosis); for the 7 cases from
                            the held-out time-critical groups and the 11 from groups in no MIMIC cohort
  in scope (13 cases)       correct commitments, escalation, wrong commitments; escalation of the 9 complicated and
                            the 4 uncomplicated cases
  per case                  the majority outcome over the 3 seeds (docs/figures/fig_casereports)
The post-hoc control is thresholded with the rule fitted on the MIMIC CDM evaluation sets at DEFER-Dx's coverage
(an external test set has no split to fit on). Intervals: 95% bootstrap over cases (seeds kept together).
Writes docs/casereports.json (no patient data; cases are public).

    python scripts/casereport_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from deferdx.baselines.selective import apply_threshold, coverage_rule  # noqa: E402
from deferdx.data.io import load_cases  # noqa: E402
from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.report import load_suite, suite_complete  # noqa: E402
from deferdx.labels import OTHER  # noqa: E402

SET = "casereports"
SYSTEMS = [("cev", "DEFER-Dx"), ("cev_seed1", "DEFER-Dx, second seed"),
           ("grpo_nodefer+thr", "GRPO control + threshold"), ("grpo_nodefer_seed1+thr", "GRPO control (seed 2) + threshold"),
           ("grpo_nodefer", "GRPO control (forced choice)"), ("deferdx", "Group-consensus reward"),
           ("zs_defer", "Zero-shot, prompted DEFER"), ("zs_nodefer", "Zero-shot, forced choice"),
           ("abl_cev_eta1", "DEFER-Dx, handoff score eta 1.0 (step 100)"), ("cev_step100", "DEFER-Dx at step 100")]


def top(r):
    if r.terminal == "commit":
        return r.diagnosis
    if r.differential_probs:
        return max(r.differential_probs, key=r.differential_probs.get)
    return r.differential[0] if r.differential else None


def escalated(r) -> bool:
    return not (r.terminal == "commit" and r.diagnosis != OTHER)


def outcome(r) -> str:
    """correct | escalated | wrong (in scope: committed to the wrong condition; out of scope: committed to one of the
    four) | none (malformed or out of budget)"""
    if r.terminal == "commit":
        if r.label == OTHER:
            return "escalated" if r.diagnosis == OTHER else "wrong"
        return "correct" if r.diagnosis == r.label else ("escalated" if r.diagnosis == OTHER else "wrong")
    return "escalated" if r.terminal == "defer" else "none"


def boot(rs, stat, n=2000, seed=0):
    if not rs:
        return None
    g = S.by_case(rs)
    ids = sorted(g)
    rng = np.random.default_rng(seed)
    vals = np.array([stat([x for i in rng.integers(0, len(ids), len(ids)) for x in g[ids[i]]]) for _ in range(n)])
    vals = vals[np.isfinite(vals)]
    return {"value": float(stat(rs)), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)),
            "n_cases": len(ids)}


def rate(fn):
    return lambda rs: float(np.mean([fn(r) for r in rs])) if rs else float("nan")


def suites(ev: Path) -> dict[str, list]:
    out = {}
    for name, _ in SYSTEMS:
        base = name.split("+")[0]
        d = ev / f"cr_{base}"
        if not suite_complete(d):  # an evaluation still running, or one that did not finish every seed
            continue
        cr = load_suite(d).get(SET, [])
        if name.endswith("+thr"):
            if not (ev / "cev").exists() or not (ev / base).exists():
                continue
            main = load_suite(ev / "cev")
            target = S.coverage([r for s in ("eval_cdm_val", "eval_cdm_test") for r in main.get(s, [])])
            ctrl = load_suite(ev / base)
            thr, keep = coverage_rule([r for s in ("eval_cdm_val", "eval_cdm_test") for r in ctrl.get(s, [])], target)
            cr = apply_threshold(cr, thr, keep)
        if cr:
            out[name] = cr
    return out


def main() -> None:
    cases = {c.case_id: c for c in load_cases(ROOT / "benchmarks/casereports/casereports.jsonl")}
    sy = suites(ROOT / "outputs/eval")
    res = {"description": __doc__.split("\n\n")[1].strip(), "n_cases": len(cases), "systems": {}, "per_case": {}}
    groups = {
        "out_of_scope": lambda c: c.label == OTHER,
        "held_out_time_critical": lambda c: c.meta.get("training_group") == "held_out",
        "new_groups": lambda c: c.meta.get("training_group") == "new",
        "in_scope": lambda c: c.label != OTHER,
        "in_scope_complicated": lambda c: c.label != OTHER and c.meta.get("complicated"),
        "in_scope_uncomplicated": lambda c: c.label != OTHER and not c.meta.get("complicated"),
    }
    for name, label in SYSTEMS:
        if name not in sy:
            continue
        rs = sy[name]
        e = {"label": label}
        for g, keep in groups.items():
            sub = [r for r in rs if keep(cases[r.case_id])]
            if not sub:
                continue
            m = {"escalated": boot(sub, rate(lambda r: outcome(r) == "escalated")),
                 "wrong": boot(sub, rate(lambda r: outcome(r) == "wrong")),
                 "no_answer": boot(sub, rate(lambda r: outcome(r) == "none"))}
            if g in ("in_scope", "in_scope_complicated", "in_scope_uncomplicated"):
                m["correct"] = boot(sub, rate(lambda r: outcome(r) == "correct"))
            else:
                esc = [r for r in sub if outcome(r) == "escalated"]
                m["handoff_top_in_scope"] = (float(np.mean([top(r) not in (None, OTHER) for r in esc]))
                                             if esc else None)
            m["tests_per_case"] = float(np.mean([r.n_tests for r in sub]))
            e[g] = m
        res["systems"][name] = e
        for cid in cases:
            outs = [outcome(r) for r in rs if r.case_id == cid]
            if outs:
                res["per_case"].setdefault(cid, {})[name] = max(set(outs), key=outs.count) if outs.count(
                    max(set(outs), key=outs.count)) > 1 else "mixed"
        oos, ins = e.get("out_of_scope"), e.get("in_scope")
        print(f"{label:44s} OOS wrong {oos['wrong']['value']*100:5.1f} escalated {oos['escalated']['value']*100:5.1f} "
              f"no answer {oos['no_answer']['value']*100:5.1f} "
              f"(held-out wrong {e['held_out_time_critical']['wrong']['value']*100:5.1f}) | in-scope correct "
              f"{ins['correct']['value']*100:5.1f} escalated {ins['escalated']['value']*100:5.1f} (complicated "
              f"{e['in_scope_complicated']['escalated']['value']*100:5.1f}, uncomplicated "
              f"{e['in_scope_uncomplicated']['escalated']['value']*100:5.1f})")
    res["cases"] = {cid: {"label": c.label, "group": c.meta.get("group"), "training_group": c.meta.get("training_group"),
                          "complicated": c.meta.get("complicated"), "time_critical": c.meta.get("time_critical")}
                    for cid, c in cases.items()}
    (ROOT / "docs/casereports.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote docs/casereports.json")


if __name__ == "__main__":
    main()
