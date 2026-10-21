#!/usr/bin/env python
"""Human-AI team analysis: what happens to the patient once the agent hands a case to a clinician?

The escalation value assumes that a deferred case is decided by a clinician who succeeds with probability tau.
Barrit et al. (medRxiv 2026, doi:10.64898/2026.10.05.26364541) showed in a randomised study of 225 physicians
that AI advocacy redirects clinical decisions (harmful decisions 2.2% with aligned vs 49.1% with adversarial AI,
an adjusted difference of 46.9 per 100) and that expertise does not protect. A handoff is AI advice. This script
models the clinician who receives it:

  AI commits     the AI's diagnosis is the team's decision
  AI escalates   with probability f (susceptibility) the clinician adopts the handoff's top-ranked diagnosis;
                 otherwise they decide independently and are right with probability a_c

a_c defaults to 0.89, the mean diagnostic accuracy of the four physicians in the MIMIC-CDM reader study (Hager et
al., Nat Med 2024: 87.5% for residents, 92.5% for a senior hospitalist, full information). For a post-hoc
threshold the handoff is the withheld commitment. Reported per system (val + test CDM cases; open-world sets):
  * team accuracy on in-set cases, and AI-caused wrong in-scope diagnoses on OTHER cases (a commitment to one of the
    four, or a clinician adopting a handed-over in-scope suggestion), as functions of f and a_c. A COMMIT of
    OTHER names no diagnosis to act on and is treated as a handoff suggesting OTHER;
  * clinician workload (share of cases escalated);
  * parameter-free handoff measures: the share of handoffs whose top suggestion is wrong, and on OTHER cases the
    share whose top suggestion is one of the four conditions (a handoff that would steer towards a wrong
    in-scope diagnosis).
Also a learning-to-defer baseline: the control escalates whenever its stated probability is below a_c (the
Bayes-optimal rejector for a calibrated model and a clinician of accuracy a_c, Mozannar & Sontag 2020), at its own
coverage. Intervals: 95% case-level bootstrap. Writes docs/team.json (aggregates only).

    python scripts/team_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.baselines.selective import apply_threshold  # noqa: E402
from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.report import crossfit_posthoc, load_suite  # noqa: E402
from deferdx.labels import OTHER  # noqa: E402

CDM = ("eval_cdm_val", "eval_cdm_test")
OW = ("eval_other_unseen", "eval_other_seen")
F_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.47, 0.5]
AC_GRID = [0.80, 0.89, 0.95]
BARRIT_F = 0.47  # adversarial-minus-aligned harmful decisions per targeted decision point (46.9 per 100)
# 100-step ablations, compared with DEFER-Dx's step-100 checkpoint (evaluated as cev_step100)
ABLATIONS = ("cev_step100", "abl_cev_eta1", "abl_cev_no_handoff", "abl_constant_defer", "abl_cev_no_cost")


def top(r) -> str | None:
    """The suggestion the clinician receives: the commitment, or the top of the handed-over differential; None for an
    episode that ended without a usable answer (malformed or out of budget)."""
    if r.terminal == "commit":
        return r.diagnosis
    if r.differential_probs:
        return max(r.differential_probs, key=r.differential_probs.get)
    return r.differential[0] if r.differential else None


def escalated(r) -> bool:
    """A case reaches a clinician unless the AI commits to one of the four conditions. A COMMIT of OTHER ("none of
    the four") names no diagnosis to act on, so it is a handoff whose suggestion is OTHER."""
    return not (r.terminal == "commit" and r.diagnosis != OTHER)


def p_correct(r, f: float, ac: float) -> float:
    """Probability that the team's final diagnosis is right on an in-set case."""
    if not escalated(r):
        return float(r.diagnosis == r.label)
    t = top(r)
    if t is None:  # nothing to anchor on: the clinician decides alone
        return ac
    return f * float(t == r.label) + (1 - f) * ac


def p_wrong_inset(r, f: float, ac: float) -> float:
    """On an OTHER case: probability that the AI causes a wrong in-scope diagnosis, by committing to one of the four
    conditions or by handing over one of them as its top suggestion to a clinician who adopts it (probability f).
    The clinician's own independent errors are not counted: they are not attributable to the AI, and the only
    physician accuracy available (Hager et al.) is for the four-way task."""
    if not escalated(r):
        return 1.0
    t = top(r)
    return f * float(t is not None and t != OTHER)


def mean(rs, fn) -> float:
    return float(np.mean([fn(r) for r in rs])) if rs else float("nan")


def boot(rs, stat, n=1000, seed=0):
    g = S.by_case(rs)
    ids = sorted(g)
    rng = np.random.default_rng(seed)
    vals = np.array([stat([x for i in rng.integers(0, len(ids), len(ids)) for x in g[ids[i]]]) for _ in range(n)])
    return {"value": stat(rs), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975))}


def systems(ev: Path, ac: float) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    raw = {n: load_suite(ev / n) for n in ("cev", "cev_seed1", "grpo_nodefer", "grpo_nodefer_seed1", "deferdx", "zs_defer",
                                           *ABLATIONS) if (ev / n).exists()}
    for n in ("cev", "cev_seed1", "deferdx", "zs_defer", *ABLATIONS):
        if n in raw:
            out[n] = raw[n]
    for ctrl in ("grpo_nodefer", "grpo_nodefer_seed1"):
        if ctrl not in raw:
            continue
        for ref in ("cev", "cev_seed1"):
            if ref in raw:
                target = S.coverage([r for s in CDM for r in raw[ref].get(s, [])])
                fitted, _ = crossfit_posthoc(raw[ctrl], target)
                out[f"{ctrl}+thr@{ref}"] = fitted
        # learning to defer: escalate when the stated probability is below the clinician's accuracy
        out[f"{ctrl}+l2d"] = {s: apply_threshold(v, ac) for s, v in raw[ctrl].items()}
    return out


def paired(x: dict, y: dict, ac: float, n_boot: int) -> dict:
    """Paired differences x - y on the same cases: team accuracy on in-set cases and AI-caused wrong in-scope
    diagnoses on never-seen OTHER, with an independent clinician (f = 0) and at Barrit's susceptibility."""
    d = {}
    for f in (0.0, BARRIT_F):
        a = [r for s in CDM for r in x.get(s, [])]
        b = [r for s in CDM for r in y.get(s, [])]
        d[f"team_acc_inset_f{f:.2f}"] = S.paired_bootstrap(a, b, lambda rs, f=f: np.mean([p_correct(r, f, ac) for r in rs]), n_boot)
        a, b = x.get("eval_other_unseen", []), y.get("eval_other_unseen", [])
        d[f"wrong_inset_unseen_f{f:.2f}"] = S.paired_bootstrap(a, b, lambda rs, f=f: np.mean([p_wrong_inset(r, f, ac) for r in rs]),
                                                              n_boot)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--ac", type=float, default=0.89)
    ap.add_argument("--out", default="docs/team.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    sy = systems(ROOT / args.eval_dir, args.ac)
    res = {"description": __doc__.split("\n\n")[1].strip(), "a_c": args.ac, "barrit_f": BARRIT_F, "systems": {}}
    for name, suite in sy.items():
        cdm = [r for s in CDM for r in suite.get(s, [])]
        unseen, seen = suite.get("eval_other_unseen", []), suite.get("eval_other_seen", [])
        esc = [r for r in cdm if escalated(r) and top(r) is not None]
        esc_ow = [r for r in unseen if escalated(r) and top(r) is not None]
        e = {"workload_inset": mean(cdm, escalated),
             "workload_unseen_other": mean(unseen, escalated),
             "handoff_top_wrong_inset": mean(esc, lambda r: top(r) != r.label),
             "handoff_top_inset_on_unseen_other": mean(esc_ow, lambda r: top(r) not in (None, OTHER)),
             "grid": {}}
        for ac in AC_GRID:
            for f in F_GRID:
                e["grid"][f"ac={ac:.2f},f={f:.2f}"] = {
                    "team_acc_inset": mean(cdm, lambda r: p_correct(r, f, ac)),
                    "wrong_inset_on_unseen_other": mean(unseen, lambda r: p_wrong_inset(r, f, ac)),
                    "wrong_inset_on_seen_other": mean(seen, lambda r: p_wrong_inset(r, f, ac))}
        for f in (0.0, BARRIT_F):
            e[f"team_acc_inset_f{f:.2f}"] = boot(cdm, lambda rs: np.mean([p_correct(r, f, args.ac) for r in rs]), args.n_boot)
            e[f"wrong_inset_unseen_f{f:.2f}"] = boot(unseen, lambda rs: np.mean([p_wrong_inset(r, f, args.ac) for r in rs]),
                                                      args.n_boot)
        res["systems"][name] = e
        print(f"{name:32s} workload {e['workload_inset']*100:5.1f}% | handoff top wrong {e['handoff_top_wrong_inset']*100:5.1f}% | "
              f"unseen OTHER handoff top in-set {e['handoff_top_inset_on_unseen_other']*100:5.1f}% | team acc f=0 "
              f"{e['team_acc_inset_f0.00']['value']*100:5.1f} f=.47 {e['team_acc_inset_f0.47']['value']*100:5.1f} | "
              f"unseen wrong f=0 {e['wrong_inset_unseen_f0.00']['value']*100:5.1f} f=.47 {e['wrong_inset_unseen_f0.47']['value']*100:5.1f}")
    # paired differences on the same cases: DEFER-Dx against the thresholded control and the consensus arm
    res["paired"] = {}
    for ref in ("cev", "cev_seed1"):
        for other in (f"grpo_nodefer+thr@{ref}", f"grpo_nodefer_seed1+thr@{ref}", "deferdx", "grpo_nodefer+l2d"):
            if ref not in sy or other not in sy:
                continue
            d = paired(sy[ref], sy[other], args.ac, args.n_boot)
            res["paired"][f"{ref} vs {other}"] = d
            print(f"{ref} vs {other:30s} " + " | ".join(f"{k} {v['diff']*100:+.1f} ({v['lo']*100:+.1f},{v['hi']*100:+.1f})" for k, v in d.items()))
    for other in ABLATIONS[1:]:
        if "cev_step100" in sy and other in sy:
            res["paired"][f"cev_step100 vs {other}"] = d = paired(sy["cev_step100"], sy[other], args.ac, args.n_boot)
            print(f"cev_step100 vs {other:24s} " + " | ".join(f"{k} {v['diff']*100:+.1f} ({v['lo']*100:+.1f},{v['hi']*100:+.1f})" for k, v in d.items()))
    (ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
