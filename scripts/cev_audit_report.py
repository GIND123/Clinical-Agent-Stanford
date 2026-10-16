#!/usr/bin/env python
"""Summarise the CEV estimator audit (scripts/cev_audit.py) into docs/cev_audit.json (aggregates only).

For each audited deferral state s with K = 16 forced continuations, V16(s) is their mean return and the credit
used in training is A = E - V_hat(s), with E = escalation value - handoff cost - nu (nu on in-set cases only;
the run's final coverage multiplier). Reports:
  noise      for K = 1, 2, 3, 4, 8: the share of random K-subsets whose credit has the same sign as with all 16,
             and the standard deviation of V_hat_K around V16
  cost       the share of states whose credit changes sign when the cost of tests ordered after s is removed from
             the continuations (deferral credited only because continuing costs tests), and how often continuations
             ordered any test
  clinician  the share of states credited (A16 > 0) when the escalation value assumes a clinician who decides
             correctly with probability a, for a from 0.50 to 1.00 (in-set: the calibrated-decision value at a;
             OTHER: a x the open-world deferral reward)
  outcomes   accuracy of the forced continuations (would the agent have been right had it not escalated?)

    python scripts/cev_audit_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.config import load_config  # noqa: E402
from deferdx.env import TestCatalog  # noqa: E402
from deferdx.rewards import SeverityMatrix  # noqa: E402
from deferdx.rewards.scoring import RewardConfig, expected_commit_reward  # noqa: E402

RUNS = {"cev_s0": "outputs/runs/cev/train_log.jsonl", "cev_s1": "outputs/runs/cev_seed1/train_log.jsonl"}
KS = (1, 2, 3, 4, 8)
ACC_GRID = [round(x, 2) for x in np.arange(0.50, 1.0001, 0.05)]


def final_nu(path: Path) -> float:
    last = None
    for line in path.open():
        last = line
    return float(json.loads(last).get("nu", 0.0))


def value(c: dict, with_cost: bool = True) -> float:
    return c["accuracy"] + c["calibration"] + c["severity"] + c["penalty"] + (c["cost"] if with_cost else 0.0)


def summarise(name: str, rcfg: RewardConfig, nu: float, rng: np.random.Generator, n_draws: int = 2000) -> dict:
    roots = [json.loads(l) for l in (ROOT / "outputs/audit" / f"{name}.jsonl").open()]
    out = {"n_states": len(roots), "nu_final": nu}
    for subset, keep in (("in_set", lambda r: r["in_set"]), ("other", lambda r: not r["in_set"]), ("all", lambda r: True)):
        rs = [r for r in roots if keep(r)]
        if not rs:
            continue
        E = np.array([r["escalation_value"] - rcfg.handoff_mu - (nu if r["in_set"] else 0.0) for r in rs])
        V = [np.array([value(c) for c in r["continuations"]]) for r in rs]
        Vn = [np.array([value(c, False) for c in r["continuations"]]) for r in rs]
        V16 = np.array([v.mean() for v in V])
        A16 = E - V16
        noise = {}
        for k in KS:
            agree, sd = [], []
            for v, e, a16 in zip(V, E, A16):
                idx = np.array([rng.choice(len(v), k, replace=False) for _ in range(n_draws)])
                vk = v[idx].mean(axis=1)
                agree.append(np.mean(np.sign(e - vk) == np.sign(a16)))
                sd.append(np.sqrt(np.mean((vk - v.mean()) ** 2)))
            noise[f"K{k}"] = {"sign_agreement_with_K16": float(np.mean(agree)), "sd_of_V_hat": float(np.mean(sd))}
        A16n = E - np.array([v.mean() for v in Vn])
        conts = [c for r in rs for c in r["continuations"]]
        cost = {"credited_with_cost": float(np.mean(A16 > 0)), "credited_without_cost": float(np.mean(A16n > 0)),
                "sign_changes_without_cost": float(np.mean(np.sign(A16) != np.sign(A16n))),
                "credited_only_because_of_cost": float(np.mean((A16 > 0) & (A16n <= 0))),
                "continuations_ordering_a_test": float(np.mean([c["n_extra_tests"] > 0 for c in conts])),
                "mean_extra_tests": float(np.mean([c["n_extra_tests"] for c in conts])),
                "mean_cost_term": float(np.mean([c["cost"] for c in conts]))}
        clin = {}
        for acc in ACC_GRID:
            Ea = np.array([(expected_commit_reward(acc, rcfg) if r["in_set"] else acc * rcfg.openworld_defer_reward)
                           + rcfg.cev_eta * r["handoff_score"] - rcfg.handoff_mu - (nu if r["in_set"] else 0.0)
                           for r in rs])
            clin[f"{acc:.2f}"] = float(np.mean(Ea - V16 > 0))
        outcomes = {"continuation_accuracy": float(np.mean([c["correct"] for c in conts])),
                    "continuation_commit_rate": float(np.mean([c["terminal"] == "commit" for c in conts])),
                    "states_where_most_continuations_err": float(np.mean([np.mean([c["correct"] for c in r["continuations"]]) < 0.5 for r in rs])),
                    "mean_V16": float(V16.mean()), "mean_E": float(E.mean()), "credited_share": float(np.mean(A16 > 0))}
        out[subset] = {"n": len(rs), "noise": noise, "cost": cost, "clinician_accuracy": clin, "outcomes": outcomes}
    return out


def main() -> None:
    cfg = load_config(ROOT / "configs/grpo_cev.yaml")
    catalog = TestCatalog.from_yaml(ROOT / cfg["test_catalog"])
    rcfg = RewardConfig.from_dict(cfg["reward"], SeverityMatrix.from_yaml(ROOT / cfg["reward"]["severity_matrix"]),
                                  total_test_cost=catalog.total_test_cost)
    rng = np.random.default_rng(0)
    res = {"description": __doc__.split("\n\n")[1].strip(), "tau": rcfg.tau,
           "tau_decision_value": expected_commit_reward(rcfg.tau, rcfg), "runs": {}}
    for name, log in RUNS.items():
        if (ROOT / "outputs/audit" / f"{name}.jsonl").exists():
            res["runs"][name] = summarise(name, rcfg, final_nu(ROOT / log), rng)
            r = res["runs"][name]
            for sub in ("in_set", "other"):
                if sub in r:
                    s = r[sub]
                    print(f"{name} {sub:6s} n={s['n']:3d} credited {s['outcomes']['credited_share']*100:5.1f}% | K3 sign agreement "
                          f"{s['noise']['K3']['sign_agreement_with_K16']*100:5.1f}% (K1 {s['noise']['K1']['sign_agreement_with_K16']*100:.1f}, "
                          f"K8 {s['noise']['K8']['sign_agreement_with_K16']*100:.1f}) | sign changes without cost "
                          f"{s['cost']['sign_changes_without_cost']*100:4.1f}% | cont. acc {s['outcomes']['continuation_accuracy']*100:5.1f}% "
                          f"| tests ordered in {s['cost']['continuations_ordering_a_test']*100:4.1f}% | credited at a=0.70 "
                          f"{s['clinician_accuracy']['0.70']*100:5.1f}%, a=0.85 {s['clinician_accuracy']['0.85']*100:5.1f}%, "
                          f"a=1.00 {s['clinician_accuracy']['1.00']*100:5.1f}%")
    (ROOT / "docs/cev_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote docs/cev_audit.json")


if __name__ == "__main__":
    main()
