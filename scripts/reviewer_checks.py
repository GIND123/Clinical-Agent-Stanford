#!/usr/bin/env python
"""Three reviewer checks that need no GPU (docs/REVIEWER_CHECKS.md). Aggregates only: no case text, no IDs.

  A. Chart artifacts. TEST(x) on a case where x was never performed returns "Not performed for this
     patient", so an agent can learn from WHICH tests a case has, not only from their results. A
     cross-validated logistic regression on the 22 availability bits alone measures how much that
     reveals: (1) the in-set diagnosis, on the CDM training cases; (2) OTHER vs in-set, on the eval
     cases built by the open-world pipeline (controls vs OTHER), so the pipeline itself is held fixed.
  B. Price sensitivity. With cost_scale: auto, ordering every test costs one correct diagnosis, so a
     test's reward cost is its price over the catalog total. Reports the weight of the placeholder
     prices and how far halving or doubling them moves any test's cost; with --train-log, the mean size
     of each reward component over a run's last 50 steps.
  C. Multiple comparisons. Holm over every paired test in docs/results.json, and within the three
     comparisons the abstract reports.

    python scripts/reviewer_checks.py --json docs/reviewer_checks.json \
        --train-log outputs/runs/deferdx_seed1/train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from deferdx.data.io import load_cases
from deferdx.env.catalog import TestCatalog

CATALOG = "configs/test_catalog.yaml"
# the paired tests behind the abstract's claims (docs/ABSTRACT.md)
ABSTRACT_TESTS = [
    ("cev@sc vs zs_nodefer@sc+thr", "selective_acc"),
    ("cev vs deferdx", "false_commit:eval_other_unseen"),
    ("cev_seed1 vs deferdx", "false_commit:eval_other_unseen"),
]


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def cv_proba(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return cross_val_predict(LogisticRegression(max_iter=2000), X, y, method="predict_proba",
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def chart_artifacts(cohorts: Path, seed: int) -> dict:
    cat = TestCatalog.from_yaml(CATALOG)
    keys = sorted(cat.tests)

    def avail(cases):
        return np.array([[cat.resolve_test(k, c) is not None for k in keys] for c in cases], dtype=float)

    train = [c for c in load_cases(cohorts / "rl_train.jsonl") if c.source == "cdm"]
    y, X = np.array([c.label for c in train]), avail(train)
    classes = sorted(set(y))
    P = cv_proba(X, y, seed)
    fit = LogisticRegression(max_iter=2000).fit(X, y)
    top = {}
    for i, cl in enumerate(fit.classes_):
        top[cl] = [{"test": keys[j], "recorded_in_class": round(X[y == cl, j].mean(), 3),
                    "recorded_in_others": round(X[y != cl, j].mean(), 3)} for j in np.argsort(-fit.coef_[i])[:3]]
    same = [c for n in ("eval_controls", "eval_other_seen", "eval_other_unseen") for c in load_cases(cohorts / f"{n}.jsonl")]
    yo = np.array([c.source != "openworld_control" for c in same], dtype=int)
    return {
        "in_set": {"n": len(y), "cv_accuracy": round(float((np.array(classes)[P.argmax(1)] == y).mean()), 3),
                   "majority_class": round(Counter(y).most_common(1)[0][1] / len(y), 3),
                   "macro_auroc": round(float(roc_auc_score(y, P, multi_class="ovr", labels=classes)), 3),
                   "most_predictive": top},
        "other_vs_in_set_same_pipeline": {"n": len(yo), "n_other": int(yo.sum()),
                                          "cv_auroc": round(float(roc_auc_score(yo, cv_proba(avail(same), yo, seed)[:, 1])), 3)},
    }


def prices(train_log: Path | None) -> dict:
    tests = yaml.safe_load(open(CATALOG))["tests"]
    price = {k: float(v["cost"]) for k, v in tests.items()}
    ph = sorted(k for k, v in tests.items() if v.get("cost_source") == "placeholder")
    total = sum(price.values())
    cost = {k: p / total for k, p in price.items()}
    out = {"n_tests": len(price), "n_placeholder": len(ph), "catalog_total_usd": total,
           "placeholder_share_of_total": round(sum(price[k] for k in ph) / total, 3),
           "placeholder_reward_cost": {k: round(cost[k], 3) for k in ph},
           "reward_cost_median": round(float(np.median(list(cost.values()))), 4),
           "reward_cost_max": {max(cost, key=cost.get): round(max(cost.values()), 3)},
           "placeholder_scaled": {}}
    for f in (0.5, 2.0):
        t2 = sum(p * (f if k in ph else 1) for k, p in price.items())
        shift = {k: abs(p * (f if k in ph else 1) / t2 - cost[k]) for k, p in price.items()}
        out["placeholder_scaled"][f"x{f}"] = {"max_shift_any_test": round(max(shift.values()), 3),
                                              "max_shift_test": max(shift, key=shift.get),
                                              "median_shift": round(float(np.median(list(shift.values()))), 4)}
    if train_log and train_log.exists():
        rows = [json.loads(line) for line in open(train_log) if line.strip()][-50:]
        out["reward_components_last50"] = {
            "run": str(train_log),
            **{k: round(float(np.mean([abs(r[k]) for r in rows if r.get(k) is not None])), 3)
               for k in ("r_accuracy", "r_calibration", "r_severity", "r_consensus", "r_cost")}}
    return out


def multiple_comparisons(results: Path) -> dict:
    paired = json.load(open(results))["paired"]
    tests = [(c, m, v["p"]) for c, ms in paired.items() for m, v in ms.items()
             if isinstance(v, dict) and v.get("p") is not None]
    adj_all = holm([p for *_, p in tests])
    idx = {(c, m): i for i, (c, m, _) in enumerate(tests)}
    fam = [tests[idx[t]][2] for t in ABSTRACT_TESTS]
    return {"n_tests": len(tests), "raw_below_0.05": sum(p < 0.05 for *_, p in tests),
            "holm_below_0.05": sum(a < 0.05 for a in adj_all),
            "abstract_tests": [{"comparison": c, "metric": m, "p": tests[idx[(c, m)]][2],
                                "holm_within_abstract": round(a, 4), "holm_all": round(adj_all[idx[(c, m)]], 4)}
                               for (c, m), a in zip(ABSTRACT_TESTS, holm(fam))]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--results", type=Path, default=Path("docs/results.json"))
    ap.add_argument("--train-log", type=Path, help="a run's train_log.jsonl, for the reward-component sizes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="write the results here")
    a = ap.parse_args()
    out = {"chart_artifacts": chart_artifacts(a.cohorts, a.seed), "prices": prices(a.train_log),
           "multiple_comparisons": multiple_comparisons(a.results)}
    print(json.dumps(out, indent=2))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
