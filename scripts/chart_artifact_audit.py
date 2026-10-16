#!/usr/bin/env python
"""Chart-artifact audit (reviewer comment: "the policy may exploit which tests were recorded").

docs/REVIEWER_CHECKS.md section A showed that the pattern of recorded tests predicts the diagnosis. This asks
whether the trained agents act on that signal. In the held-out CDM episodes (val + test, 3 seeds) of each system:
  * how often a TEST reply was "not performed" (the environment's only availability signal), per episode;
  * accuracy on answered cases and escalation, for episodes with and without such a reply;
  * for each test, the share of orders that came back "not performed".
If an agent exploited availability, its accuracy after a "not performed" reply would be higher than without one,
and it would order tests whose absence is informative. Writes docs/chart_artifacts.json (aggregates only).

    python scripts/chart_artifact_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.eval import stats as S  # noqa: E402
from deferdx.eval.report import load_suite  # noqa: E402

CDM = ("eval_cdm_val", "eval_cdm_test")
SYSTEMS = [("cev", "DEFER-Dx"), ("cev_seed1", "DEFER-Dx, second seed"), ("grpo_nodefer", "GRPO control"),
           ("grpo_nodefer_seed1", "GRPO control, second seed"), ("deferdx", "Group-consensus reward"),
           ("zs_nodefer", "Zero-shot, forced choice")]


def unavailable(r) -> int:
    return sum(1 for st in r.steps if st.action.get("type") == "TEST" and st.available is False)


def boot_diff(a, b, stat, n=2000, seed=0):
    """stat(a) - stat(b) for two groups of episodes, cases resampled within each group."""
    ga, gb = S.by_case(a), S.by_case(b)
    ia, ib = sorted(ga), sorted(gb)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        pa = [x for i in rng.integers(0, len(ia), len(ia)) for x in ga[ia[i]]]
        pb = [x for i in rng.integers(0, len(ib), len(ib)) for x in gb[ib[i]]]
        d.append(stat(pa) - stat(pb))
    d = np.array([v for v in d if v == v])
    return {"diff": stat(a) - stat(b), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}


def main() -> None:
    res = {"description": __doc__.split("\n\n")[1].strip(), "systems": {}}
    for name, label in SYSTEMS:
        d = ROOT / "outputs/eval" / name
        if not d.exists():
            continue
        rs = [r for s in CDM for r in load_suite(d).get(s, [])]
        with_na = [r for r in rs if unavailable(r) > 0]
        without = [r for r in rs if unavailable(r) == 0]
        orders, missing = Counter(), Counter()
        for r in rs:
            for st in r.steps:
                if st.action.get("type") == "TEST":
                    orders[st.action.get("test")] += 1
                    missing[st.action.get("test")] += st.available is False
        esc = lambda x: float(np.mean([r.terminal != "commit" for r in x])) if x else float("nan")  # noqa: E731
        e = {"label": label, "episodes": len(rs),
             "share_with_not_performed": len(with_na) / len(rs),
             "not_performed_per_episode": float(np.mean([unavailable(r) for r in rs])),
             "tests_per_episode": float(np.mean([r.n_tests for r in rs])),
             "selective_acc_with": S.selective_acc(with_na) if with_na else None,
             "selective_acc_without": S.selective_acc(without) if without else None,
             "escalated_with": esc(with_na), "escalated_without": esc(without),
             "not_performed_share_by_test": {t: missing[t] / orders[t] for t in sorted(orders) if orders[t] >= 20}}
        if len(with_na) >= 30 and len(without) >= 30:
            e["selective_acc_with_minus_without"] = boot_diff(with_na, without, S.selective_acc)
        res["systems"][name] = e
        diff = e.get("selective_acc_with_minus_without")
        print(f"{label:30s} episodes with a 'not performed' reply {e['share_with_not_performed']*100:5.1f}% "
              f"({e['not_performed_per_episode']:.2f}/episode; {e['tests_per_episode']:.2f} tests) | acc on answered with "
              f"{(e['selective_acc_with'] or float('nan'))*100:5.1f} vs without {(e['selective_acc_without'] or float('nan'))*100:5.1f}"
              + (f" ({diff['diff']*100:+.1f}, {diff['lo']*100:+.1f} to {diff['hi']*100:+.1f})" if diff else ""))
    (ROOT / "docs/chart_artifacts.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote docs/chart_artifacts.json")


if __name__ == "__main__":
    main()
