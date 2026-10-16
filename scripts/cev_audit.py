#!/usr/bin/env python
"""Audit of the counterfactual escalation value (CEV) estimator: inference only, no training.

The trainer credits a deferral at state s with A = E - V_hat(s), where V_hat(s) is the mean return of
K = 3 forced continuations from s (DEFER disabled). This script takes deferral states of a trained policy
on held-out cases, the exact states the evaluation recorded, and samples K_AUDIT continuations from each with
the trainer's own `branch_rollouts`, sampling as in training (temperature 1.0). For every continuation it
stores the components of its return (accuracy, proper score, severity, cost of tests ordered after s), so
that scripts/cev_audit_report.py can answer, without retraining:
  1. estimator noise: how often does a K = 3 estimate give the credit the same sign as K = 16?
  2. cost: how much of the credit for deferring comes from the test cost of continuing?
  3. clinician accuracy: how does the share of credited deferrals change if the clinician who receives the
     handoff is less reliable than the tau-reliable decision the escalation value assumes?

Writes numbers only (no case text) to outputs/audit/<name>.jsonl (git-ignored, case ids are MIMIC-derived).

    python scripts/cev_audit.py --name cev_s0 --eval outputs/eval/cev/s0.jsonl --adapter outputs/runs/cev/final
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.config import load_config  # noqa: E402
from deferdx.data.io import iter_jsonl, load_cases  # noqa: E402
from deferdx.env import EnvConfig, TestCatalog  # noqa: E402
from deferdx.env.environment import EpisodeResult  # noqa: E402
from deferdx.labels import OTHER  # noqa: E402
from deferdx.rewards import SeverityMatrix  # noqa: E402
from deferdx.rewards.scoring import RewardConfig, calibration_score, escalation_value, handoff_score  # noqa: E402
from deferdx.rollout import Rollout, branch_rollouts  # noqa: E402

SETS = ("eval_cdm_val", "eval_cdm_test", "eval_other_unseen", "eval_other_seen")


def components(cont: EpisodeResult, prefix_cost: float, rcfg: RewardConfig) -> dict:
    """The terms of scoring.continuation_value, kept apart (their sum is the continuation's return)."""
    extra_cost = max(0.0, cont.total_cost - prefix_cost)
    d = {"terminal": cont.terminal, "cost": -rcfg.cost_scale * extra_cost, "extra_cost_usd": extra_cost}
    if cont.terminal != "commit":
        d.update(accuracy=0.0, calibration=0.0, severity=0.0,
                 penalty=-(rcfg.timeout_penalty if cont.terminal == "timeout" else rcfg.invalid_penalty),
                 correct=0, diagnosis=None, probability=None)
        return d
    correct = float(cont.diagnosis == cont.label)
    p = float(cont.probability if cont.probability is not None else 0.0)
    d.update(accuracy=rcfg.alpha * correct, calibration=rcfg.calib_lambda * calibration_score(p, correct, rcfg),
             severity=0.0 if correct else -rcfg.severity_kappa * rcfg.severity(cont.label, cont.diagnosis),
             penalty=0.0, correct=int(correct), diagnosis=cont.diagnosis, probability=p)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval", required=True, help="evaluation rollouts (s<seed>.jsonl) of the policy")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--config", default="configs/grpo_cev.yaml")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-inset", type=int, default=100, help="deferral states on CDM val + test")
    ap.add_argument("--max-other", type=int, default=60, help="deferral states on OTHER (seen + unseen)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    ap.add_argument("--out", default="outputs/audit")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    catalog = TestCatalog.from_yaml(ROOT / cfg["test_catalog"])
    env_cfg = EnvConfig.from_dict(cfg.get("env", {}))
    severity = SeverityMatrix.from_yaml(ROOT / cfg["reward"]["severity_matrix"])
    rcfg = RewardConfig.from_dict(cfg["reward"], severity, total_test_cost=catalog.total_test_cost)

    cases = {c.case_id: c for s in SETS for c in load_cases(ROOT / "data" / "cohorts" / f"{s}.jsonl")}
    rows = [r for r in iter_jsonl(ROOT / args.eval) if r.get("set") in SETS]
    roots = []
    for r in rows:
        res = EpisodeResult.from_dict(r["result"])
        msgs = r.get("messages") or []
        if res.terminal == "defer" and msgs and msgs[-1]["role"] == "assistant":
            roots.append((r["set"], Rollout(r["case_id"], int(r.get("sample_idx", 0)), msgs, [], res)))
    rng = random.Random(args.seed)
    inset = [x for x in roots if x[0].startswith("eval_cdm")]
    other = [x for x in roots if x[0].startswith("eval_other")]
    picked = rng.sample(inset, min(args.max_inset, len(inset))) + rng.sample(other, min(args.max_other, len(other)))
    print(f"[cev-audit] {args.name}: {len(roots)} deferral states ({len(inset)} in-set, {len(other)} OTHER); "
          f"auditing {len(picked)} with K = {args.k}", flush=True)

    from deferdx.policy.vllm_policy import VLLMPolicy

    gen = cfg.get("generation", {})
    policy = VLLMPolicy(args.model, str(ROOT / args.adapter), max_new_tokens=int(gen.get("max_new_tokens", 1024)),
                        temperature=float(gen.get("temperature", 1.0)), top_p=float(gen.get("top_p", 1.0)),
                        top_k=gen.get("top_k"), chat_template_kwargs=cfg.get("chat_template_kwargs",
                                                                             {"enable_thinking": True}),
                        seed=args.seed, max_model_len=int(cfg.get("max_model_len", 20480)),
                        gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
                        request_seed=args.seed)
    t0 = time.time()
    branches = branch_rollouts(policy, catalog, env_cfg, [(cases[ro.case_id], ro) for _, ro in picked], args.k)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for (set_name, ro), conts in zip(picked, branches):
            res = ro.result
            rec = {
                "case_id": ro.case_id, "set": set_name, "label": res.label, "in_set": res.label != OTHER,
                "group": cases[ro.case_id].meta.get("group") or res.label,
                "n_tests_before": res.n_tests, "prefix_cost_usd": res.total_cost,
                "handoff_score": handoff_score(res), "escalation_value": escalation_value(res, rcfg),
                "handoff_top": (res.differential or [None])[0],
                "continuations": [components(c.result, res.total_cost, rcfg) | {
                    "n_extra_tests": c.result.n_tests - res.n_tests} for c in conts],
            }
            f.write(json.dumps(rec) + "\n")
    meta = {"name": args.name, "eval": args.eval, "adapter": args.adapter, "k": args.k, "n_roots": len(picked),
            "n_inset": sum(s.startswith("eval_cdm") for s, _ in picked), "seconds": round(time.time() - t0, 1),
            "sampling": {"temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 1.0)),
                         "max_new_tokens": int(gen.get("max_new_tokens", 1024))},
            "reward": {k: getattr(rcfg, k) for k in ("alpha", "calib_lambda", "severity_kappa", "cost_scale", "tau",
                                                     "handoff_mu", "openworld_defer_reward", "cev_eta")}}
    (out / f"{args.name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cev-audit] {args.name}: {len(picked) * args.k} continuations in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
