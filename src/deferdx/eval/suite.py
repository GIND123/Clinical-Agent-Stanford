"""Run one model over several evaluation sets and seeds with a single vLLM engine.

Every set is concatenated into one batch per seed, so vLLM sees the largest possible
batch at each turn. Sampling uses per-request seeds derived from (seed, case, sample,
turn): a run is reproducible for a given seed and different seeds are independent.

Writes <out>/<name>/s<seed>.jsonl (rollouts: contain MIMIC text, git-ignored) and
<out>/<name>/summary.json (aggregates per set and seed only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..data.io import load_cases, write_jsonl
from ..env.catalog import TestCatalog
from ..env.environment import EnvConfig
from ..rewards.scoring import SeverityMatrix
from ..rollout import run_episodes
from .evaluate import summarize


def set_name(path: str | Path) -> str:
    return Path(path).stem


def run_suite(model: str, adapter: str | None, sets: list[str], seeds: list[int], name: str, out_dir: str | Path,
              catalog: TestCatalog, env_cfg: EnvConfig, severity: SeverityMatrix, temperature: float = 0.6,
              top_p: float = 0.95, top_k: int | None = 20, max_new_tokens: int = 1024, max_model_len: int = 20480,
              gpu_memory_utilization: float = 0.88, chat_template_kwargs: dict | None = None,
              max_lora_rank: int = 16, limit: int | None = None) -> dict[str, Any]:
    from ..policy.vllm_policy import VLLMPolicy

    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    cases, set_of = [], {}
    for path in sets:
        cs = load_cases(path)[: limit or None]
        for c in cs:
            set_of[c.case_id] = set_name(path)
        cases.extend(cs)
    policy = VLLMPolicy(model, adapter, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                        top_k=top_k, chat_template_kwargs=chat_template_kwargs or {"enable_thinking": True},
                        seed=0, max_lora_rank=max_lora_rank, max_model_len=max_model_len,
                        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True)
    summary: dict[str, Any] = {"model": model, "adapter": adapter, "sets": [set_name(p) for p in sets],
                               "seeds": seeds, "env": env_cfg.__dict__, "sampling": {
                                   "temperature": temperature, "top_p": top_p, "top_k": top_k,
                                   "max_new_tokens": max_new_tokens}, "results": {}}
    for seed in seeds:
        t0 = time.time()
        policy.request_seed = seed
        rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=1)
        rows = []
        for ro in rollouts:
            d = ro.to_dict()
            d["set"], d["seed"] = set_of[ro.case_id], seed
            rows.append(d)
        write_jsonl(out / f"s{seed}.jsonl", rows)
        per_set = {}
        for s in summary["sets"]:
            res = [ro.result for ro in rollouts if set_of[ro.case_id] == s]
            rep = summarize(res, severity)
            rep.get("selective", {}).pop("curve", None)
            per_set[s] = rep
        summary["results"][f"s{seed}"] = per_set
        summary.setdefault("seconds", {})[f"s{seed}"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[eval-suite] {name} seed {seed}: {len(rollouts)} episodes in {time.time() - t0:.0f}s", flush=True)
    return summary
