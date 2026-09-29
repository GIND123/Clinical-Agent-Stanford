"""Stage 2: multi-turn GRPO with the group-consensus deferral reward.

Per step:
  1. sample B cases, roll out G episodes each with the current policy (temperature 1.0);
  2. score each group with `group_rewards` (p_hat comes from that same group), charging
     the coverage-constraint multiplier nu_k to deferring rollouts, then update nu;
  3. group-normalised advantages; optionally drop zero-variance groups (DAPO dynamic
     sampling), since they carry no gradient;
  4. clipped-surrogate policy gradient on every assistant turn's exact sampled tokens,
     token-level averaged over the step (DAPO), with asymmetric clip (clip-higher).

Single-process reference implementation. The reward functions are framework-agnostic;
see docs/TRAINING.md for porting them into verl for multi-node runs.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..data.io import load_cases, write_jsonl
from ..data.schema import Case
from ..env.catalog import TestCatalog
from ..env.environment import EnvConfig
from ..eval.evaluate import summarize
from ..policy.hf import HFPolicy
from ..rewards.constraint import CoverageConstraint
from ..rewards.scoring import RewardConfig, SeverityMatrix, crossover_p_hat, group_rewards
from ..rollout import group_by_case, run_episodes
from .common import (
    JsonlLogger,
    TurnSample,
    completion_logprobs,
    cosine_lr,
    load_model_and_tokenizer,
    save_checkpoint,
    set_seed,
)


def linear_schedule(step: int, spec: dict | float | None, default: float) -> float:
    if spec is None:
        return default
    if isinstance(spec, (int, float)):
        return float(spec)
    n = max(1, int(spec.get("steps", 1)))
    frac = min(1.0, step / n)
    return float(spec["start"] + frac * (spec["end"] - spec["start"]))


def group_advantages(rewards: list[float], normalize_std: bool = True, eps: float = 1e-6) -> list[float]:
    r = np.asarray(rewards, dtype=float)
    adv = r - r.mean()
    if normalize_std:
        adv = adv / (r.std() + eps)
    return adv.tolist()


def grpo_loss(model, samples: list[TurnSample], pad_id: int, n_tok_total: int, clip_low: float = 0.2,
              clip_high: float = 0.3, kl_coef: float = 0.0, entropy_coef: float = 0.0):
    """Clipped-surrogate loss for one micro-batch, normalised by the step's token count."""
    import torch

    logps, ents = completion_logprobs(model, samples, pad_id, with_entropy=entropy_coef > 0)
    total = 0.0
    clipped = 0.0
    ent_sum = 0.0
    for i, (s, lp) in enumerate(zip(samples, logps)):
        old = s.old_logprobs.to(lp.device) if s.old_logprobs is not None else lp.detach()
        ratio = torch.exp(lp - old)
        adv = torch.as_tensor(s.advantage, dtype=lp.dtype, device=lp.device)
        unclipped = ratio * adv
        clipped_obj = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv
        term = -torch.minimum(unclipped, clipped_obj).sum()
        clipped += float((unclipped > clipped_obj).float().sum())
        if kl_coef > 0 and s.ref_logprobs is not None:
            d = s.ref_logprobs.to(lp.device) - lp
            term = term + kl_coef * (torch.exp(d) - d - 1).sum()  # k3 estimator
        if ents is not None:
            term = term - entropy_coef * ents[i].sum()
            ent_sum += float(ents[i].detach().sum())
        total = total + term
    loss = total / max(1, n_tok_total)
    return loss, {"clipped_tokens": clipped, "entropy_sum": ent_sum}


def _reward_setup(cfg: dict[str, Any]):
    catalog = TestCatalog.from_yaml(cfg["test_catalog"])
    env_cfg = EnvConfig.from_dict(cfg.get("env"))
    sev_path = cfg.get("reward", {}).get("severity_matrix")
    severity = SeverityMatrix.from_yaml(sev_path) if sev_path else SeverityMatrix.uniform()
    rcfg = RewardConfig.from_dict(cfg.get("reward"), severity)
    return catalog, env_cfg, rcfg, severity


def evaluate_policy(policy, catalog, env_cfg, cases: list[Case], severity) -> tuple[dict, list]:
    rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=1)
    return summarize([r.result for r in rollouts], severity), rollouts


def train_grpo(cfg: dict[str, Any], model=None, tokenizer=None, train_cases: list[Case] | None = None,
               val_cases: list[Case] | None = None) -> Path:
    import torch

    seed = cfg.get("seed", 0)
    set_seed(seed)
    rng = random.Random(seed)
    catalog, env_cfg, rcfg, severity = _reward_setup(cfg)
    if model is None:
        model, tokenizer = load_model_and_tokenizer(cfg)
    train_cases = train_cases if train_cases is not None else load_cases(cfg["train_cases"])
    if val_cases is None and cfg.get("val_cases"):
        val_cases = load_cases(cfg["val_cases"])
    out_dir = Path(cfg["output_dir"])
    logger = JsonlLogger(out_dir / "grpo_log.jsonl")

    gen = dict(cfg.get("generation", {}))
    kw = cfg.get("chat_template_kwargs")
    policy = HFPolicy(model, tokenizer, chat_template_kwargs=kw, do_sample=True, **gen)
    greedy = HFPolicy(model, tokenizer, chat_template_kwargs=kw, do_sample=False,
                      **{k: v for k, v in gen.items() if k not in ("temperature", "top_p")})
    constraint = CoverageConstraint.from_dict(cfg.get("constraint"))
    G = int(cfg.get("group_size", 16))
    B = int(cfg.get("cases_per_step", 8))
    steps = int(cfg.get("steps", 500))
    mbs = int(cfg.get("micro_batch_size", 2))
    ppo_epochs = int(cfg.get("ppo_epochs", 1))
    clip_low, clip_high = cfg.get("clip_low", 0.2), cfg.get("clip_high", 0.3)
    kl_coef, ent_coef = float(cfg.get("kl_coef", 0.0)), float(cfg.get("entropy_coef", 0.0))
    max_seq_len = cfg.get("max_seq_len")
    ent_cfg = cfg.get("action_entropy") or {}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-6)), weight_decay=cfg.get("weight_decay", 0.0))
    total_updates = steps * ppo_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total_updates, cfg.get("warmup_ratio", 0.05)))
    if kl_coef > 0 and not hasattr(model, "disable_adapter"):
        raise NotImplementedError("kl_coef > 0 needs a LoRA model (reference = adapter disabled)")

    q = crossover_p_hat(rcfg)
    print(f"[grpo] nominal tau={rcfg.tau:.2f}; reward-induced defer crossover p_hat~{q}")

    order: list[Case] = []
    for step in range(1, steps + 1):
        t0 = time.time()
        rcfg.tau = linear_schedule(step - 1, cfg.get("tau_schedule"), rcfg.tau)
        rcfg.action_entropy_coef = float(ent_cfg.get("coef", 0.0)) if step <= int(ent_cfg.get("until_step", 0)) else 0.0
        if len(order) < B:
            extra = list(train_cases)
            rng.shuffle(extra)
            order.extend(extra)
        batch, order = order[:B], order[B:]

        # ---- 1. rollouts ------------------------------------------------------
        model.eval()
        rollouts = run_episodes(policy, catalog, env_cfg, batch, n_samples=G)
        groups = group_by_case(rollouts)
        results = [ro.result for ro in rollouts]
        defer_rate = float(np.mean([r.terminal == "defer" for r in results]))

        # ---- 2-3. rewards, constraint, advantages ------------------------------
        nu = constraint.nu
        samples: list[TurnSample] = []
        all_rewards, kept_groups, dropped_long = [], 0, 0
        for grp in groups:
            rbs = group_rewards([ro.result for ro in grp], rcfg, defer_penalty=nu)
            rewards = [rb.total for rb in rbs]
            all_rewards.extend(rewards)
            for ro, rb in zip(grp, rbs):
                ro.reward = rb.as_dict()
            if cfg.get("dynamic_sampling", True) and float(np.std(rewards)) < 1e-8:
                continue
            kept_groups += 1
            for ro, adv in zip(grp, group_advantages(rewards, cfg.get("normalize_std", True))):
                for turn in ro.turns:
                    if not turn.completion_ids:
                        continue
                    if max_seq_len and len(turn.prompt_ids) + len(turn.completion_ids) > max_seq_len:
                        dropped_long += 1
                        continue
                    samples.append(TurnSample(turn.prompt_ids, turn.completion_ids, adv))
        constraint.update(defer_rate)

        # ---- 4. policy update ---------------------------------------------------
        stats = {"loss": 0.0, "clipped_tokens": 0.0, "entropy_sum": 0.0, "grad_norm": 0.0}
        n_tok = sum(len(s.completion_ids) for s in samples)
        if samples:
            model.train()
            if ppo_epochs > 1 or kl_coef > 0:
                with torch.no_grad():
                    for m in range(0, len(samples), mbs):
                        mb = samples[m : m + mbs]
                        if ppo_epochs > 1:
                            lps, _ = completion_logprobs(model, mb, pad_id)
                            for s, lp in zip(mb, lps):
                                s.old_logprobs = lp.detach().cpu()
                        if kl_coef > 0:
                            with model.disable_adapter():
                                lps, _ = completion_logprobs(model, mb, pad_id)
                            for s, lp in zip(mb, lps):
                                s.ref_logprobs = lp.detach().cpu()
            for _ in range(ppo_epochs):
                rng.shuffle(samples)
                for m in range(0, len(samples), mbs):
                    loss, st = grpo_loss(model, samples[m : m + mbs], pad_id, n_tok, clip_low, clip_high,
                                         kl_coef, ent_coef)
                    loss.backward()
                    stats["loss"] += loss.item()
                    stats["clipped_tokens"] += st["clipped_tokens"]
                    stats["entropy_sum"] += st["entropy_sum"]
                stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0)))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        in_set = [r for r in results if r.label != "other"]
        logger.log({
            "step": step,
            "reward_mean": float(np.mean(all_rewards)),
            "reward_std": float(np.std(all_rewards)),
            "commit_acc": float(np.mean([r.correct for r in in_set])) if in_set else math.nan,
            "defer_rate": defer_rate,
            "commit_rate": float(np.mean([r.terminal == "commit" for r in results])),
            "timeout_rate": float(np.mean([r.terminal == "timeout" for r in results])),
            "invalid_rate": float(np.mean([r.terminal == "invalid" for r in results])),
            "mean_tests": float(np.mean([r.n_tests for r in results])),
            "tau": rcfg.tau,
            "nu": nu,
            "kept_groups": kept_groups,
            "turn_samples": len(samples),
            "dropped_long": dropped_long,
            "tokens": n_tok,
            "clip_frac": stats["clipped_tokens"] / max(1, n_tok * ppo_epochs),
            "entropy": stats["entropy_sum"] / max(1, n_tok * ppo_epochs),
            "loss": stats["loss"],
            "grad_norm": stats["grad_norm"],
            "lr": sched.get_last_lr()[0],
            "sec": time.time() - t0,
        })
        if cfg.get("log_rollouts_every") and step % cfg["log_rollouts_every"] == 0:
            write_jsonl(out_dir / "rollouts" / f"step_{step}.jsonl", (ro.to_dict() for ro in rollouts))

        if val_cases and cfg.get("eval_every") and step % cfg["eval_every"] == 0:
            model.eval()
            subset = val_cases[: cfg.get("eval_cases", len(val_cases))]
            rep, _ = evaluate_policy(greedy, catalog, env_cfg, subset, severity)
            cw = rep.get("closed_world", {})
            logger.log({"step": step, "eval_acc_full": cw.get("accuracy_full_coverage", math.nan),
                        "eval_selective_acc": cw.get("selective_accuracy", math.nan),
                        "eval_coverage": cw.get("coverage", math.nan),
                        "eval_aurc": rep.get("selective", {}).get("aurc", math.nan),
                        "eval_ece": rep.get("calibration", {}).get("ece", math.nan)})
        if cfg.get("save_every") and step % cfg["save_every"] == 0:
            save_checkpoint(model, tokenizer, out_dir / f"step_{step}", {"step": step, "constraint": constraint.state()})

    final = out_dir / "final"
    save_checkpoint(model, tokenizer, final, {"steps": steps, "constraint": constraint.state(), "config": cfg})
    return final
