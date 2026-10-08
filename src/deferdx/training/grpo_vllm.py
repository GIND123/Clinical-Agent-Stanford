"""Stage 2 on one GPU: colocated multi-turn GRPO with the group-consensus deferral reward.

vLLM samples and PEFT/transformers trains, on the same GPU, every step:

  1. ROLLOUT  vLLM is awake; the HF model's frozen weights are parked in pinned CPU memory.
              B cases (drawn from source pools by weight) x G episodes each, current LoRA,
              temperature 1.0, with vLLM's log-prob of every sampled token.
  2. REWARD   `group_rewards` per case: p_hat comes from the same group; the coverage
              multiplier nu is charged to deferring rollouts on in-set cases (deferring an
              OTHER case is the desired behaviour, so it is not rationed). Advantages are
              r - mean(group) by default (Dr. GRPO): dividing by the group std would blow the
              tiny Brier / cost differences inside all-correct groups up to unit scale.
              Groups whose reward spread is below `min_group_std` carry no learning signal
              and are skipped (DAPO dynamic sampling).
  3. TRAIN    vLLM sleeps (weights to CPU, KV cache freed); frozen weights return to the GPU;
              one clipped-surrogate policy-gradient step over every assistant turn's exact
              sampled tokens, token-level averaged over the step (DAPO), with truncated
              importance sampling against vLLM's log-probs (corrects the small numerical
              mismatch between the two engines; Yao et al. 2025).
  4. SWAP     adapter saved; frozen weights parked; vLLM wakes and serves the new adapter
              under a fresh id.

vLLM's LoRA outputs were checked against HF+PEFT across sleep/wake cycles before this was
written (see policy/vllm_policy.py). Logs hold aggregates only; rollout dumps contain
MIMIC text and stay under the git-ignored output directory.
"""

from __future__ import annotations

import math
import random
import shutil
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

from ..data.io import load_cases, write_jsonl
from ..data.schema import Case
from ..eval.evaluate import summarize
from ..labels import OTHER
from ..rewards.constraint import CoverageConstraint
from ..rewards.scoring import continuation_value, crossover_p_hat, escalation_value, group_rewards, handoff_score
from ..rollout import Rollout, branch_rollouts, group_by_case, run_episodes
from .common import JsonlLogger, TurnSample, completion_logprobs_lowmem, set_seed, token_budget_batches
from .grpo import _reward_setup, group_advantages, linear_schedule


# ---- case sampling ------------------------------------------------------------------------


class MixtureSampler:
    """Draw cases from source pools (case.source: cdm / openworld / openworld_control) by
    weight, without replacement within a pool until it is exhausted, then reshuffled."""

    def __init__(self, cases: list[Case], weights: dict[str, float], rng: random.Random):
        self.rng = rng
        self.pools = {k: [c for c in cases if c.source == k] for k in weights}
        self.weights = {k: float(w) for k, w in weights.items() if self.pools.get(k) and w > 0}
        if not self.weights:
            raise ValueError(f"no non-empty pool among {list(weights)}; sources present: "
                             f"{sorted({c.source for c in cases})}")
        self.queues: dict[str, deque] = {k: deque() for k in self.weights}

    def _next(self, k: str) -> Case:
        if not self.queues[k]:
            items = list(self.pools[k])
            self.rng.shuffle(items)
            self.queues[k].extend(items)
        return self.queues[k].popleft()

    def sample(self, n: int) -> list[Case]:
        names = list(self.weights)
        w = [self.weights[k] for k in names]
        out: list[Case] = []
        seen: set[str] = set()
        tries = 0
        while len(out) < n and tries < 50 * n:
            tries += 1
            c = self._next(self.rng.choices(names, w)[0])
            if c.case_id not in seen:
                seen.add(c.case_id)
                out.append(c)
        return out


# ---- GPU residency of the frozen weights ------------------------------------------------------


class FrozenWeightParker:
    """Pinned CPU copies of all frozen parameters, swapped onto / off the GPU around each
    training phase. Under LoRA the frozen weights never change, so parking only drops the
    GPU copy; nothing is copied back. Trainable (LoRA) parameters and buffers stay on GPU."""

    def __init__(self, model, device: str = "cuda"):
        import torch

        self.device = torch.device(device)
        self.frozen = []
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(self.device)
            else:
                pinned = p.data.cpu().pin_memory()
                p.data = pinned
                self.frozen.append((p, pinned))
        for mod in model.modules():
            for k, b in list(mod._buffers.items()):
                if b is not None:
                    mod._buffers[k] = b.to(self.device)
        self.on_gpu = False

    def load(self) -> None:
        import torch

        for p, pinned in self.frozen:
            p.data = pinned.to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        self.on_gpu = True

    def park(self) -> None:
        import torch

        for p, pinned in self.frozen:
            p.data = pinned
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.on_gpu = False


# ---- loss ---------------------------------------------------------------------------------------


def pg_loss(model, samples: list[TurnSample], pad_id: int, n_tok_total: int, clip_low: float, clip_high: float,
            tis_cap: float | None, chunk: int = 2048, behavior: str = "current"):
    """Clipped surrogate normalised by `n_tok_total` completion tokens. Returns (loss, stats).

    behavior="current": on-policy, ratio = 1 with gradient; tokens weighted by truncated IS
        against the sampling engine's log-probs (one optimizer step per rollout batch).
    behavior="rollout": PPO against the sampling engine's own log-probs, ratio = pi_theta / pi_vLLM,
        clipped. Used when one rollout batch feeds several optimizer steps (mini-batches): the
        later mini-batches are off-policy, and the clipped ratio is their importance correction
        (it also absorbs the small engine mismatch, so no separate IS weight is applied).
    """
    import torch

    logps, ents = completion_logprobs_lowmem(model, samples, pad_id, chunk)
    total = 0.0
    st = {"entropy_sum": 0.0, "tis_w_sum": 0.0, "tis_capped": 0.0, "mismatch_sum": 0.0, "tis_tokens": 0.0,
          "clipped": 0.0}
    for s, lp, ent in zip(samples, logps, ents):
        adv = torch.as_tensor(float(s.advantage), dtype=lp.dtype, device=lp.device)
        has_rl = s.rollout_logprobs is not None and len(s.rollout_logprobs) == len(s.completion_ids)
        if behavior == "rollout" and has_rl:
            rl = torch.tensor(s.rollout_logprobs, dtype=lp.dtype, device=lp.device)
            ok = torch.isfinite(rl)
            old = torch.where(ok, rl, lp.detach())
            ratio = torch.exp(lp - old)
            lo, hi = 1 - clip_low, 1 + clip_high
            surr = -torch.minimum(ratio * adv, torch.clamp(ratio, lo, hi) * adv)
            st["clipped"] += float(((ratio.detach() < lo) | (ratio.detach() > hi)).sum())
            st["mismatch_sum"] += float((lp.detach() - rl)[ok].abs().sum())
            st["tis_w_sum"] += float(ratio.detach().sum())
            st["tis_tokens"] += float(ok.sum())
            total = total + surr.sum()
            st["entropy_sum"] += float(ent.sum())
            continue
        old = lp.detach()
        ratio = torch.exp(lp - old)
        surr = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv)
        if tis_cap and has_rl:
            rl = torch.tensor(s.rollout_logprobs, dtype=lp.dtype, device=lp.device)
            ok = torch.isfinite(rl)
            raw = torch.exp(torch.where(ok, old - rl, torch.zeros_like(rl)))
            w = raw.clamp(max=tis_cap)
            surr = surr * w
            st["tis_w_sum"] += float(w.sum())
            st["tis_capped"] += float((raw > tis_cap).sum())
            st["mismatch_sum"] += float((old - rl)[ok].abs().sum())
            st["tis_tokens"] += float(ok.sum())
        total = total + surr.sum()
        st["entropy_sum"] += float(ent.sum())
    return total / max(1, n_tok_total), st


def _backward_with_split(model, mb: list[TurnSample], pad_id, n_tok, clip_low, clip_high, tis_cap, chunk, stats,
                         behavior: str = "current"):
    """loss.backward() on a micro-batch; on CUDA OOM, retry as two halves (recursively)."""
    import torch

    try:
        loss, st = pg_loss(model, mb, pad_id, n_tok, clip_low, clip_high, tis_cap, chunk, behavior)
        loss.backward()
        stats["loss"] += float(loss.detach())
        for k, v in st.items():
            stats[k] += v
    except torch.OutOfMemoryError:
        if len(mb) == 1:
            raise
        torch.cuda.empty_cache()
        stats["oom_splits"] += 1
        half = len(mb) // 2
        for part in (mb[:half], mb[half:]):
            _backward_with_split(model, part, pad_id, n_tok, clip_low, clip_high, tis_cap, chunk, stats, behavior)


# ---- helpers ------------------------------------------------------------------------------------


def _rollout_stats(rollouts: list[Rollout]) -> dict[str, float]:
    res = [ro.result for ro in rollouts]
    ins = [r for r in res if r.label != OTHER]
    ood = [r for r in res if r.label == OTHER]
    commits = [r for r in res if r.terminal == "commit" and r.probability is not None]
    out = {
        "commit_rate": float(np.mean([r.terminal == "commit" for r in res])),
        "defer_rate": float(np.mean([r.terminal == "defer" for r in res])),
        "invalid_rate": float(np.mean([r.terminal == "invalid" for r in res])),
        "timeout_rate": float(np.mean([r.terminal == "timeout" for r in res])),
        "mean_tests": float(np.mean([r.n_tests for r in res])),
        "mean_p": float(np.mean([r.probability for r in commits])) if commits else math.nan,
        "turns_per_episode": float(np.mean([len(ro.turns) for ro in rollouts])),
        "gen_tokens": float(sum(len(t.completion_ids or []) for ro in rollouts for t in ro.turns)),
    }
    if ins:
        out["inset_acc_full"] = float(np.mean([r.forced_prediction == r.label for r in ins]))
        out["inset_commit_acc"] = float(np.mean([r.correct for r in ins if r.terminal == "commit"] or [math.nan]))
        out["inset_defer_rate"] = float(np.mean([r.terminal == "defer" for r in ins]))
        out["inset_other_rate"] = float(np.mean([r.terminal == "commit" and r.diagnosis == OTHER for r in ins]))
    if ood:
        out["ood_false_commit"] = float(np.mean([r.terminal == "commit" and r.diagnosis != OTHER for r in ood]))
        out["ood_commit_other"] = float(np.mean([r.terminal == "commit" and r.diagnosis == OTHER for r in ood]))
        out["ood_defer_rate"] = float(np.mean([r.terminal == "defer" for r in ood]))
    return out


def _dev_metrics(rep: dict[str, Any]) -> dict[str, float]:
    cw, cal, ow = rep.get("closed_world", {}), rep.get("calibration", {}), rep.get("open_world", {})
    sel, tr = rep.get("selective", {}), rep.get("terminal_rates", {})
    return {
        "dev_acc_full": cw.get("accuracy_full_coverage", math.nan),
        "dev_mean_class_acc": cw.get("mean_class_accuracy", math.nan),
        "dev_coverage": cw.get("coverage", math.nan),
        "dev_selective_acc": cw.get("selective_accuracy", math.nan),
        "dev_div_acc": cw.get("per_class_accuracy", {}).get("diverticulitis", math.nan),
        "dev_aurc": sel.get("aurc", math.nan),
        "dev_ece": cal.get("ece", math.nan),
        "dev_brier": cal.get("brier", math.nan),
        "dev_ood_false_commit": ow.get("false_commit_rate", math.nan),
        "dev_ood_defer": ow.get("defer_rate", math.nan),
        "dev_invalid": tr.get("invalid", math.nan),
        "dev_mean_tests": rep.get("cost", {}).get("mean_tests", math.nan),
    }


def turn_advantages(groups: list[list[Rollout]], rcfg, nu: float, in_set_only: bool, min_std: float,
                    normalize_std: bool, cont_values: dict[int, float] | None = None):
    """Per-turn advantages for every rollout. Returns ([(rollout, [advantage per turn] or None)], stats).

    Group advantages (r - mean, Dr. GRPO) apply to every turn. With counterfactual escalation values
    (`cont_values`: id(rollout) -> mean return of forced continuations from its deferral state), the
    DEFER turn of that rollout instead gets the STATE-LEVEL advantage
        A = E(handoff) - V_hat(continue from the same state),
    E = escalation value - handoff cost - coverage multiplier. Groups whose rewards have no spread
    contribute only these escalation turns (None = rollout skipped)."""
    cont_values = cont_values or {}
    out, st = [], Counter()
    for grp in groups:
        label = grp[0].result.label
        pen = nu if (label != OTHER or not in_set_only) else 0.0
        rbs = group_rewards([ro.result for ro in grp], rcfg, defer_penalty=pen)
        rewards = [rb.total for rb in rbs]
        st["reward_sum"] += sum(rewards)
        st["n"] += len(rewards)
        flat = float(np.std(rewards)) < min_std
        st["skipped" if flat else "kept"] += 1
        advs = [0.0] * len(grp) if flat else group_advantages(rewards, normalize_std)
        for ro, rb, adv in zip(grp, rbs, advs):
            ro.reward = rb.as_dict()
            for k in ("accuracy", "calibration", "severity", "cost", "consensus", "penalty", "constraint"):
                st[f"r_{k}"] += getattr(rb, k)
            per_turn = None if flat else [adv] * len(ro.turns)
            if id(ro) in cont_values and ro.turns:
                e = escalation_value(ro.result, rcfg) - rcfg.handoff_mu + rb.constraint
                a = e - cont_values[id(ro)]
                per_turn = per_turn or [None] * len(ro.turns)
                per_turn[-1] = a
                st["cev_n"] += 1
                st["cev_E"] += e
                st["cev_V"] += cont_values[id(ro)]
                st["cev_A"] += a
                st["cev_A_pos"] += float(a > 0)
            out.append((ro, per_turn))
    return out, st


def piecewise(marks: list | None, i: int, default: float):
    """Value of a piecewise-constant schedule [[index, value], ...] at index i."""
    out = default
    for at, val in sorted((int(a), v) for a, v in (marks or [])):
        if i >= at:
            out = val
    return out


def ppo_update(model, samples: list[TurnSample], opt, params, pad_id: int, n_minibatches: int, mb_tokens: int,
               mb_samples: int, clip_low: float, clip_high: float, tis_cap, chunk: int, max_grad_norm: float,
               rng: random.Random) -> Counter:
    """One rollout batch -> n_minibatches optimizer steps. With one mini-batch this is the original
    on-policy step (TIS-weighted); with several, each mini-batch is a PPO step against the sampling
    engine's log-probs. Returns accumulated stats."""
    import torch

    stats: Counter = Counter()
    order = list(samples)
    if n_minibatches > 1:
        rng.shuffle(order)
    chunks = [order[i::n_minibatches] for i in range(n_minibatches)] if n_minibatches > 1 else [order]
    behavior = "rollout" if n_minibatches > 1 else "current"
    norms = []
    for part in chunks:
        if not part:
            continue
        n_tok = sum(len(s.completion_ids) for s in part)
        for mb in token_budget_batches(part, mb_tokens, max_samples=mb_samples):
            _backward_with_split(model, mb, pad_id, n_tok, clip_low, clip_high, tis_cap, chunk, stats, behavior)
        norms.append(float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm)))
        opt.step()
        opt.zero_grad(set_to_none=True)
    stats["grad_norm"] = float(np.mean(norms)) if norms else 0.0
    stats["opt_steps"] = len(norms)
    return stats


def _latest_checkpoint(out_dir: Path) -> Path | None:
    ck = sorted((out_dir / "checkpoints").glob("step_*/trainer_state.pt"))
    return ck[-1].parent if ck else None


def _lr_lambda(schedule: str, warmup: int, total: int, milestones: list | None = None):
    """LR factor for scheduler step s (training step s + 1). `milestones` = [[step, factor], ...]
    sets a piecewise-constant factor (the last milestone at or before s); warm-up ramps up to it."""
    marks = sorted((int(a), float(b)) for a, b in (milestones or [[0, 1.0]]))

    def level(s: int) -> float:
        out = marks[0][1]
        for at, fac in marks:
            if s >= at:
                out = fac
        return out

    def f(s: int) -> float:
        base = level(s)
        if s < warmup:
            return base * (s + 1) / max(1, warmup)
        if schedule == "cosine":
            prog = (s - warmup) / max(1, total - warmup)
            return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        return base

    return f


# ---- trainer ------------------------------------------------------------------------------------


def train_grpo_vllm(cfg: dict[str, Any]) -> Path:
    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM

    from ..policy.vllm_policy import VLLMPolicy, sampling_params

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    rng = random.Random(seed)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "train_log.jsonl")
    catalog, env_cfg, rcfg, severity = _reward_setup(cfg)

    train_cases = load_cases(cfg["train_cases"])
    sampler = MixtureSampler(train_cases, cfg.get("pool_weights", {"cdm": 1.0}), rng)
    dev_cases = load_cases(cfg["dev_cases"]) if cfg.get("dev_cases") else []
    if cfg.get("dev_limit"):
        dev_cases = dev_cases[: int(cfg["dev_limit"])]

    G = int(cfg.get("group_size", 8))
    B = int(cfg.get("cases_per_step", 12))
    steps = int(cfg.get("steps", 200))
    lora_cfg = cfg.get("lora") or {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    gen = cfg.get("generation", {})
    max_new = int(gen.get("max_new_tokens", 1024))
    max_model_len = int(cfg.get("max_model_len", 20480))
    kw = cfg.get("chat_template_kwargs", {"enable_thinking": True})

    # 1) vLLM first: it sizes its KV cache from the free memory at start-up.
    llm = LLM(model=cfg["model"], enable_lora=True, max_lora_rank=int(lora_cfg.get("r", 16)), max_loras=1,
              enable_sleep_mode=True, enable_prefix_caching=True, seed=seed, dtype=cfg.get("dtype", "bfloat16"),
              gpu_memory_utilization=float(cfg.get("vllm_gpu_memory_utilization", 0.80)),
              max_model_len=max_model_len)
    train_params = sampling_params(max_new, float(gen.get("temperature", 1.0)), float(gen.get("top_p", 1.0)),
                                   gen.get("top_k"), logprobs=True)
    ev = cfg.get("eval_generation", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    eval_params = sampling_params(max_new, float(ev.get("temperature", 0.6)), float(ev.get("top_p", 0.95)),
                                  ev.get("top_k", 20))
    policy = VLLMPolicy.from_llm(llm, chat_template_kwargs=kw, max_new_tokens=max_new,
                                 max_prompt_tokens=max_model_len - max_new)
    policy.params = train_params
    tokenizer = policy.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 2) HF model on CPU, LoRA, frozen weights pinned and parked.
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                                                 attn_implementation=cfg.get("attn_implementation"))
    from peft import LoraConfig, PeftModel, get_peft_model

    if cfg.get("init_adapter"):
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(lora_cfg.get("r", 16)), lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=lora_cfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM"))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    parker = FrozenWeightParker(model, "cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-5)), betas=(0.9, 0.99),
                            weight_decay=float(cfg.get("weight_decay", 0.0)))
    warmup = int(cfg.get("warmup_steps", 5))
    lr_fn = _lr_lambda(cfg.get("lr_schedule", "constant"), warmup, steps, cfg.get("lr_milestones"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    constraint = CoverageConstraint.from_dict(cfg.get("constraint"))
    in_set_only = bool((cfg.get("constraint") or {}).get("in_set_only", True))

    # 3) resume
    start = 1
    adapter_dir: Path | None = None
    ck = _latest_checkpoint(out_dir) if cfg.get("resume", True) else None
    if ck is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        state = torch.load(ck / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        # the config is the source of truth for the LR schedule: a resumed run follows the
        # schedule now in the config (base LR and milestones), not the one it was saved with
        base_lr = float(cfg.get("lr", 1e-5))
        sched.base_lrs = [base_lr for _ in opt.param_groups]
        for g in opt.param_groups:
            g["initial_lr"] = base_lr
            g["lr"] = base_lr * lr_fn(sched.last_epoch)
        for k, v in state["constraint"].items():
            setattr(constraint, k, v)
        start = int(state["step"]) + 1
        rng.seed(seed * 100003 + start)  # a fresh, still deterministic, case order after resume
        adapter_dir = ck
        print(f"[grpo-vllm] resumed from {ck} at step {start}", flush=True)
    policy.set_adapter(str(adapter_dir) if adapter_dir else None)

    q = crossover_p_hat(rcfg)
    print(f"[grpo-vllm] G={G} B={B} steps={steps}; nominal tau={rcfg.tau:.2f}; reward-induced defer "
          f"crossover p_hat~{q}; pools={ {k: len(v) for k, v in sampler.pools.items()} }", flush=True)

    tis_cap = cfg.get("tis_cap", 2.0)
    cev_k = int(cfg.get("cev_k", 3))               # forced continuations per deferral state
    cev_max_roots = int(cfg.get("cev_max_roots", 32))  # deferral states branched per step
    min_std = float(cfg.get("min_group_std", 0.02))
    normalize_std = bool(cfg.get("normalize_std", False))
    max_seq_len = int(cfg.get("max_seq_len", max_model_len))
    mb_tokens = int(cfg.get("micro_batch_tokens", 16384))
    # One unpadded sequence per micro-batch is 1.55x faster than padded batches on Qwen3-8B
    # (no attention mask, so SDPA can use its flash kernel; no padding compute) and uses less memory.
    mb_samples = int(cfg.get("micro_batch_max_samples", 1))
    chunk = int(cfg.get("logprob_chunk", 2048))
    ent_cfg = cfg.get("action_entropy") or {}
    keep_adapters = int(cfg.get("keep_adapters", 2))
    # adapters from an interrupted run count towards keep_adapters, so a resume prunes them too
    adapters: deque[Path] = deque(sorted((out_dir / "adapters").glob("step_*")))

    for step in range(start, steps + 1):
        t0 = time.time()
        rcfg.tau = linear_schedule(step - 1, cfg.get("tau_schedule"), float(cfg.get("reward", {}).get("tau", 0.85)))
        rcfg.action_entropy_coef = float(ent_cfg.get("coef", 0.0)) if step <= int(ent_cfg.get("until_step", 0)) else 0.0
        batch = sampler.sample(B)

        # ---- 1. rollouts ---------------------------------------------------------------------
        rollouts = run_episodes(policy, catalog, env_cfg, batch, n_samples=G)
        t_roll = time.time() - t0

        # ---- 1b. counterfactual escalation: branch forced continuations from each deferral state -----
        cont_values: dict[int, float] = {}
        cev_stats: Counter = Counter()
        if rcfg.defer_mode == "cev":
            tb = time.time()
            case_by_id = {c.case_id: c for c in batch}
            roots = [ro for ro in rollouts if ro.result.terminal == "defer" and ro.turns
                     and ro.messages and ro.messages[-1]["role"] == "assistant"]
            if len(roots) > cev_max_roots:
                roots = rng.sample(roots, cev_max_roots)
            if roots:
                branches = branch_rollouts(policy, catalog, env_cfg, [(case_by_id[r.case_id], r) for r in roots], cev_k)
                for r, conts in zip(roots, branches):
                    vals = [continuation_value(c.result, r.result.total_cost, rcfg) for c in conts]
                    cont_values[id(r)] = float(np.mean(vals))
                    cev_stats["cont_n"] += len(conts)
                    cev_stats["cont_commit_correct"] += sum(c.result.correct for c in conts)
                    cev_stats["cont_tests"] += sum(c.result.n_tests - r.result.n_tests for c in conts)
            cev_stats["sec"] = time.time() - tb

        # ---- 2. rewards and advantages -------------------------------------------------------------
        nu = constraint.nu
        samples: list[TurnSample] = []
        dropped_long = 0
        assigned, comp = turn_advantages(group_by_case(rollouts), rcfg, nu, in_set_only, min_std, normalize_std,
                                         cont_values)
        all_rewards = [comp["reward_sum"] / max(1, comp["n"])]
        kept, skipped = comp["kept"], comp["skipped"]
        for ro, per_turn in assigned:
            if per_turn is None:
                continue
            for turn, adv in zip(ro.turns, per_turn):
                if adv is None or not turn.completion_ids:
                    continue
                if len(turn.prompt_ids) + len(turn.completion_ids) > max_seq_len:
                    dropped_long += 1
                    continue
                samples.append(TurnSample(turn.prompt_ids, turn.completion_ids, adv, rollout_logprobs=turn.logprobs))
        res = [ro.result for ro in rollouts]
        gate = [r for r in res if r.label != OTHER] if in_set_only else res
        constraint.update(float(np.mean([r.terminal == "defer" for r in gate])) if gate else 0.0)

        # ---- 3. policy update ------------------------------------------------------------------
        t1 = time.time()
        stats: Counter = Counter()
        n_tok = sum(len(s.completion_ids) for s in samples)
        peak = 0.0
        if samples:
            llm.sleep(level=1)
            parker.load()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            n_mb = int(piecewise(cfg.get("ppo_minibatch_milestones"), step - 1, int(cfg.get("ppo_minibatches", 1))))
            stats = ppo_update(model, samples, opt, params, pad_id, n_mb, mb_tokens, mb_samples,
                               float(cfg.get("clip_low", 0.2)), float(cfg.get("clip_high", 0.28)), tis_cap, chunk,
                               float(cfg.get("max_grad_norm", 1.0)), rng)
            model.eval()
            peak = torch.cuda.max_memory_allocated() / 2**30
            adapter_dir = out_dir / "adapters" / f"step_{step:04d}"
            model.save_pretrained(adapter_dir)
            adapters.append(adapter_dir)
            while len(adapters) > keep_adapters:
                shutil.rmtree(adapters.popleft(), ignore_errors=True)
            parker.park()
            llm.wake_up()
            policy.set_adapter(str(adapter_dir))
        sched.step()
        t_train = time.time() - t1

        # ---- 4. logging ---------------------------------------------------------------------------
        defers = [ro.result for ro in rollouts if ro.result.terminal == "defer"]
        n_cev = max(1, comp["cev_n"])
        row = {"step": step, "sec": round(time.time() - t0, 1), "sec_rollout": round(t_roll, 1),
               "sec_train": round(t_train, 1), "episodes": len(rollouts),
               "reward_mean": float(np.mean(all_rewards)),
               **{k: v / max(1, len(rollouts)) for k, v in comp.items() if k.startswith("r_")},
               "handoff_score": float(np.mean([handoff_score(r) for r in defers])) if defers else math.nan,
               "cev_roots": comp["cev_n"], "cev_E": comp["cev_E"] / n_cev, "cev_V": comp["cev_V"] / n_cev,
               "cev_A": comp["cev_A"] / n_cev, "cev_A_pos": comp["cev_A_pos"] / n_cev,
               "cev_cont_acc": cev_stats["cont_commit_correct"] / max(1, cev_stats["cont_n"]),
               "cev_cont_tests": cev_stats["cont_tests"] / max(1, cev_stats["cont_n"]),
               "sec_cev": round(cev_stats["sec"], 1),
               **_rollout_stats(rollouts),
               "tau": rcfg.tau, "nu": nu, "kept_groups": kept, "skipped_groups": skipped,
               "turn_samples": len(samples), "train_tokens": n_tok, "dropped_long": dropped_long,
               "entropy": stats["entropy_sum"] / max(1, n_tok), "loss": stats["loss"],
               "grad_norm": stats.get("grad_norm", 0.0), "lr": sched.get_last_lr()[0],
               "tis_w": stats["tis_w_sum"] / max(1, stats["tis_tokens"]),
               "tis_capped_frac": stats["tis_capped"] / max(1, stats["tis_tokens"]),
               "engine_mismatch": stats["mismatch_sum"] / max(1, stats["tis_tokens"]),
               "oom_splits": stats["oom_splits"], "peak_gib": round(peak, 1),
               "opt_steps": stats.get("opt_steps", 0), "clip_frac": stats["clipped"] / max(1, stats["tis_tokens"])}
        if cfg.get("log_rollouts_every") and step % int(cfg["log_rollouts_every"]) == 0:
            write_jsonl(out_dir / "rollouts" / f"step_{step:04d}.jsonl", (ro.to_dict() for ro in rollouts))

        if dev_cases and cfg.get("eval_every") and (step % int(cfg["eval_every"]) == 0 or step == steps):
            t2 = time.time()
            policy.params = eval_params
            dev_ro = run_episodes(policy, catalog, env_cfg, dev_cases, n_samples=1)
            policy.params = train_params
            row.update(_dev_metrics(summarize([r.result for r in dev_ro], severity)))
            row["sec_eval"] = round(time.time() - t2, 1)
        logger.log(row)

        if (cfg.get("save_every") and step % int(cfg["save_every"]) == 0) or step == steps:
            ck_dir = out_dir / "checkpoints" / f"step_{step:04d}"
            model.save_pretrained(ck_dir)
            tokenizer.save_pretrained(ck_dir)
            torch.save({"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "constraint": {"nu": constraint.nu, "_violations": constraint._violations,
                                       "_integral": constraint._integral},
                        "config": cfg}, ck_dir / "trainer_state.pt")
            # keep every `keep_every`-th checkpoint (used for evaluation) plus the latest two (for resume)
            keep_every = int(cfg.get("keep_every", 25))
            cks = sorted((out_dir / "checkpoints").glob("step_*"))
            for old_ck in cks[:-2]:
                if int(old_ck.name.split("_")[1]) % keep_every:
                    shutil.rmtree(old_ck, ignore_errors=True)

    final = out_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    return final
