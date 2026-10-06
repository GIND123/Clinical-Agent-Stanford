"""Shared training utilities (torch / transformers / peft imported lazily)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..policy.hf import encode_chat


@dataclass
class TurnSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    advantage: float = 0.0
    old_logprobs: Any = None  # torch.Tensor | None
    ref_logprobs: Any = None
    rollout_logprobs: Any = None  # sampling engine's log-probs (list[float]) for truncated IS


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_and_tokenizer(cfg: dict[str, Any], trainable: bool = True):
    """Load base model (+ optional existing adapter, + optional new LoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = cfg["model"]
    init_adapter = cfg.get("init_adapter")
    tokenizer = AutoTokenizer.from_pretrained(init_adapter or model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, device_map=cfg.get("device_map", "auto"),
        attn_implementation=cfg.get("attn_implementation"),
    )
    lora = cfg.get("lora")
    if init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=trainable)
    elif lora and trainable:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=lora.get("r", 16), lora_alpha=lora.get("alpha", 32), lora_dropout=lora.get("dropout", 0.0),
            target_modules=lora.get("target_modules", "all-linear"), task_type="CAUSAL_LM",
        ))
    if trainable and cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def conversation_to_turns(messages: list[dict[str, str]], tokenizer, chat_template_kwargs: dict | None = None,
                          max_seq_len: int | None = None) -> list[TurnSample]:
    """One (prompt, completion) sample per assistant turn. The prompt is re-rendered from
    the preceding messages exactly as at generation time (templates such as Qwen3's
    strip earlier <think> blocks, so a single concatenated sequence would not match
    what the model saw)."""
    out = []
    eos = tokenizer.eos_token_id
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prompt = encode_chat(tokenizer, messages[:i], chat_template_kwargs)
        completion = list(tokenizer(msg["content"], add_special_tokens=False)["input_ids"])
        if eos is not None:
            completion.append(eos)
        if max_seq_len and len(prompt) + len(completion) > max_seq_len:
            continue
        out.append(TurnSample(prompt, completion))
    return out


def pad_batch(samples: list[TurnSample], pad_id: int):
    """Right-pad prompt+completion. Returns input_ids, attention_mask, and per-sample
    (start, length) of the completion inside the sequence."""
    import torch

    seqs = [s.prompt_ids + s.completion_ids for s in samples]
    width = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    spans = []
    for i, (s, seq) in enumerate(zip(samples, seqs)):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
        spans.append((len(s.prompt_ids), len(s.completion_ids)))
    return ids, mask, spans


def completion_logprobs(model, samples: list[TurnSample], pad_id: int, with_entropy: bool = False):
    """Per-sample tensors of log p(completion token | prefix) (and token entropies)."""
    import torch

    device = next(model.parameters()).device
    ids, mask, spans = pad_batch(samples, pad_id)
    ids, mask = ids.to(device), mask.to(device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    logps, ents = [], []
    for i, (start, length) in enumerate(spans):
        # logits at t predict token t+1
        lg = logits[i, start - 1 : start - 1 + length].float()
        tgt = ids[i, start : start + length]
        lsm = torch.log_softmax(lg, dim=-1)
        logps.append(lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        if with_entropy:
            ents.append(-(lsm.exp() * lsm).sum(-1))
    return logps, (ents if with_entropy else None)


def _frozen_head_logprob_fn():
    """Autograd function: log p(target) and entropy under a FROZEN output layer.

    Forward and backward both stream over chunks of positions, so full-vocabulary logits
    (V = 152k for Qwen3) are never stored for a whole micro-batch. Backward returns only
    d/d hidden: lm_head is frozen under LoRA, so its weight needs no gradient.
        d logp_t / d h_t = W[y_t] - softmax(W h_t) @ W
    """
    import torch

    class FrozenHeadLogprob(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, weight, target, chunk):
            lps, ents = [], []
            for c in range(0, h.shape[0], chunk):
                z = (h[c : c + chunk] @ weight.T).float()
                lse = torch.logsumexp(z, -1)
                lps.append(z.gather(-1, target[c : c + chunk, None])[:, 0] - lse)
                ents.append(lse - (torch.softmax(z, -1) * z).sum(-1))
            ctx.save_for_backward(h, weight, target)
            ctx.chunk = chunk
            ent = torch.cat(ents)
            ctx.mark_non_differentiable(ent)
            return torch.cat(lps), ent

        @staticmethod
        def backward(ctx, g_lp, g_ent):
            h, weight, target = ctx.saved_tensors
            gh = torch.empty_like(h)
            for c in range(0, h.shape[0], ctx.chunk):
                g = g_lp[c : c + ctx.chunk].float()
                gz = torch.softmax((h[c : c + ctx.chunk] @ weight.T).float(), -1) * (-g[:, None])
                gz.scatter_add_(-1, target[c : c + ctx.chunk, None], g[:, None])
                gh[c : c + ctx.chunk] = (gz.to(weight.dtype) @ weight).to(h.dtype)
            return gh, None, None, None

    return FrozenHeadLogprob


_FHL = None


def frozen_head_logprobs(hidden, weight, target, chunk: int = 2048):
    """(log-probs, entropies) of `target` tokens given final hidden states, lm_head frozen."""
    global _FHL
    if _FHL is None:
        _FHL = _frozen_head_logprob_fn()
    return _FHL.apply(hidden, weight, target, chunk)


def _backbone_and_head(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model, m.lm_head


def completion_logprobs_lowmem(model, samples: list[TurnSample], pad_id: int, chunk: int = 2048):
    """Per-sample (log-probs, entropies) of completion tokens. Same values as
    `completion_logprobs`, but the output layer is applied only at completion positions and
    through `frozen_head_logprobs`, so memory scales with the backbone, not the vocabulary.
    Requires a frozen lm_head (true for LoRA on all-linear, which excludes the output layer)."""
    backbone, head = _backbone_and_head(model)
    if head.weight.requires_grad:
        raise ValueError("completion_logprobs_lowmem needs a frozen lm_head")
    device = head.weight.device
    ids, mask, spans = pad_batch(samples, pad_id)
    ids, mask = ids.to(device), mask.to(device)
    hidden = backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    logps, ents = [], []
    for i, (start, length) in enumerate(spans):
        lp, ent = frozen_head_logprobs(hidden[i, start - 1 : start - 1 + length], head.weight,
                                       ids[i, start : start + length], chunk)
        logps.append(lp)
        ents.append(ent)
    return logps, ents


def token_budget_batches(samples: list[TurnSample], max_tokens: int, max_samples: int = 64) -> list[list[TurnSample]]:
    """Group samples (sorted by length, longest first) into micro-batches whose padded size
    (n x longest) stays within max_tokens. A sample longer than max_tokens gets its own batch."""
    order = sorted(samples, key=lambda s: len(s.prompt_ids) + len(s.completion_ids), reverse=True)
    batches: list[list[TurnSample]] = []
    cur: list[TurnSample] = []
    width = 0
    for s in order:
        n = len(s.prompt_ids) + len(s.completion_ids)
        w = max(width, n)
        if cur and (w * (len(cur) + 1) > max_tokens or len(cur) >= max_samples):
            batches.append(cur)
            cur, w = [], n
        cur.append(s)
        width = w
    if cur:
        batches.append(cur)
    return batches


def cosine_lr(step: int, total: int, warmup_ratio: float) -> float:
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(model, tokenizer, out_dir: str | Path, extra: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if extra:
        with (out_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2)


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
