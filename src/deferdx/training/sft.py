"""Stage 1: supervised fine-tuning on multi-turn trajectories (completion-only loss).

Minimal single-process loop on purpose: it shares tokenisation with the GRPO trainer,
so SFT and RL see byte-identical prompts. Scale-out: see docs/TRAINING.md.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..data.io import read_jsonl
from .common import (
    JsonlLogger,
    TurnSample,
    completion_logprobs,
    conversation_to_turns,
    cosine_lr,
    load_model_and_tokenizer,
    save_checkpoint,
    set_seed,
)


def build_samples(files: list[str], tokenizer, chat_template_kwargs: dict | None, max_seq_len: int) -> list[TurnSample]:
    samples, dropped = [], 0
    for path in files:
        for rec in read_jsonl(path):
            n_assist = sum(m["role"] == "assistant" for m in rec["messages"])
            turns = conversation_to_turns(rec["messages"], tokenizer, chat_template_kwargs, max_seq_len)
            dropped += n_assist - len(turns)
            samples.extend(turns)
    if dropped:
        print(f"[sft] dropped {dropped} turns longer than max_seq_len={max_seq_len}")
    return samples


def train_sft(cfg: dict[str, Any], model=None, tokenizer=None) -> Path:
    import torch

    set_seed(cfg.get("seed", 0))
    if model is None:
        model, tokenizer = load_model_and_tokenizer(cfg)
    out_dir = Path(cfg["output_dir"])
    logger = JsonlLogger(out_dir / "sft_log.jsonl")
    samples = build_samples(cfg["train_files"], tokenizer, cfg.get("chat_template_kwargs"), cfg.get("max_seq_len", 8192))
    if not samples:
        raise ValueError("no SFT samples")
    bs = cfg.get("batch_size", 32)
    mbs = cfg.get("micro_batch_size", 4)
    epochs = cfg.get("epochs", 2)
    steps_per_epoch = max(1, len(samples) // bs)
    total = steps_per_epoch * epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-5), weight_decay=cfg.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total, cfg.get("warmup_ratio", 0.05)))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rng = random.Random(cfg.get("seed", 0))
    step = 0
    model.train()
    for epoch in range(epochs):
        rng.shuffle(samples)
        for b in range(steps_per_epoch):
            batch = samples[b * bs : (b + 1) * bs]
            # longest-first within the batch reduces padding waste in micro-batches
            batch.sort(key=lambda s: len(s.prompt_ids) + len(s.completion_ids), reverse=True)
            n_tok = sum(len(s.completion_ids) for s in batch)
            loss_total = 0.0
            for m in range(0, len(batch), mbs):
                logps, _ = completion_logprobs(model, batch[m : m + mbs], pad_id)
                loss = -torch.cat(logps).sum() / n_tok  # token-level mean over the full batch
                loss.backward()
                loss_total += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            logger.log({"step": step, "epoch": epoch, "loss": loss_total, "grad_norm": float(grad_norm),
                        "lr": sched.get_last_lr()[0]})
            if cfg.get("save_every") and step % cfg["save_every"] == 0:
                save_checkpoint(model, tokenizer, out_dir / f"step_{step}")
    final = out_dir / "final"
    save_checkpoint(model, tokenizer, final, {"steps": step, "config": cfg})
    return final
