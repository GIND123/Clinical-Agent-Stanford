"""Local Hugging Face `transformers` policy (the one used inside GRPO training).

Runs entirely on local weights. There is deliberately no hosted-API policy in this
repo: the PhysioNet DUA restricts sending MIMIC text to third-party services.
"""

from __future__ import annotations

from typing import Any

from .base import Generation


def encode_chat(tokenizer, messages: list[dict[str, str]], chat_template_kwargs: dict | None = None,
                add_generation_prompt: bool = True) -> list[int]:
    """Render with the chat template, then tokenize. (Rendering to text first keeps
    behaviour identical across transformers versions, whose tokenize=True return
    types differ.)"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **(chat_template_kwargs or {})
    )
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def stop_token_ids(tokenizer, model=None) -> set[int]:
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    eos = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(x) for x in eos)
    return ids


class HFPolicy:
    def __init__(self, model, tokenizer, max_new_tokens: int = 768, temperature: float = 1.0, top_p: float = 1.0,
                 do_sample: bool = True, batch_size: int = 16, chat_template_kwargs: dict | None = None,
                 top_k: int | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.stop_ids = stop_token_ids(tokenizer, model)

    @classmethod
    def from_pretrained(cls, model_path: str, adapter_path: str | None = None, dtype: str = "bfloat16",
                        device_map: str | None = "auto", **kwargs) -> "HFPolicy":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map=device_map)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return cls(model, tokenizer, **kwargs)

    def generate(self, conversations: list[list[dict[str, str]]], contexts: list[Any] | None = None) -> list[Generation]:
        import torch

        prompts = [encode_chat(self.tokenizer, conv, self.chat_template_kwargs) for conv in conversations]
        out: list[Generation] = []
        device = next(self.model.parameters()).device
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            width = max(len(p) for p in chunk)
            input_ids = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, p in enumerate(chunk):  # left padding
                input_ids[i, width - len(p):] = torch.tensor(p)
                attn[i, width - len(p):] = 1
            gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=self.do_sample,
                              pad_token_id=self.pad_id, eos_token_id=sorted(self.stop_ids) or None)
            if self.do_sample:
                gen_kwargs.update(temperature=self.temperature, top_p=self.top_p)
                if self.top_k:
                    gen_kwargs["top_k"] = self.top_k
            with torch.no_grad():
                seqs = self.model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device), **gen_kwargs)
            for i, p in enumerate(chunk):
                completion = []
                for tok in seqs[i, width:].tolist():
                    completion.append(tok)
                    if tok in self.stop_ids:
                        break
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                out.append(Generation(text=text, prompt_ids=list(p), completion_ids=completion))
        return out
