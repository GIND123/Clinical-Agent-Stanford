"""Local vLLM policy: evaluation rollouts, rejection sampling, and the rollout half of the
colocated GRPO trainer (training/grpo_vllm.py). Local weights only.

The trainer builds one `LLM` and wraps it here with `VLLMPolicy.from_llm`, then swaps the
LoRA adapter each step with `set_adapter`. vLLM's LoRA log-probs match HF+PEFT to bf16
noise across sleep/wake cycles and fresh adapter ids (checked on Qwen3-0.6B, mean
|d log p| 0.024-0.030 per token vs 0.022 for the base model, against a LoRA effect of 0.45).
"""

from __future__ import annotations

from typing import Any

from .base import Generation
from .hf import encode_chat


def episode_seed(base_seed: int, case_id: str, sample_idx: int, turn: int) -> int:
    """Deterministic per-request seed: an evaluation run is reproducible for a given seed, and
    different seeds (or samples of the same case) draw independent samples."""
    import hashlib

    h = hashlib.sha256(f"{base_seed}|{case_id}|{sample_idx}|{turn}".encode()).hexdigest()
    return int(h[:8], 16)


def sampling_params(max_new_tokens: int, temperature: float, top_p: float = 1.0, top_k: int | None = None,
                    logprobs: bool = False, seed: int | None = None):
    from vllm import SamplingParams

    kw: dict[str, Any] = dict(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    if top_k:
        kw["top_k"] = top_k
    if logprobs:
        kw["logprobs"] = 0  # the sampled token's own log-prob
    if seed is not None:
        kw["seed"] = seed
    return SamplingParams(**kw)


class VLLMPolicy:
    def __init__(self, model_path: str | None = None, adapter_path: str | None = None, max_new_tokens: int = 768,
                 temperature: float = 1.0, top_p: float = 1.0, top_k: int | None = None,
                 chat_template_kwargs: dict | None = None, seed: int = 0, llm=None, return_logprobs: bool = False,
                 max_lora_rank: int = 16, max_prompt_tokens: int | None = None, request_seed: int | None = None,
                 **llm_kwargs):
        if llm is None:
            from vllm import LLM

            if adapter_path is not None:
                llm_kwargs.setdefault("max_lora_rank", max_lora_rank)
            llm = LLM(model=model_path, enable_lora=adapter_path is not None, seed=seed, **llm_kwargs)
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()
        self.params = sampling_params(max_new_tokens, temperature, top_p, top_k, return_logprobs)
        # request_seed: derive a seed per (case, sample, turn) from this base seed (evaluation).
        # None: engine-level randomness only (training rollouts).
        self.request_seed = request_seed
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.lora = None
        self._lora_id = 0
        # Prompts longer than this get an empty completion (the environment scores it as an
        # invalid turn) instead of making vLLM reject the whole batch.
        self.max_prompt_tokens = max_prompt_tokens
        if self.max_prompt_tokens is None and llm_kwargs.get("max_model_len"):
            self.max_prompt_tokens = int(llm_kwargs["max_model_len"]) - max_new_tokens
        if adapter_path:
            self.set_adapter(adapter_path)

    @classmethod
    def from_llm(cls, llm, **kwargs) -> "VLLMPolicy":
        return cls(llm=llm, **kwargs)

    def set_adapter(self, path: str | None, name: str | None = None) -> None:
        """Use a (new) LoRA adapter for later generations; None = base model. Every call gets a
        fresh adapter id, so vLLM never serves a cached copy of an older adapter's weights."""
        if path is None:
            self.lora = None
            return
        from vllm.lora.request import LoRARequest

        self._lora_id += 1
        self.lora = LoRARequest(name or f"adapter{self._lora_id}", self._lora_id, str(path))

    def _seeded(self, env):
        import copy

        p = copy.copy(self.params)
        p.seed = episode_seed(self.request_seed, env.case.case_id, getattr(env, "sample_idx", 0), len(env.result.steps))
        return p

    def generate(self, conversations: list[list[dict[str, str]]], contexts: list[Any] | None = None) -> list[Generation]:
        prompts = [encode_chat(self.tokenizer, conv, self.chat_template_kwargs) for conv in conversations]
        limit = self.max_prompt_tokens
        ok = [i for i, p in enumerate(prompts) if limit is None or len(p) <= limit]
        params = self.params
        if self.request_seed is not None and contexts is not None:
            params = [self._seeded(contexts[i]) for i in ok]
        outputs = self.llm.generate([{"prompt_token_ids": prompts[i]} for i in ok], params,
                                    lora_request=self.lora, use_tqdm=False) if ok else []
        by_index = dict(zip(ok, outputs))
        gens = []
        for i, p in enumerate(prompts):
            o = by_index.get(i)
            if o is None:
                gens.append(Generation(text="", prompt_ids=p, completion_ids=[], logprobs=[]))
                continue
            out = o.outputs[0]
            ids = list(out.token_ids)
            lps = None
            if out.logprobs is not None:
                lps = [float(d[t].logprob) if t in d else float("nan") for t, d in zip(ids, out.logprobs)]
            gens.append(Generation(text=out.text, prompt_ids=p, completion_ids=ids, logprobs=lps))
        return gens
