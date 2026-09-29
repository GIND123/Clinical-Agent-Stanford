"""Local vLLM policy for fast evaluation / rejection-sampling rollouts (not used for the
policy-gradient update itself). Local weights only."""

from __future__ import annotations

from typing import Any

from .base import Generation
from .hf import encode_chat


class VLLMPolicy:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 768,
                 temperature: float = 1.0, top_p: float = 1.0, top_k: int | None = None,
                 chat_template_kwargs: dict | None = None,
                 seed: int = 0, **llm_kwargs):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model_path, enable_lora=adapter_path is not None, seed=seed, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        # No per-request seed: it would make every sample of the same prompt identical
        # (breaking --samples N). The engine-level seed above gives run reproducibility.
        self.params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                     top_k=top_k or -1)
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("deferdx", 1, adapter_path)

    def generate(self, conversations: list[list[dict[str, str]]], contexts: list[Any] | None = None) -> list[Generation]:
        prompts = [encode_chat(self.tokenizer, conv, self.chat_template_kwargs) for conv in conversations]
        outputs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], self.params,
                                    lora_request=self.lora, use_tqdm=False)
        return [
            Generation(text=o.outputs[0].text, prompt_ids=p, completion_ids=list(o.outputs[0].token_ids))
            for p, o in zip(prompts, outputs)
        ]
