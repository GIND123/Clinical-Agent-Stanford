"""Policy interface: map a batch of chat conversations to one assistant turn each."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Generation:
    text: str
    # Exact token ids fed to / produced by the model. Required for RL training so that
    # the policy-gradient loss is computed on precisely what was sampled.
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    # log-probs of the sampled tokens under the sampling engine (vLLM), for truncated importance
    # sampling against the trainer's own log-probs. None when the engine does not report them.
    logprobs: list[float] | None = None


class Policy(Protocol):
    def generate(self, conversations: list[list[dict[str, str]]], contexts: list[Any]) -> list[Generation]:
        """`contexts` are the per-conversation environments; LLM policies ignore them,
        scripted policies (oracle/random) read the case from them."""
        ...
