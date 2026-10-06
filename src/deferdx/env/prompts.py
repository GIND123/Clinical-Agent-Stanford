"""System prompt and observation formatting."""

from __future__ import annotations

import re

from ..labels import IN_SET_LABELS, OTHER
from .catalog import TestCatalog

SYSTEM_TEMPLATE = """You are an emergency-department diagnostic agent assessing a patient with abdominal pain.
You gather information one action at a time, then either commit to a diagnosis or defer the case to a clinician.

Each turn: think briefly inside <think>...</think> (a few sentences), then output exactly one action as JSON inside <action>...</action>.

Actions:
- ASK a history/exam question: {{"type": "ASK", "topic": "<topic>"}}
  topics: {topics}
- TEST order one investigation: {{"type": "TEST", "test": "<test>"}}
  tests: {tests}
- COMMIT a final diagnosis with your probability that it is correct:
  {{"type": "COMMIT", "diagnosis": "<diagnosis>", "probability": <0-1>}}
  diagnoses: {diagnoses}
{defer_block}
Every question and test has a cost; order only what can change your decision. You may take at most {max_steps} ASK/TEST actions.
{open_world_line}Your stated probability is scored with a proper scoring rule: report your honest belief."""

DEFER_BLOCK = """- DEFER the case to a clinician with a ranked differential and a short reason:
  {"type": "DEFER", "differential": ["<most likely>", "..."], "reason": "<why a human should decide>"}
  Defer when you are likely to be wrong or the presentation does not fit the listed diagnoses. Deferral is not free: use it when it protects the patient.
"""


def system_prompt(catalog: TestCatalog, max_steps: int, allow_defer: bool = True, open_world: bool = True,
                  topics: list[str] | None = None) -> str:
    diagnoses = list(IN_SET_LABELS) + ([OTHER] if open_world else [])
    open_world_line = (
        f'The true diagnosis may be none of the four listed conditions; "{OTHER}" means a different diagnosis.\n'
        if open_world
        else ""
    )
    return SYSTEM_TEMPLATE.format(
        topics=", ".join(topics if topics is not None else catalog.asks),
        tests=", ".join(catalog.tests),
        diagnoses=", ".join(diagnoses),
        defer_block=DEFER_BLOCK if allow_defer else "",
        max_steps=max_steps,
        open_world_line=open_world_line,
    )


def initial_observation(hpi: str, history: dict[str, str] | None = None) -> str:
    extra = "".join(f"\n{k.replace('_', ' ').capitalize()}: {v.strip()}" for k, v in (history or {}).items() if v)
    return f"PATIENT PRESENTATION\n{hpi.strip()}{extra}\n\nChoose your next action."


def truncate(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n[... truncated ...]"
    return text


# CDM replaces each mention of a case's own diagnosis with "____" (four underscores);
# MIMIC's de-identification uses "___" (three). The four-underscore mask therefore marks
# "the diagnosis was named here", a label cue (docs/DATA_AUDIT.md section 5). Policies:
#   keep           the dataset as published (comparable with prior MIMIC-CDM work)
#   normalize      "____" -> "___", indistinguishable in form from any de-identified token
#   drop_sentence  remove every sentence containing the mask (the strictest ablation)
_MASK_RE = re.compile(r"_{4,}")
_SENT_RE = re.compile(r"(?<=[.;!?])\s+")
MASK_POLICIES = ("keep", "normalize", "drop_sentence")


def apply_mask_policy(text: str | None, policy: str = "keep") -> str | None:
    if text is None or policy == "keep" or "____" not in text:
        return text
    if policy == "normalize":
        return _MASK_RE.sub("___", text)
    if policy == "drop_sentence":
        lines = []
        for line in text.split("\n"):
            kept = [s for s in _SENT_RE.split(line) if not _MASK_RE.search(s)]
            lines.append(" ".join(kept))
        return "\n".join(lines)
    raise ValueError(f"unknown mask_policy {policy!r}; expected one of {MASK_POLICIES}")
