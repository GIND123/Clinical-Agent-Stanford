"""The four-action space {ASK, TEST, COMMIT, DEFER}: parsing model text and rendering
actions back to text (for SFT targets and scripted policies).

Canonical model output:

    <think> ... </think>
    <action>{"type": "TEST", "test": "ct_abdomen"}</action>

Accepted action JSON:
    {"type": "ASK",    "topic": "physical_exam"}
    {"type": "TEST",   "test": "lipase"}
    {"type": "COMMIT", "diagnosis": "appendicitis", "probability": 0.92}
    {"type": "DEFER",  "differential": ["diverticulitis", "appendicitis"], "reason": "..."}

A function-call fallback (`TEST(lipase)`, `COMMIT(appendicitis, 0.9)`,
`DEFER([a, b], reason)`) is also parsed so that zero-shot baselines are not scored as
format failures only because they ignored the JSON wrapper.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from ..labels import ALL_LABELS, normalize_label

ASK, TEST, COMMIT, DEFER, INVALID = "ASK", "TEST", "COMMIT", "DEFER", "INVALID"
TERMINAL = (COMMIT, DEFER)


@dataclass
class Action:
    type: str
    topic: str | None = None
    test: str | None = None
    diagnosis: str | None = None
    probability: float | None = None
    differential: list[str] = field(default_factory=list)
    reason: str = ""
    error: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL

    def to_json(self) -> dict:
        if self.type == ASK:
            return {"type": ASK, "topic": self.topic}
        if self.type == TEST:
            return {"type": TEST, "test": self.test}
        if self.type == COMMIT:
            return {"type": COMMIT, "diagnosis": self.diagnosis, "probability": round(float(self.probability), 3)}
        if self.type == DEFER:
            return {"type": DEFER, "differential": list(self.differential), "reason": self.reason}
        return {"type": INVALID, "error": self.error}


_ACTION_TAG = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_THINK_END = re.compile(r"</think>", re.IGNORECASE)
_CALL = re.compile(r"\b(ASK|TEST|COMMIT|DEFER)\s*\((.*?)\)", re.DOTALL | re.IGNORECASE)


def _prob(x) -> float | None:
    try:
        p = float(str(x).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if 1.0 < p <= 100.0:  # tolerate percentages
        p /= 100.0
    return p if 0.0 <= p <= 1.0 else None


def _from_json(obj: dict, valid_tests: set[str], valid_topics: set[str]) -> Action:
    kind = str(obj.get("type", "")).upper()
    if kind == ASK:
        topic = str(obj.get("topic", "")).strip().lower()
        if topic not in valid_topics:
            return Action(INVALID, error=f"unknown ASK topic {topic!r}")
        return Action(ASK, topic=topic)
    if kind == TEST:
        test = str(obj.get("test", "")).strip().lower()
        if test not in valid_tests:
            return Action(INVALID, error=f"unknown test {test!r}")
        return Action(TEST, test=test)
    if kind == COMMIT:
        dx = normalize_label(obj.get("diagnosis"))
        p = _prob(obj.get("probability"))
        if dx is None:
            return Action(INVALID, error=f"diagnosis must be one of {list(ALL_LABELS)}")
        if p is None:
            return Action(INVALID, error="COMMIT requires probability in [0, 1]")
        return Action(COMMIT, diagnosis=dx, probability=p)
    if kind == DEFER:
        diff_raw = obj.get("differential") or []
        if isinstance(diff_raw, str):
            diff_raw = re.split(r"[,;]", diff_raw)
        diff = []
        for d in diff_raw:
            lab = normalize_label(d)
            if lab and lab not in diff:
                diff.append(lab)
        return Action(DEFER, differential=diff, reason=str(obj.get("reason", "")).strip())
    return Action(INVALID, error=f"unknown action type {kind!r}")


def _from_call(kind: str, args: str, valid_tests: set[str], valid_topics: set[str]) -> Action:
    kind = kind.upper()
    args = args.strip()
    if kind in (ASK, TEST):
        key = "topic" if kind == ASK else "test"
        return _from_json({"type": kind, key: args.strip("'\" ")}, valid_tests, valid_topics)
    if kind == COMMIT:
        parts = [p.strip("'\" ") for p in args.split(",")]
        return _from_json({"type": COMMIT, "diagnosis": parts[0], "probability": parts[1] if len(parts) > 1 else None},
                          valid_tests, valid_topics)
    m = re.match(r"\[(.*?)\]\s*,?\s*(.*)", args, re.DOTALL)
    diff, reason = (m.group(1), m.group(2)) if m else (args, "")
    return _from_json({"type": DEFER, "differential": diff, "reason": reason.strip("'\" ")}, valid_tests, valid_topics)


def parse_action(text: str, valid_tests: set[str], valid_topics: set[str]) -> Action:
    """Parse the LAST action in a model turn (after any </think>)."""
    body = text
    ends = list(_THINK_END.finditer(text))
    if ends:
        body = text[ends[-1].end():]
    tags = _ACTION_TAG.findall(body)
    if tags:
        raw = tags[-1]
        obj = _loads_lenient(raw)
        if obj is None:
            m = _CALL.search(raw)
            if m:
                return _from_call(m.group(1), m.group(2), valid_tests, valid_topics)
            return Action(INVALID, error="action block is not valid JSON")
        if not isinstance(obj, dict):
            return Action(INVALID, error="action JSON must be an object")
        return _from_json(obj, valid_tests, valid_topics)
    m = None
    for m in _CALL.finditer(body):
        pass
    if m:
        return _from_call(m.group(1), m.group(2), valid_tests, valid_topics)
    # untagged JSON object with a "type" key after the reasoning
    obj = _loads_lenient(body) if '"type"' in body or "'type'" in body else None
    if isinstance(obj, dict) and "type" in obj:
        return _from_json(obj, valid_tests, valid_topics)
    if "<action" in body and "</action>" not in body:
        return Action(INVALID, error="unterminated <action> block (output truncated?)")
    return Action(INVALID, error="no <action>...</action> block found")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _loads_lenient(raw: str):
    """JSON object from a model's action block, tolerating code fences, text around the
    object, single-quoted (Python-literal) dicts and trailing commas. None if hopeless."""
    fenced = _FENCE.search(raw)
    if fenced:
        raw = fenced.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    candidates = [raw.strip()]
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    for cand in candidates:
        for fixed in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(fixed)
            except (ValueError, SyntaxError):
                pass
    return None


def render_action(action: Action, thought: str = "") -> str:
    """Canonical text for an action (SFT target / scripted policy output)."""
    think = f"<think>\n{thought.strip()}\n</think>\n\n" if thought else "<think>\n</think>\n\n"
    return f"{think}<action>{json.dumps(action.to_json())}</action>"
