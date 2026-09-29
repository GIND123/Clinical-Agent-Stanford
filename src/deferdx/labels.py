"""Diagnosis label space: the four MIMIC-CDM conditions plus the open-world OTHER."""

from __future__ import annotations

import re

APPENDICITIS = "appendicitis"
CHOLECYSTITIS = "cholecystitis"
DIVERTICULITIS = "diverticulitis"
PANCREATITIS = "pancreatitis"
OTHER = "other"

IN_SET_LABELS: tuple[str, ...] = (APPENDICITIS, CHOLECYSTITIS, DIVERTICULITIS, PANCREATITIS)
ALL_LABELS: tuple[str, ...] = IN_SET_LABELS + (OTHER,)
LABEL_INDEX = {label: i for i, label in enumerate(ALL_LABELS)}

_SYNONYMS = {
    "acute appendicitis": APPENDICITIS,
    "appendicitis": APPENDICITIS,
    "perforated appendicitis": APPENDICITIS,
    "cholecystitis": CHOLECYSTITIS,
    "acute cholecystitis": CHOLECYSTITIS,
    "calculous cholecystitis": CHOLECYSTITIS,
    "diverticulitis": DIVERTICULITIS,
    "acute diverticulitis": DIVERTICULITIS,
    "sigmoid diverticulitis": DIVERTICULITIS,
    "pancreatitis": PANCREATITIS,
    "acute pancreatitis": PANCREATITIS,
    "gallstone pancreatitis": PANCREATITIS,
    "other": OTHER,
    "none": OTHER,
    "none of the above": OTHER,
    "not listed": OTHER,
    "out of set": OTHER,
}


def normalize_label(text: str | None) -> str | None:
    """Map free text to a canonical label, or None if it cannot be mapped."""
    if text is None:
        return None
    key = re.sub(r"[^a-z ]+", " ", str(text).lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    for label in IN_SET_LABELS:
        if label in key:
            return label
    return None


def is_in_set(label: str) -> bool:
    return label in IN_SET_LABELS
