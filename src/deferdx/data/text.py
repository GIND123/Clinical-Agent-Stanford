"""Clinical-note text utilities: section splitting and radiology classification."""

from __future__ import annotations

import re

# Header aliases -> canonical history section. Matched at line starts, case-insensitive.
_HISTORY_HEADERS: dict[str, tuple[str, ...]] = {
    "hpi": ("history of present illness", "hpi", "present illness"),
    "past_medical_history": ("past medical history", "pmh", "medical history"),
    "past_surgical_history": ("past surgical history", "psh", "surgical history"),
    "medications": ("medications on admission", "home medications", "medications", "meds"),
    "allergies": ("allergies",),
    "family_history": ("family history", "fhx", "fh"),
    "social_history": ("social history", "shx", "sh"),
    "chief_complaint": ("chief complaint", "cc"),
    "physical_exam": ("physical exam", "physical examination", "admission exam", "exam"),
}

# Discharge-summary headers (regex fragments) that terminate a section we care about.
# Anything starting with "Discharge ..." matters most: e.g. "Discharge Diagnosis:"
# must never be swallowed into an earlier section, or the label leaks.
_STOP_HEADERS = (
    r"discharge[a-z /]*",
    r"pertinent results",
    r"brief hospital course",
    r"hospital course",
    r"(?:major )?surgical or invasive procedure[a-z ]*",
    r"assessment and plan",
    r"impression",
    r"labs",
    r"followup instructions",
    r"facility",
    r"attending",
    r"service",
)


def _header_regex(aliases: tuple[str, ...]) -> str:
    return "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))


_ANY_HEADER = re.compile(
    r"^[ \t]*(?P<h>"
    + _header_regex(tuple(a for al in _HISTORY_HEADERS.values() for a in al))
    + "|"
    + "|".join(_STOP_HEADERS)
    + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)


def _canonical(header: str) -> str | None:
    h = header.lower().strip()
    for name, aliases in _HISTORY_HEADERS.items():
        if h in aliases:
            return name
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split a note into canonical sections by `Header:` lines.

    Text before the first recognised header is returned under "preamble". Unknown /
    stop headers end the previous section and are dropped. First occurrence wins.
    """
    sections: dict[str, str] = {}
    matches = list(_ANY_HEADER.finditer(text or ""))
    if not matches:
        return {"preamble": (text or "").strip()}
    pre = text[: matches[0].start()].strip()
    if pre:
        sections["preamble"] = pre
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = _canonical(m.group("h"))
        body = text[m.end() : end].strip()
        if name and body and name not in sections:
            sections[name] = body
    return sections


def split_history(text: str) -> tuple[str, dict[str, str]]:
    """Return (hpi, other_sections) from a patient-history blob.

    If no headers are found the whole text is treated as the HPI, which is the
    conservative choice: the agent sees everything at reset, nothing is hidden.
    """
    sections = split_sections(text)
    hpi = sections.pop("hpi", None)
    preamble = sections.pop("preamble", "")
    sections.pop("physical_exam", None)
    if hpi is None:
        hpi = preamble
    elif preamble:
        hpi = f"{preamble}\n{hpi}"
    return hpi.strip(), sections


def admission_physical_exam(discharge_note: str) -> str:
    """Extract the admission physical exam from a discharge summary.

    MIMIC discharge summaries often contain both an admission and a discharge exam
    under one 'Physical Exam:' header; we keep text up to the first 'DISCHARGE'
    marker so the discharge exam (which may reveal the outcome) is excluded.
    """
    sections = split_sections(discharge_note)
    exam = sections.get("physical_exam", "")
    cut = re.search(r"\bdischarge\b", exam, re.IGNORECASE)
    if cut:
        exam = exam[: cut.start()]
    return re.sub(r"(?i)^\s*admission( physical)? exam[:\s]*", "", exam).strip()


def chief_complaint(discharge_note: str) -> str:
    m = re.search(r"chief complaint\s*:\s*(.*)", discharge_note or "", re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---- radiology -----------------------------------------------------------------

_MODALITY_PATTERNS = (
    ("CT", r"\bCT\b|\bCTA\b|computed tomography"),
    ("MRI", r"\bMRI?\b|\bMRCP\b|magnetic resonance"),
    ("Ultrasound", r"\bUS\b|ultrasound|sonogra|\bDUPLEX\b"),
    ("Radiograph", r"\bX-?RAY\b|radiograph|\bCXR\b|\bKUB\b|\bCHEST \(PA|\bPORTABLE\b"),
    ("Nuclear", r"\bHIDA\b|scintigra|nuclear"),
)
_REGION_PATTERNS = (
    ("Abdomen", r"\bABD|abdom|pelvi|\bRUQ\b|liver|gallbladder|pancrea|renal|kidney|appendix|\bKUB\b|\bHIDA\b"),
    ("Chest", r"chest|thora|\bCXR\b|lung"),
    ("Head", r"head|brain"),
)


def classify_radiology(text: str) -> tuple[str, str, str]:
    """Heuristic (modality, region, exam_name) from a radiology report's header."""
    head = (text or "")[:400]
    m = re.search(r"(?:EXAMINATION|EXAM|STUDY)\s*:\s*(.+)", head, re.IGNORECASE)
    exam_name = m.group(1).strip() if m else head.strip().splitlines()[0] if head.strip() else ""
    probe = exam_name or head
    modality = next((name for name, pat in _MODALITY_PATTERNS if re.search(pat, probe, re.IGNORECASE)), "Other")
    region = next((name for name, pat in _REGION_PATTERNS if re.search(pat, probe, re.IGNORECASE)), "Other")
    return modality, region, exam_name
