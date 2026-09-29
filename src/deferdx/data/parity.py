"""Text processing that mirrors how MIMIC-IV-Ext-CDM itself was built, so OTHER cases
extracted from MIMIC-IV look like CDM cases in everything except the disease.

Re-implemented from the behaviour of Hager et al.'s dataset code
(github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset, MIT licence):

* Patient history = discharge-summary text from "History of Present Illness:" up to the
  physical-exam header, with newlines replaced by spaces (dataset/discharge.py).
* Physical exam = from the PE header to "Pertinent Results:" (or "Brief Hospital
  Course:"), with everything from the first "discharge" mention removed.
* Radiology = report sections whose header starts with IMPRESSION, HISTORY, INDICATION,
  COMPARISON, ... are dropped (dataset/radiology.py), i.e. no conclusions.
* Sanitisation (dataset/dataset.py::sanitize_hadm_texts): if the history mentions a
  target-disease term the admission is discarded; in the physical exam and radiology the
  term is replaced by "____" (four underscores; MIMIC de-identification uses three).

Why it matters: CDM text carries "____" wherever its diagnosis was named and never
contains an IMPRESSION section. If OTHER cases are not processed identically, an agent
can separate them by formatting alone and the open-world results mean nothing.
"""

from __future__ import annotations

import re

MASK = "____"

# CreateDataset.py sanitize lists for the four CDM pathologies.
CDM_SANITIZE_TERMS: dict[str, list[str]] = {
    "appendicitis": ["acute appendicitis", "appendicitis", "appendectomy"],
    "cholecystitis": ["acute cholecystitis", "cholecystitis", "cholecystostomy"],
    "pancreatitis": ["acute pancreatitis", "pancreatitis", "pancreatectomy"],
    "diverticulitis": ["acute diverticulitis", "diverticulitis"],
}

_PE_END_FOR_HISTORY = [
    r"physical exam:", r"physical examination:", r"physical ___:", r"pe:", r"pe ___:",
    r"(?:pertinent|___) results:", r"hospital course:",
]
_PE_START = [r"physical exam:", r"physical examination:", r"physical ___:", r"pe:", r"pe ___:", r"pertinent results:"]

_BAD_RAD_FIELDS = (
    "CLINICAL HISTORY", "MEDICAL HISTORY", "CLINICAL INFORMATION", "COMPARISON", "COMPARISONS", "COMMENT",
    "CONCLUSION", "HISTORY", "IMPRESSION", "CLINICAL INDICATION", "INDICATION", "OPERATORS", "REASON",
    "REFERENCE", "DATE",
)


def cdm_history(discharge: str) -> str:
    text = (discharge or "").replace("\n", " ")
    for end in _PE_END_FOR_HISTORY:
        m = re.search(rf"(?:history|___) of present(?:ing)? illness:(.*?){end}", text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def cdm_physical_exam(discharge: str) -> str:
    text = (discharge or "").replace("\n", " ")
    terminal = "pertinent results:" if "pertinent results:" in text.lower() else "brief hospital course:"
    for start in _PE_START:
        m = re.search(rf"{start}(.*?){terminal}", text, re.IGNORECASE | re.DOTALL)
        if m:
            body = m.group(1)
            body = re.sub(r"(?:at|upon|on)? ?discharge.*", "", body, flags=re.IGNORECASE | re.DOTALL)
            return body.strip()
    return ""


def _parse_report(report: str) -> list[tuple[str, str]]:
    lines = (report or "").strip().split("\n")
    if lines and lines[0].isupper() and lines[0].strip() and lines[0].strip()[-1] != ":":
        lines[0] = lines[0].strip() + ":"
    lines = [ln.strip() + ":" if ln.isupper() and ":" not in ln else ln for ln in lines]
    joined = "\n".join(lines)
    pattern = r"(?m)^([A-Z \t,._-]+):((?:(?!^[A-Z \t,._-]+:).)*)"
    return [(h.strip(), b.strip()) for h, b in re.findall(pattern, joined, re.DOTALL)]


def cdm_radiology(report: str) -> str:
    """Drop conclusion/indication-type sections; '' if nothing informative remains."""
    out, informative = [], False
    for header, body in _parse_report(report):
        if any(header.startswith(bad) for bad in _BAD_RAD_FIELDS):
            continue
        informative = informative or bool(body)
        out.append(f"{header}:\n{body}\n")
    return "\n".join(out).strip() if informative else ""


def mentions(text: str, terms: list[str]) -> bool:
    return any(re.search(re.escape(t), text or "", re.IGNORECASE) for t in terms)


def mask(text: str, terms: list[str]) -> str:
    # longest first so "acute appendicitis" is masked as one span
    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(re.escape(t), MASK, text or "", flags=re.IGNORECASE)
    return text
