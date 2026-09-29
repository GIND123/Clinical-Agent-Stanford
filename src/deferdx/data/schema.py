"""Canonical case representation shared by the CDM loader, the open-world builder,
the synthetic generator and the environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LabResult:
    name: str
    value: str
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str | None = None
    fluid: str | None = None
    charttime: str | None = None
    itemid: str | None = None  # MIMIC d_labitems itemid, when known


@dataclass
class MicroResult:
    test_name: str
    specimen: str | None = None
    result: str = ""
    charttime: str | None = None
    itemid: str | None = None


@dataclass
class ImagingReport:
    modality: str
    region: str
    exam_name: str
    text: str
    charttime: str | None = None


# History sections an ASK action can reveal. `hpi` is always shown at reset.
HISTORY_SECTIONS: tuple[str, ...] = (
    "past_medical_history",
    "past_surgical_history",
    "medications",
    "allergies",
    "family_history",
    "social_history",
)


@dataclass
class Case:
    case_id: str
    label: str
    hpi: str
    history: dict[str, str] = field(default_factory=dict)
    physical_exam: str = ""
    labs: list[LabResult] = field(default_factory=list)
    microbiology: list[MicroResult] = field(default_factory=list)
    imaging: list[ImagingReport] = field(default_factory=list)
    subject_id: str | None = None
    source: str = "cdm"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        return cls(
            case_id=str(d["case_id"]),
            label=d["label"],
            hpi=d.get("hpi", ""),
            history=dict(d.get("history") or {}),
            physical_exam=d.get("physical_exam", "") or "",
            labs=[LabResult(**x) for x in d.get("labs") or []],
            microbiology=[MicroResult(**x) for x in d.get("microbiology") or []],
            imaging=[ImagingReport(**x) for x in d.get("imaging") or []],
            subject_id=None if d.get("subject_id") is None else str(d["subject_id"]),
            source=d.get("source", "cdm"),
            meta=dict(d.get("meta") or {}),
        )


def dedup_earliest(case: Case) -> Case:
    """Keep only the earliest result of each repeated test (LDTL preprocessing).

    Labs are keyed by (name, fluid), microbiology by test name, imaging by
    (modality, region). Items without a charttime keep their original order and
    sort after timed items of the same key only if no timed item exists.
    """

    def earliest(items, key_fn):
        best: dict[Any, Any] = {}
        order: list[Any] = []
        for item in items:
            k = key_fn(item)
            if k not in best:
                best[k] = item
                order.append(k)
                continue
            cur = best[k]
            if item.charttime and (not cur.charttime or item.charttime < cur.charttime):
                best[k] = item
        return [best[k] for k in order]

    case.labs = earliest(case.labs, lambda x: (x.name.lower(), (x.fluid or "").lower()))
    case.microbiology = earliest(case.microbiology, lambda x: x.test_name.lower())
    case.imaging = earliest(case.imaging, lambda x: (x.modality.lower(), x.region.lower()))
    return case
