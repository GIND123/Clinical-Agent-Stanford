"""Test/ASK catalog: maps an action argument to case content and a cost."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import load_yaml
from ..data.schema import Case, LabResult


@dataclass
class TestSpec:
    key: str
    kind: str  # lab | micro | imaging
    cost: float
    display: str
    names: list[str] = field(default_factory=list)
    fluid: str | None = None
    modality: list[str] = field(default_factory=list)
    region: list[str] = field(default_factory=list)
    itemids: set[str] = field(default_factory=set)
    cost_source: str = "unspecified"


@dataclass
class AskSpec:
    key: str
    cost: float
    display: str


def _fmt_lab(lab: LabResult) -> str:
    unit = f" {lab.unit}" if lab.unit else ""
    ref = ""
    if lab.ref_low is not None or lab.ref_high is not None:
        lo = "" if lab.ref_low is None else f"{lab.ref_low:g}"
        hi = "" if lab.ref_high is None else f"{lab.ref_high:g}"
        ref = f" (ref {lo}-{hi})"
    flag = f" [{lab.flag}]" if lab.flag else ""
    return f"{lab.name}: {lab.value}{unit}{ref}{flag}"


class TestCatalog:
    def __init__(self, tests: dict[str, TestSpec], asks: dict[str, AskSpec]):
        self.tests = tests
        self.asks = asks

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TestCatalog":
        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, d: dict) -> "TestCatalog":
        tests = {}
        for key, spec in (d.get("tests") or {}).items():
            tests[key] = TestSpec(
                key=key, kind=spec["kind"], cost=float(spec.get("cost", 0)), display=spec.get("display", key),
                names=list(spec.get("names", [])), fluid=spec.get("fluid"),
                modality=list(spec.get("modality", [])), region=list(spec.get("region", [])),
                itemids={str(i) for i in spec.get("itemids", [])}, cost_source=spec.get("cost_source", "unspecified"),
            )
        asks = {
            key: AskSpec(key=key, cost=float(spec.get("cost", 0)), display=spec.get("display", key))
            for key, spec in (d.get("ask") or {}).items()
        }
        return cls(tests, asks)

    # ---- resolution -------------------------------------------------------------

    @property
    def total_test_cost(self) -> float:
        return sum(s.cost for s in self.tests.values())

    @staticmethod
    def _matches(spec: TestSpec, name: str, fluid: str | None, itemid: str | None) -> bool:
        """itemid decides when both sides have one (robust to label synonyms and to
        same-name analytes in different fluids); otherwise fall back to name + fluid."""
        if itemid and spec.itemids:
            return itemid in spec.itemids
        if name.lower() not in {n.lower() for n in spec.names}:
            return False
        return not (spec.fluid and fluid and fluid.lower() != spec.fluid.lower())

    def _match_labs(self, spec: TestSpec, case: Case) -> list[LabResult]:
        return [lab for lab in case.labs if self._matches(spec, lab.name, lab.fluid, lab.itemid)]

    def resolve_test(self, key: str, case: Case) -> str | None:
        """Result text for TEST(key) on this case, or None if never performed."""
        spec = self.tests[key]
        if spec.kind == "lab":
            labs = self._match_labs(spec, case)
            return "\n".join(_fmt_lab(x) for x in labs) if labs else None
        if spec.kind == "micro":
            hits = [m for m in case.microbiology if self._matches(spec, m.test_name, None, m.itemid)]
            if not hits:
                return None
            return "\n".join(f"{m.test_name}{f' ({m.specimen})' if m.specimen else ''}: {m.result}" for m in hits)
        if spec.kind == "imaging":
            mods = {m.lower() for m in spec.modality}
            regs = {r.lower() for r in spec.region}
            hits = [im for im in case.imaging if im.modality.lower() in mods and im.region.lower() in regs]
            if not hits:
                return None
            return "\n\n".join(f"{im.exam_name or spec.display}:\n{im.text}" for im in hits)
        raise ValueError(f"unknown test kind {spec.kind!r}")

    def resolve_ask(self, key: str, case: Case) -> str | None:
        if key == "physical_exam":
            return case.physical_exam or None
        return case.history.get(key) or None

    def available_tests(self, case: Case) -> list[str]:
        return [k for k in self.tests if self.resolve_test(k, case) is not None]

    def unreached_lab_names(self, cases: list[Case]) -> dict[str, int]:
        """Lab names present in the data that no catalog entry covers (for coverage audits)."""
        lab_specs = [s for s in self.tests.values() if s.kind == "lab"]
        counts: dict[str, int] = {}
        for case in cases:
            for lab in case.labs:
                if not any(self._matches(s, lab.name, lab.fluid, lab.itemid) for s in lab_specs):
                    counts[lab.name] = counts.get(lab.name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
