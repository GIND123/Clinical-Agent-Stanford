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
            )
        asks = {
            key: AskSpec(key=key, cost=float(spec.get("cost", 0)), display=spec.get("display", key))
            for key, spec in (d.get("ask") or {}).items()
        }
        return cls(tests, asks)

    # ---- resolution -------------------------------------------------------------

    def _match_labs(self, spec: TestSpec, case: Case) -> list[LabResult]:
        names = {n.lower() for n in spec.names}
        out = []
        for lab in case.labs:
            if lab.name.lower() not in names:
                continue
            if spec.fluid and lab.fluid and lab.fluid.lower() != spec.fluid.lower():
                continue
            out.append(lab)
        return out

    def resolve_test(self, key: str, case: Case) -> str | None:
        """Result text for TEST(key) on this case, or None if never performed."""
        spec = self.tests[key]
        if spec.kind == "lab":
            labs = self._match_labs(spec, case)
            return "\n".join(_fmt_lab(x) for x in labs) if labs else None
        if spec.kind == "micro":
            names = {n.lower() for n in spec.names}
            hits = [m for m in case.microbiology if m.test_name.lower() in names]
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
        covered = {n.lower() for s in self.tests.values() if s.kind == "lab" for n in s.names}
        counts: dict[str, int] = {}
        for case in cases:
            for lab in case.labs:
                if lab.name.lower() not in covered:
                    counts[lab.name] = counts.get(lab.name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
