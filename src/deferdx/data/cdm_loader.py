"""Load MIMIC-IV-Ext-CDM v1.0 into canonical `Case` records.

Two on-disk layouts are supported, selected in `configs/cdm_format.yaml`:

* ``csv``    — the PhysioNet release (one table per modality, keyed by hadm_id).
* ``pickle`` — the per-pathology ``*_hadm_info_first_diag.pkl`` dictionaries produced by
  Hager et al.'s MIMIC-Clinical-Decision-Making-Framework.

File and column names are configuration, not code, because they MUST be checked
against the files you actually downloaded (`deferdx data inspect-cdm`). Everything
here runs locally; nothing is sent anywhere.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..labels import IN_SET_LABELS, normalize_label
from .schema import Case, ImagingReport, LabResult, MicroResult, dedup_earliest
from .text import classify_radiology, split_history

log = logging.getLogger(__name__)


# ---- helpers ---------------------------------------------------------------------


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _find(cdm_dir: Path, name: str) -> Path | None:
    for cand in (cdm_dir / name, cdm_dir / f"{name}.gz"):
        if cand.exists():
            return cand
    hits = list(cdm_dir.rglob(name)) + list(cdm_dir.rglob(f"{name}.gz"))
    return hits[0] if hits else None


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _itemid(x: Any) -> str:
    s = _str(x)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _str(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def inspect_cdm(cdm_dir: str | Path, nrows: int = 3) -> str:
    """Describe every table under `cdm_dir` (columns + a few rows) for config checking."""
    cdm_dir = Path(cdm_dir)
    lines = []
    for path in sorted(list(cdm_dir.rglob("*.csv")) + list(cdm_dir.rglob("*.csv.gz"))):
        df = pd.read_csv(path, nrows=nrows)
        lines.append(f"== {path.relative_to(cdm_dir)}\n   columns: {list(df.columns)}")
    for path in sorted(cdm_dir.rglob("*.pkl")):
        obj = pd.read_pickle(path)
        if isinstance(obj, dict) and obj:
            first = next(iter(obj.values()))
            keys = list(first.keys()) if isinstance(first, dict) else type(first).__name__
            lines.append(f"== {path.relative_to(cdm_dir)}  dict[{len(obj)}] -> {keys}")
        elif isinstance(obj, pd.DataFrame):
            lines.append(f"== {path.relative_to(cdm_dir)}  DataFrame columns: {list(obj.columns)}")
    return "\n".join(lines) or f"no csv/pkl files found under {cdm_dir}"


# ---- label assignment --------------------------------------------------------------


def label_from_icd(codes: list[str], icd_groups: dict[str, dict[str, list[str]]]) -> str | None:
    """Assign an in-set label if exactly one CDM condition's ICD prefixes match."""
    norm = [re.sub(r"[^A-Za-z0-9]", "", c).upper() for c in codes]
    hits = {
        label
        for label, by_version in icd_groups.items()
        for prefixes in by_version.values()
        for p in prefixes
        if any(c.startswith(p) for c in norm)
    }
    return hits.pop() if len(hits) == 1 else None


def label_from_text(text: str) -> str | None:
    t = text.lower()
    hits = {label for label in IN_SET_LABELS if label in t}
    return hits.pop() if len(hits) == 1 else None


# ---- lab-name mapping --------------------------------------------------------------


def _lab_mapping(df: pd.DataFrame | None, cols: dict[str, str]) -> dict[str, tuple[str, str | None]]:
    """itemid -> (canonical label, fluid). CDM's mapping merges equivalent itemids;
    `corresponding_ids` (if present) lists the itemids folded into each row."""
    mapping: dict[str, tuple[str, str | None]] = {}
    if df is None:
        return mapping
    for _, row in df.iterrows():
        label = _str(row.get(cols.get("label", "label")))
        fluid = _str(row.get(cols.get("fluid", "fluid"))) or None
        ids = [row.get(cols.get("itemid", "itemid"))]
        extra = row.get(cols.get("corresponding_ids", "corresponding_ids"))
        if isinstance(extra, (list, tuple)):
            ids.extend(extra)
        elif isinstance(extra, str) and extra.strip():
            ids.extend(re.findall(r"\d+", extra))
        for i in ids:
            if _str(i):
                mapping[_itemid(i)] = (label, fluid)
    return mapping


# ---- CSV layout --------------------------------------------------------------------


def load_cdm_csv(cdm_dir: str | Path, fmt: dict[str, Any], icd_groups: dict | None = None) -> list[Case]:
    cdm_dir = Path(cdm_dir)
    tables = fmt["tables"]
    key = fmt.get("key", "hadm_id")

    def table(name: str) -> pd.DataFrame | None:
        spec = tables.get(name)
        if not spec:
            return None
        path = _find(cdm_dir, spec["file"])
        if path is None:
            log.warning("CDM table %s (%s) not found under %s", name, spec["file"], cdm_dir)
            return None
        return _read_table(path)

    def col(name: str, field: str) -> str:
        return tables[name]["columns"][field]

    hist = table("history")
    if hist is None:
        raise FileNotFoundError(f"history table {tables['history']['file']} not found under {cdm_dir}")
    cases: dict[str, dict[str, Any]] = {}
    for _, r in hist.iterrows():
        cid = str(r[key])
        cases[cid] = {"history_text": _str(r[col("history", "text")]),
                      "subject_id": _str(r.get(tables["history"]["columns"].get("subject_id", "subject_id"))) or None}

    pe = table("physical_exam")
    if pe is not None:
        for _, r in pe.iterrows():
            if str(r[key]) in cases:
                cases[str(r[key])]["pe"] = _str(r[col("physical_exam", "text")])

    mapping_df = table("lab_mapping")
    lab_map = _lab_mapping(mapping_df, tables.get("lab_mapping", {}).get("columns", {}))
    labs: dict[str, list[LabResult]] = defaultdict(list)
    lab_df = table("labs")
    if lab_df is not None:
        c = tables["labs"]["columns"]
        for r in lab_df.to_dict("records"):
            cid = str(r[key])
            itemid = _str(r.get(c["itemid"]))
            itemid = _itemid(itemid)
            name, fluid = lab_map.get(itemid, (itemid, None))
            labs[cid].append(LabResult(
                name=name, value=_str(r.get(c["value"])), unit=_str(r.get(c.get("unit", ""))) or None,
                ref_low=_num(r.get(c.get("ref_low", ""))), ref_high=_num(r.get(c.get("ref_high", ""))),
                fluid=fluid, charttime=_str(r.get(c.get("charttime", ""))) or None,
            ))

    micro: dict[str, list[MicroResult]] = defaultdict(list)
    micro_df = table("microbiology")
    if micro_df is not None:
        c = tables["microbiology"]["columns"]
        for r in micro_df.to_dict("records"):
            test = _str(r.get(c["test"]))
            test = lab_map.get(test, (test, None))[0] if test.replace(".", "").isdigit() else test
            micro[str(r[key])].append(MicroResult(
                test_name=test, specimen=_str(r.get(c.get("specimen", ""))) or None,
                result=_str(r.get(c["value"])), charttime=_str(r.get(c.get("charttime", ""))) or None,
            ))

    imaging: dict[str, list[ImagingReport]] = defaultdict(list)
    rad_df = table("radiology")
    if rad_df is not None:
        c = tables["radiology"]["columns"]
        for r in rad_df.to_dict("records"):
            text = _str(r.get(c["text"]))
            h_mod, h_reg, h_name = classify_radiology(text)
            imaging[str(r[key])].append(ImagingReport(
                modality=_str(r.get(c.get("modality", ""))) or h_mod,
                region=_str(r.get(c.get("region", ""))) or h_reg,
                exam_name=_str(r.get(c.get("exam_name", ""))) or h_name,
                text=text, charttime=_str(r.get(c.get("charttime", ""))) or None,
            ))

    labels = _labels_csv(table, tables, key, fmt.get("label_source", ["discharge_diagnosis", "icd"]), icd_groups)

    out = []
    dropped = 0
    for cid, d in cases.items():
        label = labels.get(cid)
        if label is None:
            dropped += 1
            continue
        hpi, sections = split_history(d["history_text"])
        case = Case(
            case_id=cid, subject_id=d.get("subject_id"), label=label, hpi=hpi, history=sections,
            physical_exam=d.get("pe", ""), labs=labs.get(cid, []), microbiology=micro.get(cid, []),
            imaging=imaging.get(cid, []), source="cdm",
        )
        out.append(dedup_earliest(case))
    if dropped:
        log.warning("dropped %d CDM admissions with no unambiguous label", dropped)
    return out


def _labels_csv(table, tables, key, sources, icd_groups) -> dict[str, str]:
    labels: dict[str, str] = {}
    for source in sources:
        if source == "column" and "label" in tables:
            df = table("label")
            if df is not None:
                c = tables["label"]["columns"]["label"]
                for _, r in df.iterrows():
                    lab = normalize_label(_str(r[c]))
                    if lab:
                        labels.setdefault(str(r[key]), lab)
        elif source == "discharge_diagnosis" and "discharge_diagnosis" in tables:
            df = table("discharge_diagnosis")
            if df is not None:
                c = tables["discharge_diagnosis"]["columns"]["text"]
                for _, r in df.iterrows():
                    lab = label_from_text(_str(r[c]))
                    if lab:
                        labels.setdefault(str(r[key]), lab)
        elif source == "icd" and "icd_diagnosis" in tables and icd_groups:
            df = table("icd_diagnosis")
            if df is not None:
                c = tables["icd_diagnosis"]["columns"]["code"]
                grouped = df.groupby(key)[c].apply(lambda s: [_str(x) for x in s])
                for cid, codes in grouped.items():
                    # a cell may itself hold a list literal of codes
                    flat = [t for code in codes for t in re.findall(r"[A-Z]?\d[\dA-Z]*", code.upper())]
                    lab = label_from_icd(flat, icd_groups)
                    if lab:
                        labels.setdefault(str(cid), lab)
    return labels


# ---- pickle layout (Hager et al. framework) -----------------------------------------


def load_cdm_pickles(cdm_dir: str | Path, fmt: dict[str, Any]) -> list[Case]:
    cdm_dir = Path(cdm_dir)
    pk = fmt["pickle"]
    keys = pk["keys"]
    mapping_path = _find(cdm_dir, pk["lab_mapping_file"])
    lab_map = {}
    if mapping_path is not None:
        lab_map = _lab_mapping(pd.read_pickle(mapping_path), pk.get("lab_mapping_columns", {}))
    out = []
    for label in IN_SET_LABELS:
        path = _find(cdm_dir, pk["file_pattern"].format(pathology=label))
        if path is None:
            log.warning("pickle for %s not found (%s)", label, pk["file_pattern"])
            continue
        data: dict = pd.read_pickle(path)
        for hadm_id, rec in data.items():
            lows = rec.get(keys["ref_low"], {}) or {}
            highs = rec.get(keys["ref_high"], {}) or {}
            labs = []
            for itemid, value in (rec.get(keys["labs"], {}) or {}).items():
                name, fluid = lab_map.get(str(itemid), (str(itemid), None))
                labs.append(LabResult(name=name, value=_str(value), ref_low=_num(lows.get(itemid)),
                                      ref_high=_num(highs.get(itemid)), fluid=fluid))
            micro = [MicroResult(test_name=lab_map.get(str(k), (str(k), None))[0], result=_str(v))
                     for k, v in (rec.get(keys["microbiology"], {}) or {}).items()]
            imaging = []
            for rad in rec.get(keys["radiology"], []) or []:
                text = _str(rad.get("Report"))
                h_mod, h_reg, h_name = classify_radiology(text)
                imaging.append(ImagingReport(
                    modality=_str(rad.get("Modality")) or h_mod, region=_str(rad.get("Region")) or h_reg,
                    exam_name=_str(rad.get("Exam Name")) or h_name, text=text,
                ))
            hpi, sections = split_history(_str(rec.get(keys["history"])))
            case = Case(
                case_id=str(hadm_id), label=label, hpi=hpi, history=sections,
                physical_exam=_str(rec.get(keys["physical_exam"])), labs=labs,
                microbiology=micro, imaging=imaging, source="cdm",
            )
            out.append(dedup_earliest(case))
    return out


def load_cdm(cdm_dir: str | Path, fmt: dict[str, Any], icd_groups: dict | None = None) -> list[Case]:
    layout = fmt.get("layout", "csv")
    if layout == "csv":
        return load_cdm_csv(cdm_dir, fmt, icd_groups)
    if layout == "pickle":
        return load_cdm_pickles(cdm_dir, fmt)
    raise ValueError(f"unknown CDM layout {layout!r}")
