"""Local-only data audit: inventory, schema, label integrity, leakage and bias.

Runs entirely on this machine over the PhysioNet files under --root and writes
aggregates only, to <out>/DATA_AUDIT.md and <out>/data_audit.json. No row, identifier,
date or free-text value is written or printed. Any patient-derived count between 1 and
--min-cell - 1 (default 10) is shown as "<10", and a second cell is hidden ("·") wherever
a margin would otherwise let the first be recovered.

Expected layout (PhysioNet's own; what scripts/download_data.sh or the S3 access points give):
    <root>/mimic-iv-ext-cdm/1.1/*.csv, pathology_ids.json
    <root>/mimiciv/2.2/hosp/*.csv.gz        admissions, patients, diagnoses_icd, labevents, ...
    <root>/mimic-iv-note/2.2/note/*.csv.gz  discharge, radiology

    python scripts/audit_data.py --root data/physionet --out docs
    python scripts/audit_data.py --skip-mimic          # CDM only, seconds instead of minutes
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deferdx.config import load_yaml  # noqa: E402
from deferdx.data.cdm_loader import label_from_icd, label_from_text, load_pathology_ids  # noqa: E402
from deferdx.data.openworld import _prefix_sql, _stratified_sample  # noqa: E402
from deferdx.data.parity import CDM_SANITIZE_TERMS  # noqa: E402
from deferdx.data.splits import lacdm_split  # noqa: E402
from deferdx.data.text import chief_complaint  # noqa: E402
from deferdx.labels import IN_SET_LABELS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("audit")

MIN_CELL = 10
MASK_RE = r"(?<!_)_{4}(?!_)"  # CDM's sanitisation mask; MIMIC's own de-identification uses "___"
CDM_TEXT = {"hpi", "pe", "text", "discharge_diagnosis", "discharge_procedure"}
CDM_CATEGORY = {"modality", "region", "exam_name", "fluid", "category", "icd_version", "icd_title", "icd_diagnosis"}
IDENTIFIERS = {"hadm_id", "subject_id", "note_id", "itemid", "test_itemid", "spec_itemid", "icd_code"}
ATTRS = {"sex": "Sex", "age_group": "Age", "race_group": "Race", "insurance": "Insurance",
         "language": "Language", "marital_status": "Marital status"}
MODALITIES = ("CT", "Ultrasound", "Radiograph", "MRI")
# Procedures that name the answer. The pipeline never loads the procedure tables; this measures why it must not.
PROCEDURE_TERMS = {"appendectomy": r"appendectomy", "cholecystectomy": r"cholecystectomy",
                   "colectomy": r"colectomy|sigmoidectomy", "ERCP": r"\bercp\b|retrograde cholangio"}
ADJ_GAP_PP = 10.0   # label-adjusted availability gap that gets flagged
MIN_GROUP_FLAG = 50  # smallest group considered when flagging gaps


# ---- small-cell suppression and formatting ----------------------------------------


def cell(n: int) -> int | str:
    """A patient-derived count; zero is safe to show, 1..MIN_CELL-1 is not."""
    n = int(n)
    return n if n == 0 or n >= MIN_CELL else f"<{MIN_CELL}"


def rate(k: int, n: int) -> str:
    """k of n as a percentage, hiding it when k or n - k is a small cell."""
    k, n = int(k), int(n)
    if n == 0 or 0 < n < MIN_CELL:
        return "–"
    if 0 < k < MIN_CELL:
        return f"<{MIN_CELL} cases"
    if 0 < n - k < MIN_CELL:
        return f"all but <{MIN_CELL}"
    return f"{100 * k / n:.1f}%"


def suppress(ct: pd.DataFrame, row_pct: bool = False) -> pd.DataFrame:
    """Display strings for a count table. Primary suppression hides 0 < n < MIN_CELL ("<10");
    complementary suppression ("·") hides the smallest other non-zero cell in any row or column
    left with exactly one hidden cell, so no hidden value can be recovered from a margin."""
    ct = ct.fillna(0).astype(int)
    primary = (ct > 0) & (ct < MIN_CELL)
    hide = primary.copy()

    def complement(values: pd.Series, hidden: pd.Series):
        if int(hidden.sum()) != 1:
            return None
        cand = values[(~hidden) & (values > 0)]
        return cand.idxmin() if len(cand) else None

    changed = True
    while changed:
        changed = False
        for r in ct.index:
            c = complement(ct.loc[r], hide.loc[r])
            if c is not None:
                hide.loc[r, c] = changed = True
        for c in ct.columns:
            r = complement(ct[c], hide[c])
            if r is not None:
                hide.loc[r, c] = changed = True
    tot = ct.sum(axis=1)

    def show(r, c) -> str:
        n = int(ct.at[r, c])
        if primary.at[r, c]:
            return f"<{MIN_CELL}"
        if hide.at[r, c]:
            return "·"
        return f"{n:,} ({100 * n / tot[r]:.1f}%)" if row_pct and tot[r] else f"{n:,}"

    return pd.DataFrame([[show(r, c) for c in ct.columns] for r in ct.index], index=ct.index, columns=ct.columns)


def pct(k: float, n: float) -> float | None:
    return round(100.0 * k / n, 1) if n else None


def p_str(p: float | None) -> str:
    return "–" if p is None or (isinstance(p, float) and math.isnan(p)) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.1f}"
    return str(v).replace("|", "/")


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    head = [index_name or str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for idx, row in df.astype(object).iterrows():  # object keeps ints from turning into floats
        lines.append("| " + " | ".join([_fmt(idx)] + [_fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def chi2(ct: pd.DataFrame) -> dict[str, Any]:
    """Chi-square test of independence and Cramér's V, on the unsuppressed counts."""
    from scipy.stats import chi2_contingency

    t = ct.fillna(0)
    t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
    if min(t.shape) < 2:
        return {"p": None, "cramers_v": None}
    stat, p, dof, _ = chi2_contingency(t.to_numpy())
    n = float(t.to_numpy().sum())
    return {"chi2": round(float(stat), 2), "dof": int(dof), "p": float(p),
            "cramers_v": round(math.sqrt(stat / (n * (min(t.shape) - 1))), 3)}


def smd(p1: float, p2: float) -> float | None:
    """Standardised mean difference of two proportions (|SMD| > 0.1 is the usual imbalance cut-off)."""
    s = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return round((p1 - p2) / s, 3) if s else None


class Report:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.md: list[str] = []
        self.flags: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.md += ["", "#" * level + " " + text, ""]

    def p(self, text: str) -> None:
        self.md += [text, ""]

    def table(self, df: pd.DataFrame, index_name: str = "") -> None:
        self.md += [md_table(df, index_name), ""]

    def flag(self, text: str) -> None:
        self.flags.append(text)


def _hadm(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def _json(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return sorted(o)
    if isinstance(o, pd.DataFrame):
        return {str(k): {str(c): v for c, v in row.items()} for k, row in o.to_dict("index").items()}
    return str(o)


# ---- 1. inventory -----------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_inventory(root: Path, rep: Report) -> None:
    rows = []
    for sums in sorted(root.rglob("SHA256SUMS.txt")):
        base = sums.parent
        listed = {}
        for line in sums.read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                listed[parts[1].strip()] = parts[0]
        for f in sorted(p for p in base.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
            rel = f.relative_to(base).as_posix()
            want = listed.get(rel)
            status = "not listed" if want is None else ("ok" if sha256(f) == want else "MISMATCH")
            rows.append({"dataset": base.relative_to(root).as_posix(), "file": rel,
                         "MB": round(f.stat().st_size / 1e6, 1), "sha256": status})
    rep.data["inventory"] = rows
    bad = [r for r in rows if r["sha256"] == "MISMATCH"]
    if bad:
        rep.flag(f"Checksum MISMATCH for {len(bad)} file(s): " + ", ".join(r["file"] for r in bad))
    rep.h(2, "1. Inventory and provenance")
    rep.p(
        "| Dataset | Version | Access | Source |\n|---|---|---|---|\n"
        "| MIMIC-IV-Ext-CDM | 1.1 (2024-07-08) | Credentialed, PhysioNet Credentialed Health DUA 1.5.0 "
        "| https://physionet.org/content/mimic-iv-ext-cdm/1.1/ |\n"
        "| MIMIC-IV (hosp module, 7 tables) | 2.2 | Credentialed, DUA 1.5.0 | https://physionet.org/content/mimiciv/2.2/ |\n"
        "| MIMIC-IV-Note (discharge, radiology) | 2.2 | Credentialed, DUA 1.5.0 "
        "| https://physionet.org/content/mimic-iv-note/2.2/ |"
    )
    rep.p("Every file is checked against the SHA256SUMS.txt PhysioNet ships with it. Files were fetched with "
          "`aws s3 cp` from PhysioNet's S3 access points (`mimiciv-v2-2-01`, `mimic-iv-note-v2-2-01`) and, "
          "for CDM, PhysioNet's ZIP. Version 2.2 is pinned because CDM was built from MIMIC-IV 2.2.")
    rep.table(pd.DataFrame(rows).set_index("file"), "file")


# ---- 2-5. MIMIC-IV-Ext-CDM ----------------------------------------------------------


def profile_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for c in df.columns:
        s = df[c]
        row: dict[str, Any] = {"column": c, "dtype": str(s.dtype), "null_pct": pct(int(s.isna().sum()), len(s))}
        if c in CDM_TEXT:
            lens = s.dropna().astype(str).str.len()
            row["kind"] = "free text (length only)"
            row["summary"] = ("chars p5/p50/p95 = " + "/".join(f"{int(x):,}" for x in lens.quantile([0.05, 0.5, 0.95]))
                              if len(lens) else "empty")
        elif c == "valuestr":
            num = pd.to_numeric(s, errors="coerce")
            row["kind"] = "result value"
            row["summary"] = f"{pct(int(num.notna().sum()), int(s.notna().sum()))}% parse as numbers"
        elif c in IDENTIFIERS:
            row["kind"] = "identifier"
            row["summary"] = f"{s.nunique():,} distinct"
        elif c in CDM_CATEGORY:
            vc = s.value_counts()
            top = [f"{k} ({v:,})" for k, v in vc.head(8).items() if v >= MIN_CELL]
            row["kind"] = "category"
            row["summary"] = f"{len(vc):,} distinct; top: " + ", ".join(top)
        else:
            row["kind"] = "other"
            row["summary"] = f"{s.nunique():,} distinct"
        out.append(row)
    return out


def _by_case(df: pd.DataFrame | None, col: str, idx: pd.Index) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series("", index=idx, dtype=object)
    s = df.assign(_h=_hadm(df["hadm_id"])).groupby("_h")[col].apply(lambda x: "\n".join(x.dropna().astype(str)))
    return s.reindex(idx).fillna("").astype(object)


def _rates(frame: pd.DataFrame, by: str, cols: dict[str, str], order: list[str]) -> pd.DataFrame:
    rows = {}
    for g in order:
        sub = frame[frame[by] == g]
        rows[g] = {"n": cell(len(sub)), **{name: rate(int(sub[c].sum()), len(sub)) for name, c in cols.items()}}
    return pd.DataFrame(rows).T


def audit_cdm(cdm: Path, rep: Report) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Schema, label integrity, information availability and leakage for CDM. Returns per-case
    features (indexed by hadm_id, never written out) and the pathology id lists."""
    tables = {p.stem: pd.read_csv(p, low_memory=False) for p in sorted(cdm.glob("*.csv"))}
    pids = load_pathology_ids(cdm / "pathology_ids.json")
    label_of: dict[str, str] = {}
    multi = 0
    for lab, ids in pids.items():
        for i in ids:
            if i in label_of and label_of[i] != lab:
                multi += 1
            label_of.setdefault(i, lab)
    dup_within = sum(len(v) - len(set(v)) for v in pids.values())
    idx = pd.Index(list(label_of), name="hadm_id")
    feats = pd.DataFrame({"label": [label_of[i] for i in idx]}, index=idx)
    order = [lab for lab in IN_SET_LABELS if lab in set(feats["label"])]
    counts = feats["label"].value_counts()

    # -- schema
    rep.h(2, "2. MIMIC-IV-Ext-CDM: schema")
    schema: dict[str, Any] = {}
    for name, df in tables.items():
        info: dict[str, Any] = {"rows": len(df), "columns": profile_columns(df)}
        if "hadm_id" in df.columns:
            h = _hadm(df["hadm_id"])
            per = h.value_counts()
            info.update(distinct_hadm=int(h.nunique()), rows_per_hadm_p50=float(per.median()),
                        rows_per_hadm_p95=float(per.quantile(0.95)),
                        labelled_coverage_pct=pct(int(idx.isin(set(h)).sum()), len(idx)),
                        unlabelled_hadm=cell(len(set(h) - set(idx))))
        schema[name] = info
        extra = (f"; {info['distinct_hadm']:,} admissions, rows per admission p50/p95 = "
                 f"{info['rows_per_hadm_p50']:.0f}/{info['rows_per_hadm_p95']:.0f}; "
                 f"covers {info['labelled_coverage_pct']}% of labelled cases; admissions without a label: "
                 f"{info['unlabelled_hadm']}" if "hadm_id" in df.columns else "")
        rep.p(f"**{name}.csv**: {len(df):,} rows{extra}")
        rep.table(pd.DataFrame(info["columns"]).set_index("column")[["dtype", "null_pct", "kind", "summary"]], "column")
    rep.p(f"**pathology_ids.json**: {len(pids)} keys ({', '.join(f'{k} {len(v):,}' for k, v in pids.items())}); "
          f"ids listed under two classes: {multi}; duplicate ids within a class: {dup_within}.")
    rep.data["cdm_schema"] = schema

    fmt = load_yaml(REPO / "configs" / "cdm_format.yaml")
    mismatches = []
    for name, spec in fmt["tables"].items():
        df = tables.get(Path(spec["file"]).stem)
        if df is None:
            mismatches.append({"table": name, "file": spec["file"], "missing": "file not found"})
            continue
        missing = [f"{k} -> {v}" for k, v in spec["columns"].items() if v not in df.columns]
        if missing:
            mismatches.append({"table": name, "file": spec["file"], "missing": ", ".join(missing)})
    rep.data["config_mismatches"] = mismatches
    rep.h(3, "Config vs. files (configs/cdm_format.yaml)")
    if mismatches:
        rep.table(pd.DataFrame(mismatches).set_index("table"), "table")
        for m in mismatches:
            rep.flag(f"`configs/cdm_format.yaml` maps `{m['table']}` columns that `{m['file']}` does not have: {m['missing']}.")
    else:
        rep.p("Every column the loader reads exists.")

    # -- label integrity
    rep.h(2, "3. MIMIC-IV-Ext-CDM: labels")
    dist = pd.DataFrame({"cases": counts.reindex(order), "share": (100 * counts / counts.sum()).round(1).reindex(order)})
    rep.table(dist, "label")
    rep.data["label_distribution"] = {k: int(v) for k, v in counts.items()}
    ratio = counts.max() / counts.min()
    if ratio >= 2:
        rep.flag(f"Class imbalance: {counts.idxmax()} {counts.max():,} vs {counts.idxmin()} {counts.min():,} cases "
                 f"({ratio:.1f}x). Report per-class accuracy, not only case-weighted accuracy.")

    icd_cfg = load_yaml(REPO / "configs" / "openworld_icd.yaml")
    agree_rows = {}
    icd_tab = tables.get("icd_diagnosis")
    if icd_tab is not None:
        code_like = icd_tab["icd_diagnosis"].astype(str).str.match(r"^[A-Z]?\d[\dA-Z]{2,}$").mean()
        entries = icd_tab.assign(_h=_hadm(icd_tab["hadm_id"])).groupby("_h")["icd_diagnosis"].apply(
            lambda s: [str(x) for x in s.dropna()])

        def icd_label(es: list[str]) -> str | None:
            flat = [t for e in es for t in re.findall(r"\b[A-Z]?\d[\dA-Z]{2,}\b", e.upper())]
            return label_from_icd(flat, icd_cfg["cdm_conditions"]) or label_from_text(" ; ".join(es))

        feats["icd_label"] = entries.reindex(idx).map(lambda es: icd_label(es) if isinstance(es, list) else None)
        rep.data["icd_diagnosis_code_like_pct"] = round(100 * float(code_like), 1)
    dd = _by_case(tables.get("discharge_diagnosis"), "discharge_diagnosis", idx)
    feats["dd_label"] = dd.map(lambda t: label_from_text(t) if t else None)
    for src, col in (("CDM icd_diagnosis", "icd_label"), ("CDM discharge_diagnosis text", "dd_label")):
        if col not in feats:
            continue
        for lab in order:
            sub = feats[feats["label"] == lab]
            share = float((sub[col] == lab).mean())
            if share < 0.9:
                rep.flag(f"Fallback label source '{src}' agrees with pathology_ids.json on only {100 * share:.0f}% of "
                         f"{lab} cases (the rest are ambiguous or unmapped); CDM v1.0, which lacks pathology_ids.json, "
                         "would lose or mislabel these.")
            agree_rows[(src, lab)] = {
                "agrees": rate(int((sub[col] == lab).sum()), len(sub)),
                "names another condition": rate(int(sub[col].notna().sum() - (sub[col] == lab).sum()), len(sub)),
                "none or ambiguous": rate(int(sub[col].isna().sum()), len(sub)),
            }
    if agree_rows:
        t = pd.DataFrame(agree_rows).T
        t.index = [f"{a}: {b}" for a, b in t.index]
        rep.p("How often a label re-derived from CDM's own diagnosis fields matches `pathology_ids.json` "
              "(the loader's fallback label sources, `label_source` in the config):")
        rep.table(t, "source: class")
        rep.data["label_agreement"] = t

    # -- per-case features (kept in memory only)
    hpi = _by_case(tables.get("history_of_present_illness"), "hpi", idx)
    pe = _by_case(tables.get("physical_examination"), "pe", idx)
    rad_text = _by_case(tables.get("radiology_reports"), "text", idx)
    feats["hpi_chars"] = hpi.str.len()
    feats["pe_chars"] = pe.str.len()
    feats["has_pe"] = pe.str.strip().ne("")
    rad = tables.get("radiology_reports")
    if rad is not None:
        rad = rad.assign(_h=_hadm(rad["hadm_id"]))
        mods = rad.groupby("_h")["modality"].apply(lambda s: set(s.dropna().astype(str)))
        for m in MODALITIES:
            feats[f"has_{m.lower()}"] = [m in mods.get(h, set()) for h in idx]
        feats["has_imaging"] = idx.isin(set(rad["_h"]))
        feats["n_reports"] = rad.groupby("_h").size().reindex(idx).fillna(0).astype(int)
        keys = rad["modality"].astype(str).str.lower() + "|" + rad["region"].astype(str).str.lower()
        n_keys = rad.assign(_k=keys).groupby("_h")["_k"].nunique().reindex(idx).fillna(0)
        feats["reports_dropped_by_dedup"] = (feats["n_reports"] - n_keys).astype(int)
    labs = tables.get("laboratory_tests")
    if labs is not None:
        feats["n_lab_items"] = labs.assign(_h=_hadm(labs["hadm_id"])).groupby("_h")["itemid"].nunique().reindex(idx).fillna(0)
    micro = tables.get("microbiology")
    feats["has_micro"] = idx.isin(set(_hadm(micro["hadm_id"]))) if micro is not None else False

    # -- information available to the agent, by class
    rep.h(2, "4. MIMIC-IV-Ext-CDM: what the environment can reveal, by class")
    avail_cols = {"physical exam": "has_pe", **{m: f"has_{m.lower()}" for m in MODALITIES if f"has_{m.lower()}" in feats},
                  "any imaging": "has_imaging", "microbiology": "has_micro"}
    avail = _rates(feats, "label", avail_cols, order)
    if "n_lab_items" not in feats:
        feats["n_lab_items"] = 0
    avail["lab items (median)"] = [feats.loc[feats.label == lab, "n_lab_items"].median() for lab in order]
    avail["HPI chars (median)"] = [feats.loc[feats.label == lab, "hpi_chars"].median() for lab in order]
    rep.table(avail, "label")
    rep.data["availability_by_label"] = avail
    rep.p("Missing tests are informative: a test that exists for one class much more often than another lets "
          "the agent learn from *whether* a result exists, not only from what it says.")
    dropped = int(feats.get("reports_dropped_by_dedup", pd.Series(dtype=int)).sum())
    affected = int((feats.get("reports_dropped_by_dedup", pd.Series(dtype=int)) > 0).sum())
    if dropped:
        rep.p(f"`dedup_earliest` keeps one report per (modality, region). CDM has no `charttime`, so it keeps the "
              f"first in file order: **{dropped:,} reports across {cell(affected)} cases are never shown to the agent.**")
        rep.flag(f"Imaging dedup drops {dropped:,} radiology reports ({cell(affected)} cases). Without `charttime` "
                 "the 'earliest' report is simply the first row in the file.")
    rep.data["dedup"] = {"reports_total": int(feats.get("n_reports", pd.Series(dtype=int)).sum()),
                         "reports_dropped": dropped, "cases_affected": cell(affected)}

    # -- leakage
    rep.h(2, "5. MIMIC-IV-Ext-CDM: label leakage checks")
    feats["pe_mask"] = pe.str.contains(MASK_RE, regex=True)
    feats["rad_mask"] = rad_text.str.contains(MASK_RE, regex=True)
    mask_t = _rates(feats, "label", {"physical exam contains ____": "pe_mask", "radiology contains ____": "rad_mask"}, order)
    rep.p("CDM replaces each mention of the case's own diagnosis with `____`. If the mask is much more common "
          "in one class, its presence alone predicts the label.")
    rep.table(mask_t, "label")
    rep.data["mask_by_label"] = mask_t
    for col in ("pe_mask", "rad_mask"):
        r = [100 * feats.loc[feats.label == lab, col].mean() for lab in order]
        if max(r) - min(r) >= 10:
            rep.flag(f"The `____` mask in {'physical exam' if col == 'pe_mask' else 'radiology'} text ranges "
                     f"{min(r):.0f}–{max(r):.0f}% across classes; mask presence is itself a label cue "
                     f"(chi-square p {p_str(chi2(pd.crosstab(feats.label, feats[col]))['p'])}).")

    own = {}
    cross = {}
    for lab, terms in CDM_SANITIZE_TERMS.items():
        pat = "|".join(re.escape(t) for t in terms)
        for field, txt in (("HPI", hpi), ("physical exam", pe), ("radiology", rad_text)):
            feats[f"{field}:{lab}"] = txt.str.contains(pat, case=False, regex=True)
    for lab in order:
        sub = feats[feats.label == lab]
        own[lab] = {f: cell(int(sub[f"{f}:{lab}"].sum())) for f in ("HPI", "physical exam", "radiology")}
        cross[lab] = {f"mentions {o}": rate(int(sub[f"radiology:{o}"].sum()), len(sub)) for o in order}
    own_t = pd.DataFrame(own).T
    rep.p("Unmasked mentions of the case's **own** diagnosis (CDM's pipeline should leave none):")
    rep.table(own_t, "label")
    rep.data["own_term_mentions"] = own_t
    if any(isinstance(v, str) or v > 0 for v in own_t.to_numpy().ravel()):
        rep.flag("Some cases still mention their own diagnosis unmasked (see §5); check the sanitize term lists.")
    cross_t = pd.DataFrame(cross).T
    rep.p("Radiology reports that name **other** CDM conditions (rows: true label). CDM masks only the case's own "
          "diagnosis, so 'no evidence of appendicitis' survives in a cholecystitis case and hints by exclusion:")
    rep.table(cross_t, "label")
    rep.data["cross_mentions_radiology"] = cross_t
    off = [100 * feats.loc[feats.label == a, f"radiology:{b}"].mean() for a in order for b in order if a != b]
    if off and max(off) >= 10:
        rep.flag(f"Radiology names other CDM conditions in up to {max(off):.0f}% of a class's cases "
                 "(mostly negations). This is a legitimate but strong exclusion cue; the open-world cases must "
                 "keep the same kind of mentions, or the agent learns the source instead of the disease.")

    proc = (_by_case(tables.get("discharge_procedures"), "discharge_procedure", idx) + "\n"
            + _by_case(tables.get("icd_procedures"), "icd_title", idx))
    ptab = {}
    for lab in order:
        m = feats.label == lab
        ptab[lab] = {k: rate(int(proc[m].str.contains(p, case=False, regex=True).sum()), int(m.sum()))
                     for k, p in PROCEDURE_TERMS.items()}
    ptab = pd.DataFrame(ptab).T
    rep.p("Procedure tables (`discharge_procedures.csv`, `icd_procedures.csv`) name the treatment, and so the "
          "answer. No code under `src/` or `configs/` reads them (checked with grep); keep it that way:")
    rep.table(ptab, "label")
    rep.data["procedure_terms_by_label"] = ptab
    return feats, pids


# ---- 6-9. MIMIC-IV and MIMIC-IV-Note ------------------------------------------------


def _src(path: Path) -> str:
    return f"read_csv('{path.as_posix()}', header=true, all_varchar=true)"


def profile_mimic(con, path: Path, text_col: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    types = con.execute(f"DESCRIBE SELECT * FROM read_csv('{path.as_posix()}', header=true)").fetchall()
    cols = [t[0] for t in types]
    exprs = ["count(*)"] + [f'count(t."{c}")' for c in cols]
    has_h, has_s = "hadm_id" in cols, "subject_id" in cols
    if has_h:
        exprs += ["approx_count_distinct(t.hadm_id)", "count(DISTINCT c.hadm_id)"]
    if has_s:
        exprs.append("approx_count_distinct(t.subject_id)")
    if text_col:
        exprs.append(f'quantile_cont(length(t."{text_col}"), [0.05, 0.5, 0.95])')
    join = " LEFT JOIN cdm_ids c ON t.hadm_id = c.hadm_id" if has_h else ""
    row = list(con.execute(f"SELECT {', '.join(exprs)} FROM {_src(path)} t{join}").fetchone())
    n = int(row.pop(0))
    info: dict[str, Any] = {"rows": n, "columns": [
        {"column": c, "type": ty, "null_pct": pct(n - int(row[i]), n)} for i, (c, ty) in enumerate((t[0], t[1]) for t in types)]}
    row = row[len(cols):]
    if has_h:
        info["distinct_hadm_approx"] = int(row.pop(0))
        info["cdm_admissions_present"] = int(row.pop(0))
    if has_s:
        info["distinct_subject_approx"] = int(row.pop(0))
    if text_col:
        info["text_chars_p5_p50_p95"] = [int(x) for x in row.pop(0)]
    info["seconds"] = round(time.time() - t0, 1)
    log.info("profiled %s in %.0fs", path.name, info["seconds"])
    return info


def race_group(r: Any) -> str:
    r = r.upper() if isinstance(r, str) else ""
    if r.startswith("WHITE") or r == "PORTUGUESE":
        return "White"
    if r.startswith("BLACK"):
        return "Black"
    if r.startswith("HISPANIC") or r == "SOUTH AMERICAN":
        return "Hispanic/Latino"
    if r.startswith("ASIAN"):
        return "Asian"
    if r in {"", "UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
        return "Unknown"
    return "Other"


def derive_groups(adm: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    d = adm.merge(pat, on="subject_id", how="left")
    year = pd.to_numeric(d["admittime"].astype(str).str[:4], errors="coerce")
    d["age"] = pd.to_numeric(d["anchor_age"], errors="coerce") + (year - pd.to_numeric(d["anchor_year"], errors="coerce"))
    d["sex"] = d["gender"].map({"F": "Female", "M": "Male"}).fillna("Unknown")
    age = pd.cut(d["age"], [0, 30, 45, 65, 80, 200], right=False, labels=["18-29", "30-44", "45-64", "65-79", "80+"])
    d["age_group"] = age.astype(object).where(age.notna(), "Unknown")
    d["race_group"] = d["race"].map(race_group)
    d["insurance"] = d["insurance"].fillna("Unknown")
    d["language"] = np.where(d["language"].eq("ENGLISH"), "English", "Non-English/unknown")
    d["marital_status"] = d["marital_status"].fillna("Unknown")
    d["year_group"] = d["anchor_year_group"].fillna("Unknown")
    return d


def _order(s: pd.Series) -> list[str]:
    return sorted(s.dropna().unique(), key=lambda x: (x == "Unknown", str(x)))


def shortcut(frame: pd.DataFrame, y: np.ndarray, cols: list[str], seed: int = 0) -> dict[str, Any]:
    """Cross-validated logistic regression on demographics only: how much of the label they give away."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = pd.get_dummies(frame[cols].astype(str)).to_numpy(float)
    y = np.asarray(y)
    classes = np.unique(y)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(max_iter=5000), x, y, cv=cv, method="predict_proba")
    pred = classes[proba.argmax(1)]
    out: dict[str, Any] = {"features": cols, "n": int(len(y)),
                           "accuracy": round(float(accuracy_score(y, pred)), 3),
                           "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
                           "majority_accuracy": round(float(pd.Series(y).value_counts(normalize=True).iloc[0]), 3)}
    if len(classes) == 2:
        out["auroc"] = round(float(roc_auc_score(y == classes[1], proba[:, 1])), 3)
    else:
        out["macro_auroc_ovr"] = round(float(roc_auc_score(y, proba, multi_class="ovr", labels=classes)), 3)
        out["per_class_auroc"] = {str(c): round(float(roc_auc_score(y == c, proba[:, i])), 3) for i, c in enumerate(classes)}
    return out


def _profile(frame: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, m in groups.items():
        sub = frame[m]
        n = len(sub)
        q = sub["age"].quantile([0.25, 0.5, 0.75]) if n >= MIN_CELL else None
        rows[name] = {
            "n": cell(n), "female": rate(int((sub.sex == "Female").sum()), n),
            "age median (IQR)": f"{q[0.5]:.0f} ({q[0.25]:.0f}–{q[0.75]:.0f})" if q is not None else "–",
            **{r: rate(int((sub.race_group == r).sum()), n) for r in ("White", "Black", "Hispanic/Latino", "Asian", "Unknown")},
            "Medicaid": rate(int((sub.insurance == "Medicaid").sum()), n),
            "non-English/unknown": rate(int((sub.language != "English").sum()), n),
        }
    return pd.DataFrame(rows).T


def adjusted_rate(sub: pd.DataFrame, col: str, weights: pd.Series) -> float | None:
    """Direct standardisation: the group's per-class rates weighted by the cohort's class mix."""
    num = den = 0.0
    for lab, w in weights.items():
        g = sub[sub.label == lab]
        if len(g):
            num += w * float(g[col].mean())
            den += w
    return 100 * num / den if den else None


def cohort_order_is_stable(dx_path: Path, icd_cfg: dict[str, Any], runs: int = 3) -> bool:
    """Re-run build_openworld's cohort query verbatim (no ORDER BY) and compare row order across runs."""
    import duckdb

    other_case, _ = _prefix_sql(icd_cfg["other_groups"])
    cdm_case, _ = _prefix_sql(icd_cfg["cdm_conditions"])
    orders = []
    for _ in range(runs):
        con = duckdb.connect()
        con.execute(f"CREATE VIEW dx AS SELECT * FROM read_csv_auto('{dx_path.as_posix()}', all_varchar=true)")
        con.execute("CREATE VIEW dxv AS SELECT subject_id, hadm_id, CAST(seq_num AS INTEGER) AS seq_num, "
                    "CAST(icd_version AS INTEGER) AS icd_version, icd_code FROM dx")
        orders.append([r[0] for r in con.execute(f"""
            WITH tagged AS (SELECT d.subject_id, d.hadm_id, d.seq_num, {other_case} AS other_group,
                                   {cdm_case} AS cdm_group FROM dxv d),
                 cdm_any AS (SELECT DISTINCT hadm_id FROM tagged WHERE cdm_group IS NOT NULL)
            SELECT t.hadm_id FROM tagged t
            WHERE t.seq_num = 1 AND t.other_group IS NOT NULL AND t.hadm_id NOT IN (SELECT hadm_id FROM cdm_any)
        """).fetchall()])
        con.close()
    return all(o == orders[0] for o in orders)


def audit_mimic(root: Path, feats: pd.DataFrame, pids: dict[str, list[str]], rep: Report) -> None:
    import duckdb
    from scipy.stats import kruskal

    hosp = root / "mimiciv" / "2.2" / "hosp"
    note = root / "mimic-iv-note" / "2.2" / "note"
    con = duckdb.connect()
    con.register("cdm_ids_df", pd.DataFrame({"hadm_id": feats.index.astype(str)}))
    con.execute("CREATE TABLE cdm_ids AS SELECT hadm_id FROM cdm_ids_df")

    # -- schema
    rep.h(2, "6. MIMIC-IV 2.2 and MIMIC-IV-Note 2.2: schema")
    rep.p("Types are DuckDB's inference from a sample; counts are exact except distinct ids (HyperLogLog, ~2%). "
          "\"CDM admissions present\" is how many of the 2,400 CDM admissions appear in the table.")
    schema = {}
    files = [(hosp / f"{t}.csv.gz", None) for t in ("admissions", "patients", "diagnoses_icd", "d_icd_diagnoses",
                                                   "d_labitems", "microbiologyevents", "labevents")]
    files += [(note / "discharge.csv.gz", "text"), (note / "radiology.csv.gz", "text")]
    for path, text_col in files:
        if not path.exists():
            rep.p(f"**{path.name}**: not found, skipped.")
            continue
        info = profile_mimic(con, path, text_col)
        schema[path.name] = info
        bits = [f"{info['rows']:,} rows"]
        if "distinct_hadm_approx" in info:
            bits.append(f"~{info['distinct_hadm_approx']:,} admissions")
            bits.append(f"CDM admissions present: {info['cdm_admissions_present']:,}/{len(feats):,}")
        if "distinct_subject_approx" in info:
            bits.append(f"~{info['distinct_subject_approx']:,} patients")
        if text_col:
            bits.append("text chars p5/p50/p95 = " + "/".join(f"{x:,}" for x in info["text_chars_p5_p50_p95"]))
        rep.p(f"**{path.parent.name}/{path.name}**: " + "; ".join(bits))
        rep.table(pd.DataFrame(info["columns"]).set_index("column"), "column")
    rep.data["mimic_schema"] = schema

    # build_openworld fetches labs, micro and radiology by hadm_id; on CDM's own admissions, does that find
    # what CDM has? (MIMIC leaves hadm_id empty on many ED-era rows.)
    n_cdm = len(feats)
    own_key, join_key = "CDM's own tables", "hadm_id join (build_openworld)"
    join_cov = {}
    for fname, feat, what in (("labevents.csv.gz", feats["n_lab_items"] > 0, "labs"),
                              ("microbiologyevents.csv.gz", feats["has_micro"], "microbiology"),
                              ("radiology.csv.gz", feats.get("has_imaging", pd.Series(False, index=feats.index)), "radiology")):
        if fname in schema:
            join_cov[what] = {own_key: rate(int(feat.sum()), n_cdm),
                              join_key: rate(schema[fname]["cdm_admissions_present"], n_cdm)}
    if join_cov:
        t = pd.DataFrame(join_cov).T
        rep.p("**Can the open-world builder reproduce CDM's data?** `build_openworld` joins labs, microbiology and "
              "radiology on `hadm_id` (`src/deferdx/data/openworld.py`). Run on CDM's own 2,400 admissions, that join "
              "finds this share of cases with any data, against what CDM itself holds:")
        rep.table(t, "source")
        rep.data["openworld_join_coverage"] = t
        gaps = [f"{w} {v[join_key]} vs {v[own_key]}" for w, v in join_cov.items()]
        rep.flag("Open-world cases will carry less data than CDM cases: joined on `hadm_id` as `build_openworld` "
                 "does, CDM's own admissions have " + "; ".join(gaps) + " (join vs CDM). MIMIC leaves `hadm_id` "
                 "empty on many ED-era lab and microbiology rows, which CDM evidently includes. An agent can learn "
                 "'sparse data → OTHER'. Fetch by `subject_id` and a time window around the admission instead, and "
                 "re-run the source-classifier check.")

    # -- demographics
    adm = con.execute(f"SELECT subject_id, hadm_id, admittime, admission_type, admission_location, insurance, language, "
                      f"marital_status, race FROM {_src(hosp / 'admissions.csv.gz')}").df()
    pat = con.execute(f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group "
                      f"FROM {_src(hosp / 'patients.csv.gz')}").df()
    demo = derive_groups(adm, pat)
    cohort = demo.merge(feats, left_on="hadm_id", right_index=True, how="inner")
    missing = len(feats) - len(cohort)
    if missing:
        rep.flag(f"{cell(missing)} CDM admissions are not in MIMIC-IV 2.2 admissions.")
    order = [lab for lab in IN_SET_LABELS if lab in set(cohort.label)]
    ed = demo[demo.admission_location == "EMERGENCY ROOM"]

    rep.h(2, "7. Bias analysis")
    rep.p("Group definitions: sex from `patients.gender`; age = `anchor_age` + (admission year − `anchor_year`); "
          "race collapsed from `admissions.race` (White incl. Portuguese; Hispanic/Latino incl. South American; "
          "Unknown = unknown, unable to obtain, declined); `language` in MIMIC-IV 2.2 is `ENGLISH` or `?`. "
          "Chi-square p-values are uncorrected; with this many tests, treat p < 0.001 as the bar.")

    rep.h(3, "7.1 Who is in the cohort")
    path_t = suppress(pd.crosstab(cohort.admission_location, cohort.label).reindex(columns=order), row_pct=False)
    rep.p("Admission route (`admission_location`) by class:")
    rep.table(path_t, "admission location")
    rep.data["admission_location_by_label"] = path_t
    prof = pd.concat([
        _profile(cohort, {"CDM (all)": cohort.label.notna(), **{lab: cohort.label == lab for lab in order}}),
        _profile(ed, {"Baseline: ED admissions": pd.Series(True, index=ed.index)}),
        _profile(demo, {"Baseline: all admissions": pd.Series(True, index=demo.index)}),
    ])
    rep.table(prof, "cohort")
    rep.data["cohort_profiles"] = prof

    rep.h(3, "7.2 Representation: CDM vs. MIMIC-IV ED admissions")
    rep.p("SMD = standardised difference of proportions, CDM vs. ED admissions; |SMD| ≥ 0.1 is flagged.")
    repr_out = {}
    for attr, title in ATTRS.items():
        cats = _order(demo[attr])
        n_cdm = cohort[attr].value_counts().reindex(cats).fillna(0).astype(int)
        shown = suppress(pd.DataFrame({"n": n_cdm}))["n"]
        rows, off = {}, []
        for c in cats:
            p1 = n_cdm[c] / len(cohort)
            p2 = float((ed[attr] == c).mean())
            hidden = not str(shown[c]).replace(",", "").isdigit()
            rows[c] = {"CDM n": shown[c], "CDM %": "–" if hidden else f"{100 * p1:.1f}",
                       "ED admissions %": rate(int((ed[attr] == c).sum()), len(ed)),
                       "all admissions %": rate(int((demo[attr] == c).sum()), len(demo)),
                       "SMD vs ED": None if hidden else smd(p1, p2)}
            if not hidden and rows[c]["SMD vs ED"] is not None and abs(rows[c]["SMD vs ED"]) >= 0.1:
                off.append(f"{c} {100 * p1:.1f}% vs {100 * p2:.1f}% (SMD {rows[c]['SMD vs ED']:+.2f})")
        if off:
            rep.flag(f"Representation vs. ED admissions, {title.lower()} (CDM vs ED): " + "; ".join(off) + ".")
        t = pd.DataFrame(rows).T
        rep.p(f"**{title}**")
        rep.table(t, title.lower())
        repr_out[attr] = t
    rep.data["representation"] = repr_out

    rep.h(3, "7.3 Does the label depend on demographics?")
    rep.p("Row percentages: the class mix within each group. A strong association is not a bug (cholecystitis "
          "is more common in women), but it is a shortcut the agent can take instead of reasoning from findings.")
    assoc = {}
    for attr, title in ATTRS.items():
        ct = pd.crosstab(cohort[attr], cohort.label).reindex(columns=order).reindex(_order(cohort[attr]))
        st = chi2(ct)
        assoc[attr] = {"table": suppress(ct, row_pct=True), **st}
        rep.p(f"**{title}**: chi-square p {p_str(st['p'])}, Cramér's V {st['cramers_v']}")
        rep.table(assoc[attr]["table"], title.lower())
        if st["cramers_v"] is not None and st["cramers_v"] >= 0.1 and st["p"] is not None and st["p"] < 0.001:
            rep.flag(f"Label depends on {title.lower()} (Cramér's V {st['cramers_v']}, p {p_str(st['p'])}).")
    ct = pd.crosstab(cohort.year_group, cohort.label).reindex(columns=order)
    st = chi2(ct)
    rep.p(f"**Year group** (`anchor_year_group`, the patient's anchor period): chi-square p {p_str(st['p'])}, "
          f"Cramér's V {st['cramers_v']}")
    rep.table(suppress(ct, row_pct=True), "year group")
    assoc["year_group"] = {"table": suppress(ct, row_pct=True), **st}
    rep.data["label_by_group"] = assoc

    cols = list(ATTRS)
    sc_all = shortcut(cohort, cohort.label.to_numpy(), cols)
    sc_sa = shortcut(cohort, cohort.label.to_numpy(), ["sex", "age_group"])
    rep.p("**Demographics-only classifier** (5-fold CV logistic regression on one-hot groups; no clinical data):")
    rep.table(pd.DataFrame({
        "all six attributes": {k: sc_all.get(k) for k in ("accuracy", "macro_f1", "macro_auroc_ovr", "majority_accuracy")},
        "sex + age only": {k: sc_sa.get(k) for k in ("accuracy", "macro_f1", "macro_auroc_ovr", "majority_accuracy")},
    }).T.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v), "features")
    rep.p("Per-class one-vs-rest AUROC (all attributes): "
          + ", ".join(f"{k} {v:.3f}" for k, v in sc_all["per_class_auroc"].items()))
    rep.data["demographic_shortcut"] = {"all": sc_all, "sex_age": sc_sa}
    if sc_all["macro_auroc_ovr"] >= 0.6:
        best = max(sc_all["per_class_auroc"], key=sc_all["per_class_auroc"].get)
        rep.flag(f"Demographics alone predict the label with macro-AUROC {sc_all['macro_auroc_ovr']:.2f} "
                 f"({best} {sc_all['per_class_auroc'][best]:.2f}). An agent can score above chance before ordering "
                 "any test; compare against this floor.")

    rep.h(3, "7.4 Does the environment reveal less for some groups?")
    rep.p("Share of cases where each source exists, crude and label-adjusted (direct standardisation to the "
          f"cohort's class mix, so a gap is not just a different class mix). Gaps ≥ {ADJ_GAP_PP:.0f} points between "
          f"groups of ≥ {MIN_GROUP_FLAG} cases are flagged. Medians for text length and lab count.")
    weights = cohort.label.value_counts(normalize=True)
    # HPI length relative to the class median, so a gap is not just older patients having pancreatitis
    cohort["hpi_rel"] = cohort.hpi_chars / cohort.groupby("label").hpi_chars.transform("median")
    bin_cols = {"CT": "has_ct", "Ultrasound": "has_ultrasound", "MRI": "has_mri", "any imaging": "has_imaging",
                "microbiology": "has_micro"}
    bin_cols = {k: v for k, v in bin_cols.items() if v in cohort}
    info_out = {}
    for attr, title in ATTRS.items():
        rows = {}
        adj_by_col: dict[str, dict[str, float]] = {k: {} for k in bin_cols}
        for g in _order(cohort[attr]):
            sub = cohort[cohort[attr] == g]
            n = len(sub)
            row: dict[str, Any] = {"n": cell(n)}
            for name, col in bin_cols.items():
                row[name] = rate(int(sub[col].sum()), n)
                a = adjusted_rate(sub, col, weights) if n >= MIN_CELL else None
                row[f"{name} (adj.)"] = None if a is None else round(a, 1)
                if a is not None and n >= MIN_GROUP_FLAG:
                    adj_by_col[name][g] = a
            for name, col in (("lab items", "n_lab_items"), ("HPI chars", "hpi_chars"), ("PE chars", "pe_chars")):
                row[f"{name} (median)"] = float(sub[col].median()) if n >= MIN_CELL else None
            row["HPI ÷ class median (median)"] = round(float(sub.hpi_rel.median()), 2) if n >= MIN_CELL else None
            rows[g] = row
        t = pd.DataFrame(rows).T
        kw = {}
        for name, col in (("HPI chars", "hpi_rel"), ("lab items", "n_lab_items")):
            parts = [cohort.loc[cohort[attr] == g, col].dropna() for g in _order(cohort[attr])]
            parts = [p for p in parts if len(p) >= MIN_CELL]
            kw[name] = float(kruskal(*parts).pvalue) if len(parts) > 1 else None
        rep.p(f"**{title}** (Kruskal-Wallis p: HPI length within class {p_str(kw['HPI chars'])}, "
              f"lab count {p_str(kw['lab items'])})")
        rep.table(t, title.lower())
        info_out[attr] = {"table": t, "kruskal_p": kw}
        for name, vals in adj_by_col.items():
            if len(vals) > 1 and max(vals.values()) - min(vals.values()) >= ADJ_GAP_PP:
                hi, lo = max(vals, key=vals.get), min(vals, key=vals.get)
                rep.flag(f"{name} availability (label-adjusted) differs by {title.lower()}: {hi} {vals[hi]:.0f}% vs "
                         f"{lo} {vals[lo]:.0f}%.")
        if kw["HPI chars"] is not None and kw["HPI chars"] < 0.001:
            meds = {g: float(cohort.loc[cohort[attr] == g, "hpi_rel"].median()) for g in _order(cohort[attr])
                    if (cohort[attr] == g).sum() >= MIN_GROUP_FLAG}
            if meds and max(meds.values()) >= 1.2 * min(meds.values()):
                hi, lo = max(meds, key=meds.get), min(meds, key=meds.get)
                rep.flag(f"HPI length differs by {title.lower()} even within class: {hi} {meds[hi]:.2f}× vs {lo} "
                         f"{meds[lo]:.2f}× the class median; the agent sees less history for some groups.")
    rep.data["information_by_group"] = info_out

    rep.h(3, "7.5 The LA-CDM split (80/10/10, seed 269)")
    split = lacdm_split(pids)
    cohort["split"] = cohort.hadm_id.map(split)
    sp = _profile(cohort, {s: cohort.split == s for s in ("train", "val", "test")})
    rep.table(sp, "split")
    split_tests = {a: chi2(pd.crosstab(cohort[a], cohort.split)) for a in ("sex", "race_group", "insurance", "language")}
    rep.p("Chi-square across splits: " + "; ".join(f"{ATTRS[a].lower()} p {p_str(v['p'])}" for a, v in split_tests.items()))
    per_subject = cohort.groupby("subject_id").agg(n=("hadm_id", "size"), labels=("label", "nunique"),
                                                  splits=("split", "nunique"))
    repeat, mixed, leak = int((per_subject.n > 1).sum()), int((per_subject.labels > 1).sum()), int((per_subject.splits > 1).sum())
    rep.p(f"Patients with more than one CDM admission: {cell(repeat)}; with admissions under different labels: "
          f"{cell(mixed)}; **with admissions in more than one split: {cell(leak)}**.")
    rep.data["split"] = {"profiles": sp, "tests": split_tests, "patients_repeat": cell(repeat),
                         "patients_mixed_labels": cell(mixed), "patients_across_splits": cell(leak)}
    if leak:
        rep.flag(f"Patient-level leakage in the LA-CDM split: {cell(leak)} patients have admissions in more than one "
                 "split. `deferdx data build-cdm` reports 0 because CDM's CSVs carry no `subject_id` (see §2).")

    # -- ICD cross-check and open-world pool
    icd_cfg = load_yaml(REPO / "configs" / "openworld_icd.yaml")
    other_case, _ = _prefix_sql(icd_cfg["other_groups"])
    cdm_case, _ = _prefix_sql(icd_cfg["cdm_conditions"])
    con.execute(f"CREATE TABLE dx AS SELECT subject_id, hadm_id, CAST(seq_num AS INTEGER) AS seq_num, "
                f"CAST(icd_version AS INTEGER) AS icd_version, icd_code FROM {_src(hosp / 'diagnoses_icd.csv.gz')}")
    con.execute(f"CREATE TABLE tagged AS SELECT d.hadm_id, d.seq_num, {other_case} AS other_group, "
                f"{cdm_case} AS cdm_group FROM dx d")
    prim = con.execute(
        "SELECT t.hadm_id, max(CASE WHEN t.seq_num = 1 THEN t.cdm_group END) AS primary_cdm, "
        "max(CASE WHEN t.seq_num = 1 THEN t.other_group END) AS primary_other, "
        "string_agg(DISTINCT t.cdm_group, ',') AS cdm_groups "
        "FROM tagged t JOIN cdm_ids USING (hadm_id) GROUP BY t.hadm_id").df().set_index("hadm_id")
    c2 = cohort.join(prim, on="hadm_id")
    c2["cdm_groups"] = c2["cdm_groups"].fillna("")
    rep.h(2, "8. MIMIC-IV diagnoses vs. CDM labels")
    rows, cooc = {}, {}
    for lab in order:
        sub = c2[c2.label == lab]
        n = len(sub)
        rows[lab] = {
            "primary dx = this condition": rate(int((sub.primary_cdm == lab).sum()), n),
            "primary dx = another CDM condition": rate(int((sub.primary_cdm.notna() & (sub.primary_cdm != lab)).sum()), n),
            "primary dx = an OTHER group": rate(int(sub.primary_other.notna().sum()), n),
            "any code for this condition": rate(int(sub.cdm_groups.str.contains(lab).sum()), n),
        }
        cooc[lab] = {f"also coded {o}": rate(int(sub.cdm_groups.str.contains(o).sum()), n) for o in order if o != lab}
    t = pd.DataFrame(rows).T
    rep.p("Using the ICD prefixes in `configs/openworld_icd.yaml` (`cdm_conditions`, `other_groups`):")
    rep.table(t, "label")
    cooc = pd.DataFrame(cooc).T.reindex(columns=[f"also coded {o}" for o in order]).fillna("–")
    rep.table(cooc, "label")
    rep.data["icd_vs_label"] = {"primary": t, "co_coded": cooc}
    for lab in order:
        share = float((c2.loc[c2.label == lab, "primary_cdm"] == lab).mean())
        if share < 0.9:
            rep.flag(f"Only {100 * share:.0f}% of {lab} cases have it as the primary (seq 1) ICD diagnosis.")

    rep.h(2, "9. Open-world (OTHER) candidate pool")
    pool = con.execute(
        "SELECT DISTINCT t.hadm_id, t.other_group FROM tagged t WHERE t.seq_num = 1 AND t.other_group IS NOT NULL "
        "AND t.hadm_id NOT IN (SELECT hadm_id FROM tagged WHERE cdm_group IS NOT NULL) "
        "AND t.hadm_id NOT IN (SELECT hadm_id FROM cdm_ids)").df()
    con.register("pool_df", pool[["hadm_id"]])
    notes = con.execute(f"SELECT n.hadm_id, n.text FROM {_src(note / 'discharge.csv.gz')} n "
                        "JOIN pool_df p ON n.hadm_id = p.hadm_id").df()
    note_by = dict(zip(notes.hadm_id.astype(str), notes.text))
    del notes
    complaint_re = re.compile(icd_cfg["complaint_regex"], re.IGNORECASE)
    pool["has_note"] = pool.hadm_id.isin(note_by)
    pool["abdominal"] = [bool(note_by.get(h)) and bool(complaint_re.search(chief_complaint(note_by[h])))
                         for h in pool.hadm_id]
    del note_by
    counts = pool.groupby("other_group").agg(icd_candidates=("hadm_id", "size"), with_discharge_note=("has_note", "sum"),
                                             abdominal_chief_complaint=("abdominal", "sum"))
    rep.p("Same filters as `build_openworld`: primary diagnosis in an OTHER group, no CDM condition code at any "
          "position, not a CDM admission, a discharge note, and a chief complaint matching `complaint_regex`. "
          "The builder then drops HPIs that leak the diagnosis, so the final set is somewhat smaller.")
    rep.table(suppress(counts), "OTHER group")
    rep.data["openworld_pool"] = suppress(counts)
    small = [g for g, v in counts.abdominal_chief_complaint.items() if v < 100]
    if small:
        rep.flag("Open-world groups with fewer than 100 abdominal-complaint candidates (they cap the stratified "
                 f"sample): {', '.join(small)}.")

    rows = pool[pool.abdominal].sort_values("hadm_id").to_dict("records")
    sample = _stratified_sample(rows, "other_group", 800, random.Random(0))
    if not sample:
        rep.flag("No open-world candidates passed the filters; the OTHER class cannot be built from this data.")
        return
    ow = demo.merge(pd.DataFrame(sample)[["hadm_id", "other_group"]], on="hadm_id", how="inner")
    rep.p(f"Demographics of an 800-case stratified sample drawn the way `build_openworld` draws it (seed 0, input "
          f"sorted by `hadm_id`), next to CDM ({len(ow):,} sampled):")
    ow_prof = pd.concat([_profile(cohort, {"CDM (in-set)": cohort.label.notna()}),
                         _profile(ow, {"OTHER sample": ow.other_group.notna()}),
                         _profile(ow, {g: ow.other_group == g for g in sorted(ow.other_group.unique())})])
    rep.table(ow_prof, "cohort")
    both = pd.concat([cohort[cols].assign(y="in-set"), ow[cols].assign(y="other")], ignore_index=True)
    sc_ow = shortcut(both, both.y.to_numpy(), cols)
    rep.p(f"Demographics-only classifier, in-set vs. OTHER: AUROC {sc_ow['auroc']:.3f} "
          f"(accuracy {sc_ow['accuracy']:.3f}, majority {sc_ow['majority_accuracy']:.3f}).")
    rep.data["openworld_demographics"] = {"profiles": ow_prof, "shortcut": sc_ow}
    if sc_ow["auroc"] >= 0.6:
        rep.flag(f"OTHER cases differ demographically from CDM: demographics alone separate them with AUROC "
                 f"{sc_ow['auroc']:.2f}. Match or report this, or DEFER/OTHER can be learned from who the patient is.")
    stable = cohort_order_is_stable(hosp / "diagnoses_icd.csv.gz", icd_cfg)
    rep.data["openworld_cohort_order_stable"] = stable
    rep.p(f"`build_openworld` runs its cohort query without `ORDER BY`, then shuffles each group with a seeded RNG. "
          f"Re-running that exact query 3 times returned rows in {'the same' if stable else 'a different'} order. "
          "(The audit sorts by `hadm_id` before sampling, so its own numbers are deterministic.)")
    if not stable:
        rep.flag("The open-world sample is not reproducible: `build_openworld`'s cohort query has no `ORDER BY` and "
                 "DuckDB returned its rows in a different order on each of 3 runs, so the same seed draws a different "
                 "OTHER cohort. Add `ORDER BY hadm_id` before sampling.")


# ---- driver -------------------------------------------------------------------------


def run_audit(root: Path, skip_mimic: bool = False) -> Report:
    rep = Report()
    t0 = time.time()
    audit_inventory(root, rep)
    log.info("inventory done (%.0fs)", time.time() - t0)
    cdm = root / "mimic-iv-ext-cdm" / "1.1"
    feats, pids = audit_cdm(cdm, rep)
    log.info("CDM audit done (%.0fs)", time.time() - t0)
    hosp = root / "mimiciv" / "2.2" / "hosp"
    if skip_mimic or not (hosp / "admissions.csv.gz").exists() or not (hosp / "patients.csv.gz").exists():
        rep.h(2, "6-9. MIMIC-IV and MIMIC-IV-Note")
        rep.p("Skipped: " + ("--skip-mimic." if skip_mimic else "MIMIC-IV hosp admissions/patients not found."))
    else:
        audit_mimic(root, feats, pids, rep)
        log.info("MIMIC audit done (%.0fs)", time.time() - t0)
    return rep


def render(rep: Report) -> str:
    head = [
        "# Data audit",
        "",
        f"Generated {dt.date.today().isoformat()} by `scripts/audit_data.py` from the local PhysioNet files. "
        "Re-run it after any data change: `python scripts/audit_data.py --root data/physionet --out docs`.",
        "",
        "**What this file contains.** Aggregates only: counts, percentages, quantiles and test statistics. It holds "
        "no rows, identifiers, dates or note text. Patient-derived counts from 1 to "
        f"{MIN_CELL - 1} are shown as `<{MIN_CELL}`, and `·` marks a second cell hidden so the first cannot be "
        "recovered from a total. The underlying data stays under the PhysioNet Credentialed Health DUA and is "
        "never committed (`.gitignore` excludes `data/`).",
        "",
        "## Findings",
        "",
        *[f"- {f}" for f in rep.flags],
    ]
    return "\n".join(head + rep.md).rstrip() + "\n"


def main() -> None:
    global MIN_CELL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/physionet", help="directory holding the PhysioNet datasets")
    ap.add_argument("--out", default="docs", help="where DATA_AUDIT.md and data_audit.json go")
    ap.add_argument("--min-cell", type=int, default=10, help="smallest patient-derived count shown")
    ap.add_argument("--skip-mimic", action="store_true", help="CDM only (no MIMIC-IV / Note scans)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    MIN_CELL = args.min_cell
    rep = run_audit(Path(args.root), skip_mimic=args.skip_mimic)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep.data["findings"] = rep.flags
    rep.data["min_cell"] = MIN_CELL
    (out / "DATA_AUDIT.md").write_text(render(rep), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(rep.data, indent=1, default=_json), encoding="utf-8")
    log.info("wrote %s and %s (%d findings)", out / "DATA_AUDIT.md", out / "data_audit.json", len(rep.flags))


if __name__ == "__main__":
    main()
