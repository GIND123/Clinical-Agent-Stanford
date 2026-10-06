"""Build MIMIC-CDM-OW: abdominal-pain admissions whose primary diagnosis is NOT one of
the four CDM conditions (label = OTHER), extracted from MIMIC-IV v2.2 + MIMIC-IV-Note.

Deterministic ICD filtering in DuckDB over the local csv.gz files; no model, no API.

Methodological guard: if OTHER cases are extracted by a different pipeline than the
in-set CDM cases, the agent can learn to spot the *pipeline* (formatting, "____" masks,
missing IMPRESSION sections) instead of the disease. Text is therefore processed exactly
as CDM's own pipeline does (parity.py), and ``include_controls=True`` extracts in-set
controls through this same code so a source classifier can be checked against them.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

from ..labels import OTHER
from .schema import Case, ImagingReport, LabResult, MicroResult, dedup_earliest
from .parity import CDM_SANITIZE_TERMS, cdm_history, cdm_physical_exam, cdm_radiology, mask, mentions
from .text import chief_complaint, classify_radiology

log = logging.getLogger(__name__)


def _norm_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()


def _prefix_sql(groups: dict[str, dict[str, list[str]]], alias: str = "d") -> tuple[str, list[str]]:
    """SQL CASE expression mapping (icd_version, icd_code) to a group name."""
    whens = []
    for group, by_version in groups.items():
        for version, prefixes in by_version.items():
            v = int(str(version).lstrip("icd"))
            for p in prefixes:
                whens.append(
                    f"WHEN {alias}.icd_version = {v} AND upper({alias}.icd_code) LIKE '{_norm_code(p)}%' "
                    f"THEN '{group}'"
                )
    return "CASE " + " ".join(whens) + " END", list(groups)


def _path(root: Path, *parts: str) -> str:
    for suffix in ("", ".gz"):
        p = root.joinpath(*parts[:-1], parts[-1] + suffix)
        if p.exists():
            return p.as_posix()
    raise FileNotFoundError(root.joinpath(*parts))


def build_openworld(
    mimic_dir: str | Path,
    note_dir: str | Path,
    icd_cfg: dict[str, Any],
    target_n: int = 800,
    seed: int = 0,
    exclude_case_ids: set[str] | None = None,
    include_controls: bool = False,
    controls_per_label: int = 100,
    complaint_regex: str | None = None,
    hpi_leak_policy: str = "drop",
    require_cdm_inclusion: bool = True,
) -> list[Case]:
    """Return OTHER cases (and optionally same-pipeline in-set controls).

    mimic_dir: MIMIC-IV root containing ``hosp/`` (admissions, diagnoses_icd, labevents,
               d_labitems, microbiologyevents as .csv or .csv.gz).
    note_dir:  MIMIC-IV-Note root containing ``note/discharge`` and ``note/radiology``.
    require_cdm_inclusion: apply MIMIC-IV-Ext-CDM's own inclusion rule (Hager et al., check_missing):
               physical exam of at least 40 characters, at least one lab, at least one abdominal study.
    """
    import duckdb

    mimic_dir, note_dir = Path(mimic_dir), Path(note_dir)
    hosp = lambda t: _path(mimic_dir, "hosp", f"{t}.csv")  # noqa: E731
    note = lambda t: _path(note_dir, "note", f"{t}.csv")  # noqa: E731

    other_case, _ = _prefix_sql(icd_cfg["other_groups"])
    cdm_case, _ = _prefix_sql(icd_cfg["cdm_conditions"])
    complaint_re = re.compile(complaint_regex or icd_cfg.get("complaint_regex", r"abd|abdominal|belly|epigastr|flank|RLQ|RUQ|LLQ|LUQ"), re.IGNORECASE)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW dx AS SELECT * FROM read_csv_auto('{hosp('diagnoses_icd')}', all_varchar=true)")
    con.execute(
        "CREATE VIEW dxv AS SELECT subject_id, hadm_id, CAST(seq_num AS INTEGER) AS seq_num, "
        "CAST(icd_version AS INTEGER) AS icd_version, icd_code FROM dx"
    )
    # Admissions touching ANY CDM condition code are excluded from OTHER entirely.
    cohort_sql = f"""
        WITH tagged AS (
            SELECT d.subject_id, d.hadm_id, d.seq_num, d.icd_code, d.icd_version,
                   {other_case} AS other_group, {cdm_case} AS cdm_group
            FROM dxv d
        ),
        cdm_any AS (SELECT DISTINCT hadm_id FROM tagged WHERE cdm_group IS NOT NULL)
        SELECT t.subject_id, t.hadm_id, t.icd_code, t.icd_version, t.other_group, t.cdm_group
        FROM tagged t
        WHERE t.seq_num = 1 AND (
            (t.other_group IS NOT NULL AND t.hadm_id NOT IN (SELECT hadm_id FROM cdm_any))
            OR ({'TRUE' if include_controls else 'FALSE'} AND t.cdm_group IS NOT NULL)
        )
        ORDER BY t.hadm_id  -- without this DuckDB's row order varies by run, and so does the seeded sample
    """
    cohort = con.execute(cohort_sql).fetchdf()
    exclude = exclude_case_ids or set()
    cohort = cohort[~cohort["hadm_id"].astype(str).isin(exclude)]
    log.info("primary-diagnosis candidates: %d", len(cohort))
    if cohort.empty:
        return []

    con.register("cohort_df", cohort[["hadm_id"]].drop_duplicates())
    notes = con.execute(
        f"SELECT n.hadm_id, n.text FROM read_csv_auto('{note('discharge')}', all_varchar=true) n "
        "JOIN cohort_df c ON n.hadm_id = c.hadm_id ORDER BY n.hadm_id, n.note_id"
    ).fetchdf()
    note_by_hadm = dict(zip(notes["hadm_id"].astype(str), notes["text"]))

    # Keep only abdominal presentations with an extractable HPI.
    rows = []
    for r in cohort.to_dict("records"):
        r["other_group"] = _s(r["other_group"]) or None  # NULL may surface as NaN
        r["cdm_group"] = _s(r["cdm_group"]) or None
        text = note_by_hadm.get(str(r["hadm_id"]))
        if not text or not complaint_re.search(chief_complaint(text)):
            continue
        rows.append(r)
    log.info("abdominal-presentation candidates with notes: %d", len(rows))

    rng = random.Random(seed)
    others = [r for r in rows if r["other_group"] and not r["cdm_group"]]
    controls = [r for r in rows if r["cdm_group"]]
    selected = _stratified_sample(others, "other_group", target_n, rng)
    if include_controls:
        selected += _stratified_sample(controls, "cdm_group", controls_per_label * 4, rng)
    if not selected:
        return []

    ids = sorted({str(r["hadm_id"]) for r in selected})
    con.register("sel_df", __import__("pandas").DataFrame({"hadm_id": ids}))
    try:
        transfers = hosp("transfers")
    except FileNotFoundError:
        transfers = None
        log.warning("hosp/transfers not found: labs, microbiology and radiology are joined on hadm_id only, so "
                    "OTHER cases will lack the ED-era rows MIMIC-IV-Ext-CDM includes")
    if transfers:
        con.execute(
            f"""
            CREATE TEMP TABLE win AS
            SELECT t.hadm_id, any_value(t.subject_id) AS subject_id,
                   min(TRY_CAST(t.intime AS TIMESTAMP)) - INTERVAL 1 DAY AS win_start,
                   max(TRY_CAST(t.intime AS TIMESTAMP)) AS win_end
            FROM read_csv_auto('{transfers}', all_varchar=true) t JOIN sel_df s ON t.hadm_id = s.hadm_id
            GROUP BY t.hadm_id
            """
        )

    def attached(table: str, cols: str, time_col: str) -> str:
        """Rows of `table` that belong to the selected admissions: the same hadm_id, or, as MIMIC-IV-Ext-CDM
        does (Hager et al.'s fill_nan_hadm), no hadm_id but the same patient between one day before the
        admission's first transfer and its last transfer."""
        q = f"SELECT s.hadm_id AS hadm_id, {cols} FROM {table} x JOIN sel_df s ON x.hadm_id = s.hadm_id"
        if transfers:
            q += (f" UNION ALL SELECT w.hadm_id AS hadm_id, {cols} FROM {table} x JOIN win w"
                  f" ON x.hadm_id IS NULL AND x.subject_id = w.subject_id"
                  f" AND TRY_CAST(x.{time_col} AS TIMESTAMP) BETWEEN w.win_start AND w.win_end")
        return q

    lab_rows = attached(f"read_csv_auto('{hosp('labevents')}', all_varchar=true)",
                        "x.itemid, x.charttime, x.value, x.valuenum, x.valueuom, x.ref_range_lower, "
                        "x.ref_range_upper, x.flag", "charttime")
    labs = con.execute(
        f"SELECT l.*, i.label, i.fluid FROM ({lab_rows}) l "
        f"JOIN read_csv_auto('{hosp('d_labitems')}', all_varchar=true) i ON l.itemid = i.itemid "
        # a TOTAL order: ties (same item, same time) otherwise resolve differently from run to run, and
        # dedup_earliest keeps whichever tied row comes first
        "ORDER BY l.hadm_id, l.charttime, l.itemid, l.value, l.valuenum, l.valueuom, l.ref_range_lower, "
        "l.ref_range_upper, l.flag"
    ).fetchdf()
    micro = con.execute(
        attached(f"read_csv_auto('{hosp('microbiologyevents')}', all_varchar=true)",
                 "x.charttime, x.spec_type_desc, x.test_name, x.org_name, x.interpretation, x.comments", "charttime")
        + " ORDER BY hadm_id, charttime, test_name, spec_type_desc, org_name, interpretation, comments"
    ).fetchdf()
    rad_rows = attached(f"read_csv_auto('{note('radiology')}', all_varchar=true)", "x.note_id, x.charttime, x.text",
                        "charttime")
    try:  # CDM takes each report's exam name from radiology_detail (field_ordinal 1), not from the report text
        detail = note("radiology_detail")
        rad = con.execute(
            f"SELECT r.*, d.field_value AS exam_name FROM ({rad_rows}) r LEFT JOIN "
            f"(SELECT note_id, field_value FROM read_csv_auto('{detail}', all_varchar=true) "
            "WHERE field_name = 'exam_name' AND field_ordinal = '1') d ON r.note_id = d.note_id "
            "ORDER BY r.hadm_id, r.charttime, r.note_id"
        ).fetchdf()
    except FileNotFoundError:
        log.warning("note/radiology_detail not found: modality is read from report headers, which drops reports "
                    "without an EXAMINATION line")
        rad = con.execute(rad_rows + " ORDER BY hadm_id, charttime, note_id").fetchdf()

    labs_by = _group(labs)
    micro_by = _group(micro)
    rad_by = _group(rad)
    sanitize_terms = {**CDM_SANITIZE_TERMS, **icd_cfg.get("sanitize_terms", {})}
    leak_terms = icd_cfg.get("leak_terms", {})

    cases, dropped_leak, dropped_inclusion = [], 0, 0
    for r in selected:
        hadm = str(r["hadm_id"])
        text = note_by_hadm[hadm]
        label = OTHER if r["other_group"] and not r["cdm_group"] else r["cdm_group"]
        group = r["other_group"] or r["cdm_group"]
        # Same extraction + sanitisation as MIMIC-IV-Ext-CDM itself (see parity.py).
        hpi = cdm_history(text)
        if not hpi:
            continue
        terms = sanitize_terms.get(group, [])
        if mentions(hpi, terms) and hpi_leak_policy == "drop":
            dropped_leak += 1
            continue
        soft_leak = any(re.search(t, hpi, re.IGNORECASE) for t in leak_terms.get(group, []))
        imaging = []
        for x in rad_by.get(hadm, []):
            raw = _s(x["text"])
            recorded = _s(x.get("exam_name"))
            modality, region, exam_name = classify_radiology(f"EXAMINATION: {recorded}" if recorded else raw)
            body = mask(cdm_radiology(raw), terms)
            if body and modality != "Other" and region != "Other":  # CDM sanitize_rad
                imaging.append(ImagingReport(modality, region, exam_name, body, _s(x["charttime"]) or None))
        case = Case(
            case_id=hadm,
            subject_id=str(r["subject_id"]),
            label=label,
            hpi=hpi,
            history={},  # CDM keeps PMH/social/family inside the history blob
            physical_exam=mask(cdm_physical_exam(text), terms),
            labs=[
                LabResult(
                    name=_s(x["label"]), value=_s(x["value"]) or _s(x["valuenum"]), unit=_s(x["valueuom"]) or None,
                    ref_low=_f(x["ref_range_lower"]), ref_high=_f(x["ref_range_upper"]),
                    flag=_s(x["flag"]) or None, fluid=_s(x["fluid"]) or None, charttime=_s(x["charttime"]) or None,
                    itemid=_s(x["itemid"]) or None,
                )
                for x in labs_by.get(hadm, [])
            ],
            microbiology=[
                MicroResult(
                    test_name=_s(x["test_name"]), specimen=_s(x["spec_type_desc"]) or None,
                    result="; ".join(v for v in (_s(x["org_name"]), _s(x["interpretation"]), _s(x["comments"])) if v)
                    or "No growth reported",
                    charttime=_s(x["charttime"]) or None,
                )
                for x in micro_by.get(hadm, [])
            ],
            imaging=imaging,
            source="openworld" if label == OTHER else "openworld_control",
            meta={"group": group, "icd_code": r["icd_code"], "icd_version": int(r["icd_version"]),
                  "possible_label_leak": bool(soft_leak or mentions(hpi, terms))},
        )
        if require_cdm_inclusion and (len(case.physical_exam) < 40 or not case.labs
                                      or not any(im.region == "Abdomen" for im in case.imaging)):
            dropped_inclusion += 1
            continue
        cases.append(dedup_earliest(case))
    if dropped_leak:
        log.info("dropped %d admissions whose history names their own diagnosis (CDM policy)", dropped_leak)
    if dropped_inclusion:
        log.info("dropped %d admissions failing CDM's inclusion rule (PE >= 40 chars, a lab, an abdominal study)",
                 dropped_inclusion)
    return cases


def _stratified_sample(rows: list[dict], key: str, n: int, rng: random.Random) -> list[dict]:
    """Round-robin over groups so rare groups are not swamped; deterministic given rng."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    for items in by.values():
        rng.shuffle(items)
    out: list[dict] = []
    groups = sorted(by)
    while len(out) < n and any(by[g] for g in groups):
        for g in groups:
            if by[g] and len(out) < n:
                out.append(by[g].pop())
    return out


def _group(df) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in df.to_dict("records"):
        out.setdefault(str(r["hadm_id"]), []).append(r)
    return out


def _s(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
