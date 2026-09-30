import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audit_data", ROOT / "scripts" / "audit_data.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

SECRET = "ZQXSECRETNOTE"  # stands in for patient text; must never reach the report
LABELS = ("appendicitis", "cholecystitis", "diverticulitis", "pancreatitis")


def test_cell_and_rate_hide_small_counts():
    assert audit.cell(0) == 0 and audit.cell(3) == "<10" and audit.cell(10) == 10
    assert audit.rate(50, 100) == "50.0%"
    assert audit.rate(3, 100) == "<10 cases"
    assert audit.rate(95, 100) == "all but <10"
    assert audit.rate(2, 5) == "–"


def test_suppress_leaves_no_row_or_column_with_one_hidden_cell():
    ct = pd.DataFrame({"a": [5, 20, 25], "b": [30, 50, 70], "c": [40, 60, 80]}, index=["x", "y", "z"])
    out = audit.suppress(ct)
    hidden = out.isin(["<10", "·"])
    assert out.at["x", "a"] == "<10"
    assert (hidden.sum(axis=1) != 1).all() and (hidden.sum(axis=0) != 1).all()
    assert out.at["z", "c"] == "80"


def _write_cdm(root: Path, n: int = 30) -> list[int]:
    cdm = root / "mimic-iv-ext-cdm" / "1.1"
    cdm.mkdir(parents=True)
    ids = {lab: [987650000 + i * 1000 + k for k in range(n)] for i, lab in enumerate(LABELS)}
    all_ids = [h for v in ids.values() for h in v]
    lab_of = {h: lab for lab, v in ids.items() for h in v}
    (cdm / "pathology_ids.json").write_text(json.dumps(ids))
    tables = {
        "history_of_present_illness": [{"hadm_id": h, "hpi": f"{SECRET} pain x2 days"} for h in all_ids],
        "physical_examination": [{"hadm_id": h, "pe": f"{SECRET} tender" + (" ____" if k % 2 else "")}
                                 for k, h in enumerate(all_ids)],
        "laboratory_tests": [{"hadm_id": h, "itemid": 51300 + j, "valuestr": SECRET if j else "7.1",
                              "ref_range_lower": 1, "ref_range_upper": 9} for h in all_ids for j in range(2)],
        "microbiology": [{"hadm_id": h, "test_itemid": 90201, "valuestr": SECRET, "spec_itemid": 70012}
                         for h in all_ids[::3]],
        "radiology_reports": [{"hadm_id": h, "note_id": f"{h}-RR-{j}", "modality": ("CT", "Ultrasound")[j],
                               "region": "Abdomen", "exam_name": "CT ABD", "text": f"{SECRET} no evidence of appendicitis"}
                              for h in all_ids for j in range(2)],
        "discharge_diagnosis": [{"hadm_id": h, "discharge_diagnosis": f"{lab_of[h]} {SECRET}"} for h in all_ids],
        "icd_diagnosis": [{"hadm_id": h, "icd_diagnosis": f"Acute {lab_of[h]}"} for h in all_ids],
        "discharge_procedures": [{"hadm_id": h, "discharge_procedure": f"{SECRET} appendectomy"} for h in ids["appendicitis"]],
        "icd_procedures": [{"hadm_id": h, "icd_code": "0DTJ4ZZ", "icd_title": "Resection of appendix", "icd_version": 10}
                           for h in ids["appendicitis"]],
        "lab_test_mapping": [{"itemid": 51300 + j, "label": f"Test {j}", "fluid": "Blood", "category": "Chemistry",
                              "count": 100, "corresponding_ids": "[]"} for j in range(2)],
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(cdm / f"{name}.csv", index=False)
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()} {p.name}" for p in sorted(cdm.iterdir())]
    (cdm / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    return all_ids


def test_cdm_audit_emits_no_text_or_identifiers(tmp_path):
    ids = _write_cdm(tmp_path)
    rep = audit.run_audit(tmp_path, skip_mimic=True)
    md = audit.render(rep)
    js = json.dumps(rep.data, default=audit._json)
    for out in (md, js):
        assert SECRET not in out
        assert not any(str(h) in out for h in ids)
    assert "## Findings" in md and "Class imbalance" not in md  # synthetic classes are balanced
    assert all(r["sha256"] == "ok" for r in rep.data["inventory"])
