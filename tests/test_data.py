import json

import pandas as pd
import pytest

from deferdx.data.io import assign_splits, load_cases, save_cases
from deferdx.data.schema import Case, ImagingReport, LabResult, dedup_earliest
from deferdx.data.text import admission_physical_exam, chief_complaint, classify_radiology, split_history

DISCHARGE = """Chief Complaint:
abdominal pain

History of Present Illness:
45F with 2 days of RLQ pain.

Past Medical History:
HTN

Social History:
___

Physical Exam:
ADMISSION PHYSICAL EXAM:
Abd: tender RLQ
DISCHARGE PHYSICAL EXAM:
Abd: soft

Medications on Admission:
lisinopril
Discharge Medications:
lisinopril
Discharge Diagnosis:
acute appendicitis
"""


def test_sections_do_not_leak_discharge_diagnosis():
    hpi, sections = split_history(DISCHARGE)
    assert hpi.startswith("45F")
    assert sections["past_medical_history"] == "HTN"
    assert "appendicitis" not in sections.get("medications", "")
    assert "appendicitis" not in json.dumps(sections)
    assert admission_physical_exam(DISCHARGE) == "Abd: tender RLQ"
    assert chief_complaint(DISCHARGE) == "abdominal pain"


def test_classify_radiology():
    assert classify_radiology("EXAMINATION: CT ABD & PELVIS WITH CONTRAST\nFINDINGS")[:2] == ("CT", "Abdomen")
    assert classify_radiology("EXAMINATION: CHEST (PA AND LAT)\n")[:2] == ("Radiograph", "Chest")
    assert classify_radiology("EXAMINATION: US ABD LIMIT, SINGLE ORGAN\n")[:2] == ("Ultrasound", "Abdomen")


def test_dedup_earliest():
    c = Case("1", "appendicitis", "x", labs=[
        LabResult("Lipase", "90", charttime="2150-01-02"), LabResult("Lipase", "40", charttime="2150-01-01"),
        LabResult("Glucose", "100", fluid="Blood"), LabResult("Glucose", "neg", fluid="Urine"),
    ], imaging=[ImagingReport("CT", "Abdomen", "a", "late", "2150-02"), ImagingReport("CT", "Abdomen", "b", "early", "2150-01")])
    c = dedup_earliest(c)
    assert [x.value for x in c.labs] == ["40", "100", "neg"]
    assert [x.text for x in c.imaging] == ["early"]


def test_roundtrip_and_patient_level_split(tmp_path, synth):
    save_cases(tmp_path / "c.jsonl", synth)
    back = load_cases(tmp_path / "c.jsonl")
    assert [c.to_dict() for c in back] == [c.to_dict() for c in synth]
    twins = [Case(f"h{i}", "appendicitis", "x", subject_id=f"s{i // 3}") for i in range(300)]
    splits = assign_splits(twins, seed=3)
    for i in range(0, 300, 3):
        assert len({splits[f"h{i + k}"] for k in range(3)}) == 1
    assert set(splits.values()) == {"train", "val", "test"}


def test_cdm_csv_loader(tmp_path):
    from deferdx.config import load_yaml
    from deferdx.data.cdm_loader import load_cdm

    pd.DataFrame({"hadm_id": [1, 2, 3], "subject_id": [10, 20, 30],
                  "hpi": ["History of Present Illness: RLQ pain\nSocial History: smoker", "RUQ pain", "pain"]}
                 ).to_csv(tmp_path / "history_of_present_illness.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 2], "pe": ["tender RLQ", "Murphy +"]}).to_csv(tmp_path / "physical_examination.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 1, 2], "itemid": [51301, 51301, 50956], "valuestr": ["15", "18", "40"],
                  "ref_range_lower": [4, 4, 0], "ref_range_upper": [11, 11, 60]}).to_csv(tmp_path / "laboratory_tests.csv", index=False)
    pd.DataFrame({"itemid": [51301, 50956], "label": ["White Blood Cells", "Lipase"], "fluid": ["Blood", "Blood"],
                  "corresponding_ids": ["[51755]", ""]}).to_csv(tmp_path / "lab_test_mapping.csv", index=False)
    pd.DataFrame({"hadm_id": [1], "text": ["EXAMINATION: CT ABD\nappendicitis"], "modality": ["CT"], "region": ["Abdomen"],
                  "exam_name": ["CT ABD"]}).to_csv(tmp_path / "radiology_reports.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 2, 3], "discharge_diagnosis": ["Acute appendicitis", "cholecystitis", "appendicitis and pancreatitis"]}
                 ).to_csv(tmp_path / "discharge_diagnosis.csv", index=False)
    fmt = load_yaml("configs/cdm_format.yaml")
    cases = {c.case_id: c for c in load_cdm(tmp_path, fmt)}
    assert set(cases) == {"1", "2"}  # case 3 is ambiguous -> dropped
    c1 = cases["1"]
    assert c1.label == "appendicitis" and c1.hpi == "RLQ pain" and c1.history["social_history"] == "smoker"
    assert [(x.name, x.value) for x in c1.labs] == [("White Blood Cells", "15")]  # earliest kept
    assert c1.imaging[0].modality == "CT" and c1.subject_id == "10"
    assert cases["2"].labs[0].name == "Lipase"


def test_cdm_pickle_loader(tmp_path):
    from deferdx.config import load_yaml
    from deferdx.data.cdm_loader import load_cdm

    rec = {"Patient History": "RUQ pain after meal", "Physical Examination": "Murphy +",
           "Laboratory Tests": {50956: "40"}, "Reference Range Lower": {50956: 0}, "Reference Range Upper": {50956: 60},
           "Microbiology": {}, "Radiology": [{"Modality": "Ultrasound", "Region": "Abdomen", "Exam Name": "US",
                                              "Report": "gallstones"}]}
    pd.to_pickle({7: rec}, tmp_path / "cholecystitis_hadm_info_first_diag.pkl")
    pd.to_pickle(pd.DataFrame({"itemid": [50956], "label": ["Lipase"], "fluid": ["Blood"]}), tmp_path / "lab_test_mapping.pkl")
    fmt = load_yaml("configs/cdm_format.yaml")
    fmt["layout"] = "pickle"
    cases = load_cdm(tmp_path, fmt)
    assert len(cases) == 1 and cases[0].label == "cholecystitis"
    assert cases[0].labs[0].name == "Lipase" and cases[0].labs[0].ref_high == 60


def test_openworld_builder(tmp_path):
    pytest.importorskip("duckdb")
    from deferdx.config import load_yaml
    from deferdx.data.openworld import build_openworld

    hosp, note = tmp_path / "mimic" / "hosp", tmp_path / "notes" / "note"
    hosp.mkdir(parents=True)
    note.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": [1, 2, 3, 3, 4, 5],
        "hadm_id": [101, 102, 103, 103, 104, 105],
        "seq_num": [1, 1, 1, 2, 1, 1],
        "icd_code": ["K5660", "N200", "K5660", "K3580", "K3580", "I10"],
        "icd_version": [10, 10, 10, 10, 10, 10],
    }).to_csv(hosp / "diagnoses_icd.csv", index=False)
    notes = {
        101: "Chief Complaint:\nabdominal pain\nHistory of Present Illness:\ncrampy pain, distension\nPhysical Exam:\ndistended\nDischarge Diagnosis:\nSBO",
        102: "Chief Complaint:\nleft flank pain\nHistory of Present Illness:\ncolicky flank pain\nPhysical Exam:\nCVA tenderness",
        103: "Chief Complaint:\nabdominal pain\nHistory of Present Illness:\nRLQ pain\nPhysical Exam:\ntender",
        104: "Chief Complaint:\nabd pain\nHistory of Present Illness:\nRLQ pain\nPhysical Exam:\ntender",
        105: "Chief Complaint:\nheadache\nHistory of Present Illness:\nheadache",
    }
    pd.DataFrame({"note_id": list(range(5)), "subject_id": [1, 2, 3, 4, 5], "hadm_id": list(notes),
                  "text": list(notes.values())}).to_csv(note / "discharge.csv", index=False)
    pd.DataFrame({"note_id": [9], "subject_id": [1], "hadm_id": [101], "charttime": ["2150-01-01"],
                  "text": ["EXAMINATION: CT ABD & PELVIS\nFINDINGS: dilated small bowel"]}).to_csv(note / "radiology.csv", index=False)
    pd.DataFrame({"subject_id": [1, 1], "hadm_id": [101, 101], "itemid": [50956, 50956], "charttime": ["2150-01-02", "2150-01-01"],
                  "value": ["50", "30"], "valuenum": [50, 30], "valueuom": ["IU/L", "IU/L"], "ref_range_lower": [0, 0],
                  "ref_range_upper": [60, 60], "flag": ["", ""]}).to_csv(hosp / "labevents.csv", index=False)
    pd.DataFrame({"itemid": [50956], "label": ["Lipase"], "fluid": ["Blood"], "category": ["Chemistry"]}).to_csv(hosp / "d_labitems.csv", index=False)
    pd.DataFrame({"subject_id": [1], "hadm_id": [101], "charttime": ["2150-01-01"], "spec_type_desc": ["BLOOD CULTURE"],
                  "test_name": ["BLOOD CULTURE"], "org_name": [""], "interpretation": [""], "comments": [""]}).to_csv(hosp / "microbiologyevents.csv", index=False)

    icd = load_yaml("configs/openworld_icd.yaml")
    cases = build_openworld(tmp_path / "mimic", tmp_path / "notes", icd, target_n=10, include_controls=True)
    by = {c.case_id: c for c in cases}
    assert set(by) == {"101", "102", "104"}  # 103 has an appendicitis code; 105 not abdominal
    assert by["101"].label == "other" and by["101"].meta["group"] == "bowel_obstruction"
    assert by["102"].meta["group"] == "urolithiasis"
    assert by["104"].label == "appendicitis" and by["104"].source == "openworld_control"
    assert [x.value for x in by["101"].labs] == ["30"]
    assert by["101"].imaging[0].modality == "CT"
    assert "SBO" not in by["101"].hpi + by["101"].physical_exam
