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


def _write_mock_cdm(d, with_pathology_ids: bool):
    """Mock of the PhysioNet v1.1 CSVs, columns as read by Hager et al.'s ConvertPhysionet.py."""
    pd.DataFrame({"hadm_id": [1, 2, 3], "subject_id": [10, 20, 30],
                  "hpi": ["History of Present Illness: RLQ pain\nSocial History: smoker", "RUQ pain", "pain"]}
                 ).to_csv(d / "history_of_present_illness.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 2], "pe": ["tender RLQ", "Murphy +"]}).to_csv(d / "physical_examination.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 1, 2], "itemid": [51755, 51301, 50956], "valuestr": ["18", "15", "40"],
                  "ref_range_lower": [4, 4, 0], "ref_range_upper": [11, 11, 60],
                  "charttime": ["2150-01-02", "2150-01-01", "2150-01-01"]}).to_csv(d / "laboratory_tests.csv", index=False)
    pd.DataFrame({"itemid": [51301, 50956, 90201, 70012], "label": ["White Blood Cells", "Lipase", "Blood Culture, Routine", "BLOOD CULTURE"],
                  "fluid": ["Blood", "Blood", "Microbiology", "Microbiology"], "category": ["Hematology", "Chemistry", "", ""],
                  "corresponding_ids": ["[51755]", "[]", "[]", "[]"]}).to_csv(d / "lab_test_mapping.csv", index=False)
    pd.DataFrame({"hadm_id": [2], "test_itemid": [90201], "valuestr": ["NO GROWTH"], "spec_itemid": [70012]}
                 ).to_csv(d / "microbiology.csv", index=False)
    pd.DataFrame({"hadm_id": [1], "note_id": ["n1"], "text": ["FINDINGS: dilated appendix"], "modality": ["CT"],
                  "region": ["Abdomen"], "exam_name": ["CT ABD"]}).to_csv(d / "radiology_reports.csv", index=False)
    pd.DataFrame({"hadm_id": [1, 2, 3], "discharge_diagnosis": ["Acute appendicitis", "cholecystitis", "appendicitis and pancreatitis"]}
                 ).to_csv(d / "discharge_diagnosis.csv", index=False)
    pd.DataFrame({"hadm_id": [3, 3], "icd_diagnosis": ["Acute pancreatitis", "Hypertension"]}).to_csv(d / "icd_diagnosis.csv", index=False)
    if with_pathology_ids:
        (d / "pathology_ids.json").write_text(json.dumps({"appendicitis": [1], "cholecystitis": [2], "pancreatitis": [3],
                                                          "diverticulitis": []}))


@pytest.mark.parametrize("v11", [True, False])
def test_cdm_csv_loader(tmp_path, v11):
    from deferdx.config import load_yaml
    from deferdx.data.cdm_loader import load_cdm

    _write_mock_cdm(tmp_path, with_pathology_ids=v11)
    cases = {c.case_id: c for c in load_cdm(tmp_path, load_yaml("configs/cdm_format.yaml"))}
    # v1.1: pathology_ids labels case 3; v1.0: its discharge dx is ambiguous, ICD titles resolve it
    assert set(cases) == {"1", "2", "3"}
    assert cases["3"].label == "pancreatitis"
    c1 = cases["1"]
    assert c1.label == "appendicitis" and c1.hpi == "RLQ pain" and c1.history["social_history"] == "smoker"
    # 51755 folds into White Blood Cells via corresponding_ids; earliest charttime kept
    assert [(x.name, x.value, x.itemid) for x in c1.labs] == [("White Blood Cells", "15", "51301")]
    assert c1.imaging[0].modality == "CT" and c1.subject_id == "10"
    m = cases["2"].microbiology[0]
    assert (m.test_name, m.specimen, m.result) == ("Blood Culture, Routine", "BLOOD CULTURE", "NO GROWTH")


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


def _write_openworld_fixture(tmp_path):
    hosp, note = tmp_path / "mimic" / "hosp", tmp_path / "notes" / "note"
    hosp.mkdir(parents=True)
    note.mkdir(parents=True)
    pd.DataFrame({
        "subject_id": [1, 2, 3, 3, 4, 5, 6],
        "hadm_id": [101, 102, 103, 103, 104, 105, 106],
        "seq_num": [1, 1, 1, 2, 1, 1, 1],
        "icd_code": ["K5660", "N200", "K5660", "K3580", "K3580", "I10", "K5660"],
        "icd_version": [10, 10, 10, 10, 10, 10, 10],
    }).to_csv(hosp / "diagnoses_icd.csv", index=False)
    tail = "\nPertinent Results:\nlabs\nDischarge Diagnosis:\nSBO"
    notes = {
        101: "Chief Complaint:\nabdominal pain\nMajor Surgical or Invasive Procedure:\nnone\n"
             "History of Present Illness:\ncrampy pain,\ndistension\nPast Medical History:\nHTN\n"
             "Physical Exam:\ndistended, concern for SBO\nDischarge exam: soft" + tail,
        102: "Chief Complaint:\nleft flank pain\nHistory of Present Illness:\ncolicky flank pain\nPhysical Exam:\nCVA tenderness" + tail,
        103: "Chief Complaint:\nabdominal pain\nHistory of Present Illness:\nRLQ pain\nPhysical Exam:\ntender" + tail,
        104: "Chief Complaint:\nabd pain\nHistory of Present Illness:\nRLQ pain\nPhysical Exam:\ntender" + tail,
        105: "Chief Complaint:\nheadache\nHistory of Present Illness:\nheadache\nPhysical Exam:\nok" + tail,
        106: "Chief Complaint:\nabdominal pain\nHistory of Present Illness:\nknown small bowel obstruction\nPhysical Exam:\nok" + tail,
    }
    pd.DataFrame({"note_id": list(range(6)), "subject_id": [1, 2, 3, 4, 5, 6], "hadm_id": list(notes),
                  "text": list(notes.values())}).to_csv(note / "discharge.csv", index=False)
    pd.DataFrame({"note_id": [9], "subject_id": [1], "hadm_id": [101], "charttime": ["2150-01-01"],
                  "text": ["EXAMINATION: CT ABD & PELVIS\nINDICATION: eval for SBO\nFINDINGS: dilated loops, SBO pattern\n"
                           "IMPRESSION: small bowel obstruction"]}).to_csv(note / "radiology.csv", index=False)
    pd.DataFrame({"subject_id": [1, 1], "hadm_id": [101, 101], "itemid": [50956, 50956], "charttime": ["2150-01-02", "2150-01-01"],
                  "value": ["50", "30"], "valuenum": [50, 30], "valueuom": ["IU/L", "IU/L"], "ref_range_lower": [0, 0],
                  "ref_range_upper": [60, 60], "flag": ["", ""]}).to_csv(hosp / "labevents.csv", index=False)
    pd.DataFrame({"itemid": [50956], "label": ["Lipase"], "fluid": ["Blood"], "category": ["Chemistry"]}).to_csv(hosp / "d_labitems.csv", index=False)
    pd.DataFrame({"subject_id": [1], "hadm_id": [101], "charttime": ["2150-01-01"], "spec_type_desc": ["BLOOD CULTURE"],
                  "test_name": ["BLOOD CULTURE"], "org_name": [""], "interpretation": [""], "comments": [""]}).to_csv(hosp / "microbiologyevents.csv", index=False)
    return hosp, note


def test_openworld_builder(tmp_path):
    pytest.importorskip("duckdb")
    from deferdx.config import load_yaml
    from deferdx.data.openworld import build_openworld

    _write_openworld_fixture(tmp_path)
    icd = load_yaml("configs/openworld_icd.yaml")
    # the fixture's exams are shorter than CDM's 40-character minimum; inclusion is tested separately
    cases = build_openworld(tmp_path / "mimic", tmp_path / "notes", icd, target_n=10, include_controls=True,
                            require_cdm_inclusion=False)
    by = {c.case_id: c for c in cases}
    # 103: carries an appendicitis code; 105: not abdominal; 106: history names its own dx (CDM drop rule)
    assert set(by) == {"101", "102", "104"}
    c = by["101"]
    assert c.label == "other" and c.meta["group"] == "bowel_obstruction"
    assert by["102"].meta["group"] == "urolithiasis"
    assert by["104"].label == "appendicitis" and by["104"].source == "openworld_control"
    # CDM-style history blob: HPI through PMH, newlines flattened, PE excluded
    assert c.hpi == "crampy pain, distension Past Medical History: HTN" and c.history == {}
    assert c.physical_exam == "distended, concern for ____"  # masked; discharge exam cut
    rad = c.imaging[0]
    assert rad.modality == "CT" and "IMPRESSION" not in rad.text and "INDICATION" not in rad.text
    assert "____ pattern" in rad.text and "SBO" not in rad.text
    assert [(x.value, x.itemid) for x in c.labs] == [("30", "50956")]


def test_openworld_attaches_rows_without_hadm_id_like_cdm(tmp_path):
    """With hosp/transfers present, rows lacking hadm_id are attached when they are the same patient between
    one day before the first transfer and the last transfer (MIMIC-IV-Ext-CDM's fill_nan_hadm)."""
    pytest.importorskip("duckdb")
    from deferdx.config import load_yaml
    from deferdx.data.openworld import build_openworld

    hosp, note = _write_openworld_fixture(tmp_path)
    pd.DataFrame({"subject_id": [1, 1], "hadm_id": [101, 101], "transfer_id": [1, 2], "eventtype": ["ED", "admit"],
                  "careunit": ["ED", "Surgery"], "intime": ["2150-01-01 08:00:00", "2150-01-03 08:00:00"],
                  "outtime": ["2150-01-01 12:00:00", "2150-01-05 08:00:00"]}).to_csv(hosp / "transfers.csv", index=False)
    pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 2],
        "hadm_id": pd.array([101, 101, None, None, None], dtype="Int64"),  # Int64 writes "101" and "", like MIMIC
        "itemid": [50956, 50956, 50861, 50878, 50862],
        "charttime": ["2150-01-02", "2150-01-01", "2149-12-31 20:00:00",  # ED-era, inside the window -> attached
                      "2149-12-29 08:00:00",                              # before the window -> not attached
                      "2150-01-01 09:00:00"],                             # another patient -> not attached
        "value": ["50", "30", "40", "35", "4.0"], "valuenum": [50, 30, 40, 35, 4.0],
        "valueuom": ["IU/L"] * 5, "ref_range_lower": [0] * 5, "ref_range_upper": [60] * 5, "flag": [""] * 5,
    }).to_csv(hosp / "labevents.csv", index=False)
    pd.DataFrame({"itemid": [50956, 50861, 50878, 50862], "label": ["Lipase", "ALT", "AST", "Albumin"],
                  "fluid": ["Blood"] * 4, "category": ["Chemistry"] * 4}).to_csv(hosp / "d_labitems.csv", index=False)
    pd.DataFrame({"note_id": [9, 10], "subject_id": [1, 1], "hadm_id": pd.array([101, None], dtype="Int64"),
                  "charttime": ["2150-01-01", "2150-01-01 07:00:00"],
                  "text": ["EXAMINATION: CT ABD & PELVIS\nFINDINGS: dilated loops\nIMPRESSION: obstruction",
                           # no EXAMINATION header: modality must come from radiology_detail, as in CDM
                           "INDICATION: cough\nFINDINGS: clear lungs\nIMPRESSION: normal"]}
                 ).to_csv(note / "radiology.csv", index=False)
    pd.DataFrame({"note_id": [10, 10], "subject_id": [1, 1], "field_name": ["exam_name", "exam_name"],
                  "field_value": ["CHEST (PA & LAT)", "CT HEAD"], "field_ordinal": [1, 2]}
                 ).to_csv(note / "radiology_detail.csv", index=False)

    icd = load_yaml("configs/openworld_icd.yaml")
    first = build_openworld(tmp_path / "mimic", tmp_path / "notes", icd, target_n=10, include_controls=True,
                            require_cdm_inclusion=False)
    c = {x.case_id: x for x in first}["101"]
    assert sorted(x.name for x in c.labs) == ["ALT", "Lipase"]  # AST before the window and Albumin (other patient) excluded
    assert {(x.modality, x.region) for x in c.imaging} == {("CT", "Abdomen"), ("Radiograph", "Chest")}
    again = build_openworld(tmp_path / "mimic", tmp_path / "notes", icd, target_n=10, include_controls=True,
                            require_cdm_inclusion=False)
    assert [x.case_id for x in again] == [x.case_id for x in first]  # same seed, same cases, same order


def test_openworld_applies_cdm_inclusion_rule(tmp_path):
    """CDM keeps an admission only with a physical exam of >= 40 characters, a lab and an abdominal study."""
    pytest.importorskip("duckdb")
    from deferdx.config import load_yaml
    from deferdx.data.openworld import build_openworld

    hosp, note = _write_openworld_fixture(tmp_path)
    notes = pd.read_csv(note / "discharge.csv")
    long_pe = "abdomen distended and tympanic, diffusely tender without rebound, hypoactive bowel sounds"
    notes.loc[notes.hadm_id == 101, "text"] = notes.loc[notes.hadm_id == 101, "text"].str.replace(
        "distended, concern for SBO", long_pe, regex=False)
    notes.to_csv(note / "discharge.csv", index=False)
    icd = load_yaml("configs/openworld_icd.yaml")
    cases = build_openworld(tmp_path / "mimic", tmp_path / "notes", icd, target_n=10, include_controls=True)
    # 101: long exam, a lab, a CT abdomen -> kept; 102 and 104: no labs or imaging -> dropped
    assert [c.case_id for c in cases] == ["101"]
