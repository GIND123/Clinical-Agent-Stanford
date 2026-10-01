import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("diagbench_score", ROOT / "scripts" / "lambda" / "diagbench_score.py")
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def test_extract_exam_handles_every_diaggym_phrasing():
    first = ("Current diagnosis: suspected cholecystitis\nBased on the patient's initial presentation, the following "
             "investigation(s) should be performed: Abdominal ultrasound\nReason: RUQ pain")
    later = "Current diagnosis: cholecystitis\nBased on the test results, the next investigation is: HIDA scan\nReason: x"
    assert ds.extract_exam(first) == "Abdominal ultrasound"
    assert ds.extract_exam(later) == "HIDA scan"


def test_extract_exam_returns_none_when_the_model_diagnoses_instead():
    text = "The available information is sufficient to make a diagnosis.\n\nDiagnosis: Acute cholecystitis\nReason: x"
    assert ds.extract_exam(text) is None


def test_extract_diagnosis():
    text = "The available information is sufficient to make a diagnosis.\n\nDiagnosis: Acute pancreatitis\nReason: lipase"
    assert ds.extract_diagnosis(text) == "Acute pancreatitis"
    assert ds.extract_diagnosis("no structured answer") == "no structured answer"
