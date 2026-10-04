import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("diagagent_adapter", ROOT / "scripts" / "lambda" / "diagagent_adapter.py")
da = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(da)

TESTS = {"cbc", "bmp", "cmp", "renal_panel", "liver_panel", "urinalysis", "electrolyte_panel", "lipase", "amylase",
         "crp", "lactate", "coagulation", "triglycerides", "pregnancy_test", "blood_culture", "urine_culture",
         "ct_abdomen", "us_abdomen", "mri_abdomen", "xray_abdomen", "xray_chest", "hida_scan"}
TOPICS = {"physical_exam"}


def test_final_diagnosis_becomes_commit():
    a = da.translate("The available information is sufficient to make a diagnosis. \n\nDiagnosis: Acute calculous "
                     "cholecystitis\nReason: RUQ pain, Murphy sign", TESTS, TOPICS)
    assert a.type == "COMMIT" and a.diagnosis == "cholecystitis" and a.probability == 1.0


def test_out_of_set_diagnosis_is_left_invalid():
    assert da.translate("Diagnosis: Small bowel obstruction\nReason: x", TESTS, TOPICS) is None


def test_exam_recommendations_map_to_catalog_tests():
    def ask(exam):
        a = da.translate(f"Current diagnosis: abdominal pain\nBased on the patient's initial presentation, the "
                         f"following investigation(s) should be performed: {exam}\nReason: x", TESTS, TOPICS)
        return None if a is None else (a.type, a.test or a.topic)
    assert ask("CT abdomen and pelvis with contrast") == ("TEST", "ct_abdomen")
    assert ask("Right upper quadrant ultrasound") == ("TEST", "us_abdomen")
    assert ask("MRCP") == ("TEST", "mri_abdomen")
    assert ask("Chest X-ray") == ("TEST", "xray_chest")
    assert ask("Serum lipase") == ("TEST", "lipase")
    assert ask("Complete blood count (CBC)") == ("TEST", "cbc")
    assert ask("Liver function tests") == ("TEST", "liver_panel")
    assert ask("Physical examination") == ("ASK", "physical_exam")
    assert ask("Colonoscopy") is None  # nothing in the catalog: scored invalid, not guessed


def test_later_turn_phrasing_is_parsed():
    a = da.translate("Current diagnosis: pancreatitis\nBased on the test results, the next investigation is: "
                     "Abdominal CT\nReason: x", TESTS, TOPICS)
    assert (a.type, a.test) == ("TEST", "ct_abdomen")


def test_presentation_strips_environment_wrapper():
    assert da.presentation("PATIENT PRESENTATION\nRLQ pain for 2 days\n\nChoose your next action.") == "RLQ pain for 2 days"
