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


def test_open_world_maps_named_out_of_set_diagnosis_to_other():
    a = da.translate("Diagnosis: Small bowel obstruction\nReason: x", TESTS, TOPICS, open_world=True)
    assert a.type == "COMMIT" and a.diagnosis == "other" and a.probability == 1.0
    a = da.translate("Diagnosis: Acute appendicitis\nReason: x", TESTS, TOPICS, open_world=True)
    assert a.diagnosis == "appendicitis"  # in-set answers are unchanged
    assert da.translate("Diagnosis: \nReason: x", TESTS, TOPICS, open_world=True) is None  # nothing named


def test_suite_mode_writes_eval_suite_layout(tmp_path, monkeypatch):
    import json
    import sys

    from deferdx.data.io import write_jsonl
    from deferdx.data.schema import Case, ImagingReport, LabResult
    from deferdx.policy.base import Generation

    def case(cid, label, group=None):
        return Case(case_id=cid, label=label, hpi="Abdominal pain.", physical_exam="Tender abdomen on palpation.",
                    labs=[LabResult(name="White Blood Cells", value="14", fluid="Blood", itemid="51301")],
                    imaging=[ImagingReport("CT", "Abdomen", "CT ABD", "Findings.")], meta={"group": group} if group else {})

    a, b = tmp_path / "eval_cdm_test.jsonl", tmp_path / "eval_other_seen.jsonl"
    write_jsonl(a, [case("1", "appendicitis").to_dict(), case("2", "pancreatitis").to_dict()])
    write_jsonl(b, [case("3", "other", "bowel_obstruction").to_dict()])

    class Fake:  # answers "Diagnosis: Small bowel obstruction" at once; stands in for the vLLM model
        def __init__(self, model, tests, topics, open_world=False):
            self.open_world, self.params = open_world, type("P", (), {"max_tokens": 1024})()

        def generate(self, conversations, contexts=None):
            raw = "Diagnosis: Small bowel obstruction\nReason: x"
            act = da.translate(raw, set(), set(), self.open_world)
            return [Generation(text=raw + ("\n" + da.render_action(act) if act else "")) for _ in conversations]

    monkeypatch.setattr(da, "DiagAgentAdapter", Fake)
    monkeypatch.setattr(sys, "argv", ["x", "--open-world", "--suite", "da", "--out", str(tmp_path / "eval"),
                                      "--cases", str(a), str(b)])
    da.main()
    rows = [json.loads(l) for l in (tmp_path / "eval" / "da" / "s0.jsonl").read_text().splitlines()]
    assert [(r["set"], r["group"], r["result"]["diagnosis"]) for r in rows] == [
        ("eval_cdm_test", "appendicitis", "other"), ("eval_cdm_test", "pancreatitis", "other"),
        ("eval_other_seen", "bowel_obstruction", "other")]
    summary = json.loads((tmp_path / "eval" / "da" / "summary.json").read_text())
    assert summary["sets"] == ["eval_cdm_test", "eval_other_seen"] and summary["env"]["open_world"] is True
    assert summary["results"]["s0"]["eval_other_seen"]["open_world"]["commit_other_rate"] == 1.0
