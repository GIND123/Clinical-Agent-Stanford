import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("handoff_rating", ROOT / "scripts" / "handoff_rating.py")
hr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hr)


def _row(cid, label, terminal, set_name, seed=0, diff=None, probs=None):
    return {"case_id": cid, "set": set_name, "seed": seed,
            "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": f"PRESENTATION {cid}"},
                         {"role": "assistant", "content": "<think>secret reasoning</think><action>{}</action>"},
                         {"role": "user", "content": "CBC result"}],
            "result": {"case_id": cid, "label": label, "terminal": terminal, "differential": diff or [],
                       "differential_probs": probs, "reason": "unsure"}}


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_sample_is_blind_stratified_and_scores(tmp_path, monkeypatch):
    main = _write(tmp_path / "main.jsonl", [
        _row(f"c{i}", "appendicitis", "defer", "eval_cdm_test", diff=["appendicitis", "other"],
             probs={"appendicitis": 0.6, "other": 0.3}) for i in range(6)] + [
        _row(f"o{i}", "other", "defer", "eval_other_seen", diff=["cholecystitis"]) for i in range(6)] + [
        _row("z", "appendicitis", "commit", "eval_cdm_test")])
    comp = _write(tmp_path / "comp.jsonl", [_row("c0", "appendicitis", "defer", "eval_cdm_test", diff=["pancreatitis"])])
    out = tmp_path / "rate"
    monkeypatch.setattr(sys, "argv", ["x", "sample", "--rollouts", main, "--compare", comp, "--n", "12", "--out", str(out)])
    hr.main()
    sheet = (out / "sheet.html").read_text()
    assert "secret reasoning" not in sheet and "SYS" not in sheet  # only what the agent saw, no reasoning
    assert "appendicitis (60%)" in sheet or "cholecystitis" in sheet
    key = list(csv.DictReader((out / "key.csv").open()))
    assert {k["set"] for k in key} == {"eval_cdm_test", "eval_other_seen"}  # both sets represented
    assert len({k["case"] for k in key}) == 12 and "z" not in {k["case_id"] for k in key}
    for k in key:
        assert k["label"] not in sheet.split(f"Handoff {k['rating_id']}")[0][-10:]  # no label next to the id
    # fill the ratings: main handoffs useful (4), the comparator's 2, main preferred in the paired case
    rows = list(csv.DictReader((out / "ratings.csv").open()))
    by_id = {k["rating_id"]: k for k in key}
    paired_main = next((k["slot"] for k in key if k["slot"] and k["system"] == "main"), "")
    for r in rows:
        r["usefulness_1to5"] = "4" if by_id[r["rating_id"]]["system"] == "main" else "2"
        r["correct_dx_in_differential_yn"] = "y"
        r["preferred_A_or_B"] = paired_main if by_id[r["rating_id"]]["slot"] else ""
    with (out / "ratings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hr.RATING_FIELDS)
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(sys, "argv", ["x", "score", "--out", str(out)])
    rep = hr.score(type("A", (), {"out": str(out)})())
    assert rep["by_system"]["main"]["usefulness_mean"] == 4.0 and rep["by_system"]["main"]["useful_4plus"] == 1.0
    assert rep["by_system"]["compare"]["usefulness_mean"] == 2.0  # c0 is the one paired case
    assert rep["paired"] == {"cases": 1, "main_preferred": 1.0, "compare_preferred": 0.0, "same": 0.0}
