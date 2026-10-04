import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("subgroup_eval", ROOT / "scripts" / "subgroup_eval.py")
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)


def test_bootstrap_interval_contains_mean_and_is_deterministic():
    v = np.array([1.0] * 30 + [0.0] * 10)
    m, lo, hi = sg.bootstrap(v, n_boot=500, seed=1)
    assert m == pytest.approx(0.75) and lo < m < hi
    assert sg.bootstrap(v, n_boot=500, seed=1) == (m, lo, hi)


def test_small_groups_are_hidden_with_a_complement():
    frame = pd.DataFrame({"correct": [1.0] * 20 + [0.0] * 5 + [1.0] * 30,
                          "race": ["White"] * 20 + ["Unknown"] * 5 + ["Black"] * 30})
    t = sg.group_table(frame, "race", min_cell=10, n_boot=200)
    assert t.at["Unknown", "cases"] == "<10" and t.at["Unknown", "accuracy"] == "–"
    # the smallest other size is hidden too, so the total cannot give back the hidden 5
    assert t.at["White", "cases"] == "·" and t.at["White", "accuracy"] == "100.0"
    assert t.at["Black", "cases"] == "30"


def test_per_case_averages_samples(tmp_path):
    def rollout(case, diag, sample):
        return {"case_id": case, "sample_idx": sample, "messages": [],
                "result": {"case_id": case, "label": "appendicitis", "terminal": "commit", "diagnosis": diag}}
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        rollout("1", "appendicitis", 0), rollout("1", "pancreatitis", 1), rollout("2", "appendicitis", 0)]))
    frame = sg.per_case(path)
    assert frame.at["1", "correct"] == 0.5 and frame.at["1", "samples"] == 2
    assert frame.at["2", "correct"] == 1.0
    rep = sg.report(frame.assign(sex=["Female", "Male"]), min_cell=1, n_boot=100)
    assert rep["cases"] == 2 and rep["accuracy"] == 0.75 and "sex" in rep["tables"]


def test_label_adjusted_removes_class_mix():
    # both groups are right on every easy case and wrong on every hard one; group A just has more easy cases
    frame = pd.DataFrame({
        "label": ["easy"] * 18 + ["hard"] * 2 + ["easy"] * 2 + ["hard"] * 18,
        "correct": [1.0] * 18 + [0.0] * 2 + [1.0] * 2 + [0.0] * 18,
        "sex": ["A"] * 20 + ["B"] * 20})
    t = sg.group_table(frame, "sex", min_cell=10, n_boot=100, weights=frame["label"].value_counts(normalize=True))
    assert t.at["A", "accuracy"] == "90.0" and t.at["B", "accuracy"] == "10.0"
    assert t.at["A", "label-adjusted"] == t.at["B", "label-adjusted"] == "50.0"
    # a group missing a class cannot be standardised
    one_class = frame.assign(sex=["A"] * 18 + ["B"] * 22)
    t = sg.group_table(one_class, "sex", min_cell=10, n_boot=100, weights=frame["label"].value_counts(normalize=True))
    assert t.at["A", "label-adjusted"] == "–"
