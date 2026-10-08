import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cev_tvd", ROOT / "scripts" / "cev_test_vs_defer.py")
tvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tvd)


def test_windows_weight_cev_ratios_by_branched_deferrals(tmp_path):
    rows = [{"step": 1, "cev_roots": 30, "cev_cont_acc": 0.8, "cev_A_pos": 0.2, "inset_defer_rate": 0.4},
            {"step": 2, "cev_roots": 10, "cev_cont_acc": 0.4, "cev_A_pos": 0.6, "inset_defer_rate": 0.2},
            {"step": 3, "cev_roots": 0, "cev_cont_acc": 0.0, "inset_defer_rate": 0.0},
            {"step": 4, "dev_selective_acc": 0.9}]  # an evaluation row: ignored
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    loaded = tvd.load_log(tmp_path)
    assert [r["step"] for r in loaded] == [1, 2, 3]
    w = tvd.windows(loaded, 2)
    assert w[0]["steps"] == "1–2"
    assert math.isclose(w[0]["cev_cont_acc"], (0.8 * 30 + 0.4 * 10) / 40)  # weighted by branched deferrals
    assert math.isclose(w[0]["inset_defer_rate"], 0.3)                      # plain mean
    assert math.isnan(w[1]["cev_A_pos"])                                    # no data in that window
    assert "| cev steps |" in tvd.table("cev", w)
