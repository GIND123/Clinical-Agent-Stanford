import pytest

from deferdx.data.schema import Case, ImagingReport, LabResult
from deferdx.env import DiagnosticEnv, EnvConfig
from deferdx.env.actions import ASK, COMMIT, DEFER, TEST, Action


def make_case(label="appendicitis"):
    return Case(
        case_id="c1", label=label, hpi="RLQ pain.", physical_exam="McBurney tenderness.",
        history={"social_history": "Non-smoker."},
        labs=[LabResult("Lipase", "30", "IU/L", 0, 60), LabResult("White Blood Cells", "15", "K/uL", 4, 11)],
        imaging=[ImagingReport("CT", "Abdomen", "CT A/P", "Dilated appendix.")],
    )


def test_reveal_and_costs(catalog, env_cfg):
    env = DiagnosticEnv(catalog, env_cfg)
    obs = env.reset(make_case())
    assert "RLQ pain" in obs
    out = env.step(Action(TEST, test="ct_abdomen"))
    assert "Dilated appendix" in out.observation and not out.done
    out = env.step(Action(TEST, test="lipase"))
    assert "Lipase: 30 IU/L (ref 0-60)" in out.observation
    out = env.step(Action(TEST, test="us_abdomen"))  # never performed
    assert "Not performed" in out.observation
    assert env.result.total_cost == catalog.tests["ct_abdomen"].cost + catalog.tests["lipase"].cost + catalog.tests["us_abdomen"].cost
    out = env.step(Action(COMMIT, diagnosis="appendicitis", probability=0.9))
    assert out.done and env.result.correct and env.result.n_tests == 3


def test_repeat_is_free(catalog, env_cfg):
    env = DiagnosticEnv(catalog, env_cfg)
    env.reset(make_case())
    env.step(Action(TEST, test="cbc"))
    cost = env.result.total_cost
    out = env.step(Action(TEST, test="cbc"))
    assert "already provided" in out.observation and env.result.total_cost == cost


def test_budget_then_timeout(catalog):
    env = DiagnosticEnv(catalog, EnvConfig(max_steps=1))
    env.reset(make_case())
    out = env.step(Action(ASK, topic="physical_exam"))
    assert "must be COMMIT or DEFER" in out.observation
    out = env.step(Action(TEST, test="cbc"))
    assert out.done and env.result.terminal == "timeout"


def test_invalid_limit(catalog, env_cfg):
    env = DiagnosticEnv(catalog, env_cfg)  # max_invalid=1
    env.reset(make_case())
    assert not env.step_text("garbage").done
    out = env.step_text("more garbage")
    assert out.done and env.result.terminal == "invalid"


def test_forced_choice_modes(catalog):
    env = DiagnosticEnv(catalog, EnvConfig(allow_defer=False, open_world=False, max_invalid=5))
    env.reset(make_case())
    assert "DEFER" not in env.system_prompt.split("Actions:")[1].split("Every question")[0]
    assert not env.step(Action(DEFER, differential=["appendicitis"])).done
    assert not env.step(Action(COMMIT, diagnosis="other", probability=0.5)).done
    assert env.result.n_invalid == 2


def test_defer_result(catalog, env_cfg):
    env = DiagnosticEnv(catalog, env_cfg)
    env.reset(make_case("diverticulitis"))
    out = env.step(Action(DEFER, differential=["diverticulitis", "appendicitis"], reason="unclear"))
    assert out.done and env.result.terminal == "defer"
    assert env.result.forced_prediction == "diverticulitis" and not env.result.correct


def test_itemid_matching_beats_names(catalog):
    case = Case("c2", "pancreatitis", "epigastric pain", labs=[
        LabResult("Glucose", "neg", fluid="Urine", itemid="51084"),
        LabResult("Glucose", "140", fluid="Blood", itemid="50931"),
        LabResult("Leukocyte Count", "13", itemid="51301"),  # synonym label, known itemid
        LabResult("Lipase", "900"),                          # no itemid -> name fallback
    ])
    bmp = catalog.resolve_test("bmp", case)
    assert "140" in bmp and "neg" not in bmp
    assert "neg" in catalog.resolve_test("urinalysis", case)
    assert "Leukocyte Count: 13" in catalog.resolve_test("cbc", case)
    assert "900" in catalog.resolve_test("lipase", case)
    assert catalog.unreached_lab_names([case]) == {}


def test_auto_cost_scale(catalog):
    from deferdx.rewards import RewardConfig

    rc = RewardConfig.from_dict({"cost_scale": "auto", "alpha": 1.0}, total_test_cost=catalog.total_test_cost)
    assert abs(rc.cost_scale * catalog.total_test_cost - 1.0) < 1e-9
    with pytest.raises(ValueError):
        RewardConfig.from_dict({"cost_scale": "auto"})
