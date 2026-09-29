from deferdx.env.actions import ASK, COMMIT, DEFER, INVALID, TEST, Action, parse_action, render_action

TESTS = {"lipase", "ct_abdomen", "cbc"}
TOPICS = {"physical_exam", "social_history"}


def p(text):
    return parse_action(text, TESTS, TOPICS)


def test_json_actions():
    assert p('<action>{"type": "TEST", "test": "lipase"}</action>').test == "lipase"
    assert p('<action>{"type": "ask", "topic": "physical_exam"}</action>').type == ASK
    a = p('<think>hmm</think><action>{"type":"COMMIT","diagnosis":"Acute appendicitis","probability":0.8}</action>')
    assert (a.type, a.diagnosis, a.probability) == (COMMIT, "appendicitis", 0.8)
    d = p('<action>{"type":"DEFER","differential":["diverticulitis","bogus","appendicitis"],"reason":"x"}</action>')
    assert d.type == DEFER and d.differential == ["diverticulitis", "appendicitis"]


def test_percent_probability_and_other():
    a = p('<action>{"type":"COMMIT","diagnosis":"none of the above","probability":"85%"}</action>')
    assert a.diagnosis == "other" and abs(a.probability - 0.85) < 1e-9


def test_invalid_cases():
    assert p("no action here").type == INVALID
    assert p('<action>{"type":"TEST","test":"mri_brain"}</action>').type == INVALID
    assert p('<action>{"type":"COMMIT","diagnosis":"appendicitis"}</action>').type == INVALID
    assert p('<action>{"type":"COMMIT","diagnosis":"gout","probability":0.5}</action>').type == INVALID
    assert p('<action>{"type":"COMMIT","diagnosis":"appendicitis","probability":1.7e3}</action>').type == INVALID
    assert p("<action>not json</action>").type == INVALID


def test_action_inside_think_is_ignored():
    text = ('<think>maybe <action>{"type":"TEST","test":"cbc"}</action></think>'
            '<action>{"type":"TEST","test":"lipase"}</action>')
    assert p(text).test == "lipase"


def test_function_call_fallback():
    assert p("I will order TEST(ct_abdomen)").test == "ct_abdomen"
    a = p("COMMIT(pancreatitis, 0.7)")
    assert (a.diagnosis, a.probability) == ("pancreatitis", 0.7)
    d = p("DEFER([appendicitis, diverticulitis], unclear)")
    assert d.type == DEFER and d.differential == ["appendicitis", "diverticulitis"]


def test_render_roundtrip():
    for act in (Action(TEST, test="cbc"), Action(ASK, topic="social_history"),
                Action(COMMIT, diagnosis="cholecystitis", probability=0.66),
                Action(DEFER, differential=["pancreatitis"], reason="unclear")):
        back = p(render_action(act, "reasoning"))
        assert back.to_json() == act.to_json()
