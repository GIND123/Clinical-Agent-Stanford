"""Run DiagAgent in its own response format inside the unchanged DEFER-Dx environment.

DiagAgent-14B was trained on DiagGym's format ("Current diagnosis / ... should be performed: X"
or "Diagnosis: Y"). Prompted with this environment's action format it produced no valid action in
2,400 episodes (docs/BASELINES.md). This adapter is policy-side only:

  * it shows DiagAgent DiagGym's own system prompt, the case presentation, and each environment
    observation as "Here is the test result: ...", keeping DiagAgent's raw replies as its history;
  * it translates each reply into one environment action: "Diagnosis: Y" -> COMMIT (label via
    `normalize_label`), a recommended exam -> TEST/ASK via keyword rules over the catalog;
  * a diagnosis outside the four classes, or an exam no catalog test matches, becomes no action, so
    the environment scores it as invalid, as it would any unusable reply.

DiagAgent states no probability, so commits carry probability 1.0; its calibration metrics are
not meaningful. The environment, catalog, prompts for other models and scoring are unchanged.

    python scripts/lambda/diagagent_adapter.py --cases data/cdm/all.jsonl --out outputs/da_adapter.jsonl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deferdx.data import load_cases  # noqa: E402
from deferdx.data.io import write_jsonl  # noqa: E402
from deferdx.env import EnvConfig, TestCatalog  # noqa: E402
from deferdx.env.actions import ASK, COMMIT, TEST, Action, render_action  # noqa: E402
from deferdx.labels import IN_SET_LABELS, normalize_label  # noqa: E402
from deferdx.policy.base import Generation  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# DiagGym/DiagAgent/eval/single_turn/instructions/diagnose.txt (github.com/MAGIC-AI4Med/DiagGym), verbatim.
DIAGNOSE_PROMPT = """You are a medical AI assistant. Help the doctor with diagnosis by analyzing patient information, suggesting relevant tests, and providing a final diagnosis when sufficient information is available.

RESPONSE FORMAT:

If more information is needed:
```
Current diagnosis: [your diagnosis according to the information provided]
Based on the patient's initial presentation, the following investigation(s) should be performed: [one additional test]
Reason: [reason for the test]
```

If sufficient information exists for diagnosis:
```
The available information is sufficient to make a diagnosis.

Diagnosis: [Diagnosis result]
Reason: [Diagnosis reason]
```"""

# Ordered: the first matching rule wins (specific before generic).
EXAM_RULES: list[tuple[str, str]] = [
    (r"physical exam|physical examination|\bexam(ination)? of the abdomen", "ASK:physical_exam"),
    (r"\bhida\b|hepatobiliary|cholescint", "hida_scan"),
    (r"\bmrcp\b|\bmri\b|magnetic resonance", "mri_abdomen"),
    (r"\bct\b|computed tomograph|cat scan", "ct_abdomen"),
    (r"ultrasound|sonograph|\bus\b|\bruq us\b|duplex", "us_abdomen"),
    (r"(x-?ray|radiograph|\bcxr\b|film).*chest|chest.*(x-?ray|radiograph|film)|\bcxr\b", "xray_chest"),
    (r"x-?ray|radiograph|\bkub\b|abdominal film", "xray_abdomen"),
    (r"lipase", "lipase"),
    (r"amylase", "amylase"),
    (r"c-?reactive|\bcrp\b", "crp"),
    (r"lactate|lactic", "lactate"),
    (r"triglyceride", "triglycerides"),
    (r"\bhcg\b|pregnan", "pregnancy_test"),
    (r"blood culture", "blood_culture"),
    (r"urine culture", "urine_culture"),
    (r"urinalysis|urine analysis|\bua\b", "urinalysis"),
    (r"coagul|\binr\b|prothrombin|\bptt\b|\bpt\b", "coagulation"),
    (r"comprehensive metabolic|\bcmp\b", "cmp"),
    (r"liver|\blfts?\b|hepatic|bilirubin|\balt\b|\bast\b|alkaline phosphatase", "liver_panel"),
    (r"basic metabolic|\bbmp\b|chem-?7", "bmp"),
    (r"renal|kidney function|creatinine|\bbun\b", "renal_panel"),
    (r"electrolyte", "electrolyte_panel"),
    (r"complete blood|\bcbc\b|blood count|white (blood )?cell|\bwbc\b|hemoglobin", "cbc"),
]


def extract_exam(text: str) -> str | None:
    """Text after the last colon on the line starting "Based on the" (DiagGym's recommendation line)."""
    for line in text.splitlines():
        if line.strip().lower().startswith("based on the") and ":" in line:
            return line.rsplit(":", 1)[1].strip() or None
    return None


def extract_final_diagnosis(text: str) -> str | None:
    """The final "Diagnosis:" (not "Current diagnosis:")."""
    m = re.search(r"(?<!Current )\bDiagnosis:\s*(.*?)(?:\n\s*Reason:|$)", text, re.S)
    return m.group(1).strip() if m else None


def map_exam(exam: str, valid_tests: set[str], valid_topics: set[str]) -> Action | None:
    e = exam.lower()
    for pattern, target in EXAM_RULES:
        if re.search(pattern, e):
            if target.startswith("ASK:"):
                topic = target.split(":", 1)[1]
                return Action(ASK, topic=topic) if topic in valid_topics else None
            return Action(TEST, test=target) if target in valid_tests else None
    return None


def translate(raw: str, valid_tests: set[str], valid_topics: set[str]) -> Action | None:
    """One DiagAgent reply -> one environment action, or None (scored invalid)."""
    dx = extract_final_diagnosis(raw)
    if dx is not None:
        label = normalize_label(dx)
        return Action(COMMIT, diagnosis=label, probability=1.0) if label in IN_SET_LABELS else None
    exam = extract_exam(raw)
    return map_exam(exam, valid_tests, valid_topics) if exam else None


def presentation(first_user_message: str) -> str:
    text = first_user_message.replace("PATIENT PRESENTATION\n", "", 1)
    return text.replace("\n\nChoose your next action.", "").strip()


class DiagAgentAdapter:
    def __init__(self, model: str, valid_tests: set[str], valid_topics: set[str], max_new_tokens: int = 1024,
                 seed: int = 0):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, seed=seed, max_model_len=16384, gpu_memory_utilization=0.92)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)  # DiagAgent's own decoding
        self.valid_tests, self.valid_topics = valid_tests, valid_topics
        self.history: dict[int, list[dict[str, str]]] = {}

    def generate(self, conversations, contexts):
        keys, prompts = [], []
        for conv, env in zip(conversations, contexts):
            k = id(env)
            if len(conv) == 2 or k not in self.history:  # new episode: system + presentation
                self.history[k] = [{"role": "system", "content": DIAGNOSE_PROMPT},
                                   {"role": "user", "content": presentation(conv[1]["content"])}]
            else:
                self.history[k].append({"role": "user", "content": "Here is the test result: " + conv[-1]["content"]})
            prompts.append(self.tok.apply_chat_template(self.history[k], tokenize=False, add_generation_prompt=True))
            keys.append(k)
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        gens = []
        for k, o in zip(keys, outs):
            raw = o.outputs[0].text.strip().replace("```", "")
            self.history[k].append({"role": "assistant", "content": raw})
            action = translate(raw, self.valid_tests, self.valid_topics)
            gens.append(Generation(text=raw + ("\n" + render_action(action) if action else "")))
        return gens


def main() -> None:
    from deferdx.config import load_config
    from deferdx.rollout import run_episodes

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--model", default="Henrychur/DiagAgent-14B")
    ap.add_argument("--config", default=str(REPO / "configs/base.yaml"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = dict(cfg.get("env", {}), allow_defer=False, open_world=False)  # same as --no-defer --closed-world
    catalog, env_cfg = TestCatalog.from_yaml(cfg["test_catalog"]), EnvConfig.from_dict(env)
    cases = [c for p in args.cases for c in load_cases(p)][: args.limit]
    topics = set(env_cfg.ask_topics or catalog.asks)
    policy = DiagAgentAdapter(args.model, set(catalog.tests), topics)
    rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=1,
                            on_turn=lambda t, n: print(f"  turn {t}: {n} active", file=sys.stderr))
    n = write_jsonl(args.out, (ro.to_dict() for ro in rollouts))
    print(f"{n} rollouts -> {args.out}")


if __name__ == "__main__":
    main()
