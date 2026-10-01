"""Score DiagBench generations with DiagGym's own judge prompts and a local judge model.

The paper judges with GPT-4o through the OpenAI API. That would send the MIMIC subset to a
third party (PhysioNet DUA), so this uses the same prompt files
(instructions/accuracy.txt, instructions/hit_ratio.txt) with an open model on the same
GPU. Numbers are "reproduced with an open judge", not the paper's exact protocol.

The exam ground truth is each case's `recommended_exam_names` from the public release;
the paper's metric script instead reads raw MIMIC events from a file that is not public.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SINGLE = Path.home() / "DiagGym/DiagAgent/eval/single_turn"


def extract_diagnosis(text: str) -> str:
    m = re.search(r"Diagnosis:\s*(.*?)(?:\n\s*Reason:|$)", text, re.S)
    return (m.group(1) if m else text).strip()


def extract_exam(text: str) -> str | None:
    """The recommended exam: what follows the last colon on the line starting "Based on the".

    DiagGym phrases the first recommendation "...should be performed: X" but later turns
    differently. The 2026-10-01 run matched only the first phrasing (750 of 3,735 replies), so its
    exam hit ratio was discarded; this version finds an exam in 3,055 of those 3,735 replies (the
    rest diagnose instead) and in 3,687 of 3,735 reference turns.
    """
    for line in text.splitlines():
        if line.strip().lower().startswith("based on the") and ":" in line:
            return line.rsplit(":", 1)[1].strip() or None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Qwen/Qwen3-8B")
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    acc_tpl = (SINGLE / "instructions/accuracy.txt").read_text()
    hit_tpl = (SINGLE / "instructions/hit_ratio.txt").read_text()
    llm = LLM(model=args.judge, max_model_len=8192, gpu_memory_utilization=0.9, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=16)
    no_think = {"enable_thinking": False}

    summary = {"judge": args.judge, "protocol": "DiagGym judge prompts, local open judge", "subsets": {}}
    for path in sorted(Path(args.gen_dir).glob("diagbench_*.json")):
        rows = json.loads(path.read_text())
        subset = path.stem.split("_", 1)[1]
        prompts, keys = [], []
        for r in rows:
            if r["task"].startswith("final"):
                prompts.append(acc_tpl.format(pred_diag=extract_diagnosis(r["generated"]), gt_diag=r["ground_truth"]))
                keys.append((r["task"], "acc"))
            else:
                exam = extract_exam(r["generated"])
                if exam is None:  # format failure or the model chose to diagnose instead
                    keys.append((r["task"], "no_exam"))
                    prompts.append(None)
                    continue
                prompts.append(hit_tpl.format(pred_exam=exam, gt_exam=", ".join(r["recommended_exam_names"])))
                keys.append((r["task"], "hit"))
        todo = [[{"role": "user", "content": p}] for p in prompts if p is not None]
        outs = iter(llm.chat(todo, params, chat_template_kwargs=no_think, use_tqdm=False))
        verdicts = []
        for p, (task, kind) in zip(prompts, keys):
            if p is None:
                verdicts.append((task, kind, False))
                continue
            text = next(outs).outputs[0].text.strip().lower()
            ok = ("correct" in text and "wrong" not in text) if kind == "acc" else ("same" in text and "different" not in text)
            verdicts.append((task, kind, ok))
        res = {}
        for task in ("final_published", "final_fixed", "exam"):
            v = [ok for t, _, ok in verdicts if t == task]
            res[task] = {"n": len(v), "score": round(sum(v) / len(v), 4) if v else None}
        res["exam"]["no_exam_extracted"] = sum(1 for t, k, _ in verdicts if t == "exam" and k == "no_exam")
        summary["subsets"][subset] = res
        print(subset, json.dumps(res), flush=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
