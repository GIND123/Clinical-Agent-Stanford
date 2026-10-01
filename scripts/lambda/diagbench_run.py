"""DiagBench single-turn inference for DiagAgent, reusing DiagGym's own conversation builder.

Conversations come from DiagGym/DiagAgent/eval/single_turn: `load_dataset` builds them and
`TransformersDiagnoser._format_messages` renders them, so prompts match the published
code exactly. Generation runs on vLLM instead of HF generate (speed); sampling matches
theirs: temperature 0.0, max 2048 tokens.

Three tasks per subset:
  final_published  - final diagnosis exactly as inference_final_diagnose.py builds it. Its
                     load_dataset leaves the final assistant turn, which contains the
                     ground-truth diagnosis, in the prompt.
  final_fixed      - the same with that final assistant turn removed (messages[:-1]).
  exam             - next-examination recommendation at every non-final assistant turn,
                     exactly as inference_process_exam.py slices it (messages[:msg_idx]).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SINGLE = Path.home() / "DiagGym/DiagAgent/eval/single_turn"
sys.path.insert(0, str(SINGLE))

import inference_final_diagnose as fd  # noqa: E402
import inference_process_exam as pe  # noqa: E402

SUBSETS = ("MIMICIV", "PMCOA", "MTSamples", "DDXPlus")


class _Holder:
    """Stands in for a TransformersDiagnoser so we can call its _format_messages unchanged."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True, help="DiagBench_<subset>.json converted to lists")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.92, seed=0)
    holder = _Holder(llm.get_tokenizer())
    params = SamplingParams(temperature=0.0, max_tokens=2048)
    instruction = str(SINGLE / "instructions/diagnose.txt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subset in SUBSETS:
        data = fd.load_dataset(str(Path(args.data_dir) / f"DiagBench_{subset}.json"), instruction)
        if args.limit:
            data = data[: args.limit]
        if not data:  # load_dataset keeps only cases with recommended_exam_names (MIMIC-IV subset)
            print(f"{subset}: no single-turn cases, skipped", flush=True)
            continue
        jobs = []  # (task, record, prompt)
        for i, item in enumerate(data):
            msgs = item["messages"]
            assert msgs[-1]["role"] == "assistant", "expected the final diagnosis as the last turn"
            base = {"subset": subset, "item_idx": i, "note_id": item.get("note_id", f"item_{i}"),
                    "ground_truth": item.get("final_diagnosis", ""),
                    "recommended_exam_names": item.get("recommended_exam_names", [])}
            jobs.append(("final_published", base,
                         fd.TransformersDiagnoser._format_messages(holder, copy.deepcopy(msgs))))
            jobs.append(("final_fixed", base,
                         fd.TransformersDiagnoser._format_messages(holder, copy.deepcopy(msgs[:-1]))))
            for j, m in enumerate(msgs):
                if m["role"] == "assistant" and j != len(msgs) - 1:
                    jobs.append(("exam", {**base, "msg_idx": j, "reference_turn": m["content"]},
                                 pe.TransformersDiagnoser._format_messages(holder, copy.deepcopy(msgs[:j]))))
        outs = llm.generate([p for _, _, p in jobs], params)
        rows = [{"task": t, **rec, "generated": o.outputs[0].text.strip().replace("```", "")}
                for (t, rec, _), o in zip(jobs, outs)]
        (out_dir / f"diagbench_{subset}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"{subset}: {len(data)} cases, {len(rows)} generations", flush=True)


if __name__ == "__main__":
    main()
