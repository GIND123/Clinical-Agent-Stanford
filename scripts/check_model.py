"""Pre-data sanity check of a real local model in the DEFER-Dx loop (synthetic cases only).

Verifies the pieces that unit tests with a fake tokenizer cannot:
  1. the model's zero-shot action-format compliance under our system prompt;
  2. that <think> / </think> survive decoding (if they are stripped, earlier reasoning
     is not removed from context by Qwen3's template and parsing falls back);
  3. that SFT prompts re-rendered from messages are token-identical to the prompts
     the model actually saw at generation time (SFT/RL consistency).

Usage (CPU is fine for a 0.6B model):
    python scripts/check_model.py --model Qwen/Qwen3-0.6B --cases 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deferdx.data.synthetic import synth_cases  # noqa: E402
from deferdx.env import EnvConfig, TestCatalog, parse_action  # noqa: E402
from deferdx.env.actions import INVALID  # noqa: E402
from deferdx.policy.hf import HFPolicy  # noqa: E402
from deferdx.rollout import run_episodes  # noqa: E402
from deferdx.training.common import conversation_to_turns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float32", help="float32 on CPU; bfloat16 on GPU")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--save-rollouts", default=None, help="optional JSONL of full rollouts (synthetic cases only)")
    args = ap.parse_args()

    catalog = TestCatalog.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "test_catalog.yaml")
    env_cfg = EnvConfig(max_steps=args.max_steps, max_invalid=2)
    cases = synth_cases(n=args.cases, n_other=max(1, args.cases // 4), seed=7)[: args.cases]
    kw = {"enable_thinking": True}
    policy = HFPolicy.from_pretrained(args.model, dtype=args.dtype, device_map=None,
                                      max_new_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20,
                                      batch_size=args.cases, chat_template_kwargs=kw)
    tok = policy.tokenizer
    t0 = time.time()
    rollouts = run_episodes(policy, catalog, env_cfg, cases, n_samples=1,
                            on_turn=lambda t, n: print(f"turn {t}: {n} active ({time.time() - t0:.0f}s)", flush=True))

    turns = [t for ro in rollouts for t in ro.turns]
    parsed = [parse_action(t.text, set(catalog.tests), set(catalog.asks)) for t in turns]
    think_kept = sum("</think>" in t.text for t in turns)
    truncated = sum(len(t.completion_ids or []) >= args.max_new_tokens
                    and (t.completion_ids or [None])[-1] not in policy.stop_ids for t in turns)
    prompt_match = 0
    n_checked = 0
    for ro in rollouts:
        rebuilt = conversation_to_turns(ro.messages, tok, kw)
        for gen_turn, sft_turn in zip(ro.turns, rebuilt):
            n_checked += 1
            prompt_match += gen_turn.prompt_ids == sft_turn.prompt_ids
    report = {
        "model": args.model,
        "turns": len(turns),
        "valid_action_rate": sum(a.type != INVALID for a in parsed) / max(1, len(parsed)),
        "invalid_reasons": Counter(a.error for a in parsed if a.type == INVALID).most_common(5),
        "action_types": dict(Counter(a.type for a in parsed)),
        "terminals": dict(Counter(ro.result.terminal for ro in rollouts)),
        "think_tags_survive_decoding": f"{think_kept}/{len(turns)}",
        "truncated_turns": f"{truncated}/{len(turns)}",
        "invalid_examples": [t.text[-300:] for t, a in zip(turns, parsed) if a.type == INVALID][:3],
        "sft_prompt_token_identical": f"{prompt_match}/{n_checked}",
        "mean_completion_tokens": sum(len(t.completion_ids or []) for t in turns) / max(1, len(turns)),
        "seconds": round(time.time() - t0, 1),
        "example_turn": turns[0].text[-600:] if turns else "",
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.save_rollouts:
        with open(args.save_rollouts, "w", encoding="utf-8") as f:
            for ro in rollouts:
                f.write(json.dumps(ro.to_dict()) + "\n")


if __name__ == "__main__":
    main()
