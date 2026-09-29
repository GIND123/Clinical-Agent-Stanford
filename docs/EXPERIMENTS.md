# Experiment runbook (stanford_idea §6)

All commands assume `data/cdm/{train,val,test}.jsonl` and `data/openworld/other.jsonl` exist. `M=Qwen/Qwen3-8B`. Evaluation rollouts are greedy (`--temperature 0`, the default).

## §6.1 Baselines

| # | Baseline | How |
|---|---|---|
| 1 | Zero-shot sequential (ReAct-style) | `deferdx rollout --policy vllm --model $M --no-defer --cases data/cdm/test.jsonl --out outputs/eval/zs.jsonl` |
| 2 | SFT, forced commit (**primary control**) | SFT data built with `--no-defer`, then `deferdx rollout --policy vllm --model $M --adapter outputs/sft_nodefer/final --no-defer ...` |
| 3 | GRPO accuracy + cost only, no defer | `deferdx train grpo --set env.allow_defer=false reward.brier_lambda=0 output_dir=outputs/grpo_nodefer` |
| 4 | LA-CDM | external: github.com/dharouni/LA-CDM (report its own numbers on the same split) |
| 5 | DiagAgent-14B | external weights; can also run inside this env with `--policy vllm --model <path>` once prompts are aligned |
| 6 | **Post-hoc threshold on #3** (critical ablation) | rollouts of #3 on val and test, then `deferdx baseline --method coverage --target 0.8 --val ... --test ...` (also `--method risk --target 0.05`) |
| 7 | Conformal-style (SGR) | `deferdx baseline --method sgr --target 0.05 --delta 0.05 --val ... --test ...` |
| — | Random planner floor | `deferdx rollout --policy random --samples 5 ...` |

For a fair #6 against DEFER-Dx, compare at **matched coverage**: take DEFER-Dx's test coverage from `closed_world.coverage` and use it as `--target` for `--method coverage`.

## §6.2 Metrics → report keys

| Metric | Key in `deferdx evaluate` JSON |
|---|---|
| Per-class accuracy, leaderboard mean accuracy, macro-F1 | `closed_world.per_class_accuracy`, `closed_world.accuracy_full_coverage`, `closed_world.macro_f1` |
| Risk-coverage curve, AURC, acc@{70,80,90} | `selective.curve`, `selective.aurc`, `selective.accuracy@70` … |
| ECE, Brier (committed cases) | `calibration.ece`, `calibration.brier` |
| Deferral precision/recall vs "would have been wrong" | `deferral.*` (pass `--counterfactual <no-defer rollouts>` for a true counterfactual) |
| Confident-error rate (p > 0.8) | `safety.confident_errors_per_case@0.8`, `safety.error_rate_among_confident@0.8` |
| Severity-weighted error | `safety.severity_weighted_error_per_case` |
| Diverticulitis accuracy (headline) | `closed_world.per_class_accuracy.diverticulitis` (+ `per_class_coverage`) |
| False-commit on OTHER, OOD AUROC | `open_world.false_commit_rate`, `open_world.ood_auroc` |
| Tests/case, termination-step distribution | `cost.mean_tests`, `cost.investigations_histogram`, `cost.one_step_terminations` |

## τ sweep → risk-coverage curve from training

Each τ gives a separate policy (an operating point). Run `deferdx crossover` first to see the effective threshold each τ induces.

```bash
for t in 0.75 0.80 0.85 0.90 0.95; do
  deferdx train grpo --set reward.tau=$t tau_schedule=null output_dir=outputs/grpo_tau$t
done
```

## Open-world ablations (§4.2)

- OTHER in test only: train on `data/cdm/train.jsonl`.
- OTHER partly in training: split `data/openworld/other.jsonl` by patient and add the train part to `train_cases`. Keep the evaluation OTHER cases disjoint by `subject_id`.
- Pipeline-leak check: train a bag-of-words classifier to separate `data/openworld/controls.jsonl` from CDM cases of the same labels. An AUROC well above 0.5 means the open-world numbers are confounded.

## Scaling out

`training/grpo.py` is a single-process reference implementation. For 8B with G=16 and full runs (plan §5.2: 2–5 days per run), port it to verl:

- Environment: `DiagnosticEnv.reset/step_text` is a pure lookup, so wrap it as a verl multi-turn tool/interaction.
- Reward: call `rewards.group_rewards` on each prompt's group of G rollouts. It needs the whole group for p̂, so compute rewards at group level, not per sample.
- Keep the constraint's ν as trainer state, updated once per batch with `CoverageConstraint.update`.
