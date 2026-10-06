# Experiment runbook

This is the grid that produces docs/RESULTS.md, run on one 48 GB GPU by two sequential queues. Every step is idempotent: a step whose `outputs/queue/<name>.done` marker exists is skipped, and training resumes from its latest checkpoint. Both queues wait for a free GPU.

```bash
source scripts/env_gpu.sh
setsid nohup bash scripts/queue.sh > outputs/logs/queue.log 2>&1 &                                   # main grid
WAIT_PID=<queue.sh pid> setsid nohup bash scripts/queue_ablations.sh > outputs/logs/queue_ablations.log 2>&1 &
python scripts/make_report.py && python scripts/make_figures.py                                      # any time
python scripts/plot_training.py --runs deferdx=outputs/runs/deferdx nodefer=outputs/runs/nodefer
```

## Main grid (scripts/queue.sh)

| Step | What | Time (approx.) |
|---|---|---|
| `train_deferdx` | DEFER-Dx, 150 GRPO steps, B = 12 cases × G = 8 (configs/grpo_deferdx.yaml) | ~18 h |
| `eval_deferdx` | 5 evaluation sets × 3 seeds | ~1 h |
| `eval_zs_nodefer`, `eval_zs_defer` | zero-shot Qwen3-8B: forced choice; prompted DEFER | ~1 h each |
| `train_nodefer` | control: identical, DEFER not offered (configs/grpo_nodefer.yaml) | ~18 h |
| `eval_grpo_nodefer` | as above | ~1 h |
| `eval_*_maskdrop` | both trained models with every sentence containing CDM's `____` mask removed | ~30 min each |
| `eval_zs_closed` | zero-shot, closed world (four labels only, as in prior work) | ~30 min |

## Ablations (scripts/queue_ablations.sh; 100 steps, compared with the main run's step-100 checkpoint)

| Ablation | Config | Question |
|---|---|---|
| group std normalisation | configs/ablations/std_norm.yaml | Does deferral collapse as Che et al. (2026) predict when advantages are std-normalised? |
| CDM-only training | configs/ablations/cdm_only.yaml | Does deferral learned from in-set difficulty alone transfer to out-of-set presentations? |
| no coverage constraint | configs/ablations/no_constraint.yaml | What does the Lagrangian floor prevent? |
| leave-one-out p̂ | configs/ablations/forced_loo.yaml | Does removing p̂'s commit-selection bias change deferral quality? |
| training seed 1 | configs/ablations/seed1.yaml | Training-seed variance of the main result (150 steps) |

## Evaluation sets (data/cohorts, built by `deferdx data cohorts`)

`eval_cdm_val` (240) and `eval_cdm_test` (240) are the exact LA-CDM splits, never used for training or any training-time decision. Also evaluated: `eval_other_seen` (206), `eval_other_unseen` (187: time-critical groups held out of training) and `eval_controls` (258: same-pipeline in-set cases). Development decisions use only `dev` (212), a held-out part of the training split.

## Comparisons and how they are made

- **Learned vs post-hoc deferral (the critical ablation, plan §6.1 #6).** DEFER-Dx against the GRPO control plus a confidence threshold at DEFER-Dx's own coverage. The threshold is cross-fitted: fit on val and applied to test, and vice versa, so no case is scored by a threshold that saw it. The comparison is a paired bootstrap on the same cases.
- **Conformal-style (plan #7).** SGR at 5% selective risk, δ = 0.05, cross-fitted the same way.
- **Prompted vs learned deferral.** Zero-shot with DEFER in the prompt against DEFER-Dx.
- **Self-consistency.** Majority vote over the 3 evaluation samples for every system. Agreement is the confidence: the inference-time version of the group-consensus signal.
- **Published systems.** As context only (different environments or splits); see RESULTS.md §4.

## Reporting rules

Report every number with its 95% case-level bootstrap interval. Pool val + test (480 cases) for headline numbers, and give the test-only row for comparability with LA-CDM. Per-class claims need the pooled set (51 diverticulitis cases); the test split alone has 25. Groups with fewer than 10 cases are suppressed.
