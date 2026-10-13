# Experiment runbook (final, as run)

All runs used one shared 48 GB GPU. The final budget runs to **Wednesday, October 14, 2026, 05:30 IST (Tuesday, October 13, 20:00 US Eastern)**, through `scripts/queue_v8.sh`. This page records what ran, what did not and why, and which comparisons the results can support. The numbers are in [RESULTS.md](RESULTS.md); figures and captions are in the README §3.

## Runs

| Run | Config | Steps | Outcome | Role |
|---|---|---|---|---|
| **DEFER-Dx**: counterfactual escalation value, scored handoff, PI coverage controller | `configs/grpo_cev.yaml` | 150 | trained and evaluated on 5 sets × 3 seeds (`scripts/queue_v4.sh`, then `queue_v5.sh`) | the method |
| Case-level group-consensus reward (TIAR/KARL-style) | `configs/grpo_deferdx.yaml` | 150 | trained and evaluated | the published mechanism, as comparator |
| Zero-shot Qwen3-8B, forced choice and prompted DEFER | (none) | (none) | evaluated | baselines, plus cross-fitted post-hoc thresholds, SGR and self-consistency |
| Zero-shot Qwen3-8B, closed world (4 labels, as in prior work) | (none) | (none) | `queue_v5.sh`, CDM sets only | context against published closed-world numbers |
| DEFER-Dx with every sentence containing CDM's `____` mask removed | (none) | (none) | `queue_v5.sh`, CDM sets only | robustness to the mask cue |
| GRPO control, identical training without DEFER | `configs/grpo_nodefer.yaml` | 150 | trained and evaluated (`queue_v7.sh`, Oct 11; the first attempt died of an out-of-memory error at step 24 and was resumed from step 20) | learned vs post-hoc deferral on an identically trained model |
| Constant deferral reward | `configs/ablations/constant_defer.yaml` | 100 | trained and evaluated (`queue_v7.sh`, Oct 11–12), against DEFER-Dx's step-100 checkpoint | does the counterfactual credit matter? |
| DEFER-Dx, second training seed | `configs/ablations/cev_seed1.yaml` (seed 1, otherwise identical) | 150 | trained and evaluated (`queue_v8.sh`, Oct 12–13; the first attempt was stopped by another user's GPU job, the second resumed from step 50); compared with the control thresholded at its own coverage | training-seed variance |
| No-handoff ablation (η = 0) | `configs/ablations/cev_no_handoff.yaml` | 100 | trained and evaluated (`queue_v8.sh`, Oct 13), against DEFER-Dx's step-100 checkpoint | does the scored handoff matter? |
| Held-out evaluation of the dual-ascent run | (none) | (none) | evaluated (`queue_v7.sh`, Oct 12) | record of the controller failure |
| gpt-oss-20b zero-shot | (none) | (none) | **not run**: both attempts in `queue_v8.sh` failed at engine start-up (vLLM's FlashInfer kernels rejected the Blackwell GPU, "requires sm75 or higher"); an optional baseline, not retried | larger open model |
| A second seed of the control | (none) | (none) | **not run** | — |

**Records, not results.** These two runs appear only through their training logs (README Fig. 8).
- `cev_dualascent` is the first CEV run. Its plain dual-ascent coverage cap ran a limit cycle. It also predates the rule that deferrals left unbranched get no DEFER-turn update.
- `cev_pi_crashed` is the first PI-controller attempt, which died of an out-of-memory error at step 29.

## How the queues evolved

| Queue | Dates (IST) | Why it was replaced |
|---|---|---|
| `queue.sh`, `queue_ablations.sh` | Oct 6–7 | The consensus reward turned out to be published (TIAR, KARL, AWA-RL). The method became the counterfactual escalation value (CEV). |
| `queue_v2.sh` | Oct 7–8 | Under CEV, plain dual ascent on the coverage multiplier wound up and collapsed deferral, so it was replaced by a PI controller. |
| `queue_v3.sh` | Oct 8–9 | The CEV run died of an out-of-memory error (another process on the shared GPU), and the queue moved on without the method. |
| `queue_v4.sh` | Oct 9–10 | Final code; every step retried from its checkpoint; vLLM memory share lowered (0.72 training, 0.78 evaluation; memory only). It trained DEFER-Dx. |
| `queue_v5.sh` | Oct 10 | Fit a Saturday budget. Replaced the same morning when the budget was extended to Monday for the journal experiments. |
| `queue_v6.sh` | Oct 10 | The journal run, to Mon 18:00 IST. Replaced the same day so the Stanford-abstract evaluations ran first. |
| `queue_v7.sh` | Oct 10–12 | Stanford-abstract block first (DEFER-Dx, closed-world zero-shot, mask-drop), then the control, the constant-deferral ablation with the step-100 evaluation, and the dual-ascent evaluation. The second seed did not fit its deadline. |
| `queue_v8.sh` | Oct 12–13 | Budget extended to Wed 05:30 IST: the second DEFER-Dx seed and the no-handoff ablation; gpt-oss-20b failed at start-up. Finished Tue 21:57 IST. |
| `queue_v9.sh` | Oct 13–15 | Started Tue 21:58 IST: the control's second training seed, and the consensus arm's second seed if its adapter arrives in time; deadline Thu Oct 15, 12:00 IST. |

## Comparisons the results support

- **Against the published mechanism.** DEFER-Dx against the case-level consensus arm, paired on the same cases. This is the novelty claim.
- **Learned against prompted deferral.** DEFER-Dx against zero-shot Qwen3-8B with DEFER in the prompt, paired.
- **Learned deferral against a post-hoc threshold on an identically trained model.** DEFER-Dx against the GRPO control with a confidence threshold cross-fitted at DEFER-Dx's own coverage, paired. Each DEFER-Dx seed is compared at its own coverage.
- **Learned deferral against a post-hoc threshold on a zero-shot model.** The same, on zero-shot forced choice, with one sample and with three-sample voting.
- **How thresholds match coverage.**
  - **Fitting:** each threshold is fit on val and applied to test, and vice versa.
  - **Ties:** stated probabilities are coarse (0.80 alone covers 14–17% of the control's cases), so a plain cut overshoots the target. Commitments tied at the cut are kept with a probability fitted to reach it, decided by a fixed hash of the case and evaluation seed (`coverage_rule` in `src/deferdx/baselines/selective.py`).
  - **Draws:** RESULTS.md §5 uses one pre-specified draw, and §5b the range over 20 draws.
  - **Before Oct 13:** reports used a plain cut, which put the control at 79% coverage for both seeds (targets 77.6% and 73.2%). That overstated the learned-versus-threshold advantage: +3.1 instead of +2.3, and +3.2 instead of +1.1.
- **Open world.** False commitments, deferral and naming OTHER on 206 OTHER cases from groups seen in training and 187 from five time-critical groups never seen. Ruptured AAA and ectopic pregnancy have fewer than 10 cases each: they are pooled, never reported separately.

**Not answered, and said so.** The control's and the consensus arm's seed variance (one seed each). Two DEFER-Dx seeds show that handoff content and the in-set comparison with the threshold vary between seeds, but cannot estimate by how much. Anything in the queue that does not finish by the deadline is listed here when the run ends.

## Second seeds of the control and the consensus arm (proposed Oct 13; the control's is queued)

The control and the consensus arm each have one training seed, so the learned-versus-threshold comparison and the comparison with the published mechanism each rest on one run of the comparator. Two second seeds close that gap. They are independent jobs, split across two GPUs so that both finish in about a day.

| Job | Who, where | Training | Time |
|---|---|---|---|
| Control (no DEFER), seed 1 | Dr. George, on the shared 48 GB GPU: **queued** in `scripts/queue_v9.sh`, which starts when `queue_v8.sh` ends (by Wed Oct 14, 05:30 IST) and stops by Thu Oct 15, 12:00 IST | `deferdx train grpo-vllm --config configs/ablations/nodefer_seed1.yaml --set vllm_gpu_memory_utilization=0.72`; the config differs from `configs/grpo_nodefer.yaml` only in `seed: 1` and `output_dir: outputs/runs/nodefer_seed1` | 150 × 466 s ≈ 19.4 h, plus ≈ 1.7 h evaluation |
| Consensus arm, seed 1 | Shreeya, on a Lambda Cloud GPU (48 GB A6000, or an 80 GB H100) | `deferdx train grpo-vllm --config configs/grpo_deferdx.yaml --set seed=1 output_dir=outputs/runs/deferdx_seed1` | 150 × 520 s ≈ 21.7 h on the shared GPU; the Lambda speed is measured by a 3-step smoke run first |

- **Why this way round:** the control is the comparator for the main claim, so its second seed stays on the hardware that every DEFER-Dx and control run used. The consensus arm stands for a published mechanism, where a change of hardware matters least.
- **Seeds and code:** `seed=1`, as in `configs/ablations/cev_seed1.yaml` (case order, rollout sampling, LoRA initialisation); everything else as in the seed-0 runs. The training code and both configs are unchanged since `d8a3f0b` (Oct 9), so either job can run from the current `main`.
- **Environment on Lambda:** vLLM 0.11.0 with torch 2.8.0+cu128, the pin that Lambda's CUDA 12.8 driver needs (`scripts/lambda/setup.sh`). Library versions may differ from the shared GPU's; the run records them.
- **Evaluation:** the Lambda run's adapter (`outputs/runs/deferdx_seed1/final`; weights only, no case text) is evaluated on the shared GPU with `deferdx eval-suite --model Qwen/Qwen3-8B --adapter outputs/runs/deferdx_seed1/final --name deferdx_seed1` (≈ 1.7 h). Every system is then evaluated by the same engine, and the paired comparisons run on one machine. The control's evaluation: `deferdx eval-suite --model Qwen/Qwen3-8B --adapter outputs/runs/nodefer_seed1/final --name grpo_nodefer_seed1 --no-defer`.
- **Report:** done Oct 13. `scripts/make_report.py` knows both systems (`deferdx_seed1`, `grpo_nodefer_seed1`). Each DEFER-Dx seed is compared with each comparator seed, and each control is thresholded at that DEFER-Dx seed's coverage, with the tie-break range in §5b. A dry run with stand-in suites reproduced the existing comparisons.
- **Queue:** `queue_v9.sh` trains the control's second seed, then evaluates it (3 seeds) and reports. If the consensus arm's adapter is already in `outputs/runs/deferdx_seed1/final` when the queue gets there, the queue evaluates and reports it too. Otherwise, run the evaluation command above after the queue.
- **Data:** the Lambda run needs `data/cohorts/rl_train.jsonl` and `dev.jsonl` on the VM, and these contain MIMIC text. Confirm with the PI that cloud compute is acceptable under the PhysioNet DUA before uploading. Only aggregates come back to the repository.
- **If one does not finish:** each is reported as it stands, and the other still counts.

## Reporting rules

- **Intervals:** every number with its 95% case-level bootstrap interval (2,000 resamples; all seeds of a case resampled together). Comparisons are paired bootstraps on the same cases.
- **Pooling:** val + test (480 cases) for headline numbers, with test-only rows for comparability with LA-CDM. Per-class claims use the pooled set (51 diverticulitis cases).
- **Small cells:** groups with fewer than 10 cases are suppressed.
- **Incomplete evaluations:** an evaluation counts only if every requested seed finished (`suite_complete`).
