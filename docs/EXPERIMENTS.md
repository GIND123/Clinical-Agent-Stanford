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
| DEFER-Dx, second training seed | `configs/ablations/cev_seed1.yaml` (seed 1, otherwise identical) | 150 | did not fit `queue_v7.sh`; training in `queue_v8.sh` (first attempt stopped by another user's GPU job, second resumed from step 50); compared with the control thresholded at its own coverage | training-seed variance |
| No-handoff ablation (η = 0) | `configs/ablations/cev_no_handoff.yaml` | 100 | `queue_v8.sh`, after the second seed; against DEFER-Dx's step-100 checkpoint | does the scored handoff matter? |
| Held-out evaluation of the dual-ascent run | (none) | (none) | evaluated (`queue_v7.sh`, Oct 12) | record of the controller failure |
| gpt-oss-20b zero-shot | (none) | (none) | last in `queue_v8.sh`, only if it fits | larger open model |
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
| `queue_v8.sh` | Oct 12–14 | Budget extended to Wed 05:30 IST: the second DEFER-Dx seed, the no-handoff ablation, gpt-oss-20b if it fits. |

## Comparisons the results support

- **Against the published mechanism.** DEFER-Dx against the case-level consensus arm, paired on the same cases. This is the novelty claim.
- **Learned against prompted deferral.** DEFER-Dx against zero-shot Qwen3-8B with DEFER in the prompt, paired.
- **Learned deferral against a post-hoc threshold on a zero-shot model.**
  - **Setup:** DEFER-Dx against zero-shot forced choice with a confidence threshold cross-fitted at DEFER-Dx's own coverage (fit on val, applied to test and vice versa), paired.
  - **Single sample:** tied stated probabilities can keep the threshold from reaching the target coverage. RESULTS.md reports the coverage actually reached.
  - **Self-consistency:** with agreement over three samples as the confidence, coverage can be matched.
- **Open world.** False commitments, deferral and naming OTHER on 206 OTHER cases from groups seen in training and 187 from five time-critical groups never seen. Ruptured AAA and ectopic pregnancy have fewer than 10 cases each: they are pooled, never reported separately.

**Not answered, and said so.** The control's seed variance (one control seed). Anything in the queue that does not finish by the deadline is listed here when the run ends.

## Reporting rules

- **Intervals:** every number with its 95% case-level bootstrap interval (2,000 resamples; all seeds of a case resampled together). Comparisons are paired bootstraps on the same cases.
- **Pooling:** val + test (480 cases) for headline numbers, with test-only rows for comparability with LA-CDM. Per-class claims use the pooled set (51 diverticulitis cases).
- **Small cells:** groups with fewer than 10 cases are suppressed.
- **Incomplete evaluations:** an evaluation counts only if every requested seed finished (`suite_complete`).
