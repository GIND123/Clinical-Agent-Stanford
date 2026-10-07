# Methods (as implemented)

This is the method that the code in this repository runs, with the constants it uses. It is written to be lifted into a paper. The research plan is in [stanford_idea_md.md](../stanford_idea_md.md); where the implementation departs from the plan, the reason is given here and in the README.

## 1. Task and environment

**Decision process.** Each episode is one hospital admission. The hidden state is the diagnosis y ∈ {appendicitis, cholecystitis, diverticulitis, pancreatitis} ∪ {OTHER}. At reset the agent sees the patient's full history (the MIMIC-IV-Ext-CDM history text, which contains the history of present illness and past, social and family history). On each turn it emits one action as JSON inside `<action>…</action>`, after reasoning inside `<think>…</think>`.

| Action | Effect |
|---|---|
| `ASK(physical_exam)` | reveals the admission physical examination (free) |
| `TEST(x)`, x in 22 tests | reveals the earliest result of that test for this admission, or "not performed"; charged at its 2025 BIDMC standard charge (placeholder prices where marked in `configs/test_catalog.yaml`) |
| `COMMIT(d, p)` | terminal: diagnosis d ∈ D ∪ {OTHER} with stated probability p ∈ [0, 1] |
| `DEFER(differential, reason)` | terminal: hand the case to a clinician with a ranked differential |

At most 8 ASK/TEST actions are allowed per episode, and at most 2 malformed turns. Observations are looked up from the record, never generated, so the environment needs no simulator and no external model. Tests are individual (CBC, CMP, lipase, CT abdomen, …), not the three coarse categories of LDTL. Every COMMIT carries a numeric probability, not a binary confidence flag.

## 2. Reward

**Commit.** For COMMIT(d, p) on a case with label y:

R_commit = α·1[d = y] − λ·(p − 1[d = y])² − κ·C[y, d] − c·Σ_t cost(a_t)

with α = 1, λ = 1 (Brier score, a strictly proper scoring rule), κ = 0.5 and C the 5×5 asymmetric severity matrix in `configs/severity_matrix.yaml`. C is a time-to-harm proxy (for example, a missed OTHER case costs 1.0, a missed appendicitis 0.8). The cost scale c = α / Σ_x cost(x) = 6.81 × 10⁻⁵ per US$, so ordering every test in the catalog costs one correct diagnosis (LA-CDM's convention); a CT abdomen costs 0.089.

**Counterfactual escalation value (CEV; the contribution).** Every published deferral or abstention reward we found prices escalation by how likely the policy is to be wrong on the *case*. Some use the share of a GRPO group that was correct (TIAR, KARL, and our own earlier version below); some use a prior estimate of the success rate (AWA-RL); some use cases designated in advance (TrustMed-RL). CEV prices it by what would happen if the agent did not escalate **at the exact state where it escalated**.

1. **Branch.** For an episode that ends with DEFER at state s (the conversation and the tests revealed so far), the environment is rebuilt by deterministic replay. The trainer then samples K = 3 *forced continuations* from s. In these, escalating is not an option: a DEFER is executed as COMMIT of the differential's top, at its stated probability, but further tests remain allowed within the budget.
2. **Value of not escalating.** Their mean clinical return V̂(s) = mean_k [α·1[correct] − λ(p − 1[correct])² − κ·C[y, d] − c·cost(tests after s)] estimates the value of continuing from s: deciding now or investigating further, whichever the policy would do.
3. **Value of escalating.** E = V_τ + η·S(q, y) − μ − ν. V_τ is the expected commit reward of a calibrated decision with success probability τ (in-set cases), or 1.2 (OTHER cases). S is the normalised multi-class Brier score of the **handed-over probabilistic differential** q, which is strictly proper, so honest differentials are optimal; η = 0.3. μ = 0.1 is the handoff cost and ν the coverage multiplier.
4. **Credit.** The DEFER turn's advantage is the state-level counterfactual A = E − V̂(s). Earlier turns keep the usual group advantage, computed from the episode reward E − c·cost(tests).

The consequence is the distinction a clinician makes:
- **reducible uncertainty** (another test would settle it) gives a high V̂(s), so escalation is discouraged and the agent learns to investigate;
- **irreducible uncertainty** (continuing would err) gives a low V̂(s), so escalation is credited;
- because V̂ includes the severity matrix and the scoring rule, an escalation is worth more where the agent's errors would be dangerous or confidently wrong;
- because S rewards the handoff's content, deferral is a calibrated, informative handoff rather than an "I don't know".

Config: `configs/grpo_cev.yaml` (`reward.defer_mode: cev`, `cev_k`, `cev_max_roots`, `env.handoff_probs: true`). Ablations: without the scored handoff (`configs/ablations/cev_no_handoff.yaml`), and a constant deferral reward (`configs/ablations/constant_defer.yaml`).

**Group-consensus deferral (case-level baseline arm, published mechanism).** This is the reward first implemented here, now known to match prior work (TIAR's λ(1 − 2p̂) advantage adjustment is the case τ = 0.5; KARL and AWA-RL are close variants). It is kept as the comparator that isolates what the counterfactual, state-level estimate adds. GRPO samples G episodes of the same case. Let p̂ be the fraction of the group's committing episodes that were correct: an on-policy, label-verified estimate of how likely the current policy is to be wrong on this case. For a DEFER on an in-set case:

R_defer = γ·(τ − p̂) − μ − c·Σ_t cost(a_t)

with γ = 2, handoff cost μ = 0.1 and the safety dial τ. Deferring pays when the group usually gets the case wrong and is penalised when it usually gets it right. If no episode in the group committed, p̂ = τ (deferral then costs only μ). On an OTHER case, R_defer = 1.2 − μ, which beats a perfectly confident correct COMMIT(OTHER) (≤ 1): "none of these" warrants escalation.

**What τ means.** For a calibrated committer with success probability q, the expected commit reward equals the expected defer reward at q*: τ = 0.85 gives q* ≈ 0.645 and τ = 0.95 gives q* ≈ 0.70 (`deferdx crossover`). The policy learns to defer when it estimates q < q*, not q < τ. Both numbers are reported.

**Alternative estimator (ablation).** `reward.p_hat_mode: forced_loo` estimates p̂ for deferring episode i from the forced predictions of the other G − 1 episodes (commit, or the top of a deferral differential). It is not biased by which episodes chose to commit, and it stays defined when most of the group defers. A deferring episode's own differential never changes its own reward.

**Coverage constraint.** Over-deferral is a documented failure mode (Abstain-R1). The rate of DEFER on in-set cases in each batch is held below ρ_max = 0.30 by projected dual ascent on a multiplier ν (lr 2.0, after 2 consecutive violations; ν ≤ 5). ν is charged to deferring episodes on in-set cases only. Subtracting the same constant from every episode in a group would cancel in the advantage, so it is not done. Deferring an OTHER case is the desired behaviour and is not rationed.

**Format.** An episode ending in malformed actions or the step budget scores −1 minus its investigation cost.

## 3. Data

**MIMIC-IV-Ext-CDM v1.1** (Hager et al., *Nature Medicine* 2024): 2,400 admissions (957 appendicitis, 648 cholecystitis, 257 diverticulitis, 538 pancreatitis), labels from `pathology_ids.json`. The exact LA-CDM 80/10/10 split (stratified, seed 269) is kept so that LA-CDM's numbers are comparable on the same test set. When a test repeats within an admission, only the earliest result is kept. CDM's CSVs carry no `subject_id`; it is attached from MIMIC-IV `admissions`.

**MIMIC-CDM-OW (open world).** `deferdx data build-openworld` selects MIMIC-IV 2.2 admissions with an abdominal chief complaint whose principal (seq 1) ICD diagnosis falls in one of 9 non-CDM groups. Admissions carrying any CDM-condition code at any position are excluded. Each case is built with CDM's own text pipeline (`data/parity.py`):

- the history blob and physical-exam windows;
- the radiology section filter, and the exam name taken from `radiology_detail`;
- `____` masking of the case's own diagnosis;
- CDM's inclusion rule;
- earliest-value deduplication under a total row order.

The build is deterministic and byte-identical across runs. Same-pipeline **controls** are admissions with a CDM condition as principal diagnosis, built by the same pipeline: 648 controls next to 713 OTHER cases.

**Cohorts** (`deferdx data cohorts`, patient-disjoint):

| Set | Cases | Use |
|---|---|---|
| RL train | 2,404 = 1,760 CDM + 279 OTHER + 365 controls | GRPO |
| dev | 212 = 160 CDM + 32 OTHER + 20 controls | monitoring only |
| CDM val + test | 240 + 240 | evaluation; never seen in training |
| OTHER, seen groups | 206 (bowel obstruction, gastroenteritis/colitis, GI bleed, urolithiasis) | evaluation |
| OTHER, unseen groups | 187 (mesenteric ischaemia, perforated/bleeding ulcer, AAA, DKA, ectopic) | evaluation; groups never seen in training |
| controls | 258 | evaluation (source-shortcut check) |

No training case shares a patient with an open-world evaluation case. The time-critical groups are held out entirely, so the open-world evaluation separates recognising a group seen in training from escalating an unfamiliar, dangerous presentation. Training on controls removes the shortcut "built by the open-world pipeline ⇒ OTHER". Cases are drawn from source pools by weight (CDM 0.75, OTHER 0.15, controls 0.10), which also keeps the controls' cholecystitis-heavy mix from shifting the class balance.

## 4. Training

**Model.** Qwen3-8B (thinking mode), LoRA r = 16, α = 32, all linear layers of the transformer blocks (the output layer is frozen).

**Algorithm.** Multi-turn GRPO on batches of B = 12 cases × G = 8 episodes, rollout temperature 1.0. Steps 1–50 take one on-policy gradient step per batch; from step 51, each batch is split into 4 PPO mini-batches with an optimizer step after each. The loss for those is the clipped ratio π_θ / π_vLLM against the sampling engine's own log-probs, which corrects both for staleness and for the engine mismatch. The constituent choices:

- **Advantages:** r − mean(group), with no division by the group standard deviation (Dr. GRPO). Dividing would inflate the tiny Brier and cost differences inside all-correct groups to unit scale and distort the designed trade-offs.
- **Group filter:** groups whose reward spread is below 0.05 are skipped (dynamic sampling).
- **Loss:** a clipped surrogate (ε = 0.2 / 0.28, clip-higher) over every assistant turn's exact sampled tokens, averaged at the token level over the batch (DAPO).
- **Engine correction:** truncated importance sampling (cap 2) against the sampling engine's own token log-probs.
- **Optimiser:** AdamW, gradient-norm clip 1.0, no KL term. Learning rate 1 × 10⁻⁵ for steps 1–25 (after 5 warm-up steps), then 5 × 10⁻⁵. After 25 steps at 1 × 10⁻⁵ the adapter had moved the weights by only ~1.3 × 10⁻⁴ of their norm and batch metrics were flat, so the rate was raised. Both schedules (`lr_milestones`, `ppo_minibatch_milestones`) are part of the config, so the control arm and every ablation follow them exactly. After 50 steps the update had grown 4.6× (still ~6 × 10⁻⁴ of the weights' norm) with dev metrics flat within noise; mini-batching multiplies the updates per rollout at almost no extra compute.
- **Curricula:** τ is annealed 0.95 → 0.85 over 100 steps (deferral cheap early, §5.4 of the plan), with a small terminal-action entropy bonus (0.02) for the first 30 steps.

**Turns.** Qwen3's chat template strips earlier `<think>` blocks, so every assistant turn is trained against exactly the prompt it was generated from.

**Single-GPU colocation** (`training/grpo_vllm.py`). vLLM generates with the current LoRA adapter while the HF model's frozen weights sit in pinned CPU memory. vLLM then sleeps (weights to CPU, KV cache freed), the frozen weights return to the GPU, and one policy-gradient step runs. The new adapter is served under a fresh id.

- **Correctness check:** vLLM's LoRA log-probs match HF+PEFT within bf16 noise across sleep/wake cycles (mean |Δ log p| 0.024–0.030 per token, against a LoRA effect of 0.45), and the training-time engine mismatch stays at about 0.02.
- **Memory:** the output layer's log-probs are computed by a custom autograd function that recomputes logits chunk by chunk in the backward pass (gradient-checked against the naive computation), so memory scales with the backbone, not the 152k vocabulary.
- **Throughput:** one unpadded sequence per micro-batch is 1.55× faster than padded micro-batches. A step takes about 6–7 minutes on one RTX PRO 5000 (48 GB).

**Control arm.** Identical data, reward (including the Brier term), compute and seed, but DEFER is not offered. Thresholding this model's stated probability is the fair answer to "why not just threshold?".

## 5. Evaluation

**Protocol.** Each model runs on all five evaluation sets with 3 seeds. Sampling uses Qwen3's thinking-mode settings (T 0.6, top-p 0.95, top-k 20), never greedy. Per-request seeds derived from (seed, case, sample, turn) make runs reproducible.

**Metrics.**
- **Closed world** (CDM val + test, 480 cases): accuracy at full coverage (a deferral counts through the top of its differential), mean-class accuracy (LA-CDM's convention), macro-F1, per-class accuracy, coverage, selective accuracy, risk-coverage curve and AURC, accuracy at 70/80/90% coverage, ECE and Brier on committed cases.
- **Clinical safety:**
  - unflagged errors (wrong commits per case) and confident errors (wrong commits with p > 0.8);
  - severity-weighted error;
  - deferral precision (deferrals whose best guess was wrong) and recall (would-be errors that were deferred);
  - tests and dollars per case.
- **Open world:** false-commit rate (an OTHER case committed to one of the four), confident false commits, deferral rate, and OOD AUROC.

**Statistics.** 95% intervals come from a case-level bootstrap with 2,000 resamples; all seeds of a case are resampled together. Comparisons are paired bootstraps on the same cases.

**Baselines.** All are run here on the same cases and seeds:
- zero-shot Qwen3-8B, forced choice and with prompted DEFER;
- the GRPO control;
- post-hoc confidence thresholds on each forced-choice model at DEFER-Dx's coverage, cross-fitted (fit on val, applied to test and vice versa, so no case is scored by a threshold that saw it);
- self-consistency over the 3 samples (majority vote, with agreement as confidence: the inference-time twin of the group-consensus reward);
- DiagAgent-14B and published LA-CDM / LDTL numbers as context (`docs/BASELINES.md`).

**Robustness.**
- **The `____` diagnosis mask:** each mask is a label cue (`docs/DATA_AUDIT.md` §5). Evaluation is repeated with every masked sentence removed (`--mask-policy drop_sentence`).
- **Source shortcut:** the in-set controls show whether "answers OTHER" means "recognises OTHER" or "recognises the pipeline".

**Data governance.** No MIMIC text leaves the machine, and every model runs on local weights. Rollouts stay in the git-ignored `outputs/`. Committed documents hold aggregates only, with patient-derived counts of 1–9 suppressed.
