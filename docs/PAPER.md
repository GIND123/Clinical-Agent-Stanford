# Training Clinical Agents to Ask, Test, Commit, and Defer

*Draft for a journal submission (and the source of the Stanford AI+HEALTH abstract). Figures: `docs/figures/` (vector PDFs; TikZ sources in `docs/figures/tikz/`), numbered and captioned in README §3. Every `[[...]]` is a placeholder that is filled only from `docs/RESULTS.md`. Methods are written from the code; see `docs/METHODS.md` for every constant.*

## Abstract
[[Written last, from the results. See docs/ABSTRACT.md for the conference version.]]

## 1. Introduction

A patient arrives at the emergency department with abdominal pain. An interactive diagnostic agent asks for the examination, orders a CT and names a diagnosis. Every such agent published on the MIMIC-IV clinical-decision-making benchmark (MIMIC-CDM; Hager et al. 2024) must name one of four conditions (appendicitis, cholecystitis, diverticulitis or pancreatitis), even when the patient has a perforated ulcer or mesenteric ischaemia, and even when its own evidence is split. None can hand the case back.

Three developments make this gap pressing. Regulators increasingly ask that clinical AI escalate to a human when uncertain: the FDA's Digital Health Advisory Committee (November 2025) emphasised timely human intervention for generative-AI devices, and predetermined change-control plans call for monitorable quantities such as abstention rates. The field's own leaderboard shows the cost of optimising accuracy alone: the strongest reported MIMIC-CDM agent (LDTL) raises case-weighted accuracy to 93.4% while its accuracy on the minority condition, diverticulitis, falls to 78.8%, below a random test-ordering policy (90.4%). And recent theory argues that the obvious fix, a learned "abstain" action trained by reinforcement learning, collapses into refusing everything, and that one should instead threshold a calibrated confidence report after training (Che et al. 2026).

We ask whether an agent can *learn* when to defer, without collapsing, and whether learned deferral beats thresholding a confidence report produced by an identically trained agent. Our contributions:

1. **DEFER-Dx.** An agent trained with multi-turn group-relative policy optimisation over four actions: ask, order one of 22 individual tests, commit with a stated probability, or defer to a clinician with a ranked differential.
2. **The counterfactual escalation value.** At each state where the agent escalates, forced continuations branched from that exact state estimate the clinical value of not escalating, by deciding now or investigating further. The escalation is credited with the difference, so the agent learns to separate uncertainty that more testing would resolve from uncertainty that warrants a handoff. Prior deferral rewards price escalation by case-level difficulty (TIAR, KARL, AWA-RL) or by designated cases (TrustMed-RL). The handoff is a probabilistic differential trained with a strictly proper scoring rule.
3. **An open-world benchmark (MIMIC-CDM-OW).** 713 abdominal-pain admissions whose principal diagnosis is none of the four, built with MIMIC-CDM's own text pipeline, and 648 same-pipeline in-set controls. Time-critical diagnosis groups are held out of training entirely.
4. **A head-to-head test of learned versus post-hoc deferral.** The comparison is at matched coverage, with paired bootstrap inference, against an identically trained no-deferral agent whose confidence is thresholded: exactly the repair theory recommends.

## 2. Related work

**Sequential diagnosis on MIMIC-CDM.** Hager et al. (2024) released MIMIC-IV-Ext-CDM and showed that LLMs fall short of clinicians when they must gather information. LA-CDM (Bani-Harouni et al., ICLR 2026) trains a hypothesis-driven planner/diagnoser pair with RL and an uncertainty-expression reward (81.3% mean-class accuracy on its test split). LDTL (Shen et al. 2026) learns latent diagnostic trajectories with three coarse action categories and reports 93.4% case-weighted accuracy on an unpublished split. DiagAgent/DiagGym (Qiu et al. 2025) trains on a learned EHR world model, and CDPR (Peng et al. 2026) adds counterfactual process rewards for cost-aware workups. All are closed-world and forced-choice.

**Abstention and deferral.** Learning to defer (Madras et al. 2018; Mozannar & Sontag 2020) and selective prediction (Geifman & El-Yaniv 2017) are well developed for classifiers. For LLMs, MediQ (Li et al. 2024) and MedAbstain (Machcha et al. 2026) prompt or wrap abstention, and Safe-to-Stop (2026) uses conformal stopping rules. Che et al. (2026) prove that a discrete abstain action trained under error-penalised RL drifts to refusing everything, aggravated by GRPO's group std normalisation, and recommend thresholding a proper-scored confidence report instead. TrustMed-RL (Zhan et al. 2026) trains a clinical VLM agent whose deferral is rewarded on instances designated by construction (evidence-corruption probes, out-of-knowledge tasks). KARL (Gao et al. 2026) rewards QA abstention from within-group response statistics: abstaining is rewarded only when no rollout in the group is correct. Our comparator arm generalises this to a continuous case-level reward γ(τ − p̂) with an explicit clinical operating point (identical to TIAR at τ = 0.5). DEFER-Dx instead credits each escalation at the state where it happens, against what continuing from that state would have achieved. As with KARL, and unlike TrustMed-RL, no case is labelled "should defer".

**Credit assignment by branching.** Resetting a language environment to an intermediate state and rolling out from it gives Monte Carlo value estimates for step-level credit (VinePPO, Kazemnejad et al. 2024, arXiv 2410.01679; tree-structured variants such as TreeRL and Tree-GRPO). In sequential diagnosis, CDPR (Peng et al. 2026) uses short counterfactual rollouts to score the investigation chosen against its alternatives, but it has no deferral action and no handoff. CEV applies branching to the escalation decision itself: the counterfactual is continuing without the option to escalate, valued in clinical units (accuracy, proper score, severity, test cost), and the escalation carries a properly scored handoff. The branching is not new; what it is applied to, and how the escalation is valued, are.

**Calibration by RL.** Rewarding Doubt (Bani-Harouni et al. 2025) and related work train verbalised confidence with proper scoring rules. We use the Brier score on every commitment, so the control arm is a calibrated-report baseline in the sense of Che et al.

## 3. Problem formulation

Each episode is one hospital admission, modelled as a partially observable decision process (Figure 1). The hidden state is the diagnosis y ∈ 𝒟 ∪ {OTHER}, with 𝒟 = {appendicitis, cholecystitis, diverticulitis, pancreatitis}. At reset, the agent sees the history at presentation: the history of present illness and the past, social and family history. On each turn it reasons, then emits one action:

| Action | Effect |
|---|---|
| ASK(physical exam) | reveals the admission physical examination; free |
| TEST(x), x one of 22 individual tests | reveals the earliest recorded result of that test for this admission, or "not performed"; charged at its 2025 standard charge |
| COMMIT(d, p) | terminal: diagnosis d ∈ 𝒟 ∪ {OTHER} with stated probability p ∈ [0, 1] |
| DEFER(q, reason) | terminal: hands the case to a clinician with a probabilistic differential q over 𝒟 ∪ {OTHER} |

At most eight investigations (ASK or TEST) are allowed, and at most two malformed turns. Observations are looked up from the record and never generated, so the environment needs neither a simulator nor an external model. Tests are individual (complete blood count, metabolic panel, lipase, CT abdomen and so on), not coarse categories, and every commitment states a numeric probability.

## 4. Method

### 4.1 Rewards
**Commitments.** COMMIT(d, p) on a case with label y earns

R = α·1[d = y] − λ·(p − 1[d = y])² − κ·C[y, d] − c·Σ_t cost(a_t),

with α = 1 and λ = 1, so the second term is the Brier score, a strictly proper scoring rule. κ = 0.5, and C is a 5 × 5 asymmetric severity matrix: a time-to-harm proxy in which, for example, a missed OTHER case costs 1.0 and a missed appendicitis 0.8. The cost scale c makes ordering every test in the catalogue cost as much as one correct diagnosis, so a CT abdomen costs 0.089. An episode that ends in malformed actions or exhausts its budget scores −1 minus its investigation cost.

**The counterfactual escalation value** (Figure 2). Published deferral rewards price escalation by how likely the policy is to be wrong on the *case*: the share of a sampled group that was correct (TIAR, KARL), a prior estimate of the success rate (AWA-RL), or cases designated in advance (TrustMed-RL). We price it by what would happen if the agent did not escalate, *at the exact state where it escalated*.
1. **Branch.** For an episode that ends with DEFER at state s (the conversation and the results revealed so far), the environment is rebuilt by deterministic replay, and K = 3 forced continuations are sampled from s. In them escalation is disabled: a DEFER is executed as COMMIT of the differential's top choice at its stated probability. Further tests remain allowed within the budget.
2. **Value of not escalating.** Their mean clinical return, V̂(s) = (1/K) Σ_k [α·1[d_k = y] − λ(p_k − 1[d_k = y])² − κ·C[y, d_k] − c·cost(tests after s)], estimates the value of continuing from s, by deciding now or by investigating further, whichever the policy would do.
3. **Value of escalating.** E = V_τ + η·S(q, y) − μ − ν.
   - V_τ is the expected commitment reward of a calibrated decision that succeeds with probability τ on in-set cases, and 1.2 on OTHER cases.
   - S is the normalised multi-class Brier score of the handed-over differential q. It is strictly proper, so an honest differential maximises it. η = 0.3.
   - μ = 0.1 is the handoff cost and ν the coverage multiplier (§4.2).
4. **Credit.** The DEFER turn's advantage is the state-level counterfactual A = E − V̂(s). Earlier turns keep the usual group-relative advantage. At most 32 deferrals per step are branched, chosen at random. The DEFER turn of any other deferral gets no update, because its only signal would be E, which is nearly constant across cases.

**What the credit separates.** Where another test would settle the case, the continuations test, commit correctly and earn a high V̂(s), so escalating is discouraged and the agent learns to investigate. Where continuing would err, V̂(s) is low and escalation is credited. Because V̂ includes the severity matrix and the scoring rule, an escalation is worth more where the agent's errors would be dangerous or confidently wrong. Because S scores the handoff's content, a deferral must be an informative probabilistic differential, not an "I don't know".

**Comparator arm: case-level group consensus.** It is the published mechanism, and was this project's first design. GRPO samples G episodes of each case, and p̂ is the fraction of the group's committing episodes that were correct. A deferral on an in-set case earns γ·(τ − p̂) − μ − c·Σ_t cost(a_t), with γ = 2. TIAR's advantage adjustment is the special case τ = 0.5, and KARL and AWA-RL are close variants. On an OTHER case a deferral earns 1.2 − μ, which exceeds a perfectly confident correct COMMIT(OTHER). The arm is kept to isolate what the state-level counterfactual adds.

**The deferral threshold τ** is annealed from 0.95 to 0.85 over the first 100 steps, so deferral is cheap early. For a calibrated committer, τ = 0.85 corresponds to deferring when the estimated success probability falls below about 0.645; we report both numbers.

**Why this design avoids the collapse of Che et al.** Their collapse needs three things: (i) abstention scored 0 while blanket answering loses in expectation; (ii) a KL anchor whose restoring force shares the abstain gate; (iii) group std normalisation in the sparse-answer regime. DEFER-Dx removes each:

- **(i)** The deferral credit depends on the state. Wherever continuing succeeds, V̂(s) exceeds E and deferring is penalised, so blanket deferral is not optimal for any policy that solves some cases from where it stands.
- **(ii)** No KL anchor is used. Coverage is instead held by an explicit constraint on the in-set deferral rate (§4.2).
- **(iii)** Advantages are centred but not std-normalised (Dr. GRPO), so the designed penalties keep their scale.

Every commitment still carries a proper-scored probability, so the agent can also be thresholded after training: DEFER-Dx contains the recommended repair as a special case.

### 4.2 Training
**Model and algorithm** (Figure S1). Qwen3-8B in thinking mode, with LoRA (rank 16, α = 32) on every linear layer of the transformer blocks. Training is multi-turn GRPO on batches of 12 cases × 8 episodes, sampled at temperature 1.0, for 150 steps.
- **Advantages:** reward minus the group mean, without division by the group standard deviation (Dr. GRPO), so the designed trade-offs keep their scale. Groups whose reward spread is below 0.05 are skipped.
- **Loss:** a clipped surrogate (ε = 0.2 below, 0.28 above) over every assistant turn's sampled tokens, averaged at the token level (DAPO), with truncated importance weights (cap 2) against the sampling engine's own log-probabilities. Each turn is trained against exactly the prompt it was generated from.
- **Optimisation:** AdamW, gradient-norm clipping at 1.0, no KL term. The learning rate is 1 × 10⁻⁵ for steps 1–25 (after 5 warm-up steps), then 5 × 10⁻⁵. From step 51 each batch is split into four PPO mini-batches. A small terminal-action entropy bonus (0.02) applies for the first 30 steps.

**One GPU.** Generation (vLLM, with the current adapter) and training (PEFT) share a single 48 GB GPU: vLLM sleeps while the policy is updated, and the new adapter is then served under a fresh id. vLLM's LoRA log-probabilities match the training model's within bf16 noise across sleep–wake cycles: mean |Δ log p| is 0.024–0.030 per token, against a LoRA effect of 0.45. Output-layer log-probabilities are computed chunk by chunk with a custom autograd function, so memory does not scale with the 152k-token vocabulary. A step takes 5–10 minutes.

**Coverage controller.** Over-deferral is a documented failure of learned abstention, so the in-set deferral rate is held at or below ρ_max = 0.30 by a multiplier ν charged to deferring episodes on in-set cases only; deferring an OTHER case is never rationed.
- **The method:** ν is set by a PI controller (Stooke et al. 2020): e_k = ρ_k − ρ_max, I_k = clip(I_{k−1} + e_k, 0, ν_max/K_i), ν_k = clip(K_p·e_k + K_i·I_k, 0, ν_max), with K_p = 1.5, K_i = 0.2 and ν_max = 2.
- **Why not dual ascent:** our first DEFER-Dx run used plain projected dual ascent, as the comparator arm does (rate 2.0, ν ≤ 5). Under the escalation value the unconstrained policy wants to defer well above the cap, so the cap binds. Dual ascent is integral-only, and the policy answered ν several steps late, so ν wound up to 4.6 against an escalation margin near 1. In-set deferral then collapsed to about 5% for some 20 steps, out-of-set deferral fell with it, and the cycle repeated (Figure 8).
- **Why PI fixes it:** the proportional term reacts at once and vanishes with the violation, so the integral term and ν_max can stay small.
- **The comparator arm** keeps dual ascent: its cap barely bound (ν ≤ 0.19).

## 5. Data
**MIMIC-IV-Ext-CDM 1.1** (Hager et al. 2024) has 2,400 admissions with acute abdominal pain: 957 appendicitis, 648 cholecystitis, 538 pancreatitis and 257 diverticulitis. We keep the exact LA-CDM 80/10/10 split, so LA-CDM's numbers are comparable on its test set. When a test repeats within an admission, the earliest result is used.

**MIMIC-CDM-OW** (this work; Figure 3) adds the open world. It selects MIMIC-IV 2.2 admissions with an abdominal chief complaint whose principal diagnosis falls in one of nine non-CDM groups, excluding any admission that carries a CDM-condition code at any position. Each case is built with MIMIC-CDM's own text pipeline, so that format cannot reveal the source:
- the same history and physical-examination windows;
- the same radiology section filter;
- the same masking of the case's own diagnosis;
- the same inclusion rule;
- earliest-value deduplication under a total row order.

The build is deterministic, and rebuilds are byte-identical. It yields 713 OTHER cases and 648 same-pipeline in-set controls: admissions with a CDM condition as principal diagnosis, built by the same pipeline. Training on the controls removes the shortcut "built by the open-world pipeline, so OTHER".

**Cohorts** are patient-disjoint: no training case shares a patient with an evaluation case.

| Set | Cases | Use |
|---|---|---|
| RL train | 2,404 = 1,760 CDM + 279 OTHER + 365 controls | training |
| Dev | 212 = 160 CDM + 32 OTHER + 20 controls | monitoring only |
| CDM val + test | 240 + 240 | evaluation; never used for a training decision |
| OTHER, seen groups | 206 (bowel obstruction, gastroenteritis or colitis, GI bleeding, urolithiasis) | evaluation |
| OTHER, unseen groups | 187 (mesenteric ischaemia, perforated or bleeding ulcer, ruptured AAA, DKA, ectopic pregnancy) | evaluation; held out of training entirely |
| In-set controls | 258 | evaluation (source-shortcut check) |

The five time-critical groups never appear in training, so the open-world evaluation separates recognising a group seen in training from escalating an unfamiliar, dangerous presentation. Ruptured AAA and ectopic pregnancy have fewer than 10 cases each and are only reported pooled. Training cases are drawn from the source pools by weight: CDM 0.75, OTHER 0.15, controls 0.10.

## 6. Experimental setup

**Systems.** All share the base model, the environment and the evaluation protocol:

- zero-shot forced choice;
- zero-shot with prompted DEFER;
- the GRPO control (identical training without DEFER);
- the case-level group-consensus arm (the published mechanism: TIAR/KARL-style);
- DEFER-Dx with the counterfactual escalation value, two training seeds;
- an ablation: a constant deferral reward, against DEFER-Dx at 100 steps (the no-handoff ablation was not run within the compute budget);
- each forced-choice system plus a post-hoc threshold at DEFER-Dx's coverage, cross-fitted on val/test;
- each forced-choice system plus SGR (5% selective risk, δ = 0.05);
- self-consistency over three samples for every system.

DiagAgent-14B, LA-CDM and LDTL appear as context (§7.4).

**Metrics.** These fall into two families. Each is reported with a 95% case-level bootstrap interval; comparisons use paired bootstraps on the same cases (docs/METHODS.md §5).
- **Machine-learning benchmark:** accuracy at full coverage, mean-class accuracy, macro-F1, coverage, selective accuracy, AURC, accuracy at fixed coverage, ECE and Brier, OOD AUROC.
- **Clinical safety:**
  - minority-class accuracy;
  - unflagged errors (wrong commits per case) and confident errors (wrong commits with p > 0.8);
  - severity-weighted error;
  - false commits on never-seen time-critical diagnoses;
  - deferral precision and recall, and test cost.

## 7. Results

### 7.1 Closed world
All numbers are on CDM val + test (480 admissions, 3 evaluation seeds) with 95% case-level bootstrap intervals; differences are paired on the same cases (Figure 5, Figure 6, Figure S3; docs/RESULTS.md Tables 1, 2 and 5).

**Learned against post-hoc deferral on an identically trained model.** DEFER-Dx answered 77.6% (74.3–80.6) of in-set cases at 95.2% (93.4–96.8) accuracy. The GRPO control, trained identically but without DEFER and thresholded on its stated probability at DEFER-Dx's coverage (cross-fitted), answered 79.2% at 92.1% (89.9–94.2). The coverage difference was −1.6 points (−4.8 to 1.7), so the comparison is at matched coverage.
- **Accuracy on answered cases:** DEFER-Dx was 3.1 points (0.9 to 5.4; p = 0.007) more accurate.
- **Unflagged errors:** wrong commitments with no escalation fell from 6.2 to 3.8 per 100 cases (−2.5, −4.4 to −0.8; p = 0.007).
- **Calibration:** Brier score 0.047 against 0.066. Confident errors, AURC and ECE did not differ significantly.
- **Three-sample voting** for both systems (coverage 79.8% against 78.8%): the accuracy gap narrowed to 2.2 points (−0.9 to 5.5), and confident errors fell by 3.5 per 100 (−5.8 to −1.5).

**What it costs.** When a deferral is scored by the top of its handed-over differential, DEFER-Dx was less accurate than the control:
- **full coverage:** 78.2% against 82.4% (−4.2, −7.2 to −1.2);
- **diverticulitis:** 69.3% against 76.5% (−7.2, −15.5 to 0.0).

The reason is its handoffs: 68.4% of its in-set handoffs rank OTHER first, against 8.6% for the consensus arm and 27.6% for prompted deferral. Its in-set handoff quality is correspondingly lower (−20.4 points against the control's thresholded deferrals). DEFER-Dx also ordered fewer tests: 0.84 per case, against 1.48 for the control and 1.83 for the consensus arm, and 0.67 before deferring. It escalated early rather than investigating, so this run does not show the "investigate further" half of the mechanism.

**Against the deferral reward from prior work** (the identically trained case-level consensus arm):
- **Same:** accuracy on answered cases (+1.1, −0.5 to 2.9) and unflagged errors (−1.2, −2.6 to 0.2).
- **Lower coverage:** 77.6% against 82.6% (−5.1).
- **Lower when handoffs are counted:** full-coverage accuracy (−7.6) and diverticulitis accuracy (−16.3, −24.3 to −8.8).

**Against zero-shot Qwen3-8B.** Unflagged errors were 3.8 per 100 against 24.1 (forced choice) and 9.1 (prompted DEFER; paired −5.3, −7.5 to −3.3).

### 7.2 Open world
Figure 7, Figure S4; docs/RESULTS.md Tables 3a–3d.

**Never-seen time-critical diagnoses** (187 cases: mesenteric ischaemia, perforated or bleeding ulcer, DKA, and pooled AAA and ectopic pregnancy). DEFER-Dx escalated 87.0% (83.2–90.4) and committed to one of the four conditions in 6.8% (4.3–10.0). For comparison:
- **The control without a threshold:** 15.0% (paired −8.2, −12.1 to −4.6).
- **The consensus arm:** 11.4% (−4.6, −8.0 to −1.4).
- **Zero-shot:** 17.6% forced choice and 14.4% with prompted DEFER.
- **The thresholded control:** 8.4%, a difference of −1.6 (−5.0 to 1.6) that is not significant.
- **Context, not a head-to-head:** an untrained Qwen3-14B with a confidence threshold reaches 6.6% (docs/BASELINES.md §7).

On OTHER cases, DEFER-Dx's handoff contained OTHER 93.9% of the time, against 14.7% for the consensus arm.

**OTHER from groups seen in training** (206 cases). False commits were 6.5% for DEFER-Dx and for the thresholded control, against 13.4% for the consensus arm (paired −7.0, −9.9 to −4.4). DEFER-Dx escalated 89.0% of these cases and named OTHER for only 4.5%; the thresholded control named OTHER for 47.4%.

**Same-pipeline in-set controls** (258 cases). DEFER-Dx named OTHER for 0.3% and answered 70.0% at 88.7% accuracy, so its open-world behaviour is not the source shortcut.

**Separability.** The OOD AUROC uses the stated probability alone to separate OTHER from in-set cases. It is lower for DEFER-Dx (86.8) than for the control (92.6): its open-world safety comes from escalating, not from a better-separated confidence.

### 7.3 Risk-coverage and calibration
Figure 4, Figure S2.
- **Risk–coverage:** DEFER-Dx's curve lies on the consensus arm's and above the zero-shot curves. The area under it (AURC) is 0.052 (0.038–0.072), against 0.048 for the thresholded control and 0.045 for the consensus arm; neither difference is significant. At 70% coverage it is 96.5% accurate.
- **Calibration:** of its stated probabilities on commitments, ECE 0.042 (0.027–0.059) and Brier 0.047 (0.034–0.061), the lowest Brier score of all systems.

### 7.6 Ablations
[[Filled when the runs finish (Mon–Tue): DEFER-Dx at 100 steps against the constant deferral reward and against no scored handoff (Figure S5; docs/RESULTS.md Table 5).]]

### 7.7 Training-seed variance and training dynamics
[[Second DEFER-Dx training seed (Tue).]]

**Controller** (Figure 8). Under plain dual ascent, the coverage multiplier ν reached 4.6, 2.5 and 1.5 in three successive overshoots of the first DEFER-Dx run. After each overshoot, in-set deferral collapsed, and dev coverage swung between 48% and 88% depending on where an evaluation fell in the cycle. With the PI controller, the final run's ν peaked at 1.49 and did not wind up. In-set deferral still overshot the cap once, around steps 45–60, and dev coverage dipped to 37% at step 50; from step 75 on, it held at 69–75%.

### 7.4 Context against published systems
On the LA-CDM test split (240 cases), with deferrals counted through the top of their differential, mean-class accuracy is:
- **DEFER-Dx:** 76.4% (71.1–81.0);
- **the control:** 82.7%;
- **the consensus arm:** 86.3%;
- **zero-shot Qwen3-8B:** 74.4% when OTHER is offered, and 86.9% when restricted to the four labels.

LA-CDM reports 81.3% in its closed-world environment, which uses an LLM-written summary, 12 tests and a different base model. LDTL reports 93.4% case-weighted accuracy on its own unpublished split. Neither is a head-to-head comparison.

### 7.5 Robustness
**The diagnosis mask.** MIMIC-CDM replaces the case's own diagnosis with "____" in some sentences, which is a label cue. With every such sentence removed, DEFER-Dx answers 76.0% (72.8–79.2) of cases at 96.2% (94.4–97.6) accuracy, against 77.6% at 95.2%. Paired on the same cases, the differences are −1.0 points (−2.0 to 0.0) in accuracy on answered cases and +1.6 (−0.2 to 3.5) in coverage, and AURC is unchanged. The agent does not rely on the mask.

**Source shortcut.** On the 258 same-pipeline in-set controls, DEFER-Dx names OTHER for 0.3% of cases and answers 70.0% at 88.7% accuracy. Its open-world behaviour is therefore not "built by the open-world pipeline, so OTHER".

**Closed world.** Zero-shot Qwen3-8B restricted to the four labels, as in prior work, reaches 86.9% (83.1–90.4) mean-class accuracy on the LA-CDM test split under this protocol, against 74.4% when OTHER is offered.

## 8. Discussion
**What learned deferral buys.** Against a confidence threshold on an identically trained model at matched coverage, DEFER-Dx was 3.1 points more accurate on the cases it answered, and let 2.5 fewer errors per 100 cases through unflagged. The effect is modest, and with three-sample voting the accuracy gap is not significant. The larger differences are in the open world:
- it escalated 87% of never-seen time-critical presentations;
- the thresholded control escalated 18.5% and named OTHER for most of the rest;
- it committed wrongly far less often than the same model without a threshold, or than the deferral reward from prior work.

The counterfactual escalation value therefore produces an agent that escalates where a confidence threshold would answer.

**What it does not buy.**
- **Less specific handoffs:** the agent learned to escalate early (0.67 tests before deferring) and to label uncertain in-set cases "not one of the four". So its in-set handoffs are less specific than those of the consensus arm or the control, and counting them as diagnoses lowers full-coverage and diverticulitis accuracy.
- **Not separable from the handoff design:** whether this follows from the counterfactual itself, or from the OTHER value in the escalation reward (1.2), is what the no-handoff and constant-reward ablations (§7.6) bear on.
- **No more investigation:** fewer tests were ordered than by the comparators, so we do not claim that the agent learned to investigate more.

**Implications.**
- **Evaluate in the open world:** for deployment, the relevant comparison is with a thresholded agent, and it must include presentations outside the training label set, where the two approaches differ most.
- **Score the handoff on in-set cases:** handoffs need their own pressure toward in-set specificity, for example by scoring them against a clinician's differential, not only the recorded diagnosis.

**Limitations.**
- **Single centre, one organ system.** MIMIC (BIDMC), abdominal pain only.
- **Proxy labels:** principal ICD codes for OTHER cases, and CDM's curated labels.
- **Proxy cost and severity:** the severity matrix is a time-to-harm proxy, not clinician-built; some test prices are placeholders (marked in the catalog).
- **No clinician adjudication yet.**
- **Small samples:** two training seeds for DEFER-Dx and one for every other arm, and 51 diverticulitis cases in the evaluation set.
- **Single training seed for the comparators:** the control and the consensus arm have one seed each.
- **Shared, single GPU:** runs were interrupted by out-of-memory errors caused by other processes, and once by a machine restart; each resumed from its newest checkpoint. Memory settings changed between parts; sampling and the algorithm did not.
- **The environment reveals only what was recorded:** test availability is informative.

## 9. Data and code availability

Code: this repository. Data: PhysioNet credentialed access (MIMIC-IV 2.2, MIMIC-IV-Note 2.2, MIMIC-IV-Ext-CDM 1.1). Under the data use agreement, no MIMIC text was sent to any third-party service; all models ran on local weights. Model weights trained on MIMIC are released only through PhysioNet's credentialed channel.
