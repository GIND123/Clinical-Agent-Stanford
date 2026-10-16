# Training Clinical Agents to Ask, Test, Commit, and Defer

*Draft for a journal submission (and the source of the Stanford AI+HEALTH abstract). Figures: `docs/figures/` (vector PDFs; TikZ sources in `docs/figures/tikz/`), numbered and captioned in README §3. Every number is taken from `docs/RESULTS.md`. Methods are written from the code; see `docs/METHODS.md` for every constant.*

## Abstract
**Background.** Interactive diagnostic agents for acute abdominal pain are evaluated in a closed world: they must name one of four diagnoses and cannot hand a case to a clinician. Abstention learned by reinforcement learning is known to collapse into refusal, and theory recommends thresholding a calibrated confidence instead.

**Methods.** DEFER-Dx is a Qwen3-8B agent trained with multi-turn GRPO to examine, order any of 22 priced tests, commit to a diagnosis with a stated probability, or defer to a clinician with a probabilistic differential. Each deferral is credited by a counterfactual escalation value: continuations branched from the state where the agent deferred, with deferral disabled, estimate the clinical value of continuing, and the handoff is scored with a strictly proper rule. A PI controller caps in-set deferral at 30%. We evaluate on 480 held-out MIMIC-IV-Ext-CDM admissions and on MIMIC-CDM-OW, a new open-world set built with the same text pipeline, with five time-critical diagnoses held out of training.

**Results.** DEFER-Dx answered 77.6% of in-set cases at 95.2% accuracy (95% CI 93.4–96.8), with 3.8 unflagged errors per 100 cases (zero-shot: 24.1). Against an identically trained agent without deferral whose confidence was thresholded at matched coverage, accuracy on answered cases was higher in all four pairings of two training seeds per arm (by 1.1 to 5.3 points; pooled 3.2, 1.6 to 5.1), with 2.4 fewer unflagged errors per 100 (1.1 to 3.9). On 187 time-critical admissions never seen in training, the two seeds escalated 87% of cases and committed wrongly in 6.8% and 7.1%, against 11.4% for an identically trained case-level deferral reward from prior work (paired −4.6, −8.0 to −1.4; and −4.3, −7.7 to −1.1). What the handoffs named differed between seeds.

**Conclusions.** Crediting escalation against the counterfactual of continuing trains a stable deferral policy that commits wrongly on unfamiliar, dangerous presentations less often than a case-level deferral reward. A confidence threshold on an identically trained agent recovers part of this benefit on familiar cases, but how much varies between training runs.

## 1. Introduction

A patient arrives at the emergency department with abdominal pain. An interactive diagnostic agent asks for the examination, orders a CT and names a diagnosis. Every such agent published on the MIMIC-IV clinical-decision-making benchmark (MIMIC-CDM; Hager et al. 2024) must name one of four conditions (appendicitis, cholecystitis, diverticulitis or pancreatitis), even when the patient has a perforated ulcer or mesenteric ischaemia, and even when its own evidence is split. None can hand the case back.

Three developments make this gap pressing. Regulators expect clinical decision support to leave the decision with a clinician who can independently review the basis of its recommendations rather than rely on them (FDA Clinical Decision Support Software guidance, revised January 2026), and a randomised study of 225 physicians has just shown how fragile that safeguard is: conversational AI with a concealed objective raised harmful decisions from 2% to 49%, and clinical expertise did not protect (Barrit et al. 2026). The field's own leaderboard shows the cost of optimising accuracy alone: the strongest reported MIMIC-CDM agent (LDTL) raises case-weighted accuracy to 93.4% while its accuracy on the minority condition, diverticulitis, falls to 78.8%, below a random test-ordering policy (90.4%). And recent theory argues that the obvious fix, a learned "abstain" action trained by reinforcement learning, collapses into refusing everything, and that one should instead threshold a calibrated confidence report after training (Che et al. 2026).

We ask whether an agent can *learn* when to defer, without collapsing, and whether learned deferral beats thresholding a confidence report produced by an identically trained agent. Our contributions:

1. **DEFER-Dx.** An agent trained with multi-turn group-relative policy optimisation over four actions: ask, order one of 22 individual tests, commit with a stated probability, or defer to a clinician with a ranked differential.
2. **The counterfactual escalation value.** At each state where the agent escalates, forced continuations branched from that exact state estimate the clinical value of not escalating, by deciding now or investigating further. The escalation is credited with the difference, so that the agent can learn to separate uncertainty that more testing would resolve from uncertainty that warrants a handoff. Prior deferral rewards price escalation by case-level difficulty (TIAR, KARL, AWA-RL) or by designated cases (TrustMed-RL). The handoff is a probabilistic differential trained with a strictly proper scoring rule.
3. **An open-world benchmark (MIMIC-CDM-OW).** 713 abdominal-pain admissions whose principal diagnosis is none of the four, built with MIMIC-CDM's own text pipeline, and 648 same-pipeline in-set controls. Time-critical diagnosis groups are held out of training entirely.
4. **A head-to-head test of learned versus post-hoc deferral.** The comparison is at matched coverage, with paired bootstrap inference, against an identically trained no-deferral agent whose confidence is thresholded: exactly the repair theory recommends.

**Findings in brief.** Learned deferral was stable, and on never-seen time-critical presentations it committed wrongly less often than the deferral reward from prior work, in both training seeds. Against thresholding an identically trained agent at matched coverage, it was more accurate on the cases it answered in every pairing of two training seeds per arm (pooled +3.2 points). The size of that margin depended mainly on the control's seed. A threshold is only as good as the confidence it cuts, and identically trained controls differed in how well their stated probabilities ranked their errors; DEFER-Dx's accuracy on answered cases did not (95.2% and 95.3%).

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
- the GRPO control (identical training without DEFER), two training seeds;
- the case-level group-consensus arm (the published mechanism: TIAR/KARL-style);
- DEFER-Dx with the counterfactual escalation value, two training seeds;
- two ablations against DEFER-Dx at 100 steps: a constant deferral reward, and no handoff score (η = 0);
- each forced-choice system plus a post-hoc threshold at DEFER-Dx's coverage, cross-fitted on val/test (each DEFER-Dx seed is compared with each control seed thresholded at that DEFER-Dx seed's coverage);
- each forced-choice system plus SGR (5% selective risk, δ = 0.05);
- self-consistency over three samples for every system.

DiagAgent-14B, LA-CDM and LDTL appear as context (§7.4).

**Matching coverage.** Stated probabilities are coarse (0.80, 0.85, …): for the control, 0.80 alone covers 14–17% of cases, so a plain cut jumps from 64% to 78–81% coverage. A post-hoc threshold therefore keeps each commitment tied at the cut with a probability fitted to reach the target coverage, decided by a fixed hash of the case and evaluation seed. The threshold and that probability are fitted on one split and applied to the other. We report one pre-specified draw and the range of the point difference over 20 draws.

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

**Learned against post-hoc deferral on an identically trained model.** DEFER-Dx answered 77.6% (74.3–80.6) of in-set cases at 95.2% (93.4–96.8) accuracy. The GRPO control (first training seed), trained identically but without DEFER and thresholded on its stated probability at DEFER-Dx's coverage (cross-fitted; §6), answered 77.7% at 92.9% (90.7–94.9). The coverage difference was −0.1 points (−3.3 to 3.1).
- **Accuracy on answered cases:** DEFER-Dx was 2.3 points (0.1 to 4.6; p = 0.037) more accurate; over 20 tie-break draws, +2.2 to +2.9.
- **Unflagged errors:** wrong commitments with no escalation fell from 5.6 to 3.8 per 100 cases (−1.8, −3.5 to −0.1; p = 0.046); over the draws, −2.3 to −1.7.
- **Calibration:** Brier score 0.047 against 0.061. Confident errors, AURC and ECE did not differ significantly.
- **Three-sample voting** for both systems (coverage 79.8% against 78.5%): the accuracy gap was 2.2 points (−0.9 to 5.5), not significant, and confident errors fell by 3.5 per 100 (−5.8 to −1.5).
- **Across training seeds** (§7.7, Figure S6): with two seeds per arm, the four pairings gave +1.1 to +5.3 points, and three of the four intervals exclude 0. Pooled over them, accuracy on answered cases was +3.2 (1.6 to 5.1) and unflagged errors −2.4 per 100 (−3.9 to −1.1).

The in-set advantage over an identically trained, thresholded model is therefore consistent in sign but varies in size, mostly with the control's training seed.

**What it costs, in this seed.** When a deferral is scored by the top of its handed-over differential, DEFER-Dx was less accurate than the control:
- **full coverage:** 78.2% against 82.4% (−4.2, −7.2 to −1.2);
- **diverticulitis:** 69.3% against 76.5% (−7.2, −15.5 to 0.0).

The reason is its handoffs: 68.4% of its in-set handoffs rank OTHER first, against 8.6% for the consensus arm and 27.6% for prompted deferral. Its in-set handoff quality is correspondingly lower (−19.3 points, −26.8 to −11.7, against the control's thresholded deferrals). The second training seed does not pay this cost (§7.7). DEFER-Dx also ordered fewer tests: 0.84 per case, against 1.48 for the control and 1.83 for the consensus arm, and 0.67 before deferring. It escalated early rather than investigating, so this run does not show the "investigate further" half of the mechanism.

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
- **The thresholded control:** 7.5%, a difference of −0.7 (−3.7 to 2.3) that is not significant. It escalated 21.0% of these cases and committed to OTHER in 64.3%.
- **Context, not a head-to-head:** an untrained Qwen3-14B with a confidence threshold reaches 6.6% (docs/BASELINES.md §7).

On OTHER cases, DEFER-Dx's handoff contained OTHER 93.9% of the time, against 14.7% for the consensus arm.

**OTHER from groups seen in training** (206 cases). False commits were 6.5% for DEFER-Dx and 6.1% for the thresholded control (paired +0.3, −2.1 to 2.8), against 13.4% for the consensus arm (paired −7.0, −9.9 to −4.4). DEFER-Dx escalated 89.0% of these cases and named OTHER for only 4.5%; the thresholded control named OTHER for 45.0%.

**Same-pipeline in-set controls** (258 cases). DEFER-Dx named OTHER for 0.3% and answered 70.0% at 88.7% accuracy, so its open-world behaviour is not the source shortcut.

**Separability.** The OOD AUROC uses the stated probability alone to separate OTHER from in-set cases. It is lower for DEFER-Dx (86.8) than for the control (92.6): its open-world safety comes from escalating, not from a better-separated confidence.

### 7.3 Risk-coverage and calibration
Figure 4, Figure S2.
- **Risk–coverage:** DEFER-Dx's curve lies on the consensus arm's and above the zero-shot curves. The area under it (AURC) is 0.052 (0.038–0.072), against 0.047 for the thresholded control and 0.045 for the consensus arm; neither difference is significant. At 70% coverage it is 96.5% accurate.
- **Calibration:** of its stated probabilities on commitments, ECE 0.042 (0.027–0.059) and Brier 0.047 (0.034–0.061), the lowest Brier score of all systems.

### 7.4 Context against published systems
On the LA-CDM test split (240 cases), with deferrals counted through the top of their differential, mean-class accuracy is:
- **DEFER-Dx:** 76.4% (71.1–81.0), and 85.2% (80.8–89.3) for its second training seed, whose handoffs rarely rank OTHER first;
- **the control:** 82.7%;
- **the consensus arm:** 86.3%;
- **zero-shot Qwen3-8B:** 74.4% when OTHER is offered, and 86.9% when restricted to the four labels.

LA-CDM reports 81.3% in its closed-world environment, which uses an LLM-written summary, 12 tests and a different base model. LDTL reports 93.4% case-weighted accuracy on its own unpublished split. Neither is a head-to-head comparison.

### 7.5 Robustness
**The diagnosis mask.** MIMIC-CDM replaces the case's own diagnosis with "____" in some sentences, which is a label cue. With every such sentence removed, DEFER-Dx answers 76.0% (72.8–79.2) of cases at 96.2% (94.4–97.6) accuracy, against 77.6% at 95.2%. Paired on the same cases, the differences are −1.0 points (−2.0 to 0.0) in accuracy on answered cases and +1.6 (−0.2 to 3.5) in coverage, and AURC is unchanged. The agent does not rely on the mask.

**Source shortcut.** On the 258 same-pipeline in-set controls, DEFER-Dx names OTHER for 0.3% of cases and answers 70.0% at 88.7% accuracy. Its open-world behaviour is therefore not "built by the open-world pipeline, so OTHER".

**Closed world.** Zero-shot Qwen3-8B restricted to the four labels, as in prior work, reaches 86.9% (83.1–90.4) mean-class accuracy on the LA-CDM test split under this protocol, against 74.4% when OTHER is offered.

### 7.6 Ablations
Both ablations are trained for 100 steps and compared with DEFER-Dx's step-100 checkpoint, evaluated on 2 seeds (Figure S5; docs/RESULTS.md Table 5).

**Counterfactual credit against a constant deferral reward.** The constant reward is matched to the consensus reward's crossover at the current τ, and trained with the same PI coverage controller. At 100 steps:
- **No difference:** accuracy on answered cases (94.5% against 94.9%; −0.4, −2.3 to 1.5), unflagged errors (3.8 against 4.0 per 100), and confident errors.
- **Never-seen time-critical diagnoses:** wrong commitments do not differ either (10.2% against 10.7%).
- **Where the credit differs:**
  - it defers more: coverage 68.1% against 77.5%, and 77.8% against 62.8% of never-seen time-critical cases;
  - its handoffs name OTHER far more often on those cases (89.0% against 24.7%);
  - it makes fewer false commits on OTHER from seen groups (−4.4, −7.5 to −1.5);
  - its calibration is better (ECE −0.028, −0.045 to 0.000).
- **What it costs:** lower full-coverage accuracy (−6.5, −9.0 to −4.0) and diverticulitis accuracy (−11.8).

In this seed, then, the OTHER-leaning handoffs came with the counterfactual credit, not with deferral reward as such. They are not a necessary consequence of it: the second DEFER-Dx seed ranks OTHER first in 4.2% of in-set handoffs at 150 steps (§7.7). On the main in-set metrics at 100 steps, the credit is not distinguishable from a well-matched constant reward. DEFER-Dx's advantage on never-seen time-critical diagnoses at 150 steps (6.8%) appeared in the second half of training.

**Without the scored handoff (η = 0).** The same training with the handoff score removed from the escalation value, so a deferral is credited whatever its differential says. At 100 steps:
- **No difference:** accuracy on answered cases (94.5% against 95.6%; −1.1, −2.8 to 0.6), unflagged errors (3.8 against 3.3 per 100) and confident errors.
- **Where the handoff score differs:**
  - its in-set handoffs are more informative: handoff quality +6.5 points (3.5 to 9.3);
  - it defers more: coverage 68.1% against 75.9% (−7.8, −10.9 to −5.0);
  - it makes fewer false commits on OTHER from seen groups (6.3% against 9.5%; −3.2, −6.3 to −0.2). On never-seen time-critical diagnoses, 10.2% against 12.8% (−2.7, −5.9 to 0.5) is not significant.
- **What it does not cause:** without the score, handoffs rank OTHER first more often (55.1% of in-set handoffs, against 35.1%).

The scored handoff therefore does what it was designed for: it makes handoffs more informative, and it raises the value of escalating, so the agent defers more. The OTHER-leaning handoffs come with the counterfactual credit itself, with or without the score (35.1% and 55.1% at 100 steps, against 12.8% for a constant reward), and how far they develop depends on the seed (§7.7).

### 7.7 Training-seed variance and training dynamics
**Second training seed.** A second DEFER-Dx run differs only in its seed (case order, sampling and LoRA initialisation) and was evaluated in the same way. Against the control, it is compared at its own coverage (73.2%; the control thresholded to 72.1%).
- **What replicates:**
  - accuracy on answered cases of 95.3% (93.2–97.1), at 73.2% coverage, with 3.5 unflagged errors per 100 (first seed: 95.2% at 77.6%, and 3.8);
  - 87.0% of never-seen time-critical cases escalated, and 7.1% (4.3–10.3) committed wrongly (first seed: 87.0% and 6.8%);
  - against the consensus arm, fewer false commits on never-seen time-critical cases (−4.3, −7.7 to −1.1) and on OTHER from seen groups (−6.6, −9.9 to −3.6), at the same accuracy on answered cases (+1.2, −0.4 to 2.9);
  - few tests: 0.89 per case.
- **What varies:** the in-set advantage over the thresholded control. Against the first control seed, accuracy on answered cases was +1.1 (−0.8 to 3.3) and unflagged errors −0.8 per 100 (−2.4 to 0.8), neither significant (over 20 tie-break draws, +1.1 to +2.2 and −1.7 to −0.8). Against the second control seed, the differences were +4.1 (1.9 to 6.5) and −3.0 (−4.8 to −1.3).
- **What differs: the handoffs.**
  - Only 4.2% of this seed's in-set handoffs rank OTHER first (first seed: 68.4%). Counting handoffs, its full-coverage accuracy is 86.3% (83.6–88.9): above the control (+4.0, 1.6 to 6.5), level with the consensus arm (+0.5, −1.3 to 2.4), and 83.0% on diverticulitis.
  - On OTHER cases, however, its handoff contains OTHER for only 35.2% of never-seen time-critical cases (first seed: 93.9%). It escalates them just as often but hands over a differential of in-set diagnoses.
  - Its stated probabilities are less calibrated (ECE 0.059 against 0.042).

Escalation behaviour and the advantage over the prior deferral reward replicate. The in-set gain over a threshold varies in size, and what the handoff names depends on the seed. The handoff score (η = 0.3) does not pin down which kind of differential the agent learns. Two seeds show that the variation exists but cannot estimate its size.

**Second training seed of the control.** The GRPO control was retrained with seed 1, otherwise identically, and evaluated in the same way.
- **Similar:** accuracy at full coverage, 81.7% against 82.4%, and mean-class accuracy on the LA-CDM test split, 81.3% against 82.7%.
- **Different:**
  - it committed on every in-set case, where the first seed left 6.2% of episodes without a commitment (malformed or out of budget), which count as not answered;
  - its stated probabilities ranked its errors worse: AURC 0.074 against 0.044, Brier 0.130 against 0.091, and 7.0 confident errors per 100 against 2.5;
  - it ordered more tests: 2.18 against 1.48 per case.
- **So its threshold recovers less:** at DEFER-Dx's coverage it was 89.9% accurate on answered cases, against 92.9% for the first control seed and 95.2% for DEFER-Dx.

**All four pairings** of DEFER-Dx and control seeds (Figure S6; docs/RESULTS.md Table 5):
- **Same direction everywhere:** DEFER-Dx was more accurate on answered cases (+1.1 to +5.3; three of the four intervals exclude 0) and let fewer errors through unflagged (−0.8 to −4.1).
- **Pooled over the pairings**, with cases resampled and the training runs held fixed:
  - accuracy on answered cases: +3.2 points (1.6 to 5.1);
  - unflagged errors: −2.4 per 100 (−3.9 to −1.1);
  - confident errors: −2.1 per 100 (−3.3 to −1.0);
  - AURC: lower, −0.014 (−0.031 to −0.005);
  - ECE: higher, +0.036 (0.014 to 0.050).
- **No difference:** full-coverage accuracy (+0.2) and wrong commitments on never-seen time-critical cases (−1.0, −3.3 to 1.2).

The pooled interval reflects case sampling only. With two runs per arm, training-seed variance is shown by the pairings, not estimated.

**Controller** (Figure 8). Under plain dual ascent, the coverage multiplier ν reached 4.6, 2.5 and 1.5 in three successive overshoots of the first DEFER-Dx run. After each overshoot, in-set deferral collapsed, and dev coverage swung between 48% and 88% depending on where an evaluation fell in the cycle. With the PI controller, the final run's ν peaked at 1.49 and did not wind up. In-set deferral still overshot the cap once, around steps 45–60, and dev coverage dipped to 37% at step 50; from step 75 on, it held at 69–75%.

## 8. Discussion
**What learned deferral buys.** On familiar cases, a margin over a threshold that is consistent in sign but variable in size. Against a confidence threshold on an identically trained model at matched coverage, DEFER-Dx was more accurate on the cases it answered in all four pairings of two training seeds per arm (pooled +3.2 points, 1.6 to 5.1). The margin depended mainly on the control's seed:
- **The threshold varied:** identically trained controls differed in how well their stated probabilities ranked their errors (AURC 0.044 and 0.074).
- **DEFER-Dx did not:** its accuracy on answered cases was the same in both runs (95.2% and 95.3%), though its coverage and handoffs were not.

Che et al. are right that a proper-scored confidence makes thresholding a strong baseline. With one control run it can look nearly as good as learned deferral (+1.1 to +2.3); with another, clearly worse (+4.1 to +5.3). Two runs per arm show this variation but cannot estimate it.

What replicates across both seeds is the comparison with the deferral reward from prior work. At the same accuracy on answered cases, DEFER-Dx committed wrongly less often:
- on never-seen time-critical diagnoses: −4.6 and −4.3 points;
- on OTHER from groups seen in training: −7.0 and −6.6.

On never-seen time-critical presentations it also behaves differently from the threshold:
- **DEFER-Dx** escalated 87% of them, in both seeds.
- **The thresholded control** committed wrongly about as rarely (7.5%), but it escalated only 21.0% and committed to "not one of the four" for 64.3%.

A committed OTHER label is not a handoff: the clinician receives no differential and no flag that the case was uncertain. The counterfactual escalation value therefore produces an agent that escalates where a confidence threshold would answer "none of these".

**What it does not buy.**
- **Reliable handoff content:** the first seed labelled uncertain in-set cases "not one of the four" (68.4% of in-set handoffs), which lowers full-coverage and diverticulitis accuracy when handoffs are counted as diagnoses. The second did not (4.2%), but it rarely named OTHER when handing over a true OTHER case. A constant deferral reward with the same OTHER value and controller produced neither pattern at 100 steps, and removing the handoff score made OTHER-first handoffs more common (§7.6). The score makes handoffs more informative but does not fix which differential the agent learns, so handoffs need stronger pressure toward the right content.
- **More investigation:** both seeds ordered fewer tests than the comparators (0.84 and 0.89 per case; 0.67 before deferring in the first), so we do not claim that the agent learned to investigate more. The "another test would settle it" half of the mechanism is not shown.

**Implications.**
- **Compare with a threshold on the same model:** learned deferral should be compared with a calibrated threshold on an identically trained model, at exactly matched coverage. Coarse stated probabilities make an unmatched comparison easy to report by accident: with a plain cut, our first-seed advantage read 3.1 points and our second-seed advantage 3.2. The comparator also needs more than one training run: our two control seeds gave margins of +2.3 and +5.3 against the same DEFER-Dx run.
- **Evaluate in the open world:** the two approaches differ most outside the training label set, in whether an unfamiliar case is escalated or labelled.
- **Score the handoff more strongly:** for example against a clinician's differential, not only the recorded diagnosis.

**Limitations.**
- **Single centre, one organ system.** MIMIC (BIDMC), abdominal pain only.
- **Proxy labels:** principal ICD codes for OTHER cases, and CDM's curated labels.
- **Proxy cost and severity:** the severity matrix is a time-to-harm proxy, not clinician-built; some test prices are placeholders (marked in the catalog).
- **No clinician adjudication yet.**
- **Small samples:** two training seeds each for DEFER-Dx and the control, which differ in handoff content and in how well the control's confidence thresholds, and 51 diverticulitis cases in the evaluation set.
- **One training seed for the consensus arm:** its comparisons, including the open-world advantage that replicated across DEFER-Dx seeds, rest on one run of the comparator.
- **Coarse confidences:** post-hoc thresholds reach the target coverage only with randomised tie-breaking at the cut; we report the range over 20 draws.
- **Shared, single GPU:** runs were interrupted by out-of-memory errors caused by other processes, and once by a machine restart; each resumed from its newest checkpoint. Memory settings changed between parts; sampling and the algorithm did not.
- **The environment reveals only what was recorded:** test availability is informative.

## 9. Data and code availability

Code: this repository. Data: PhysioNet credentialed access (MIMIC-IV 2.2, MIMIC-IV-Note 2.2, MIMIC-IV-Ext-CDM 1.1). Under the data use agreement, no MIMIC text was sent to any third-party service; all models ran on local weights. Model weights trained on MIMIC are released only through PhysioNet's credentialed channel.
