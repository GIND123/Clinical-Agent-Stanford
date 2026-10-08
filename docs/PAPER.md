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

[[From docs/METHODS.md §1: the POMDP over a single admission; hidden state y ∈ D ∪ {OTHER}; reveal-on-request observations from the record; the action table; the 8-investigation budget.]] Figure 1 (`docs/figures/fig_overview.pdf`).

## 4. Method

### 4.1 Rewards
[[docs/METHODS.md §2: the counterfactual escalation value (branching, V̂(s), E, the state-level advantage); the scored handoff; the case-level consensus comparator arm and why it is prior art (TIAR is identical at τ = 0.5).]] Figure 2 (`fig_cev.pdf`).

[[docs/METHODS.md §2: commit reward; group-consensus deferral reward; the τ → crossover table (τ = 0.85 ⇒ defer when the estimated success probability is below 0.645); the open-world deferral reward; the coverage constraint.]]

**Why this design avoids the collapse of Che et al.** Their collapse needs three things: (i) abstention scored 0 while blanket answering loses in expectation; (ii) a KL anchor whose restoring force shares the abstain gate; (iii) group std normalisation in the sparse-answer regime. DEFER-Dx removes each:

- **(i)** The deferral reward depends on the case. On cases the group solves (p̂ ≥ τ) deferring scores below committing, so blanket deferral is not optimal for any policy that solves some cases.
- **(ii)** No KL anchor is used. Coverage is instead held by an explicit Lagrangian constraint on the in-set deferral rate.
- **(iii)** Advantages are centred but not std-normalised (Dr. GRPO), so the designed penalties keep their scale.

Every commitment still carries a proper-scored probability, so the agent can also be thresholded after training: DEFER-Dx contains the recommended repair as a special case.

### 4.2 Training
[[docs/METHODS.md §4: Qwen3-8B + LoRA; colocated vLLM/PEFT GRPO on one GPU; turn-level samples; truncated importance sampling; curricula; the PI coverage controller and why plain dual ascent failed under CEV.]] Figure S1 (`fig_training.pdf`); Figure 8 (`fig_training_dynamics.pdf`).

## 5. Data
[[docs/METHODS.md §3, with the cohort table and the parity rules that keep OTHER cases indistinguishable from CDM cases by format.]] Figure 3 (`fig_data.pdf`).

## 6. Experimental setup

**Systems.** All share the base model, the environment and the evaluation protocol:

- zero-shot forced choice;
- zero-shot with prompted DEFER;
- the GRPO control (identical training without DEFER);
- the case-level group-consensus arm (the published mechanism: TIAR/KARL-style);
- DEFER-Dx with the counterfactual escalation value;
- ablations: no scored handoff; constant deferral reward;
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
[[Table 1 (ML) and Table 2 (clinical) from docs/RESULTS.md. Lead with the paired matched-coverage comparison of DEFER-Dx against the thresholded control.]] Figure 5 (`fig_paired_differences.pdf`), Figure 6 (`fig_clinical_safety.pdf`), Figure S3 (`fig_per_class.pdf`).

### 7.2 Open world
[[Tables 3a–3d: seen vs never-seen groups, the per-group breakdown, and the controls.]] Figure 7 (`fig_open_world.pdf`), Figure S4 (`fig_unseen_groups.pdf`).

### 7.3 Risk-coverage and calibration
Figure 4 (`fig_risk_coverage.pdf`), Figure S2 (`fig_reliability.pdf`). Ablations: Figure S5 (`fig_ablations.pdf`).

### 7.4 Context against published systems
[[Table 4. LA-CDM rows are on the same test split but in a different environment; LDTL numbers come from its own unpublished split and are case-weighted. Neither is a head-to-head.]]

### 7.5 Robustness
[[The mask ablation (drop every sentence containing CDM's "____" diagnosis mask) and the source-shortcut check on same-pipeline controls.]]

## 8. Discussion
[[What learned deferral buys over thresholding, stated only as strongly as the paired intervals allow; when it does not.]]

**Limitations.**
- **Single centre, one organ system.** MIMIC (BIDMC), abdominal pain only.
- **Proxy labels:** principal ICD codes for OTHER cases, and CDM's curated labels.
- **Proxy cost and severity:** the severity matrix is a time-to-harm proxy, not clinician-built; some test prices are placeholders (marked in the catalog).
- **No clinician adjudication yet.**
- **Small samples:** single training seed per arm (see the seed replicate, if run), and 51 diverticulitis cases in the evaluation set.
- **The environment reveals only what was recorded:** test availability is informative.

## 9. Data and code availability

Code: this repository. Data: PhysioNet credentialed access (MIMIC-IV 2.2, MIMIC-IV-Note 2.2, MIMIC-IV-Ext-CDM 1.1). Under the data use agreement, no MIMIC text was sent to any third-party service; all models ran on local weights. Model weights trained on MIMIC are released only through PhysioNet's credentialed channel.
