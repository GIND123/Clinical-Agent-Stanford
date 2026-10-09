# Abstract draft: Stanford AI+HEALTH 2026

Status: **DRAFT. Every `[[...]]` is a placeholder filled only from docs/RESULTS.md / docs/results.json once the final run finishes** (DEFER-Dx around Oct 10, 16:00 IST; the no-DEFER control and the paired comparison around Oct 11, 10:30 IST). Nothing below is a result until its placeholder is replaced by a number from that file.

## What the submission form asks for

Read from the call ([aimi.stanford.edu/aihealth26](https://aimi.stanford.edu/aihealth26)) on 2026-10-09.

| | |
|---|---|
| Format | **Lightning talk (5 minutes)** |
| Deadline | **Thursday, October 15, 2026, 11:59 pm PT** (Friday Oct 16, 12:29 IST) |
| Where | The Google Form linked from the call |
| Fields | talk title · 1–2 sentence description · **abstract, up to 300 words** · focus area · work stage · relevance to attendees · collaborators / co-authors · previous or upcoming presentations and publications |
| Focus areas (pick one) | The State of Health AI · AI in Clinical Practice · Implementing AI in Healthcare · Foundations in Health AI |

## Title options
1. **Training Clinical Agents to Ask, Test, Commit, and Defer**
2. Clinical Agents That Learn When to Hand a Case Back
3. Learned Deferral for Interactive Diagnostic Agents under an Open-World Label Space

## 1–2 sentence description

DEFER-Dx is a reinforcement-learning diagnostic agent that asks, orders tests, commits to a diagnosis or hands the case to a clinician, and is trained to escalate only where continuing would do worse. On emergency admissions from MIMIC-IV, including dangerous diagnoses it never saw in training, we test whether learned escalation beats thresholding a model's own confidence.

## Abstract (up to 300 words)

The previous draft was 318 words before any numbers. With each outcome's Results and Conclusion, and numbers in place of the placeholders, this draft is about 294 words (A), 270 (B) or 272 (C), counting the section labels. The optional agreement-gate sentence (about 20 words) fits only with B or C, or after trimming A. Re-count after filling it in: `python3 -c "import sys; print(len(sys.stdin.read().split()))"` on the pasted text.

**Background.** Regulators increasingly expect clinical AI to escalate uncertain cases; the FDA's Digital Health Advisory Committee (Nov 2025) emphasised timely human intervention for generative-AI devices. Yet published interactive diagnostic agents are closed-world and forced-choice: they must name one of a fixed set of diagnoses and cannot hand a case back. On the MIMIC-IV clinical-decision-making benchmark, the strongest reported agent reaches 93.4% accuracy but 78.8% on diverticulitis, below a random test-ordering policy.

**Methods.** DEFER-Dx is a Qwen3-8B agent trained with multi-turn reinforcement learning (GRPO) to examine, order any of 22 priced tests, commit to a diagnosis with a stated probability, or defer to a clinician with a probabilistic differential. Deferral is trained against a counterfactual: at the state where the agent escalates, continuations branched from that state with escalation disabled estimate the clinical value of deciding now or testing further, and escalation is credited only where a handoff beats continuing. Commitments are scored for accuracy, calibration, error severity and test cost; handoffs with a proper scoring rule. We train on MIMIC-IV-Ext-CDM (2,400 admissions, four abdominal conditions) and a new open-world extension of 713 admissions with other diagnoses, holding five time-critical diagnoses out of training.

**Results.** *Pick the version that matches the paired result; see the next section.*

**Conclusion.** *Pick the version that matches.*

## Results and Conclusion, written for each outcome

The outcome is decided by the paired difference `cev vs grpo_nodefer+thr` in `docs/results.json` (`selective_acc` and `unsafe_errors`, same 480 cases). Use only what its interval supports.

**A: learned deferral beats the threshold** (the interval for the selective-accuracy difference excludes 0).
> **Results.** On 480 held-out admissions, DEFER-Dx answered [[cov]]% of cases at [[sel]]% accuracy (95% CI [[sel_lo]]–[[sel_hi]]), against [[thr_sel]]% for an identically trained agent without deferral whose confidence was thresholded at matched coverage (paired difference [[d]] points, [[d_lo]] to [[d_hi]]). Unflagged errors fell from [[thr_unfl]] to [[unfl]] per 100 cases, and diverticulitis accuracy was [[div]]%. On time-critical diagnoses never seen in training, it committed to a wrong condition in [[f]]% of cases, against [[g]]%.
>
> **Conclusion.** Escalation credited against the counterfactual of continuing gives a safer diagnostic agent than thresholding an identically trained model's confidence, including on presentations it never saw in training.

**B: a tie on accuracy, better on safety** (the accuracy interval includes 0, but the unflagged-error or time-critical difference excludes 0).
> **Results.** On 480 held-out admissions, DEFER-Dx answered [[cov]]% of cases at [[sel]]% accuracy, matching a confidence threshold on an identically trained agent at the same coverage ([[d]] points, [[d_lo]] to [[d_hi]]). It reduced [[unflagged errors from [[thr_unfl]] to [[unfl]] per 100 cases / wrong commitments on never-seen time-critical diagnoses from [[g]]% to [[f]]%]].
>
> **Conclusion.** Learned escalation matches confidence thresholding on accuracy while [[the safety gain]]. Evaluating agents in an open world with held-out time-critical diagnoses exposes failure modes that forced-choice benchmarks miss.

**C: no advantage over the threshold.** Then lead with what training and the benchmark do show.
> **Results.** Reinforcement learning with deferral reduced unflagged errors from 24.1 per 100 cases (zero-shot Qwen3-8B) to [[unfl]], at [[sel]]% accuracy on the [[cov]]% of cases answered. A confidence threshold on an identically trained agent performed similarly ([[d]] points, [[d_lo]] to [[d_hi]]).
>
> **Conclusion.** Training an agent to escalate makes it substantially safer than zero-shot agents, but a calibrated threshold on an equally trained model remains a strong baseline. Open-world evaluation with held-out time-critical diagnoses is needed to tell them apart.

**Optional sentence, if word count allows** (from `scripts/agreement_gate.py`, applied to every system alike). Escalating the cases on which three samples disagree further reduced missed time-critical diagnoses to [[gate_f]]%, at [[gate_cov]]% coverage.

### Where each number comes from (`docs/results.json`)

| Placeholder | Key | Note |
|---|---|---|
| `[[cov]]`, `[[sel]]` (`[[sel_lo]]`–`[[sel_hi]]`) | `tables.cev.ml.coverage`, `tables.cev.ml.selective_acc` | × 100, one decimal |
| `[[thr_sel]]` | `tables["grpo_nodefer+thr"].ml.selective_acc` | the control, thresholded at DEFER-Dx's coverage (cross-fitted) |
| `[[d]]` (`[[d_lo]]` to `[[d_hi]]`) | `paired["cev vs grpo_nodefer+thr"].selective_acc` | `diff`, `lo`, `hi`; also report `p` |
| `[[unfl]]`, `[[thr_unfl]]` | `tables.cev.clinical.unsafe_errors`, `tables["grpo_nodefer+thr"].clinical.unsafe_errors` | × 100 = per 100 cases |
| `[[div]]` | `tables.cev.clinical.acc_diverticulitis` | 51 cases; quote with its interval in the talk |
| `[[f]]`, `[[g]]` | `tables.cev.clinical.unseen_false_commit`, `tables["grpo_nodefer+thr"].clinical.unseen_false_commit` | the 187 never-seen time-critical cases |
| against the published mechanism | `paired["cev vs deferdx"]` | the consensus arm; CEV must beat it for the novelty claim |

The 24.1 in version C is `tables.zs_nodefer.clinical.unsafe_errors`, already in the file.

## Other form fields

**Focus area.** *Foundations in Health AI* (a new training method and benchmark). Alternative: *AI in Clinical Practice*, if the talk leads with the safety results.

**Work stage.** Research. A retrospective evaluation on de-identified MIMIC-IV data; not deployed.

**Relevance to attendees.** A clinical AI system that cannot say "a clinician should take this" is a deployment risk. The talk shows how to train and evaluate diagnostic agents that know when to hand a case back. It measures unflagged errors, missed time-critical diagnoses and the quality of the handoff, not accuracy alone. That matters to clinicians, developers and regulators designing human-in-the-loop systems.

**Collaborators / co-authors.** [[Names and affiliations, confirmed by the team.]]

**Previous or upcoming presentations and publications.** [[None, or list. The code repository is public on GitHub; the data and trained weights are not, under the PhysioNet DUA.]]

## Novelty wording (docs/RESEARCH.md §9)
Claim: escalation trained against a state-level counterfactual of not escalating (branched forced continuations), valued in clinical units, with a properly scored probabilistic handoff. Do NOT claim:
- a group-statistics or case-difficulty deferral reward (TIAR, KARL, AWA-RL);
- the first RL-trained clinical deferral (TrustMed-RL);
- branched rollouts for credit assignment as such (VinePPO, TreeRL/Tree-GRPO);
- counterfactual rollouts in sequential diagnosis (CDPR, which scores test choices and has no deferral).

Re-check novelty within 72 hours of submission.

## Claims checklist (each must trace to docs/RESULTS.md)
- [ ] Abstract is 300 words or fewer after the numbers are in
- [ ] Selective accuracy and coverage of DEFER-Dx, with 95% CI (Table 1)
- [ ] Paired difference vs matched-coverage threshold on the GRPO control (Table 5)
- [ ] Paired difference vs the case-level consensus arm (the published mechanism; Table 5)
- [ ] Handoff quality (proper score of the handed-over differential)
- [ ] Unflagged / confident errors (Table 2)
- [ ] Diverticulitis accuracy (Table 2; n = 51 in val + test)
- [ ] Time-critical false commits, unseen groups (Table 3b). Ruptured AAA and ectopic pregnancy have fewer than 10 cases each: they count in the pooled unseen numbers, but no per-group claim; per-group numbers exist only for mesenteric ischaemia, perforated/bleeding ulcer and DKA
- [ ] LA-CDM-split context row (Table 4), stated as context, not head-to-head
- [ ] The 93.4 / 78.8 LDTL numbers are quoted "as reported" (different split, case-weighted metric)
- [ ] Untrained Qwen3-14B with a confidence threshold misses fewer unseen time-critical cases (6.6%) than the trained consensus arm (11.4%) (docs/BASELINES.md §7). So frame time-critical results against the identically trained control, not against zero-shot 8B alone
