# Abstract draft: Stanford AI+HEALTH 2026

Status: **FINAL TEXT, numbers filled from docs/results.json** (report of Oct 10, 2026, 09:29 IST: DEFER-Dx and every comparator evaluated on 480 CDM cases and the open-world sets, 3 seeds, 95% case-level bootstrap intervals). The identically trained no-DEFER control is a journal run (`scripts/queue_v7.sh`), so the submission uses outcome **D** below, which does not depend on it.

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

DEFER-Dx is a reinforcement-learning diagnostic agent that asks, orders tests, commits to a diagnosis or hands the case to a clinician, and is trained to escalate only where continuing would do worse. On emergency admissions from MIMIC-IV, including dangerous diagnoses it never saw in training, we compare learned escalation with a deferral reward from prior work and with thresholding a model's own confidence.

## Submission text (final: 298 words with section labels; limit 300)

**Title.** Training Clinical Agents to Ask, Test, Commit, and Defer

**Background.** Regulators increasingly expect clinical AI to escalate uncertain cases; the FDA's Digital Health Advisory Committee (Nov 2025) emphasised timely human intervention for generative-AI devices. Yet published interactive diagnostic agents are closed-world and forced-choice: they must name one of a fixed set of diagnoses and cannot hand a case back.

**Methods.** DEFER-Dx is a Qwen3-8B agent trained with multi-turn reinforcement learning (GRPO) to examine, order any of 22 priced tests, commit to a diagnosis with a stated probability, or defer to a clinician with a probabilistic differential. Deferral is trained against a counterfactual: at the state where the agent escalates, continuations branched from that state with escalation disabled estimate the clinical value of deciding now or testing further, and escalation is credited only where a handoff beats continuing. We train on MIMIC-IV-Ext-CDM (2,400 admissions, four abdominal conditions) and a new open-world extension of 713 admissions with other diagnoses, holding five time-critical diagnoses out of training.

**Results.** On 480 held-out admissions, DEFER-Dx answered 77.6% of cases at 95.2% accuracy (95% CI 93.4–96.8), with 3.8 unflagged errors per 100 cases (zero-shot 24.1). With three-sample voting for both, it was 11.4 points (7.4–15.7) more accurate on answered cases than a confidence threshold on the zero-shot model at matched coverage. On 187 time-critical admissions never seen in training, it escalated 87.0% and wrongly committed in 6.8%, against 11.4% for an identically trained group-consensus deferral reward from prior work (paired difference −4.6 points, −8.0 to −1.4). Its handoffs favoured "other diagnosis", lowering accuracy when they are counted (78.2% vs 85.8% for the consensus reward).

**Conclusion.** Crediting escalation against the counterfactual of continuing yields an agent that rarely errs without flagging the case and commits wrongly on never-seen dangerous presentations less often than a prior deferral reward, at the cost of less specific handoffs.

### Where each number comes from (`docs/results.json`, `docs/RESULTS.md`)

| Number in the abstract | Source |
|---|---|
| 480 admissions; 77.6% answered; 95.2% (93.4–96.8) | `tables.cev.ml.coverage`, `tables.cev.ml.selective_acc` (RESULTS §1) |
| 3.8 vs 24.1 unflagged errors per 100 | `tables.cev.clinical.unsafe_errors`, `tables.zs_nodefer.clinical.unsafe_errors` (§2) |
| +11.4 points (7.4–15.7) at matched coverage, three-sample voting | `paired["cev@sc vs zs_nodefer@sc+thr"].selective_acc`; coverage difference −0.2 (−4.6 to 4.0), so coverage is matched (§5) |
| 187 unseen; escalated 87.0%; wrong commitments 6.8% vs 11.4% | `tables.cev.eval_other_unseen.defer`, `.false_commit`; `tables.deferdx.eval_other_unseen.false_commit` (§3b) |
| −4.6 points (−8.0 to −1.4) | `paired["cev vs deferdx"]["false_commit:eval_other_unseen"]`, p = 0.013 (§5) |
| 78.2% vs 85.8% when handoffs are counted | `tables.cev.ml.acc_full`, `tables.deferdx.ml.acc_full` (§1) |
| handoffs favour "other diagnosis" | 68.4% of DEFER-Dx's in-set handoffs rank OTHER first (consensus arm 8.6%, prompted DEFER 27.6%); computed from the evaluation suites, aggregates only |

### What not to claim (in the abstract or the talk)

- **That DEFER-Dx learns to investigate more.** It orders fewer tests than the consensus arm, not more: 0.84 vs 1.83 per case, and 0.67 vs 2.10 before deferring. It escalates early, so the "test more" half of the mechanism is not shown.
- **Deferral precision (80.5%) as "escalations that would have been errors".** That metric counts a deferral as justified when the handoff's top choice is wrong, and a handoff ranking OTHER first on an in-set case always counts.
- **Lower unflagged errors than the consensus arm.** 3.8 vs 4.9 per 100, paired −1.2 (−2.6 to 0.2), which is not significant. Same for AURC and ECE against it.
- **Better full-coverage or diverticulitis accuracy.** Both are lower than the consensus arm: 78.2 vs 85.8, and 69.3 vs 85.6 with paired −16.3 (−24.3 to −8.8).
- **Fewer missed time-critical diagnoses than every baseline.** An untrained Qwen3-14B with a confidence threshold reaches 6.6% (docs/BASELINES.md §7), against DEFER-Dx's 6.8%. The claim holds against the same 8B base model and against the deferral reward from prior work.
- **The single-sample threshold comparison (+17.5 points).** It is not at matched coverage: tied stated probabilities kept the threshold at 92.6% coverage against 77.6%.

## Results and Conclusion, written for each outcome (A–C need the GRPO control; superseded by the final text above, outcome D)

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
- [x] Abstract is 300 words or fewer after the numbers are in (298 with section labels)
- [x] Selective accuracy and coverage of DEFER-Dx, with 95% CI (Table 1): 95.2 (93.4–96.8) at 77.6 (74.3–80.6)
- [ ] Paired difference vs matched-coverage threshold on the GRPO control (Table 5): **journal run, not in the abstract**; the zero-shot threshold at matched coverage (three-sample voting) is used instead
- [x] Paired difference vs the case-level consensus arm (Table 5): unseen false commits −4.6 (−8.0 to −1.4); selective accuracy +1.1 (−0.5 to 2.9, n.s.)
- [x] Handoff quality: lower than the consensus arm in-set (44.1 vs 68.6), because handoffs favour OTHER; on OTHER cases the handoff contains the truth 93.9–95.6% vs 14.7–15.0%. Stated as a cost in the abstract
- [x] Unflagged / confident errors (Table 2): 3.8 / 2.3 per 100 (zero-shot 24.1 / 7.9)
- [x] Diverticulitis accuracy (Table 2; n = 51): 69.3 (58.1–80.0), lower than the consensus arm; not claimed
- [x] Time-critical false commits, unseen groups (Table 3b): 6.8 (4.3–10.0). Ruptured AAA and ectopic pregnancy have fewer than 10 cases each: they count in the pooled unseen numbers, but no per-group claim; per-group numbers exist only for mesenteric ischaemia, perforated/bleeding ulcer and DKA
- [x] LA-CDM-split context row (Table 4): not used in the abstract (DEFER-Dx 76.4 mean-class accuracy at full coverage on that split)
- [x] The LDTL numbers were cut from the abstract for length; quote them "as reported" in the talk if used
- [ ] Untrained Qwen3-14B with a confidence threshold misses fewer unseen time-critical cases (6.6%) than the trained consensus arm (11.4%) (docs/BASELINES.md §7). So frame time-critical results against the identically trained control, not against zero-shot 8B alone
