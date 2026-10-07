# Abstract draft: Stanford AI+HEALTH 2026 (deadline Oct 15, 2026, 11:59 pm PT)

Status: **DRAFT. Every `[[...]]` is a placeholder filled only from docs/RESULTS.md once the runs finish.** Nothing below is a result until its placeholder is replaced by a number from that file.

## Title options
1. **Training Clinical Agents to Ask, Test, Commit, and Defer**
2. Clinical Agents That Learn When to Hand a Case Back
3. Learned Deferral for Interactive Diagnostic Agents under an Open-World Label Space

## Full version (~280 words)

**Background.** Regulators increasingly expect clinical AI to escalate to a clinician when it is uncertain; the FDA's Digital Health Advisory Committee (Nov 2025) emphasised timely human intervention for generative-AI devices. Yet published interactive diagnostic agents are closed-world, forced-choice systems: they must name one of a fixed set of diagnoses and cannot hand a case back. On the MIMIC-IV clinical-decision-making benchmark, the strongest reported agent raises mean accuracy to 93.4% while its accuracy on the minority condition, diverticulitis, falls to 78.8%, below a random test-ordering policy.

**Methods.** DEFER-Dx is a Qwen3-8B agent trained with multi-turn group-relative policy optimisation over four actions: ask, order one of 22 individual tests, commit to a diagnosis with a stated probability, or defer to a clinician with a ranked differential. Deferral is learned against a *counterfactual*. At the exact state where the agent escalates, continuations branched from that state estimate the clinical value of deciding now or ordering further tests instead, and the escalation is credited only where a handoff beats continuing. The handoff is a probabilistic differential scored with a proper scoring rule. The commit reward combines accuracy, a proper scoring rule, an asymmetric severity cost and test cost. We train and evaluate on MIMIC-IV-Ext-CDM (2,400 admissions) and on a new open-world extension of [[713]] abdominal admissions whose diagnosis is none of the four, holding out time-critical diagnoses (mesenteric ischaemia, perforated ulcer, ruptured AAA, DKA, ectopic pregnancy) entirely from training.

**Results.** On 480 held-out admissions, DEFER-Dx committed on [[coverage]]% of cases at [[selective accuracy]]% accuracy ([[95% CI]]), versus [[x]]% for a matched-coverage confidence threshold on an identically trained no-deferral control. Unflagged errors fell from [[a]] to [[b]] per 100 cases and diverticulitis accuracy reached [[d]]%. On unseen time-critical presentations it committed to a wrong in-set diagnosis in [[f]]% of cases versus [[g]]% for the control.

**Conclusion.** [[One sentence, written after the results: what learned deferral buys over thresholding, stated only as strongly as the paired intervals allow.]]

## Short version (~150 words)

[[Cut from the full version once the numbers are in.]]

## Novelty wording (docs/RESEARCH.md §9)
Claim: escalation trained against a state-level counterfactual of not escalating (branched forced continuations), valued in clinical units, with a properly scored probabilistic handoff. Do NOT claim a group-statistics or case-difficulty deferral reward (TIAR, KARL, AWA-RL) or the first RL-trained clinical deferral (TrustMed-RL). Re-check novelty within 72 hours of submission.

## Claims checklist (each must trace to docs/RESULTS.md)
- [ ] Selective accuracy and coverage of DEFER-Dx, with 95% CI (Table 1)
- [ ] Paired difference vs matched-coverage threshold on the GRPO control (Table 5)
- [ ] Paired difference vs the case-level consensus arm (the published mechanism; Table 5)
- [ ] Handoff quality (proper score of the handed-over differential)
- [ ] Unflagged / confident errors (Table 2)
- [ ] Diverticulitis accuracy (Table 2; n = 51 in val + test)
- [ ] Time-critical false commits, unseen groups (Table 3b)
- [ ] LA-CDM-split context row (Table 4), stated as context, not head-to-head
- [ ] The 93.4 / 78.8 LDTL numbers are quoted "as reported" (different split, case-weighted metric)
