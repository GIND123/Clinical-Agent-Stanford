# Reviewer checks (Oct 14, 2026)

A reviewer at the University of Florida commented on the two-page summary of Oct 11. This page does two things:
- runs the three checks from those comments that need no GPU;
- records the status of every comment.

Every number comes from [reviewer_checks.json](reviewer_checks.json), written by:

```
python scripts/reviewer_checks.py --json docs/reviewer_checks.json \
    --train-log outputs/runs/deferdx_seed1/train_log.jsonl
```

The script reads `data/cohorts`, `configs/test_catalog.yaml`, `docs/results.json` and one training log. It reports aggregates only.

## A. Chart artifacts: does knowing which tests were recorded give the diagnosis away?

**Why it matters.** When the agent orders a test that was never performed on a patient, the environment replies "Not performed for this patient; no result available." An agent can therefore learn from *which* tests a case has, not only from their results. Which tests were recorded reflects what the treating clinicians suspected.

**Method.** A logistic regression sees only the 22 recorded/not-recorded flags, never a result, and is scored by 5-fold cross-validation.

| Question | Cases | Result |
|---|---|---|
| In-set diagnosis (four classes) | 1,760 CDM training cases | 69.4% correct, against 39.9% for always guessing the most common class; macro AUROC 0.879 |
| OTHER against in-set, with both built by the open-world pipeline (controls vs OTHER) | 651 evaluation cases (393 OTHER) | AUROC 0.840 |

The strongest signals:
- **Cholecystitis:** abdominal ultrasound was recorded in 87.6% of cases, against 38.3% of the others.
- **Pancreatitis:** triglycerides in 40.6% of cases (2.1% of the others), and amylase in 51.0% (12.5%).
- **Appendicitis:** abdominal CT in 94.2% of cases (50.6% of the others), and a pregnancy test in 32.5% (14.3%).

**What this shows.** The shortcut is real, and it is open to every agent on MIMIC-IV-Ext-CDM that orders tests, including published ones.

**What it does not show.** That DEFER-Dx uses it. DEFER-Dx orders 0.84 tests per case, so it sees few "not performed" replies.

**Next:**
- Count the "not performed" replies in the evaluation episodes (CPU, on the machine that holds them).
- Compare accuracy after such a reply with accuracy after a result.
- To remove the shortcut altogether, the environment would have to answer unrecorded tests the same way for every case, for example with an imputed normal result. That change needs retraining.

## B. Test prices: do the placeholder prices or the cost term drive the results?

**How cost enters the reward.** With `cost_scale: auto`, ordering all 22 tests costs as much as one correct diagnosis (1.0). A test's reward cost is therefore its price divided by the catalog total, $14,674.

**The placeholder prices:**
- 10 of the 22 tests have placeholder prices, and together they are 29.6% of the catalog total.
- Nine of them are laboratory tests costing 0.007 to 0.017 each.
- The tenth, the HIDA scan, costs 0.204. Its $3,000 placeholder is the only one that matters.
- Across all tests, the median reward cost is 0.014, and the maximum is 0.332 (abdominal MRI, a BIDMC price).

**Rescaling them.** Halving or doubling every placeholder price changes:
- the HIDA scan's cost by 0.084 (halved) or 0.111 (doubled);
- the typical test's cost by 0.004 to 0.006.

At 0.84 tests per case, that is under 0.005 of reward per case.

**The size of each reward component** in the consensus arm's second training seed, averaged over its last 50 steps:

| Accuracy | Consensus deferral | Cost | Calibration | Severity |
|---|---|---|---|---|
| 0.642 | 0.157 | 0.094 | 0.058 | 0.023 |

**Reading:**
- Except for the HIDA scan, the placeholder prices are too small to shape the policy.
- The cost term as a whole is not negligible. Abdominal CT costs 0.089 and abdominal ultrasound 0.088 (BIDMC prices).
- Under CEV, a forced continuation's extra test costs count against continuing.
- So the early deferral (0.84 tests per case) could partly come from cost. Only a training ablation can settle this: the cost term set to zero, or prices perturbed (GPU).

## C. Multiple comparisons

**Across all paired tests in `docs/results.json`:** 218 tests. 107 have p < 0.05, and 58 remain below 0.05 after Holm correction.

**The three comparisons the abstract reports:**

| Comparison | p | Holm, within the three | Holm, across all 218 |
|---|---|---|---|
| Accuracy on answered cases, against a threshold on the zero-shot model (three-sample voting, matched coverage) | < 0.001 | < 0.001 | < 0.001 |
| Wrong commitments on never-seen time-critical cases, against the consensus arm (first DEFER-Dx seed) | 0.013 | 0.020 | 1.0 |
| The same, second DEFER-Dx seed | 0.010 | 0.020 | 1.0 |

**Reading:**
- The zero-shot comparison survives any correction.
- The time-critical result survives correction within the abstract's three claims, but not across all 218 tests.
- Its stronger support is that two independently trained DEFER-Dx seeds each show it.
- In the paper, mark the secondary comparisons exploratory and report the Holm-adjusted p.

## Status of every comment (updated Oct 16)

### Methods supervisor

| Comment | Status |
|---|---|
| Placeholder prices shape the reward | §B: only the HIDA price matters. The CEV audit (docs/cev_audit.json) finds that removing the test cost of forced continuations changes the sign of the escalation credit in 0% and 1% of deferral states. The training ablation without the cost term (`configs/ablations/cev_no_cost.yaml`) runs in `queue_v10.sh`. |
| Early deferral may come from cost | Not supported by the audit: forced continuations ordered a test in only 23% and 18% of cases and were right in 32% and 67%, so deferral reflects cases the agent cannot resolve. The no-cost ablation tests it directly (queue_v10). |
| The policy may exploit chart artifacts | Audited (`scripts/chart_artifact_audit.py`, docs/chart_artifacts.json): DEFER-Dx saw a "not performed" reply in 12–17% of episodes, and its accuracy on answered cases was lower after one (89.7 vs 95.7; 88.5 vs 96.2), as for every trained model. Observational: harder cases draw more tests. |
| Rewards score outcomes, not reasoning | Still needs clinicians: the blinded rating sheet exists (`scripts/handoff_rating.py`). |
| The 30% cap is engineered | Ablation without the cap (`configs/ablations/cev_no_constraint.yaml`, rho_max 1.0) runs in `queue_v10.sh`. |
| Only one training run | Two seeds each for DEFER-Dx and the control, evaluated; learned vs threshold reported for all four pairings and pooled (+3.2, 1.6 to 5.1). The consensus arm's second seed is trained but its adapter is not on this machine. |
| The win disappears under majority vote | With two seeds per arm the single-sample difference holds in all four pairings; with three-sample voting it is +2.2 (−0.9 to 5.5) for the first pairing (PAPER results). |
| Systems answer at different rates | Thresholds match coverage exactly (`coverage_rule`); risk–coverage curves in Extended Data Fig. 3. |
| Pooled runs inflate significance | The bootstrap resamples cases with all their runs; pooled seed comparisons hold training runs fixed and say so. |
| No multiple-comparison correction | Holm column in RESULTS.md §5 (218 tests, Wald p from the bootstrap SE). Primary tests (the paper's two questions): Holm p ≤ 0.02; across all 218: pooled accuracy 0.052, time-critical comparisons not significant. Stated in PAPER Methods. |
| The gap to the published reward is not significant on headline measures | Stated: accuracy on answered cases is the same; the difference is on unfamiliar presentations. |
| The time-critical result rests on about 13 of 187 cases | Intervals reported; replicated in the second seed (−4.6 and −4.3); does not survive Holm across all 218 tests, said so. |
| CEV uses three noisy rollouts | Audited with K = 16 (`scripts/cev_audit.py`): K = 3 gives the credit the same sign in 88% and 92% of in-set states (K = 8: 95%, 96%). Extended Data Fig. 7. |
| CEV assumes the clinician is right | Three answers: the credit's sensitivity to the assumed clinician accuracy (46% of deferrals credited at 0.70 vs 60% at 0.85); a human-AI team model with a clinician of accuracy 0.89 (Hager et al.) who can be steered by the handoff (Barrit et al. 2026); and a learning-to-defer baseline (defer when p < clinician accuracy). PAPER, "What the handoff says". |
| Handoffs list OTHER first 68% of the time | First seed only (second: 4.2%); the team model shows each seed's handoff pattern costs patients under clinician susceptibility. Moved to the main results. |
| Unflagged errors depend on the case mix | Open-world results given per 100 cases in the same units (AI-caused wrong in-scope diagnoses per 100). |
| Sample sizes vary | One cohort table: PAPER Extended Data Table 1. |
| The fairness analysis has no results | Done: docs/fairness/ for both DEFER-Dx seeds, the consensus reward, the control and prompted DEFER. |
| The FDA citation concerns mental-health devices | Fixed in ABSTRACT.md and PAPER.md: FDA Clinical Decision Support Software guidance (revised January 2026). |
| The caption refers to the README | Paper figures have self-contained legends (docs/paper/DEFER-Dx.pdf). |
| The DiagBench leakage claim | Our re-run is in BASELINES.md. Contacting the authors is the team's decision. |

### Clinical supervisor

| Comment | Status |
|---|---|
| Acute poisoning should be recognised as OTHER and deferred | MIMIC-CDM-CR includes lead, Amanita and thallium poisoning from published case reports (benchmarks/casereports/); results per case in docs/casereports.json and PAPER Fig. 5. |
| Unflagged errors (3.8 per 100) and false commits on time-critical cases (6.8%) are the harm | Reported with intervals, replicated in the second seed, and modelled through the clinician who receives the handoff. |
| No staging (WSES/Hinchey, Revised Atlanta, Tokyo) | Evaluated, not built: in-set cases labelled complicated by guideline proxies from admission codes and labs; a threshold escalates complicated disease less often than uncomplicated (16.7% vs 24.0%), DEFER-Dx does not, and no system escalates it preferentially (docs/severity.json, Extended Data Fig. 6). Published cases carry the report's guideline grade. A staging output needs graded labels and retraining: future work, stated in the paper. |
| The tool is static: no empirical treatment, monitoring or reassessment | Stated as a limitation. The environment reveals recorded results only; MIMIC-IV hosp has no time series of vital signs, so evolving states cannot be simulated faithfully. MIMIC-CDM-CR includes one patient at two visits two days apart (CR26, CR27). |
