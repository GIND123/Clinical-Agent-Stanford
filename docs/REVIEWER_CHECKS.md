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

## Status of every comment

| Comment | Status |
|---|---|
| Placeholder prices shape the reward | §B. Only the HIDA scan's price matters. A price perturbation run is still needed (GPU). |
| Early deferral may come from cost | §B: plausible. Needs the cost ablation (GPU). |
| The policy may exploit chart artifacts | §A: the shortcut exists. Auditing whether DEFER-Dx uses it needs the evaluation episodes (CPU). |
| Rewards score outcomes, not reasoning | Needs clinicians. The blinded rating sheet exists (`scripts/handoff_rating.py`). |
| The 30% cap is engineered | `configs/ablations/no_constraint.yaml` exists but has not been run (GPU). |
| Only one training run | Two seeds each for DEFER-Dx and the control, both evaluated (PAPER §7.7). The consensus arm's second seed is trained, and its evaluation is pending ([CONSENSUS_SEED1.md](CONSENSUS_SEED1.md)). |
| The win disappears under majority vote | True of the one pairing in the Oct 11 summary. With two seeds per arm, single-sample accuracy on answered cases is +3.2 (1.6 to 5.1), pooled over the four pairings (PAPER §7.7). State the voting result up front. |
| Systems answer at different rates | Thresholds are matched to coverage exactly (`coverage_rule`). Risk–coverage curves are in PAPER §7.3. |
| Pooled runs inflate significance | The bootstrap already resamples each case with all its runs (EXPERIMENTS, reporting rules). The summary should say so. |
| No multiple-comparison correction | §C. |
| The gap to the published reward is not significant on headline measures | Text fix. ABSTRACT.md already lists it under "what not to claim". |
| The time-critical result rests on about 13 of 187 cases | Report the interval, 6.8% (4.3 to 10.0), and the replication in the second seed (−4.3, −7.7 to −1.1). |
| CEV uses three noisy rollouts | Needs a GPU run: the CEV estimate's variance against the number of rollouts. |
| CEV assumes the clinician is right | Needs design work: a clinician-accuracy term and a learning-to-defer baseline. |
| Handoffs list OTHER first 68% of the time | First seed only. The second seed does this in 4.2% of handoffs (PAPER §7.6–7.7). Move it up in the summary. |
| Unflagged errors depend on the case mix | Text fix: give open-world wrong commitments per 100 cases as well. |
| Sample sizes vary | Text fix: one table of splits. |
| The fairness analysis has no results | Deferral-equity tables exist for the baselines (BASELINES §7). For DEFER-Dx, run `scripts/subgroup_eval.py` on its evaluation outputs (CPU), or drop the claim. |
| The FDA citation concerns mental-health devices | Text fix. Cite FDA's clinical decision support software guidance instead, or drop the regulator sentence. ABSTRACT.md and PAPER.md still cite the advisory committee. |
| The caption refers to the README | Text fix in the summary. |
| The DiagBench leakage claim | Our re-run is in BASELINES.md. Contacting the authors is the team's decision. |
