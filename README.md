# DEFER-Dx

An interactive clinical diagnostic agent trained with reinforcement learning over four actions: **ASK, TEST, COMMIT, DEFER**. Deferring to a clinician is a learned action. Its reward is the **counterfactual escalation value (CEV)**: at the exact state where the agent escalates, forced continuations branched from that state estimate the clinical value of *not* escalating, by deciding now or investigating further. The DEFER turn is credited with the difference. The handoff is a probabilistic differential trained with a strictly proper scoring rule. The label space is open-world: the four MIMIC-CDM abdominal conditions plus **OTHER**.

![DEFER-Dx: an agent that asks, tests, commits or defers; commitments are scored in clinical units and deferrals are credited against the counterfactual of not deferring](docs/figures/fig_overview.png)

- Original research plan: [stanford_idea_md.md](stanford_idea_md.md) (its consensus-reward method is superseded; target: Stanford AI+HEALTH 2026 abstract, due Oct 15, 2026)
- The method as implemented, with every constant: [docs/METHODS.md](docs/METHODS.md)
- Paper draft: [docs/PAPER.md](docs/PAPER.md) · abstract scaffold: [docs/ABSTRACT.md](docs/ABSTRACT.md)
- Results: [docs/RESULTS.md](docs/RESULTS.md) (generated as evaluations finish) · all figures with captions: [§3](#3-results-and-figures), sources in [docs/figures/](docs/figures/)

> **PhysioNet DUA.** MIMIC text never reaches a third-party service. Every model runs on local weights (`transformers` / `vllm`), and the repo has no hosted-API clients. `.gitignore` excludes `data/`, `outputs/`, `.env`, `*.jsonl`, `*.csv*` and `*.pkl`. Rollout files contain MIMIC text, so treat them as credentialed data. Committed documents hold aggregates only, and patient-derived counts of 1–9 are suppressed. Weights trained on MIMIC are not pushed to public hubs; PhysioNet's credentialed hosting is the DUA-consistent release route.

---

## 1. Status (2026-10-16)

| Component | State |
|---|---|
| Data: MIMIC-IV-Ext-CDM 1.1, MIMIC-IV 2.2 hosp, MIMIC-IV-Note 2.2 | downloaded; SHA256 and gzip verified |
| Open-world set MIMIC-CDM-OW: 713 OTHER + 648 same-pipeline controls | built; deterministic (byte-identical rebuilds) |
| **MIMIC-CDM-CR: 32 published case reports** ([benchmarks/casereports/](benchmarks/casereports/)) | built Oct 16 from open-access reports (2020–2025), cited case by case; no patient data, shareable; every model evaluated on it (queue_v10) |
| Patient-disjoint cohorts (`data/cohorts/`) | built |
| **DEFER-Dx with the counterfactual escalation value** (`configs/grpo_cev.yaml`, the method) | two training seeds, each trained 150 steps and evaluated (Oct 10, Oct 13). Records, not results: `cev_dualascent` (plain dual-ascent cap, limit cycle) and `cev_pi_crashed` (out of memory at step 29) |
| GRPO control (no DEFER), the key comparator | two training seeds, trained and evaluated (Oct 11, Oct 14) |
| Case-level group-consensus arm (the published mechanism, a comparator) | trained and evaluated. Its second seed was trained on Lambda Cloud (Oct 13–14; [docs/CONSENSUS_SEED1.md](docs/CONSENSUS_SEED1.md)); the adapter is not on this machine, so it is **not evaluated** |
| Ablations at 100 steps | constant deferral reward and no handoff score: done. Without the test-cost term and without the coverage cap: training in `scripts/queue_v10.sh` (Oct 16–17). DEFER-Dx with the handoff score weighted 1.0: queued in `scripts/queue_v11.sh` |
| Reviewer and supervisor checks | CEV estimator audit, severity strata, human-AI team model (Barrit et al. 2026), learning-to-defer baseline, chart-artifact audit, subgroup equity, Holm correction: done (§3) |
| Manuscript | [docs/PAPER.md](docs/PAPER.md), typeset as [docs/paper/DEFER-Dx.pdf](docs/paper/DEFER-Dx.pdf) (`bash scripts/build_paper.sh`) |
| Figures | 18 publication figures in [docs/figures/](docs/figures/) (PDF, PNG, SVG; TikZ sources) |
| Tests | 119 pass: 117 test functions, some parametrised (`pytest`; CPU only) |

**GPU plan: finish every queued run** (user, Oct 17), hard stop Wed Oct 21, 12:00 IST. `scripts/queue_v10.sh`, then `scripts/queue_v11.sh`:

| Job | Expected (IST) |
|---|---|
| CEV estimator audit, both DEFER-Dx seeds | done Fri 11:12 |
| Every model on MIMIC-CDM-CR (3 seeds) | done Sat 07:23 (two reboots on Oct 16 stopped the queue at 11:31) |
| Ablation without the test-cost term (100 steps) and its evaluation | Sun ~07:20 (stopped at step 31 by the Oct 17 reboot; resumed 21:03 from step 30) |
| Ablation without the coverage cap (100 steps) and its evaluation | Sun ~21:30 |
| DEFER-Dx with the handoff score weighted 1.0 (100 steps), held-out and published-case evaluation | Mon ~13:00 |

The machine restarted four times on Oct 16–17 (desktop updates and restarts), each time stopping the queue. A crontab `@reboot` entry now waits for the data disk and runs `scripts/resume_after_reboot.sh`, which restarts the sync daemon and any unfinished queue. The rules are unchanged: a fixed list; a job starts only if its estimate fits before the deadline; every attempt is killed
at the deadline; at most two attempts, training resuming from its newest checkpoint; whatever finishes is reported.

**Hugging Face.** Uploads still fail: the account's private storage is full, mostly from other projects. The sync daemon
retries every 15 minutes, so the adapters go up as soon as there is space; until then local archives under
`outputs/archive/` are the backup.

**Stanford AI+HEALTH abstract.** The FDA sentence was replaced on Oct 16 (reviewer: the Nov 2025 advisory committee
concerned mental-health devices); see [docs/ABSTRACT.md](docs/ABSTRACT.md).

## 2. Is it novel? Is it beating SOTA? (honest assessment)

### 2.1 Novelty

Three literature checks (2026-10-06 and 2026-10-07; details in [docs/RESEARCH.md](docs/RESEARCH.md) §7–9) found that the reward first built here was **already published**. Deferral rewarded by the policy's estimated case-level success rate γ(τ − p̂) is:
- **TIAR** (arXiv 2605.25850): identical at τ = 0.5;
- **KARL** (2604.22779): a binary variant;
- **AWA-RL** (2607.10738): the same idea in multi-turn agents, with a refusal-rate penalty;
- **TrustMed-RL** (2610.04387): RL-trained clinical deferral on designated cases.

The method was therefore redesigned around a quantity none of them uses.

**Counterfactual escalation value (CEV).** At the exact state where the agent escalates, K forced continuations are branched by replaying the environment, with escalation disabled and further tests allowed. Their mean clinical return V̂(s) (accuracy, proper score, severity, cost of further tests) is the value of *not* escalating there. The DEFER turn's advantage is E − V̂(s), where E includes a strictly proper score of the handed-over probabilistic differential.

| Closest work | What it does | Difference from CEV |
|---|---|---|
| TIAR, KARL, AWA-RL | Abstention reward from the policy's **case-level** success rate (group statistics or a prior checkpoint) | CEV is **state-level and counterfactual**: escalating versus continuing *from the same state*, so "another test would settle it" (escalation discouraged) is separated from "continuing would err" (escalation credited). Valued in clinical units: severity, calibration and workup cost. |
| TrustMed-RL | Deferral rewarded on cases designated by construction | No case is labelled "should defer"; the target is counterfactual and on-policy. |
| Tree / branched RL (Tree-GRPO, SIPO, Counterfactual Rollout Replay, ASCT); CDPR | Branching for generic step-level credit; CDPR scores investigative actions | None targets an **escalation** action or values it against continuation, and none has a handoff. |
| Signed Rescue Routing | Escalation value in model cascades, at inference time | CEV *trains* an agent's escalation, with on-policy counterfactuals. |
| Decide/Ask/Defer, Safe to Stop?, MedAbstain, MediQ | Evaluation, conformal stopping or prompting | Learned by RL. |
| — | (none found) | **The scored handoff:** deferral must hand over a probabilistic differential, rewarded with a strictly proper scoring rule. |

**Verdict.** To our knowledge, based on these searches, the CEV mechanism (state-level counterfactual escalation credit plus a properly scored handoff) is new, and so is MIMIC-CDM-OW, the open-world benchmark with never-seen time-critical diagnoses. Novelty against an open literature cannot be proven absolutely, so the plan re-checks within 72 hours of submission. The previously built consensus reward is kept only as a **comparator arm**: it is the published mechanism, and CEV must beat it.

### 2.2 SOTA

Published MIMIC-CDM numbers use different splits, metrics and environments, so only some comparisons are like-for-like. Rows marked "this protocol" are the held-out evaluation of this repository ([docs/RESULTS.md](docs/RESULTS.md) §1 and §4; 3 seeds, 95% bootstrap intervals). The closed-world zero-shot row was re-run under this protocol (`eval_zs_closed`) and reproduces the earlier runs (87.2, [docs/BASELINES.md](docs/BASELINES.md)).

| System | Split | Mean-class acc | Case-weighted acc | Diverticulitis | Selective acc (coverage), CDM val + test |
|---|---|---|---|---|---|
| LA-CDM, trained (ICLR 2026, reported) | LA-CDM test | 81.3 | – | 75.0 | – |
| LDTL (reported; current SOTA) | own unpublished 70/10/20 | – | **93.4** | 78.8 | – |
| Random planner (from LDTL) | own split | – | 84.8 | 90.4 | – |
| DiagAgent-14B (run here) | LA-CDM test | 71.9 | 77.9 | 56.0 | – |
| Qwen3-8B zero-shot, **closed world** (4 labels, as in prior work; this protocol) | LA-CDM test | 86.9 (83.1–90.4) | 87.2 (83.6–90.6) | 92.0 (83.3–98.7) | 89.9 (95.8%) |
| Qwen3-8B zero-shot, open world, forced choice (this protocol) | LA-CDM test | 74.4 (69.1–79.5) | 76.7 (72.2–81.1) | 76.0 (60.8–90.1) | 75.5 (98.4%) |
| Qwen3-8B zero-shot, open world, prompted DEFER (this protocol) | LA-CDM test | 78.3 (73.7–82.4) | 79.2 (75.1–83.1) | 82.7 (70.7–92.4) | 87.9 (75.1%) |
| Case-level consensus arm, open world (published mechanism; this protocol) | LA-CDM test | **86.3** (82.2–90.2) | 87.1 (83.5–90.6) | **90.7** (79.4–98.6) | **94.0** (82.6%) |
| **DEFER-Dx (CEV)**, open world (this protocol) | LA-CDM test | 76.4 (71.1–81.0) | 79.0 (74.3–83.2) | 77.3 (63.0–90.3) | **95.2** (77.6%) |
| DEFER-Dx, second training seed (this protocol) | LA-CDM test | 85.2 (80.8–89.3) | 87.1 (83.3–90.7) | 86.7 (74.7–96.4) | **95.3** (73.2%) |
| GRPO control, no DEFER (this protocol) | LA-CDM test | 82.7 (77.9–87.1) | 85.4 (81.7–89.0) | 76.0 (60.8–89.5) | 87.8 (93.8%) |

**Verdict so far:**
- **Open world costs accuracy.** Offering OTHER as a fifth answer costs the untrained model about 13 points: 86.9 closed world against 74.4 open world on the same test split, because it answers OTHER on many in-set cases. Every trained system here works in the open world.
- **Above LA-CDM:** the trained comparator arm reaches 86.3 mean-class accuracy on LA-CDM's test split in the open world, against LA-CDM's 81.3 in its closed world. The environments differ (full history versus a summary, 22 tests versus 12, a different base model), so this is context, not a head-to-head.
- **Above LDTL on diverticulitis:** 90.7 versus 78.8, on a different split.
- **Below LDTL's 93.4 at full coverage:** the comparator's case-weighted 87.1 is the one SOTA number not reached, and LDTL's split is unpublished.
- **Selective target ("> 95 at 85% coverage"):** DEFER-Dx reaches 95.2 (93.4–96.8), but at 77.6% coverage, not 85%; its second seed reaches 95.3 at 73.2%. At full coverage, counting its handoffs, the first seed is below the comparator arm (78.2 vs 85.8), because 68% of its in-set handoffs rank OTHER first. The second seed is not (86.3; 4% OTHER first).
- **Learned deferral against thresholding an *identically trained* model**, at exactly matched coverage, with two training seeds per arm: more accurate on answered cases in all four pairings (+1.1 to +5.3 points; pooled +3.2, 1.6 to 5.1). The size of the margin depends mainly on the control's seed. Against a zero-shot model thresholded at matched coverage (three-sample voting), DEFER-Dx is 11.1 points (7.0–15.3) more accurate on answered cases.

The plan itself (§6.4) says not to claim SOTA on LDTL's full-coverage metric; the pitch is a Pareto improvement on safety axes nobody else reports.

## 3. Results and figures

**Held-out results** (CDM val + test, 480 cases × 3 seeds; paired bootstrap on the same cases; [docs/RESULTS.md](docs/RESULTS.md)):

| | DEFER-Dx | DEFER-Dx, second seed | GRPO control + threshold at matched coverage | Group-consensus reward (prior work) | Zero-shot, prompted DEFER | Zero-shot, forced choice |
|---|---|---|---|---|---|---|
| Coverage (in-set cases answered) | 77.6 (74.3–80.6) | 73.2 (69.7–76.5) | 77.7 (74.7–80.6) | 82.6 (79.5–85.6) | 75.1 (72.0–78.1) | 98.4 |
| Accuracy on answered cases | 95.2 (93.4–96.8) | 95.3 (93.2–97.1) | 92.9 (90.7–94.9) | 94.0 (92.0–95.9) | 87.9 (85.2–90.4) | 75.5 |
| Unflagged errors per 100 cases ↓ | 3.8 (2.4–5.1) | 3.5 (2.2–5.0) | 5.6 (4.0–7.2) | 4.9 (3.4–6.6) | 9.1 (7.2–11.1) | 24.1 |
| Wrong commitments, 187 unseen time-critical cases ↓ | 6.8 (4.3–10.0) | 7.1 (4.3–10.3) | 7.5 (4.6–10.7) | 11.4 (7.8–15.3) | 14.4 (10.2–19.1) | 17.6 |
| Deferred, unseen time-critical cases | 87.0 | 87.0 | 21.0 | 70.4 | 32.8 | 0 |
| Accuracy counting handoffs (full coverage) | 78.2 (75.0–81.4) | 86.3 (83.6–88.9) | 82.4 (79.6–85.1) | 85.8 (83.1–88.4) | 77.3 | 74.3 |
| In-set handoffs ranking OTHER first | 68.4 | 4.2 | – | 8.6 | 27.6 | – |
| Tests per case | 0.84 | 0.89 | 1.48 | 1.83 | 1.92 | 1.93 |

The control (first training seed) is thresholded at DEFER-Dx's first-seed coverage; the second DEFER-Dx seed is compared with the control thresholded at its own (72.1%). The control's second seed, thresholded at 77.6%, answers at 89.9% accuracy with 7.8 unflagged errors per 100.

**What the paired comparisons support:**
- **Against the deferral reward from prior work** (identically trained), in both training seeds: fewer wrong commitments on never-seen time-critical diagnoses, −4.6 (−8.0 to −1.4) and −4.3 (−7.7 to −1.1), and on seen OTHER groups, −7.0 (−9.9 to −4.4) and −6.6 (−9.9 to −3.6). Accuracy on answered cases is the same (+1.1 and +1.2, intervals include 0).
- **Against thresholding an identically trained model** (the GRPO control), at exactly matched coverage, two training seeds per arm (Fig. S6):

  | Accuracy on answered cases (unflagged errors per 100) | Control, first seed | Control, second seed |
  |---|---|---|
  | DEFER-Dx, first seed | +2.3 (0.1 to 4.6); −1.8 | +5.3 (3.0 to 7.7); −4.1 |
  | DEFER-Dx, second seed | +1.1 (−0.8 to 3.3); −0.8 | +4.1 (1.9 to 6.5); −3.0 |

  - **Pooled over the four pairings:** +3.2 points (1.6 to 5.1), with −2.4 (−3.9 to −1.1) unflagged and −2.1 (−3.3 to −1.0) confident errors per 100. The interval reflects case sampling, with the training runs held fixed.
  - **Why the margin varies:** a threshold is only as good as the confidence it cuts. The control's second seed ranks its errors worse (AURC 0.074 vs 0.044; 7.0 vs 2.5 confident errors per 100), and the first leaves 6.2% of in-set episodes without a commitment, which count as not answered. DEFER-Dx's accuracy on answered cases is 95.2 and 95.3 in its two seeds.
  - **Not different:** wrong commitments on never-seen time-critical cases (pooled −1.0, −3.3 to 1.2), where DEFER-Dx escalates 87% and the thresholded control mostly commits to OTHER (64%).
- **Matched coverage needs care.** Stated probabilities are coarse, so a plain cut on the control overshoots the target coverage (79% instead of 73% for the second seed). With that overshoot the advantage reads +3.1 and +3.2. Every threshold now keeps commitments tied at the cut with a fitted probability. RESULTS.md §5b gives the range over 20 tie-break draws.
- **Seed-dependent handoffs.** The first seed ranks OTHER first in 68% of in-set handoffs, lowering full-coverage accuracy (78.2) and diverticulitis accuracy. The second does so in 4% (86.3 at full coverage), but its handoff names OTHER for only 35% of true OTHER cases (first seed: 94%).
- **Against post-hoc thresholding of the zero-shot model at matched coverage:** +12.4 points (9.4–15.7) accuracy on answered cases. With three-sample voting for both, as in the abstract: +11.1 (7.0–15.3), and −8.8 (−12.3 to −5.4) unflagged errors per 100.
- **Ablation: counterfactual credit against a constant deferral reward** (100 steps each, same cases). No difference in accuracy on answered cases (−0.4), unflagged errors, or wrong commitments on never-seen time-critical cases (10.2% vs 10.7%). The credit defers more, puts OTHER in its handoffs far more often (89% vs 25%), cuts false commits on seen OTHER groups (−4.4, −7.5 to −1.5), and lowers diverticulitis accuracy (−11.8).
- **Ablation: without the scored handoff** (η = 0; 100 steps, same cases).
  - **No difference:** accuracy on answered cases (−1.1, −2.8 to 0.6) and unflagged errors (3.8 vs 3.3 per 100).
  - **What the score adds:** more informative in-set handoffs (+6.5, 3.5 to 9.3), more deferral (coverage 68.1% vs 75.9%), and fewer false commits on seen OTHER groups (−3.2, −6.3 to −0.2).
  - **What it doesn't cause:** the OTHER-leaning handoffs. They are more common without the score (55% vs 35% of in-set handoffs, against 13% with a constant reward), so they come from the counterfactual credit, to a degree that depends on the seed.
- **Robustness to the mask cue.** With every sentence containing CDM's `____` diagnosis mask removed, DEFER-Dx answers 76.0% at 96.2% accuracy, against 77.6% at 95.2%. All paired differences are within 1.6 points and AURC is unchanged, so it does not rely on the mask.
- **Complicated disease** (Fig. S7; [docs/severity.json](docs/severity.json)). 114 of 480 in-set cases are complicated by guideline proxies from admission codes and labs (WSES/AAST, Revised Atlanta, Tokyo 2018). A confidence threshold escalates complicated disease less often than uncomplicated (16.7% vs 24.0%; −7.4, −14.1 to −0.5); DEFER-Dx does not (+1.6, −6.1 to 9.8). On complicated cases DEFER-Dx escalates more than the thresholded control in all four seed pairings (+4.4 to +7.0) and than the consensus reward (+7.6, +12.0). No system escalates complicated disease preferentially.
- **The handoff and the clinician who reads it** (Fig. S8; [docs/team.json](docs/team.json); Barrit et al. 2026). With an independent clinician (accuracy 0.89, the MIMIC-CDM physicians) every system's team accuracy is about 94%. If the clinician adopts the handoff's top suggestion with the probability measured under adversarial AI advocacy (0.47), handoff content decides: the first seed's team accuracy falls to 86.8% (its in-set handoffs rank OTHER first) against 91.5% for the thresholded control, and the second seed's handoffs suggest an in-scope diagnosis for 84% of never-seen time-critical cases (37.0% AI-caused wrong in-scope diagnoses). A learning-to-defer rule (defer when the stated probability is below the clinician's accuracy) is safest on these terms, at 43% clinician workload.
- **The escalation credit** (Fig. S9; [docs/cev_audit.json](docs/cev_audit.json)). With 16 forced continuations per deferral state, the K = 3 estimate used in training has the same sign in 88% and 92% of in-set states. Removing test cost changes the sign in 0% and 1%: early deferral is not driven by the cost of tests. If forced to continue, the first seed would have been right in 32% of its in-set deferral states, the second in 67%.
- **Chart artifacts** ([docs/chart_artifacts.json](docs/chart_artifacts.json)). DEFER-Dx receives a "not performed" reply in 12–17% of episodes, and its accuracy on answered cases is lower after one (89.7 vs 95.7): no sign that it exploits test availability.
- **Equity** ([docs/fairness/](docs/fairness/)). Escalation and unflagged errors do not differ systematically by sex, race, insurance or language; patients aged 80+ are escalated more often.
- **Multiple comparisons.** Holm over all 218 paired tests in RESULTS.md §5: the zero-shot comparison survives (< 0.001); the pooled learned-versus-threshold difference reaches 0.052; the time-critical comparisons do not survive and rest on their replication across two seeds. Within the four primary tests (the paper's two questions), Holm p ≤ 0.02.
- **Published case reports** (Fig. S10; [docs/casereports.json](docs/casereports.json); 32 cases × 3 seeds). The second DEFER-Dx seed escalated all 19 out-of-scope cases; the first was misled by two reported traps (ruptured aneurysm with an Alvarado score of 9; heterotopic pregnancy), 8.8% of out-of-scope episodes. The control gave no answer on 21% of out-of-scope episodes, exhausting its test budget on the five cases whose deciding test is outside the catalogue (porphyria, Amanita and thallium poisoning, adrenal crisis, MI without chest pain); the consensus reward called the gallstone ileus cholecystitis. DEFER-Dx escalated complicated in-scope disease (44% and 19% of episodes on 9 cases) and never uncomplicated disease (4 cases). Descriptive: too few cases for precise rates.

**The figures.** Every figure is below (the paper renumbers them: [docs/PAPER.md](docs/PAPER.md)). Vector PDFs for the manuscript, SVG and 300-dpi PNG are in [docs/figures/](docs/figures/).
- **Data figures:** drawn by [scripts/make_figures.py](scripts/make_figures.py) from aggregates only (`docs/results.json`, numbers-only training logs). They are regenerated with every report, so they fill in as runs finish. A figure whose inputs don't exist yet shows a labelled placeholder.
- **Schematics:** TikZ, in [docs/figures/tikz/](docs/figures/tikz/), with shared styles in `deferdx-figures.sty`; build with `bash scripts/build_tikz.sh`.
- **Colour:** each system has one colour in every figure, and DEFER-Dx is always blue. Every series also has a label and its own marker.
- **Intervals:** all are 95% case-level bootstrap intervals (2,000 resamples).

**Figure 1 | DEFER-Dx.**
- **The episode:** one admission. The agent sees the history at presentation. Turn by turn it asks for the examination, orders one of 22 individual tests (each charged at its price, at most 8 investigations), commits to a diagnosis with a stated probability, or defers to a clinician with a ranked, probabilistic differential.
- **Scoring:** a commitment is scored in clinical units: accuracy, Brier score, severity of the error and test cost. A deferral is credited against the counterfactual of not deferring (Fig. 2).
- **Label space:** open, the four MIMIC-CDM conditions or OTHER.

![Figure 1](docs/figures/fig_overview.png)

**Figure 2 | The counterfactual escalation value.**
- **Branching:** when an episode ends in DEFER at state *s*, the environment is replayed to *s*. K = 3 forced continuations are sampled in which escalation is disabled (a DEFER executes as COMMIT of the differential's top) and further tests stay allowed.
- **Credit:** their mean clinical return *V̂(s)* is the value of not escalating from *s*. The DEFER turn's advantage is *A = E − V̂(s)*, where *E* values the escalation itself, including a strictly proper score of the handed-over differential.
- **Effect:** escalation is credited where continuing would err, and discouraged where another test would settle the case.

![Figure 2](docs/figures/fig_cev.png)

**Figure 3 | Data and evaluation design.** Two sources, patient-disjoint cohorts, and what is measured. MIMIC-CDM-OW, built here with MIMIC-CDM's own text pipeline, supplies the OTHER cases and same-pipeline in-set controls. The five time-critical OTHER groups are held out of training entirely.

![Figure 3](docs/figures/fig_data.png)

**Figure 4 | Risk–coverage on CDM val + test.**
- **a:** accuracy on answered in-set cases as the stated-probability threshold is lowered. Each curve ends (filled marker) at the system's own coverage, because a deferral never counts as answered. Hollow markers are forced-choice systems with a post-hoc threshold cross-fitted at the reference system's coverage (vertical line); commitments tied at the cut are kept with a fitted probability, so the threshold reaches that coverage.
- **b:** area under the risk–coverage curve (lower is better).

![Figure 4](docs/figures/fig_risk_coverage.png)

**Figure 5 | Paired differences on the same cases.**
- **Panels:** each compares DEFER-Dx with one comparator, at matched coverage for the thresholded ones. The zero-shot threshold panel uses three-sample voting for both systems, as in the abstract; the single-sample version is in RESULTS.md §5.
- **Two seeds:** circles are the reported run. Diamonds in the first two panels are the second training seed, against the control thresholded at that seed's own coverage. Against the control, the first seed's in-set advantage is just significant and the second's is not. Against the consensus reward, both seeds commit wrongly less often on OTHER cases.
- **Points:** paired bootstrap differences in percentage points (AURC and ECE × 100). They are oriented so that positive values favour the reference system: the sign is flipped for lower-is-better metrics (↓). Filled points have an interval that excludes 0.

![Figure 5](docs/figures/fig_paired_differences.png)

**Figure 6 | Clinical safety on CDM val + test.**
- **a:** unflagged errors, wrong commitments per case.
- **b:** confident errors, wrong commitments stated with p > 0.8.
- **c:** severity-weighted error, from the time-to-harm severity matrix.
- **d:** tests ordered per case.

Hollow markers: forced choice with a post-hoc threshold at matched coverage.

![Figure 6](docs/figures/fig_clinical_safety.png)

**Figure 7 | Open world.** Outcomes on OTHER cases from (a) diagnosis groups seen in training and (b) five time-critical groups never seen in training. The four outcomes are a false commitment to one of the four conditions, a confident false commitment (p > 0.8), deferral to a clinician, and naming OTHER.

![Figure 7](docs/figures/fig_open_world.png)

**Figure 8 | Training dynamics and the coverage controller.**
- **Panels:** (a) coverage multiplier ν; (b) in-set deferral rate in training batches (5-step mean) against the 30% cap; (c) out-of-set deferral rate; (d) in-set coverage on the 212-case dev set at each evaluation.
- **Dual ascent:** with plain dual ascent, ν wound up while the policy lagged, and in-set deferral collapsed after each overshoot: a limit cycle. Dev coverage then depended on where an evaluation fell in the cycle.
- **PI controller:** the method is trained with a PI controller instead (Fig. S1). The consensus arm's cap barely bound.

![Figure 8](docs/figures/fig_training_dynamics.png)

**Figure S1 | One training step.** Multi-turn GRPO with the counterfactual escalation value, and the PI controller that holds in-set deferral at or below the cap.

![Figure S1](docs/figures/fig_training.png)

**Figure S2 | Reliability of stated probabilities** on committed CDM val + test cases. Only bins with at least 10 commitments are drawn, and the diagonal is perfect calibration. ECE is shown with its 95% interval.

![Figure S2](docs/figures/fig_reliability.png)

**Figure S3 | Accuracy at full coverage by condition**, CDM val + test. A deferral counts through the top of its differential.

![Figure S3](docs/figures/fig_per_class.png)

**Figure S4 | Unseen time-critical OTHER groups.** False commitments and deferral by diagnosis group, for groups with at least 10 cases. Ruptured AAA and ectopic pregnancy have fewer and are suppressed under the small-cell rule.

![Figure S4](docs/figures/fig_unseen_groups.png)

**Figure S5 | Ablations at 100 training steps.** DEFER-Dx's step-100 checkpoint against the same training with a constant deferral reward, and without the scored handoff (η = 0). Both are evaluated on 2 seeds with 95% intervals.

![Figure S5](docs/figures/fig_ablations.png)

**Figure S6 | Learned versus post-hoc deferral across training seeds.** Each row compares one DEFER-Dx training seed with one seed of the GRPO control, thresholded at that DEFER-Dx seed's coverage (paired, same cases, matched coverage). The last row pools the four pairs: cases are resampled and the training runs held fixed, so its interval reflects case sampling, and the four rows show the variation between training runs. Positive values favour DEFER-Dx; filled markers have an interval that excludes 0.

![Figure S6](docs/figures/fig_seed_pairings.png)

**Figure S7 | Escalation and unflagged errors by severity.** Complicated (filled) and uncomplicated (hollow) in-set cases; complicated disease from admission codes and laboratory values by the guideline for each condition.

![Figure S7](docs/figures/fig_severity.png)

**Figure S8 | The clinician who receives the handoff.** Team accuracy on in-set cases (a) and AI-caused wrong in-scope diagnoses on never-seen time-critical cases (b) as the probability f that the clinician adopts the handoff's top suggestion rises; otherwise the clinician decides alone with accuracy 0.89. The vertical line marks the adversarial-advocacy effect of Barrit et al. 2026.

![Figure S8](docs/figures/fig_team.png)

**Figure S9 | The escalation-value estimator.** (a) Share of in-set deferral states whose credit has the same sign with K forced continuations as with 16; (b) share of in-set deferrals credited as the accuracy assumed for the receiving clinician varies.

![Figure S9](docs/figures/fig_cev_audit.png)

**Figure S10 | MIMIC-CDM-CR.** Majority outcome over three seeds for each published case and each system: correct, escalated, wrong, or no majority.

![Figure S10](docs/figures/fig_casereports.png)

## 4. Method in brief

- **Actions.** `ASK(physical_exam)`; `TEST(x)` for 22 individual tests (CBC, CMP, lipase, CT abdomen, …) at 2025 BIDMC charges; `COMMIT(d, p)` with d ∈ {4 conditions, OTHER} and stated probability p; `DEFER(differential, reason)`. At most 8 investigations. The full history is shown at reset.
- **Commit reward:** α·1[d = y] − λ(p − 1[d = y])² − κ·C[y, d] − c·Σcost, with α = 1, λ = 1 (Brier), κ = 0.5, C the severity matrix, and c = 1/Σ(catalog prices).
- **Escalation (CEV):** the DEFER turn's advantage is E − V̂(s).
  - V̂(s) is the mean clinical return of K = 3 forced continuations from the same state (no escalation; tests allowed).
  - E = V_τ (OTHER: 1.2) + η·S(handoff) − μ − ν, with S the normalised Brier score of the handed-over differential, η = 0.3, μ = 0.1.
  - τ is annealed 0.95 → 0.85.
- **Comparator arm (published mechanism):** a case-level reward γ(τ − p̂) − μ, with p̂ the group's commit accuracy.
- **Coverage constraint:** in-set deferral ≤ 30%; ν is charged to deferring rollouts only. Projected dual ascent for the comparator; a PI Lagrangian (K_p 1.5, K_i 0.2, ν ≤ 2) for the escalation-value arms, where plain dual ascent wound up and collapsed deferral.
- **Training:** multi-turn GRPO, B = 12 cases × G = 8.
  - Dr. GRPO advantages (r − mean, no std division); groups with reward spread < 0.05 are skipped.
  - Clip-higher (0.2 / 0.28), DAPO token-level loss, no KL.
  - LoRA r = 16 / α = 32 on all linear layers; temperature 1.0 rollouts.

## 5. Decisions made during the build, and why

1. **Colocated vLLM + PEFT trainer** (`training/grpo_vllm.py`), replacing the HF-generate GRPO reference loop, which was too slow for multi-turn RL on one GPU.
   - **How it works:** vLLM samples with the current LoRA while the frozen weights wait in pinned CPU memory; vLLM then sleeps and PEFT trains.
   - **Checked:** vLLM's LoRA log-probs match HF+PEFT within bf16 noise across sleep/wake cycles (mean |Δ log p| 0.024–0.030 per token, against a LoRA effect of 0.45).
2. **Truncated importance sampling** against vLLM's own token log-probs: the engines differ by about 0.02 nats per token.
3. **Memory and speed of the training pass.**
   - **Memory:** a custom frozen-head log-prob autograd function recomputes logits chunk by chunk in the backward pass (gradient-checked), so memory doesn't scale with the 152k vocabulary.
   - **Speed:** one unpadded sequence per micro-batch is 1.55× faster than padded batches (benchmarked: 1,945 vs 1,259 tokens/s).
4. **Learning-rate schedule (all runs inherit it).** Steps 1–25 at 1e-5, then 5e-5 (`lr_milestones`). After 25 steps at 1e-5 the adapter had moved the weights by only 1.3 × 10⁻⁴ of their norm and metrics were flat.
5. **PPO mini-batches (all runs inherit it).** One optimizer step per rollout batch up to step 50, then 4 PPO mini-batches against vLLM's behaviour log-probs (`ppo_minibatch_milestones`). At step 50 the update was still only 6 × 10⁻⁴ of the weights' norm. This is what produced the step-75 gains in §3.
6. **Checkpoints every 5 steps** (multiples of 25 kept). The machine rebooted at step 86; the run resumed from step 75.
7. **Constant-deferral-reward ablation replaces the no-constraint ablation.**
   - The constraint never bound, so a no-constraint run would replicate the main one.
   - The constant arm matches the consensus reward's operating point (0.503 → 0.412 over the τ curriculum), so it isolates the value of the consensus signal itself.
8. **Open-world set made deterministic.** Its SQL row order was not total, so tied lab results resolved differently between builds: 21 of 713 OTHER cases differed between two builds. It is now byte-identical across rebuilds.
9. **`build-openworld --n 2400` is the default.** It samples before its drop rules: `--n 800` yields 259 OTHER cases, not 713.
10. **Cohorts are patient-disjoint.**
    - The time-critical OTHER groups are held out of training entirely.
    - Same-pipeline controls are in training, so "open-world pipeline ⇒ OTHER" cannot be learned.
    - Cases are drawn from source pools by weight (CDM 0.75 / OTHER 0.15 / controls 0.10).

11. **Redesigned the deferral reward for novelty (2026-10-07).**
    - The case-level group-consensus reward turned out to be published: TIAR is identical at τ = 0.5; KARL and AWA-RL are close variants.
    - It was replaced as the method by the counterfactual escalation value with a scored handoff (`configs/grpo_cev.yaml`, `rollout.branch_rollouts`, `training.grpo_vllm.turn_advantages`).
    - The running consensus run became a comparator arm.
    - Ablations now isolate the new parts: no scored handoff, and a constant reward.

### Departures from the original plan

1. **The coverage penalty applies to deferring rollouts only.** A constant subtracted from a whole GRPO group cancels in the advantage.
2. **The open-world DEFER reward is 1.2, not 1.1.** After the handoff cost μ = 0.1, 1.1 would exactly tie a perfect COMMIT(OTHER).
3. **τ is not the effective threshold.** τ = 0.85 means defer below a success estimate of about 0.645; report both numbers.
4. **p̂ with no commits in the group is "neutral" (= τ).** Setting it to "zero" would reward all-defer groups.
5. **There are two "Mean Acc" conventions.**
   - LA-CDM uses the per-class mean; LDTL's own rows are case-weighted.
   - Both are reported (`accuracy_full_coverage`, `mean_class_accuracy`).
   - The plan's "93.4 vs 81.3" crosses both metrics and splits.
6. **Full-coverage accuracy for DEFER** counts the top of the deferral differential.
7. **RL samples are per turn.** Qwen3's template strips earlier `<think>` blocks.
8. **ASK covers the physical exam only, and the full history is shown at reset**, as in LA-CDM and LDTL.
9. **Evaluation samples** at T 0.6, top-p 0.95, top-k 20, as Qwen3 recommends; it never decodes greedily.
10. **No SFT stage.** The base model already uses DEFER when it's offered (25% of CDM cases zero-shot), so RL starts from it directly.
11. **The LDTL "official 70/10/20 split" doesn't exist.** The exact LA-CDM 80/10/10 split is used, and val + test are pooled for power (cross-fitted thresholds).

## 6. Evaluation protocol

- **Sets:**
  - CDM val + test (480; never used in training);
  - OTHER from seen groups (206) and from unseen, time-critical groups (187);
  - same-pipeline controls (258).
  - 3 seeds each, with reproducible per-request seeds.
- **Baselines, all run here on the same cases:**
  - zero-shot Qwen3-8B, forced choice and with prompted DEFER;
  - the GRPO control;
  - post-hoc thresholds at DEFER-Dx's coverage (cross-fitted: fit on val, applied to test and vice versa);
  - SGR (5% risk, δ 0.05);
  - self-consistency over 3 samples;
  - gpt-oss-20b (planned; not run, see §1);
  - DiagAgent-14B and the published LA-CDM / LDTL numbers as context.
- **Machine-learning metrics:** accuracy at full coverage, mean-class accuracy, macro-F1, coverage, selective accuracy, AURC, accuracy at 70/80/90% coverage, ECE, Brier, OOD AUROC.
- **Clinical metrics:**
  - per-class accuracy (diverticulitis);
  - unflagged errors and confident errors (p > 0.8);
  - severity-weighted error;
  - false commits on never-seen, time-critical diagnoses, overall and per diagnosis group (groups ≥ 10 cases);
  - deferral precision and recall;
  - tests and dollars per case.
- **Statistics:** 95% case-level bootstrap intervals; paired bootstrap differences on the same cases.
- **Robustness:**
  - every sentence containing CDM's `____` diagnosis mask removed;
  - same-pipeline controls (source shortcut);
  - closed-world zero-shot, as in prior work.

## 7. Reproduce

```bash
pip install -e ".[train,data,dev]"        # CPU tools + tests: pytest
# GPU stack (one 48 GB GPU; Python 3.12): torch, vllm, transformers, peft in .venv-gpu
uv venv --python 3.12 .venv-gpu && uv pip install --python .venv-gpu/bin/python vllm -e ".[train,data,dev]"
source scripts/env_gpu.sh                 # reads only the HF token from .env; caches in .cache/ (git-ignored)

# Data (credentialed). Interactive prompt; resumes and checks PhysioNet SHA256SUMS:
bash scripts/download_physionet_interactive.sh cdm hosp note
bash scripts/verify_physionet_downloads.sh
deferdx data build-cdm --cdm-dir data/physionet/mimic-iv-ext-cdm/1.1 --out data/cdm      # exact LA-CDM split
deferdx data build-openworld --mimic-dir data/physionet/mimiciv/2.2 --note-dir data/physionet/mimic-iv-note/2.2 \
    --exclude-cases data/cdm/all.jsonl --controls --controls-per-label 3000 --n 2400 --out data/openworld_xl
deferdx data cohorts                      # -> data/cohorts/*.jsonl + manifest.json

# Everything else runs through one idempotent queue (waits for a free GPU; resumes after crashes or reboots):
setsid nohup bash scripts/queue_v4.sh >> outputs/logs/queue_v4.log 2>&1 &

# Single pieces:
deferdx train grpo-vllm --config configs/grpo_cev.yaml            # DEFER-Dx, counterfactual escalation (the method)
deferdx train grpo-vllm --config configs/grpo_deferdx.yaml         # case-level consensus comparator arm
deferdx train grpo-vllm --config configs/grpo_nodefer.yaml         # control
deferdx eval-suite --model Qwen/Qwen3-8B --adapter outputs/runs/deferdx/final --name deferdx
python scripts/make_report.py              # docs/RESULTS.md, docs/results.json and the data figures in docs/figures/
bash scripts/build_tikz.sh                 # the TikZ schematics (pdflatex with tikz, standalone, sansmath)
python scripts/plot_training.py --runs deferdx=outputs/runs/deferdx nodefer=outputs/runs/nodefer
deferdx crossover                                                   # tau -> effective deferral threshold
```

**Keeping GitHub and the Hugging Face Hub in sync** (a CPU-only process alongside the queue):
```bash
setsid nohup python scripts/sync_daemon.py --interval 900 >> outputs/logs/sync.log 2>&1 &
python scripts/push_hf.py --run cev          # or push one run by hand
```
- **GitHub:** every 15 minutes the regenerated, aggregate-only results (`docs/RESULTS.md`, `docs/results.json`, `docs/figures/`) are committed and pushed to `main`. An allow-list means nothing else can be staged. While a queue runs, a push that would rebase remote code changes into the working tree is held back until the queue ends.
- **Hugging Face:** each finished run's LoRA adapter goes to a **private** model repo under the account of the `hf` token in `.env`, with a model card (provenance, DUA status, intended use, held-out metrics) and the numbers-only training log. Cards refresh whenever results change.
- **Why private:** the weights derive from PhysioNet credentialed data, so a public release belongs on PhysioNet's credentialed channel. Rollouts, cases and MIMIC text are never uploaded.

| Run | Hugging Face repo (private) |
|---|---|
| Comparator (case-level consensus) | `GOVINDFROM/deferdx-consensus-qwen3-8b-lora` |
| DEFER-Dx (counterfactual escalation) | `GOVINDFROM/deferdx-cev-qwen3-8b-lora` (when trained); the first, dual-ascent run's checkpoints are on its `dual-ascent-*` branches |
| No-defer control | `GOVINDFROM/deferdx-nodefer-control-qwen3-8b-lora` (when trained) |
| Ablations | `GOVINDFROM/deferdx-abl-*-qwen3-8b-lora` (when trained) |

**After a reboot,** relaunch the queue command and the sync command above. Steps marked done in `outputs/queue/*.done` are skipped, and training resumes from its newest checkpoint (one is saved every 5 steps).

**Synthetic smoke tests (no MIMIC data):** `bash scripts/smoke_synthetic.sh`; `deferdx data synth`.

## 8. Layout

| Path | What it does |
|---|---|
| [src/deferdx/data/](src/deferdx/data/) | CDM loader (earliest value per repeated test), open-world builder (DuckDB; CDM-parity text pipeline in `parity.py`), patient-disjoint cohorts (`cohorts.py`), LA-CDM split, synthetic cases |
| [src/deferdx/env/](src/deferdx/env/) | Reveal-on-request POMDP; 22-test catalog matched by MIMIC itemid; action parser; `mask_policy` for the `____` cue |
| [src/deferdx/rewards/](src/deferdx/rewards/) | Commit reward, group-consensus deferral reward (`p_hat_mode`, `defer_mode`), crossover analysis, coverage constraint |
| [src/deferdx/policy/](src/deferdx/policy/) | Local vLLM (LoRA hot-swap, per-request seeds, token log-probs) and HF policies; scripted oracle/random |
| [src/deferdx/training/grpo_vllm.py](src/deferdx/training/grpo_vllm.py) | Colocated GRPO: sleep/wake, frozen-weight parking, PPO mini-batches, truncated IS, LR and mini-batch milestones, OOM split-and-retry, checkpoint/resume |
| [src/deferdx/training/common.py](src/deferdx/training/common.py) | Frozen-head log-prob autograd function, token-budget micro-batching |
| [src/deferdx/eval/](src/deferdx/eval/) | Metrics; evaluation suites; bootstrap statistics and clinical metrics; self-consistency; cross-fitted post-hoc thresholds; per-group tables |
| [src/deferdx/baselines/](src/deferdx/baselines/) | Post-hoc threshold and SGR |
| [configs/](configs/) | Environment, reward, catalog, severity matrix, ICD lists; `grpo_deferdx.yaml`, `grpo_nodefer.yaml`, `ablations/` |
| [scripts/](scripts/) | Queues, report, figures, training plots, data audit, downloads, Lambda baseline jobs |
| [docs/figures/](docs/figures/) | Publication figures (PDF, SVG, PNG); TikZ sources and shared styles in `tikz/` |
| [tests/](tests/) | 117 CPU tests (`pytest`) |

## 9. Documents

| Doc | What it holds |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Environment, reward with constants, cohorts, training, evaluation protocol |
| [docs/RESULTS.md](docs/RESULTS.md) | ML-benchmark and clinical tables with intervals, open world, published context, paired tests (generated) |
| [docs/figures/](docs/figures/) | All figures; captions in §3 of this README |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | The run grid, queues, ablations and the question each answers |
| [docs/CONSENSUS_SEED1.md](docs/CONSENSUS_SEED1.md) | The consensus arm's second training seed (Lambda Cloud): setup, cost, data handling, in-training dev evaluations, every training step; no held-out numbers yet |
| [docs/REVIEWER_CHECKS.md](docs/REVIEWER_CHECKS.md) | Reviewer comments of Oct 14: chart-artifact audit, test-price sensitivity, Holm correction, and the status of every comment |
| [docs/PAPER.md](docs/PAPER.md), [docs/ABSTRACT.md](docs/ABSTRACT.md) | Paper draft and abstract scaffold (placeholders until results) |
| [docs/RESEARCH.md](docs/RESEARCH.md) | Facts checked against primary sources; literature re-checks and the narrowed novelty claim |
| [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md) | Inventory, schemas, label and leakage checks, bias analysis |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Metrics and best published results per component |
| [docs/BASELINES.md](docs/BASELINES.md) | Earlier inference-only baselines (Lambda A100 runs) |

## 10. Checked vs. still to verify

**Resolved:**
- **The dataset:** CDM v1.1 files, columns and labels (`pathology_ids.json`); no official split exists, so LA-CDM's split is reproduced exactly; CDM's text pipeline and inclusion rule, replicated for the open world; BIDMC costs for the panels and imaging.
- **Catalog coverage:** the core panels resolve for ≥ 99.8% of cases; `hida_scan` never resolves (CDM has no HIDA reports). Urine blood, bacteria and yeast are never reachable, a faithful copy of Hager's urinalysis panel.
- **Source shortcut:** a classifier separates same-pipeline controls from CDM cases only weakly (AUROC 0.563); the physical exam alone reaches 0.673.

**Still to verify:**
- **Placeholder costs:** tests marked `cost_source: placeholder` need real prices.
- **The severity matrix** is a time-to-harm proxy; it needs a clinician-built version.
- **Clinician review** of the ICD lists and sanitize terms.
- **Clinician adjudication** of deferrals (plan §6.3).
- **Informative test availability:** a test's existence can hint at the diagnosis. Report `charge_unavailable` both ways.

## 11. Limitations and not built yet

- **Scope:** single centre (BIDMC), abdominal pain only; proxy labels (principal ICD for OTHER).
- **Statistics:** 51 diverticulitis cases in val + test; one training seed per arm until the seed replicate finishes.
- **Not run:**
  - **MedGemma-27B:** gated; the configured HF account has not accepted its terms.
  - **LA-CDM re-run:** its repo has no released weights.
  - **The MIMIC-IV-ED triage variant, a Med-PRM verifier, a multi-GPU/verl port.**
