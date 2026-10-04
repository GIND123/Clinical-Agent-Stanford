# Evaluation landscape: metrics, benchmarks and state of the art

Checked 2026-09-30. This answers two questions: which metrics and benchmarks evaluate each part of DEFER-Dx, and who holds the best published result on each. The third step, running existing models on these benchmarks without any training, is in §3 and needs GPUs.

**How the numbers were checked.** Numbers come from each paper's own arXiv page (abstract, HTML tables or text). A few come only from a search summary and are marked †; confirm those against the PDF before citing. Extraction was done by reading tool summaries of the pages, so re-check any number that goes into the abstract.

## 1. Component → metrics → benchmarks

| Component of DEFER-Dx | Standard metrics | Benchmarks that measure it |
|---|---|---|
| **End-to-end sequential diagnosis** | per-class accuracy, mean accuracy, macro-F1 | **MIMIC-CDM** (our environment), DiagBench, AgentClinic, MediQ, DDXPlus |
| **TEST: choosing investigations** | tests per case, test cost ($), steps to decision, one-step terminations; exam-recommendation precision/recall/F1 | MIMIC-CDM (LA-CDM cost, LDTL steps), DiagBench, MIMIC-ED-Assist (ED-Copilot, time cost) |
| **ASK: gathering history** | accuracy vs. number of questions, accuracy-turn AUC | MediQ (iMedQA, iCraft-MD), interactive DDXPlus, AgentClinic. In our environment ASK reveals the physical exam, so MIMIC-CDM covers it; these test dialogue-style asking. |
| **COMMIT confidence** | ECE, Brier, AUROC of confidence (correct vs. wrong) | MIMIC-CDM (LA-CDM reports ECE), MedQA (Rewarding Doubt) |
| **DEFER: handing off** | risk-coverage curve, AURC, selective error / accuracy at a coverage, deferral precision/recall, confident-error rate | MIMIC-derived abdominal pain (Safe-to-Stop), MedAbstain, Safe-Psych, severity-aware conformal planning (DDXPlus, MediQ) |
| **Open world (OTHER)** | false-commit rate on out-of-set cases, OOD AUROC | **none exists**; MIMIC-CDM-OW is our contribution |
| **Safety** | severity-weighted error, severe/high-risk error rate, missed red flags | severity-aware conformal planning (DDXPlus), ResidencyRL (internal) |
| **Fairness** | per-group accuracy, coverage and deferral gaps | no leaderboard; see [DATA_AUDIT.md](DATA_AUDIT.md) §7 for the cohort's own gaps |
| **Base model** (context only) | accuracy / rubric score | MedQA, HealthBench |

Two definitions to fix before comparing anything (details in [RESEARCH.md](RESEARCH.md) §1-3):
- "Mean accuracy" is the unweighted mean of per-class accuracy in LA-CDM, but case-weighted in LDTL's own rows.
- There is no official MIMIC-CDM split. LA-CDM uses 80/10/10 (seed 269, reproduced in `splits.py`); LDTL uses an unpublished 70/10/20.

## 2. State of the art per benchmark

### 2.1 MIMIC-CDM, sequential (our benchmark)

| Method | Trained? | App. | Chol. | Div. | Panc. | Mean | F1 | Other | Split | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-Chat, in LDTL's loop | no (API) | 99.4 | 97.6 | **92.3** | 92.5 | **96.6** | 96.6 | | 70/10/20 | LDTL Tab. 3 |
| Claude-Sonnet-4, in LDTL's loop | no (API) | 99.3 | 98.4 | 90.3 | 88.8 | 95.8 | 95.8 | | 70/10/20 | LDTL Tab. 3 |
| GPT-4o, in LDTL's loop | no (API) | 99.4 | 98.4 | 88.4 | 88.8 | 95.6 | 95.7 | | 70/10/20 | LDTL Tab. 3 |
| Gemini-3-Flash, in LDTL's loop | no (API) | 99.4 | 94.6 | 90.3 | 90.7 | 95.2 | 95.3 | | 70/10/20 | LDTL Tab. 3 |
| GPT-4o-mini, in LDTL's loop | no (API) | 94.7 | 91.5 | 82.6 | 86.1 | 90.5 | 90.6 | | 70/10/20 | LDTL Tab. 3 |
| **LDTL** (Llama-3-8B, LoRA) | yes | 98.9 | 95.4 | 78.8 | 87.9 | **93.4** | 91.7 | ≤3 steps; 318 one-step stops | 70/10/20 | LDTL Tab. 1 |
| **LA-CDM** (ICLR 2026) | yes (SFT+RL) | 93.1 | 83.6 | 75.0 | 73.5 | 81.3 | 81.3 (macro) | $1,295.61 tests/case; ECE 0.069 → 0.037 | 80/10/10 | LA-CDM Tab. 1 |
| ReAct | no | 90.2 | 79.7 | 66.7 | 62.9 | 74.9 | 74.8 (macro) | $1,480.32 | 80/10/10 | LA-CDM Tab. 1 |
| LA-CDM, zero-shot | no | 73.5 | 55.1 | 72.0 | 57.5 | 64.5 | 64.5 (macro) | $1,521.73 | 80/10/10 | LA-CDM Tab. 1 |
| SFT-all (all information; upper bound) | yes | 98.4 | 89.8 | 95.8 | 87.5 | 92.8 | 92.9 (macro) | $3,792.79 | 80/10/10 | LA-CDM Tab. 1 |

Related results that are **not** on the same footing:
- **DxChain** (arXiv 2604.23605, inference-only GPT-4.1-mini): 90.67% primary-diagnosis accuracy, but with **all information given at once** on all 2,400 cases. It isn't sequential, so it's not comparable.
- **Safe-to-Stop** (arXiv 2609.09678, Sept 2026): an **inference-only** risk-constrained stopping layer on a MIMIC-derived abdominal-pain set (1,834 train / 367 eval episodes). It reports **16.9% selective error at 78.8% coverage, 0.68 tests per case**, against 30.8% error at 100% coverage and 1.53 tests for the base agent. Its state-error AUROC is 0.853, against 0.715 for max-probability. It is the closest published work to our DEFER axis on this data; the authors call it feasibility evidence, not a confirmatory result.

What this means for the abstract:
1. **The best mean accuracy on MIMIC-CDM comes from untrained frontier models (96.6)**, not from a trained agent. It's also the best diverticulitis number (92.3).
2. Those frontier rows **cannot be reproduced on MIMIC text through normal APIs** (PhysioNet DUA). Our comparisons are therefore against open weights, plus the numbers as reported.
3. Nobody reports AURC or false-commit-on-OTHER on MIMIC-CDM. Only LA-CDM reports calibration (ECE), and Safe-to-Stop is the only paper with selective-risk numbers.

### 2.2 DiagBench (DiagGym, arXiv 2510.24654)
2,257 physician-validated cases: MIMIC-IV 750, PMC-OA 631, MTSamples 379, DDXPlus 497. Public on Hugging Face (MIT licence), but the MIMIC-IV subset is MIMIC-derived; treat it as credentialed.

| Setting (MIMIC-IV, in-domain) | DiagAgent-14B (SOTA) |
|---|---|
| Single turn: exam hit ratio / diagnosis accuracy | 68.49% / 87.87% |
| End-to-end: exam-recommendation F1 / diagnosis accuracy | 43.72% / ~62.6% (the paper gives 62.63% in text; a table reading gave 62.67%) |
| End-to-end: weighted rubric score | 32.86% |
| Out-of-domain: accuracy / rubric score | 63.84% / 50.27, vs Claude-4-Sonnet 60.12% / 39.07 |

The paper headlines "+11.20% diagnostic accuracy, +17.58% exam-recommendation F1 over 11 LLMs". Baseline rows are in its Supplementary Table 4, which I haven't checked.

### 2.3 AgentClinic (arXiv 2405.07960 v5; npj Digital Medicine 2026)
Diagnostic accuracy with an LLM-simulated patient:

| Variant | Best | Next best |
|---|---|---|
| AgentClinic-MedQA | Claude-3.5-Sonnet 62.1 ± 3.3 | OpenBioLLM-70B 58.3 ± 4.2 (human physicians: 54 ± 28.5) |
| AgentClinic-MIMIC-IV | Claude-3.5 42.9 ± 3.3 | OpenBioLLM-70B 38.1 ± 3.2 |
| AgentClinic-NEJM (images) | Claude-3.5-Sonnet 37.2 ± 2.2 | GPT-4 27.7 ± 2.0 |

On AgentClinic-MedQA as an out-of-distribution test, MedExAgent-8B (arXiv 2605.07058) scores 37.8% strict accuracy against Aloe-Beta-70B's 39.5%, under its own protocol. ResidencyRL (arXiv 2608.07418, Aug 2026) trains on AgentClinic-style simulators, but its abstract gives no AgentClinic number.

### 2.4 MediQ (arXiv 2406.00922, NeurIPS 2024)

| | iMedQA | iCraft-MD |
|---|---|---|
| All information given (upper bound, GPT-4) | 79.7 | 91.4 |
| Initial information only (GPT-4) | 54.5 | 67.9 |
| Best interactive (GPT-4 + abstention "Scale+RG+SC") | 66.1 | 84.3 |

Severity-aware conformal planning (arXiv 2608.27847, Aug 2026, Llama-3.1-8B, inference-only) reports 66.18% accuracy on MediQ with 6.57 questions, 91.54% stop-time coverage and a 1.30-point gap to all-information.

### 2.5 DDXPlus, interactive
- Severity-aware conformal planning (arXiv 2608.27847): **Llama-3.1-8B 98.57% accuracy, 4.77 questions, severe-error 0.00%**; Qwen3-4B 97.76%; Mistral-7B 97.55%. Also DDx-F1 ≈ 0.60.
- MEDDxAgent (ACL 2025, arXiv 2502.19175): GTPA@1 **0.86** (GPT-4o, 15 questions). The same system gets 0.54 on iCraft-MD and 0.56 on RareBench.
- The protocols differ (starting evidence, question budget), so these two aren't directly comparable.

### 2.6 Abstention and calibration

| Work | What it measures | Best reported |
|---|---|---|
| MedAbstain (arXiv 2601.12471) | MedQA + AMBOSS MCQA with an abstain option: accuracy, conformal set size (LAC, APS), abstention rate | No leaderboard. Human-evaluated abstention precision 71.43%, recall 13.16%. No AURC. |
| Safe-Psych (arXiv 2607.13036, May 2026) | Psychiatry notes revealed in stages, with DIAGNOSE / CLARIFY / ABSTAIN labels | Under-abstention above 60% for most models |
| Abstain-R1 (arXiv 2604.17073, ACL 2026) | General-domain RL for abstention (Abstain-Test, Abstain-QA, SelfAware) | Precedent for RL-learned abstention; not medical |
| Rewarding Doubt (arXiv 2503.02623, Llama-3-8B) | Calibration via a log-score RL reward: ECE, AUROC | MedQA (zero-shot transfer): ECE 0.1145, AUROC 0.6649. TriviaQA: ECE 0.0226, AUROC 0.8592. |
| LA-CDM | ECE on MIMIC-CDM | 0.069 → 0.037 after calibration training |

### 2.7 Test cost and emergency department
- ED-Copilot (ICML 2024, MIMIC-ED-Assist, RL test selection): halves average time cost from about 4 h to 2 h while improving prediction accuracy†. Exact F1/AUC not extracted.
- On MIMIC-CDM, cost is reported only by LA-CDM (§2.1 table, 2025 BIDMC charges) and step counts by LDTL.

### 2.8 Context only: not part of our environment
- Doctor-R1 (arXiv 2510.04284, ICLR 2026): HealthBench average 36.29, against GPT-4.1 31.18, Grok-4 33.03 and Claude Sonnet 4 25.69†. This measures conversational inquiry quality.
- MedQA and HealthBench leaderboards: the aggregator sites disagree with each other and I found no primary source, so no number is given here.

## 3. Inference-only runs (step 3, needs GPUs)

**Run on 2026-10-01; results are in [BASELINES.md](BASELINES.md).** That run covered Qwen3-8B and DiagAgent-14B on MIMIC-CDM, the post-hoc baselines and DiagBench. Qwen3-14B, MedGemma-27B, gpt-oss-20b, MediQ, DDXPlus and AgentClinic are not run yet.

| Benchmark | Data access | Runs in this repo? | Models (open weights) | Priority |
|---|---|---|---|---|
| MIMIC-CDM: all 2,400 cases, plus the LA-CDM split (240 val / 240 test) for comparability | credentialed (have it) | yes: `deferdx rollout --policy vllm --model <m> --no-defer`, then `scripts/subgroup_eval.py` | Qwen3-8B, Qwen3-14B, DiagAgent-14B, MedGemma-27B-text-it, gpt-oss-20b | **1** |
| Post-hoc threshold (#6) and SGR conformal (#7) on those rollouts | same | yes: `deferdx baseline` (CPU only) | — | **1** |
| DiagBench (to check our harness reproduces DiagAgent-14B's published numbers) | Hugging Face, public; MIMIC subset treated as credentialed | no (DiagGym repo) | DiagAgent-14B | 2 |
| MediQ, DDXPlus (external check on ASK-style behaviour) | public | no (their repos) | same open models | 3 |
| AgentClinic | public | no; needs an LLM patient simulator | same | 4 |

Rules for these runs:
- Frontier API models (GPT-4o, Claude, DeepSeek) may be run only on the public benchmarks, never on MIMIC text, unless the team uses a PhysioNet-approved route.
- Before MIMIC data goes onto a Lambda Cloud VM, check that the institution's DUA terms allow third-party cloud compute (listed as open in `stanford_idea_md.md`).
- Report mean ± sd over 3 seeds with Qwen3's sampling settings (`docs/EXPERIMENTS.md`). Read every MIMIC-CDM accuracy against the demographics-only floor of **47.8%** accuracy (majority class 39.9%; macro-AUROC 0.68). That floor is from `DATA_AUDIT.md` §7.3.

### 3.1 Data readiness (checked 2026-09-30)

| Data | Status | Where (git-ignored) | Notes |
|---|---|---|---|
| MIMIC-IV-Ext-CDM 1.1 | ✅ checksums match | `data/physionet/mimic-iv-ext-cdm/1.1/` | built into `data/cdm/{all,train,val,test}.jsonl` (LA-CDM split 1,920 / 240 / 240) |
| MIMIC-IV 2.2 hosp (admissions, patients, transfers, diagnoses_icd, labevents, microbiologyevents, dictionaries) | ✅ checksums match | `data/physionet/mimiciv/2.2/hosp/` | demographics for per-group evaluation; open-world cohort (`transfers` gives CDM's time window) |
| MIMIC-IV-Note 2.2 (discharge, radiology, radiology_detail) | ✅ checksums match | `data/physionet/mimic-iv-note/2.2/note/` | open-world cohort (`radiology_detail` gives CDM's exam names) |
| MIMIC-CDM-OW (OTHER cases) | ✅ built 2026-10-04 | `data/openworld/` | 713 OTHER cases + 51 same-pipeline controls (`--n 2400`); identical across two builds |
| DiagBench (4 subsets, 2,257 cases) | ✅ | `data/public/diagbench/` | the MIMIC-IV subset is MIMIC text; treat it as credentialed |
| MediQ (iMedQA `all_dev_good`, iCraft-MD) | ✅ | `data/public/mediq/` | |
| DDXPlus (test, validate, evidences, conditions) | ✅ | `data/public/ddxplus/` | training split not downloaded (no training) |
| AgentClinic (MedQA, NEJM, extended) | ✅ | `data/public/agentclinic/` | **AgentClinic-MIMIC-IV is not public**; it needs a credentialed rebuild |
| Model weights | ❌ not downloaded | — | 8B–27B don't fit this laptop (24 GB, already swapping); download them on the GPU machine |

Checksums for the public files are in `data/public/MANIFEST.sha256`.

### 3.2 Evaluation protocol required by the data audit
These follow from [DATA_AUDIT.md](DATA_AUDIT.md) and from running the pipeline on the real data.

1. **Score on all 2,400 cases, not only the 240-case test set.** Zero-shot baselines see no training data, so every case is a fair test case. The test split has 25 diverticulitis cases (±~16 points at 80% accuracy); all cases give 257 (±~5). Still report the LA-CDM test split so the numbers compare with LA-CDM.
2. **Intervals and groups:** run `scripts/subgroup_eval.py` on every rollout file. It gives 95% bootstrap intervals and accuracy per class, sex, age, race, insurance and language, with small groups hidden. The audit found label-adjusted differences in what the environment reveals: CT 61% (age 18–29) vs 78% (65–79), and microbiology 69% vs 93%. Per-group accuracy shows whether models inherit these gaps.
3. **The mask cue:** `subgroup_eval.py` also splits accuracy by whether a case's radiology contains CDM's `____` mask (present in 27–51% of cases depending on class). Higher accuracy with the mask, within a class, means the model reads the mask rather than the findings.
4. **Environment limits, the same for every baseline; decide before the runs whether to change them, not midway:**
   - `dedup_earliest` hides 1,337 radiology reports in 412 cases. CDM has no `charttime`, so it keeps the first in file order.
   - `hida_scan` never returns a result: CDM has 0 HIDA or nuclear reports.
   - Anion Gap (in 1,917 of 1,920 training cases), eGFR, Magnesium, LDH and Troponin exist but no catalog test reaches them.

   Fixing these changes the action space relative to LA-CDM.
5. **Baseline naming:** the repo's `random` policy also diagnoses at random. On the real test split it scores 25.8%, which is chance, not LDTL's 84.8% "random planner" (random tests plus an LLM diagnosis). Don't put them in the same row.
6. **Contamination:**
   - DiagAgent-14B was trained on 118k MIMIC-IV records. Its overlap with CDM's 2,400 admissions can't be checked; report its MIMIC-CDM numbers with that caveat.
   - DiagBench's public MIMIC subset contains **19 CDM admissions** (matched through MIMIC-IV-Note `note_id`), and 62 of its 750 cases have a final diagnosis naming a CDM condition. It has been public since Oct 2025, so newer models may have seen those cases.
7. **OTHER metrics:** false-commit and OOD AUROC need MIMIC-CDM-OW. On 2026-10-04 `build_openworld` was changed to build it the way CDM was built:
   - A sorted cohort query, so the sample is reproducible. It was not before.
   - Rows without a `hadm_id`, timed from one day before the first transfer to the last. Before this, the join found microbiology for 32% of CDM admissions against CDM's 77%; now it finds exactly what CDM holds.
   - Exam names from `radiology_detail`. Before this, 52% of OTHER cases lost all imaging.
   - CDM's inclusion rule.

   OTHER cases now match CDM on imaging, labs and exam length. A text classifier separates same-pipeline controls from label-matched CDM cases at AUROC 0.563, with the physical exam alone still at 0.673. Results are in [BASELINES.md](BASELINES.md).
8. **Pipeline checks already run on real data:**
   - The oracle policy scores 100% (environment and scoring are consistent).
   - Qwen3-0.6B completes episodes end to end (2 of 8 smoke-test episodes broke the action format, expected at that size).
   - `coverage` resolves the core panels for ≥99.8% of cases.

## 4. Where DEFER-Dx still has open ground
- **Learned deferral on MIMIC-CDM:** Safe-to-Stop does *inference-only* stopping with risk control on the same kind of data. Nobody *trains* a DEFER action with its own reward. Cite Safe-to-Stop as the closest prior work and compare at matched coverage.
- **Calibration on MIMIC-CDM:** only LA-CDM reports it (ECE). Nobody reports Brier, AURC or the confident-error rate.
- **Open world:** there is no out-of-set benchmark for any sequential diagnosis agent.
- **Watch list:** Safe-to-Stop (2609.09678), severity-aware conformal planning (2608.27847, ask/commit with severity-weighted risk) and ResidencyRL (2608.07418, RL with safety rewards) all appeared in Aug–Sep 2026. Re-check arXiv before submitting.

## Sources
- LDTL: https://arxiv.org/abs/2604.05116 (tables: https://arxiv.org/html/2604.05116)
- LA-CDM: https://arxiv.org/abs/2506.13474 (tables: https://arxiv.org/html/2506.13474v3); ICLR 2026 poster https://iclr.cc/virtual/2026/poster/10011252
- DxChain: https://arxiv.org/abs/2604.23605
- Safe-to-Stop: https://arxiv.org/abs/2609.09678
- DiagGym / DiagBench: https://arxiv.org/abs/2510.24654, https://huggingface.co/datasets/Henrychur/DiagBench
- AgentClinic: https://arxiv.org/abs/2405.07960, https://agentclinic.github.io/
- MedExAgent: https://arxiv.org/abs/2605.07058
- ResidencyRL: https://arxiv.org/abs/2608.07418
- MediQ: https://arxiv.org/abs/2406.00922
- Severity-aware conformal planning: https://arxiv.org/abs/2608.27847
- MEDDxAgent: https://arxiv.org/abs/2502.19175
- MedAbstain: https://arxiv.org/abs/2601.12471
- Safe-Psych: https://arxiv.org/abs/2607.13036
- Abstain-R1: https://arxiv.org/abs/2604.17073
- Rewarding Doubt: https://arxiv.org/abs/2503.02623
- ED-Copilot: https://arxiv.org/abs/2402.13448
- Doctor-R1: https://arxiv.org/abs/2510.04284
