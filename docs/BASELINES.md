# Inference-only baselines (no training)

Round 1 ran on 2026-10-01 and round 2 on 2026-10-04. This is step 3 of the evaluation plan in [BENCHMARKS.md](BENCHMARKS.md): run existing models as-is, with no fine-tuning, so the trained agent has numbers to beat. Every number here is an aggregate. Rollout files contain MIMIC text and stay in git-ignored `outputs/`.

## Setup

| | |
|---|---|
| Hardware | Lambda Cloud, 1× A100 40 GB (SXM4), us-east-1. Round 1 ran on one instance (two earlier launches never started the job). Round 2 ran on two: the first was terminated after the open-world runs so it wouldn't bill unattended, and the rest resumed on a second. All are terminated and their disks deleted. |
| Software | vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6. The pins are needed: the latest vLLM pulls a CUDA 13 torch that Lambda's 570 driver can't run, and transformers 5.x removed a tokenizer attribute vLLM 0.11 reads. |
| Code | Round 1: this repo at `d32371b`, with [`scripts/lambda/run_all.sh`](../scripts/lambda/run_all.sh). Round 2: `67a6bb8`, then `1deb557` for the resumed run (it only adds `SKIP` to the job script), with [`scripts/lambda/run_all2.sh`](../scripts/lambda/run_all2.sh). The environment is unchanged in both rounds. |
| Data on the VM | Built case files only: CDM (`all.jsonl`, `test.jsonl`), the open-world set (`other.jsonl`, `controls.jsonl`) and DiagBench. No raw MIMIC tables. The uploaded case files are byte-identical to the local builds (same SHA-1). |
| MIMIC-CDM settings | `deferdx rollout --policy vllm --no-defer --closed-world` with the default catalog, `max_steps` 8 and 1,024 new tokens per turn. Qwen3 models use their recommended thinking-mode sampling (T 0.6, top-p 0.95, top-k 20). Qwen3-14B also caps the context at 16,384 tokens to leave room for the KV cache; the longest Qwen3-8B episode was 5,253 tokens. DiagAgent-14B uses greedy decoding, as its authors do. |
| Scoring | `deferdx evaluate`, `deferdx baseline`, `scripts/subgroup_eval.py` |
| Cost | GPU: round 1 about $9.0; round 2 about $9.4 ($3.9 and $5.5 on its two instances). About $18.4 of the $20 budget in total, estimated from instance time at $1.99/h. No paid API was used. |

Run log (UTC):
- **Round 1:** Qwen3-8B, all cases 00:37–01:39 · DiagAgent-14B, all cases 01:41–02:35 · DiagBench generation and scoring 02:36–02:46 · Qwen3-8B seeds 1–2 on the test split 02:46–03:01.
- **Round 2:** DiagAgent-14B through its adapter, all cases 19:00–19:54 · open world without DEFER 19:54–20:23 · open world with DEFER 20:23–20:50 · (first instance terminated; resumed on a second) · DiagBench re-score 21:44–21:48 · gpt-oss-20b smoke test 21:48–21:50 · Qwen3-14B smoke test 21:50–21:54 · Qwen3-14B, all cases 21:54–00:22.

## 1. MIMIC-CDM in the DEFER-Dx environment

Forced commit, four classes. "Mean-class" is the unweighted mean of per-class accuracy (LA-CDM's convention); "case" is case-weighted. Invalid episodes count as wrong.

| Model | Cases | Seeds | Mean-class acc. | Case acc. | App. / Chol. / Div. / Panc. | Macro-F1 | ECE | Brier | AURC | Invalid | Tests / case | $ / case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B, zero-shot | all 2,400 | 1 | 84.1 | 85.8 (95% CI 84.3–87.1) | 95.1 / 80.9 / 84.8 / 75.5 | 85.0 | 0.067 | 0.085 | 0.037 | 4.0% | 1.95 | 1,221 |
| Qwen3-8B, zero-shot | LA-CDM test (240) | 3 | **87.2 ± 2.4** | 88.3 ± 1.1 | 93.8 / 85.1 / 86.7 ± 12.9 / 83.3 | 87.9 ± 1.8 | 0.102 | 0.070 | 0.021 | 4.3% | 1.99 | 1,192 |
| Qwen3-14B, zero-shot | all 2,400 | 1 | 85.0 | 86.4 (95% CI 84.9–87.8) | 92.5 / 85.8 / 84.4 / 77.1 | 86.4 | 0.103 | 0.091 | 0.036 | 4.8% | 2.22 | 1,268 |
| Qwen3-14B, zero-shot | LA-CDM test (240) | 1 | 88.5 | 88.3 | 92.7 / 89.2 / 96.0 / 75.9 | 88.9 | 0.138 | 0.083 | 0.024 | 5.4% | 2.30 | 1,262 |
| DiagAgent-14B, through its format adapter | all 2,400 | 1 (greedy) | 69.0 | 75.6 (95% CI 73.9–77.3) | 92.3 / 70.1 / 48.2 / 65.6 | 77.7 | n/a | n/a | n/a | 15.0% | 5.49 | 2,630 |
| DiagAgent-14B, through its format adapter | LA-CDM test (240) | 1 (greedy) | 71.9 | 77.9 | 94.8 / 73.8 / 56.0 / 63.0 | 80.8 | n/a | n/a | n/a | 15.4% | 5.70 | 2,644 |
| DiagAgent-14B, in this environment's action format (round 1) | all 2,400 | 1 | 0.0 | 0.0 | — | — | — | — | — | **100%** | 4.13 | — |

The LA-CDM test rows for Qwen3-14B and the DiagAgent adapter are the test-split cases of their all-case runs, not separate runs.

For reference, reported in the papers and not re-run:

| | Split | Metric | Number |
|---|---|---|---|
| LA-CDM, trained (ICLR 2026) | LA-CDM test | mean-class | 81.3 |
| LA-CDM, zero-shot | LA-CDM test | mean-class | 64.5 |
| ReAct | LA-CDM test | mean-class | 74.9 |
| LDTL, trained | its own 70/10/20 split | case-weighted | 93.4 |

How to read this:

- **Qwen3-8B with no training scores 87.2 on the LA-CDM test split, above LA-CDM's trained 81.3. That is not a like-for-like win.** The environments differ: this one shows the full history at the start (LA-CDM gives an LLM-written summary), offers 22 tests instead of 12, and uses a different prompt and base model (Qwen3-8B with thinking, against Qwen2.5-7B). The fair conclusion is that this environment plus Qwen3-8B is a strong starting point. The trained agent should be compared with it, not with 81.3.
- **Diverticulitis varies 96 / 92 / 72 across the three seeds, on 25 test cases.** On all 2,400 cases (257 diverticulitis) it is 84.8 (CI 80.5–89.1). Use the all-case numbers for per-class claims.
- **Qwen3-14B is not reliably better than Qwen3-8B.** On all 2,400 cases it scores 86.4 against 85.8 (one seed each).
  - Case by case, 154 are right only with 14B and 139 only with 8B (McNemar p ≈ 0.38).
  - It does better on cholecystitis (85.8 against 80.9) and worse on appendicitis (92.5 against 95.1).
  - Its stated confidence is less well calibrated (ECE 0.103 against 0.067).
  - Model size alone moves this environment little.
- **DiagAgent-14B, run in its own format, reaches 75.6% case accuracy (69.0 mean-class).** That is 10 points below zero-shot Qwen3-8B. In round 1 it was prompted with this environment's action format and never produced a valid action (0 of 2,400). The adapter ([`scripts/lambda/diagagent_adapter.py`](../scripts/lambda/diagagent_adapter.py)) changes only what DiagAgent sees and how its replies are read:
  - It shows DiagGym's own system prompt, and each test result as "Here is the test result: …".
  - It reads "Diagnosis: Y" as COMMIT, and a recommended exam as TEST or ASK, using keyword rules over the catalog.

  The environment, catalog and scoring are unchanged.
- **When DiagAgent commits it is right 92.1% of the time. Most of the gap is answers this environment can't accept.** 15.0% of episodes are invalid:
  - 218 final diagnoses outside the four classes: 62 other biliary or gallstone conditions, 28 that name one of the four but don't normalise to it, and 128 other conditions.
  - 40 recommended exams that no catalog test matches.
  - 102 replies with neither a diagnosis nor an exam line.

  Diverticulitis is hit hardest: 38.1% of its episodes are invalid. Counting as correct the 25 replies that name only the right class lifts case accuracy to 76.6, so the adapter's strictness is not what holds it back.
- **DiagAgent orders almost three times as many tests:** 5.5 per case ($2,630 per case), against 1.95 ($1,221) for Qwen3-8B.
- **DiagAgent was trained on 118k MIMIC-IV records.** Its overlap with CDM's admissions can't be checked ([BENCHMARKS.md](BENCHMARKS.md) §3.2 item 6). It states no confidence, so the adapter gives every commit probability 1.0, and its calibration and selective metrics are left out.
- **gpt-oss-20b ran only as a 20-case smoke test,** with the Qwen3 sampling settings. It loads and runs in vLLM 0.11 on an A100, and 19 of 20 episodes committed (1 invalid). Its replies carry its own analysis and final channel markers, but the action tag still parses. A full run is not done.

## 2. Post-hoc deferral (baselines #6 and #7)

These are the "why not just threshold?" baselines from `docs/EXPERIMENTS.md`. Thresholds are calibrated on all 2,160 non-test cases (seed 0); zero-shot runs see no training data, so every non-test case is fair calibration data. They're then applied to the test split.

| Model | Method | Threshold on stated p | Test coverage | Accuracy on committed | Deferral precision / recall |
|---|---|---|---|---|---|
| Qwen3-8B | Answer everything | — | 100% | 88.3 ± 1.1 | — |
| Qwen3-8B | Threshold, coverage target 0.8 (#6) | ≥ 0.70 | **85.8 ± 2.3%** (3 seeds) | **95.5 ± 0.7** (3 seeds) | 0.36 / 0.30 (seed 0) |
| Qwen3-8B | Threshold, risk target 5% (#6) | ≥ 0.75 | 70.0% | 95.8 | 0.21 / 0.43 |
| Qwen3-8B | SGR, 5% risk at δ 0.05 (#7) | ≥ 0.85 | 60.0% | 97.9 | 0.20 / 0.57 |
| Qwen3-14B | Answer everything | — | 100% | 88.3 | — |
| Qwen3-14B | Threshold, coverage target 0.8 (#6) | ≥ 0.65 | 79.6% | 96.9 | 0.25 / 0.32 |
| Qwen3-14B | Threshold, risk target 5% (#6) | ≥ 0.70 | 76.2% | 97.3 | 0.23 / 0.36 |
| Qwen3-14B | SGR, 5% risk at δ 0.05 (#7) | ≥ 0.85 | 58.8% | 97.2 | 0.13 / 0.39 |

**On average, untrained Qwen3-8B plus a confidence cutoff already meets the plan's headline target of "> 95 at 85% coverage".** Learned deferral has to beat this at matched coverage, using the same test cases and seeds.

Calibrated on only the 240 validation cases, SGR certified nothing and deferred everything (0% coverage). Calibrate zero-shot baselines on all non-test cases. DiagAgent-14B states no confidence, so it has no post-hoc baseline.

## 3. Checks from the data audit (all 2,400 cases)

**`____` mask cue.** CDM masks the case's own diagnosis in radiology. Accuracy by whether the mask is present, within each class:

| Class | Qwen3-8B, no mask | Qwen3-8B, mask | Qwen3-14B, no mask | Qwen3-14B, mask | DiagAgent-14B, no mask | DiagAgent-14B, mask |
|---|---|---|---|---|---|---|
| Appendicitis (n 570 / 387) | 95.4 | 94.6 | 92.5 | 92.5 | 91.2 | 93.8 |
| Cholecystitis (n 485 / 163) | 81.0 | 80.4 | 86.6 | 83.4 | 70.7 | 68.1 |
| Diverticulitis (n 131 / 126) | 80.9 | 88.9 | 80.9 | 88.1 | 42.0 (CI 34.3–50.4) | 54.8 (CI 46.0–62.7) |
| Pancreatitis (n 367 / 171) | **71.1** (CI 66.5–75.5) | **84.8** (CI 78.9–90.1) | **73.3** (CI 68.7–77.7) | **85.4** (CI 80.1–90.6) | 63.8 (CI 58.9–68.4) | 69.6 (CI 62.6–76.0) |

For both Qwen3 models on pancreatitis the gap is clear (the intervals don't overlap) and about the same size, 12–14 points. For DiagAgent-14B the gaps point the same way but the intervals overlap. It is an association, not proof the model reads the token: a masked report usually named the diagnosis, which also means its findings were clear. Only an ablation that replaces the mask can separate the two.

**Groups.** Label-adjusted accuracy: per-class accuracy within the group, weighted by the cohort's class mix (direct standardisation). This removes the effect of a group simply having more of the easy classes. Crude accuracy and 95% intervals are in each run's `subgroup_eval.py` output. Groups with ≥ 50 cases:

| Group | Qwen3-8B | Qwen3-14B | DiagAgent-14B |
|---|---|---|---|
| Age 18–29 / 30–44 / 45–64 / 65–79 / 80+ | 86.9 / 85.8 / 87.2 / 81.6 / 81.2 | 88.6 / 86.6 / 87.5 / 83.5 / 78.8 | 80.1 / 77.1 / 77.1 / 70.6 / 64.4 |
| Insurance Medicaid / Medicare / Other | 85.9 / 80.2 / 87.6 | 86.1 / 82.0 / 87.5 | 68.9 / 67.2 / 78.4 |
| Race Asian / Black / Hispanic-Latino / Other / White | 82.4 / 83.6 / 87.2 / 86.8 / 86.3 | 81.9 / 83.2 / 91.6 / 80.5 / 87.2 | 72.9 / 77.4 / 80.7 / 70.8 / 75.4 |
| Sex Female / Male | 85.5 / 85.8 | 85.8 / 86.8 | 75.3 / 75.7 |

- **Most of the crude age gap is class mix.** For Qwen3-8B the crude gap is 18 points; appendicitis is young and easiest, pancreatitis older and hardest.
- **About 6 points remain for Qwen3-8B between the youngest and oldest groups, and about 7 between Medicare and other insurance** (these overlap).
- **Qwen3-14B's age gap is larger: about 10 points after adjustment** (88.6 against 78.8). Its crude intervals for 18–29 (90.4–95.2) and 80+ (73.0–83.9) don't overlap. Its Medicare gap is about 5 points.
- **DiagAgent-14B's gaps are larger: about 16 points by age and 11 by insurance after adjustment.** The crude 95% intervals for ages 18–29 (87.7–93.2) and 80+ (51.3–63.5) are far apart.
- After adjustment, race groups span about 5 points for Qwen3-8B (82.4–87.2), 11 for Qwen3-14B (80.5–91.6) and 10 for DiagAgent-14B (70.8–80.7). Sex differences are about 1 point or less for all three.
- The audit found that older patients have more tests available and longer histories, even relative to their class (`DATA_AUDIT.md` §7.4).

## 4. DiagBench reproduction (DiagAgent-14B, MIMIC-IV subset, single-turn)

Only the MIMIC-IV subset (750 cases) has single-turn tasks; DiagGym's loader skips the other three subsets, which the paper uses for end-to-end and rubric evaluation (not run here). Prompts were built by DiagGym's own `load_dataset` and `_format_messages`. Generations are from round 1; round 2 re-scored them with a fixed exam extractor.

| Task | This run | Paper (single-turn MIMIC-IV) |
|---|---|---|
| Diagnosis accuracy, DiagGym's public script as written (n 750) | 98.1 | 87.87 |
| Diagnosis accuracy, final answer turn removed (n 750) | **80.1** | |
| Exam-recommendation hit ratio, all turns (n 3,735) | **39.0** | 68.49 |
| Exam-recommendation hit ratio, turns whose reply names an exam (n 3,055) | 47.7 | |

- **The public `inference_final_diagnose.py` leaves the case's final assistant turn, which contains "Diagnosis: <answer>", in the prompt.** Run that way, 319 of 750 replies didn't give a final diagnosis at all; they restated a "Current diagnosis" taken from context and asked for another test. With that turn removed, all 750 replies give a final "Diagnosis:". The paper's 87.87 falls between the two protocols. From the public code we can't tell which protocol produced it, or how much of the gap is the judge.
- **The exam hit ratio is well below the paper's, but it doesn't measure the same thing.** Of the 3,735 replies, 680 diagnose instead of recommending an exam; these count as misses.
  - Our ground truth is each turn's `recommended_exam_names` from the public release. The paper's metric script reads raw MIMIC events from a file that isn't public.
  - Treat 39.0 / 47.7 as "reproduced with the public data and an open judge", not as a failure to reproduce.
- **Judge:** DiagGym's own `accuracy.txt` and `hit_ratio.txt` prompts on a local Qwen3-8B, with thinking off and greedy decoding. The paper used GPT-4o, which would send MIMIC-derived text to OpenAI's regular API (PhysioNet DUA).
- **Re-scoring moved the no-answer-turn result from 79.9 (round 1) to 80.1, two verdicts out of 750.** The judge and generations were the same. The batch was not: it now also held the 3,055 exam prompts, and vLLM's greedy decoding isn't bit-identical across batch compositions.
- **The round 1 exam extractor found an exam in only 750 of 3,735 replies.** That was a bug, now fixed in `scripts/lambda/diagbench_score.py`, and its number was discarded.

## 5. Open world: OTHER (Qwen3-8B, seed 0)

The open-world set is built by `deferdx data build-openworld` the same way CDM was built ([BENCHMARKS.md](BENCHMARKS.md) §3.2 item 7). Qwen3-8B is offered OTHER as a fifth answer (no `--closed-world`) on three groups of cases:
- the 240 LA-CDM test cases;
- 713 OTHER cases (abdominal-pain admissions whose principal diagnosis is none of the four);
- 51 controls: same pipeline as OTHER, but with one of the four as the principal diagnosis.

It is run once with DEFER disabled and once with DEFER allowed.

| Cases | Outcome | DEFER disabled | DEFER allowed |
|---|---|---|---|
| CDM test (240) | correct | 75.4 | 64.6 |
| | answered OTHER | 17.1 | 5.8 |
| | wrong in-set class | 5.4 | 2.9 |
| | deferred | — | 25.4 |
| | invalid | 2.1 | 1.2 |
| | *accuracy on committed* | *77.0* | *88.1* |
| OTHER (713) | answered OTHER (correct) | 81.9 | 44.3 |
| | **false commit to one of the four** | **14.6** | **10.2** |
| | deferred | — | 43.5 |
| | invalid | 3.5 | 2.0 |
| Controls (51) | correct | 39.2 | 45.1 |
| | answered OTHER | 43.1 | 11.8 |
| | wrong in-set class | 17.6 | 7.8 |
| | deferred | — | 33.3 |
| | invalid | 0.0 | 2.0 |
| In-set vs OTHER | OOD AUROC (`deferdx evaluate`) | 0.849 | 0.840 |

How to read this:

- **Offering OTHER costs in-set accuracy.** With OTHER available and DEFER off, Qwen3-8B answers OTHER on 17.1% of CDM test cases: appendicitis 7.3, cholecystitis 20.0, pancreatitis 25.9 and diverticulitis 28.0. Test accuracy drops from 87.5 (closed world, same seed) to 75.4.
- **False commits come mostly from two groups, and mostly go to diverticulitis.** Gastroenteritis or colitis (219 cases) is falsely committed 23.3% of the time, and mesenteric ischaemia (94) 24.5%. Urolithiasis is 1.3% and GI bleeding 6.3%. Two OTHER groups have fewer than 10 cases and aren't shown separately.
- **OTHER answers also track how atypical a case is, not only its true label.** Qwen3-8B answers OTHER on 43.1% of the controls, against 17.1% of CDM test cases, although both have an in-set principal diagnosis.
  - Controls only need the diagnosis at ICD sequence 1, so they are less typical than CDM's curated cases.
  - Text still separates the sources: a classifier tells controls from label-matched CDM cases at AUROC 0.563 overall, and 0.673 on the physical exam alone.
  - Report controls next to OTHER so that "answers OTHER" isn't read as "recognises OTHER". With n = 51, the control rates have intervals of about ±14 points.
- **Prompted DEFER against a threshold at matched coverage.**
  - With DEFER allowed, Qwen3-8B commits on 73.3% of CDM test cases, at 88.1% accuracy, and falsely commits 10.2% of OTHER cases.
  - Thresholding the no-DEFER run's stated probability at ≥ 0.75 gives 74.2% coverage, 82.6% accuracy and 8.1% false commits.
  - So prompted DEFER is more accurate on what it answers, and the threshold lets fewer OTHER cases through as in-set answers. This is one seed, and both gaps are about 1.5 standard errors. Treat it as a trade-off to test, not a result.
  - The threshold here was chosen on the test cases themselves to match coverage, which favours it slightly.
- With DEFER allowed, deferral precision and recall on the in-set cases are 0.45 and 0.50, against the forced-prediction counterfactual.

## 6. Takeaways for the trained agent
1. **The floor is high.** Zero-shot Qwen3-8B reaches 84–88% mean-class accuracy in this environment, and 95.5% at 86% coverage with a simple threshold. Qwen3-14B is not significantly better on all cases (86.4 against 85.8). Report DEFER-Dx against these numbers on the same cases and seeds.
2. **Per-class claims should use all 2,400 cases.** Twenty-five test cases moved diverticulitis by 24 points between seeds.
3. **Check the mask cue before claiming reasoning gains,** especially for pancreatitis.
4. **Track the age and Medicare gaps through training:** after adjusting for class, about 6–7 points for Qwen3-8B, 10 by age for Qwen3-14B, and 11–16 for DiagAgent-14B.
5. **The domain-trained agent is not the strongest baseline here.** DiagAgent-14B is limited by answering outside the label set and by test-heavy trajectories. That supports keeping OTHER and per-test cost in the reward.
6. **For the open world, report three things.** First, OTHER results next to the same-pipeline controls. Second, false commits by OTHER group. Third, learned DEFER against a threshold at matched coverage, with the threshold calibrated on non-test cases.
7. **Still open:**
   - the mask ablation;
   - more seeds on all 2,400 cases;
   - a full gpt-oss-20b run and MedGemma-27B;
   - frontier models through a PhysioNet-approved route.

## 7. Round 3: zero-shot baselines on the trained models' evaluation cohorts (2026-10-09)

Sections 1–6 used all 2,400 CDM cases and an earlier open-world build. The trained models are evaluated on `data/cohorts` instead (`deferdx data cohorts`):
- CDM val + test, 480 cases;
- OTHER, groups seen in training, 206;
- OTHER, time-critical groups never seen in training, 187;
- same-pipeline controls, 258.

I rebuilt them with the README's commands, and every set has exactly the size the trained models were evaluated on. The systems below run through the same evaluation suite, with OTHER allowed and DEFER off, and are scored by `scripts/make_report.py`. Rows marked † come from `docs/results.json` on main (`301e60f`), with the same cohorts and scoring. Intervals are 95% case-level bootstraps.

| System | Accuracy, CDM (480) | Diverticulitis | Unflagged errors / 100 | Time-critical OTHER missed (187) | False commit, seen OTHER (206) | OOD AUROC | Tests / case | $ / case |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B zero-shot, 3 seeds † | 74.3 (70.9–77.6) | 73.9 (63.5–83.7) | 24.1 (21.0–27.4) | 17.6 (13.0–22.6) | 12.8 (9.2–16.5) | 88.1 | 1.93 | 1,158 |
| **DiagAgent-14B**, through its format adapter, greedy | 76.2 (72.5–80.1) | 58.8 (46.2–72.5) | 22.9 (19.2–26.6) | **9.6 (5.9–14.4)** | 12.6 (8.3–17.0) | 85.3 | 5.64 | 2,693 |
| Group-consensus deferral arm, trained, defers † | 85.8 (83.1–88.4) | 85.6 (76.9–93.2) | 4.9 (3.4–6.6) | 11.4 (7.8–15.3) | 13.4 (9.4–17.6) | 88.2 | 1.83 | 1,298 |
| **Qwen3-14B** zero-shot, 3 seeds | 74.2 (70.8–77.5) | 68.6 (56.6–80.0) | 24.7 (21.5–28.0) | **10.3 (7.1–14.1)** | **8.1 (5.2–11.2)** | 90.3 | 2.18 | 1,228 |

How the adapter handles the open world: DiagAgent names a specific diagnosis and has no "none of these" option. So `--open-world` counts any named diagnosis outside the four as OTHER. On an in-set case that includes near-misses such as "biliary colic", which count as wrong. Confidence-based metrics (AURC, ECE, Brier, confident errors) aren't reported for DiagAgent, because every one of its answers carries probability 1.0.

How to read this:
- **A domain-trained model with no DEFER action misses fewer time-critical OTHER cases than untrained Qwen3-8B** (9.6 against 17.6), and its interval overlaps the trained consensus arm's (11.4). Naming specific diagnoses is enough to recognise "none of the four" on these presentations. On its own, "time-critical OTHER missed" doesn't show what deferral adds, so report it next to this row.
- **It doesn't defer, and it pays for that on in-set cases.** It makes 22.9 unflagged errors per 100 cases, against 4.9 for the trained arm. It gets 58.8% on diverticulitis and orders three times as many tests ($2,693 per case).
- **DiagAgent was trained on MIMIC-IV records.** Its overlap with these admissions can't be checked ([BENCHMARKS.md](BENCHMARKS.md) §3.2 item 6).
- **Qwen3-14B diagnoses the four conditions no better than Qwen3-8B** (74.2 against 74.3), the same finding as on all 2,400 cases in §1. **But it is much better at recognising "none of the four":** it misses 10.3% of time-critical OTHER cases against 17.6, and falsely commits on 8.1% of the seen OTHER groups against 12.8.

**Learned deferral against a bigger model with a confidence threshold.** This uses the report's own cross-fitted post-hoc threshold (`crossfit_posthoc`), with the target set to the trained arm's coverage. Qwen3-14B states 0.7 on so many cases that the closest achievable coverage is 86.2%, not 82.6%.

| At about the trained arm's coverage | Coverage | Accuracy on answered cases | Unflagged errors / 100 | Time-critical OTHER missed |
|---|---|---|---|---|
| Qwen3-14B zero-shot + threshold ≥ 0.7 | 86.2 (83.9–88.3) | 77.7 (74.1–81.0) | 19.2 (16.4–22.2) | **6.6 (3.9–9.8)** |
| Group-consensus deferral arm, trained † | 82.6 (79.5–85.6) | **94.0 (92.0–95.9)** | **4.9 (3.4–6.6)** | 11.4 (7.8–15.3) |

- **On the four conditions, training wins clearly:** 94.0% accuracy on the cases answered, against 77.7. It makes 4.9 unflagged errors per 100 cases, against 19.2.
- **On unseen time-critical presentations, it doesn't.** A thresholded zero-shot Qwen3-14B misses 6.6% of them, against the trained 8B arm's 11.4%, with overlapping intervals. The open-world claim therefore needs the trained method's own threshold comparison, against its 8B no-DEFER control. A bigger untrained model already closes most of this particular gap.

**Prompted DEFER: a bigger model asked to defer, against the trained arm.** Qwen3-14B ran once with DEFER offered in the prompt (seed 0 only, to stay within budget). Qwen3-8B's row and the trained arm's are from main's report.

| CDM val + test (480) | Coverage | Accuracy on answered cases | Unflagged errors / 100 | Time-critical OTHER missed | Deferral precision | Errors caught |
|---|---|---|---|---|---|---|
| Qwen3-8B zero-shot, prompted DEFER, 3 seeds † | 75.1 (72.0–78.1) | 87.9 (85.2–90.4) | 9.1 (7.2–11.1) | 14.4 (10.2–19.1) | 51.9 (44.8–58.2) | 53.5 (46.7–59.8) |
| Qwen3-14B zero-shot, prompted DEFER, 1 seed | 65.2 (61.0–69.4) | 89.5 (86.0–92.7) | 6.9 (4.8–9.4) | **6.4 (3.2–10.2)** | 45.0 (37.2–52.6) | 64.3 (55.8–73.0) |
| Group-consensus deferral arm, trained † | **82.6 (79.5–85.6)** | **94.0 (92.0–95.9)** | **4.9 (3.4–6.6)** | 11.4 (7.8–15.3) | 52.2 (44.0–60.5) | 62.7 (52.9–71.2) |

- **Prompted to defer, Qwen3-14B gets close to the trained arm's unflagged-error rate** (6.9 against 4.9 per 100), but by deferring on 35% of in-set cases rather than 17%. The trained arm answers 17 more cases in every 100 at higher accuracy. That coverage, at the same safety, is what training buys here.
- **It also defers on 42–51% of OTHER cases** (seen: 51.0; unseen: 42.2), so its low time-critical miss rate comes partly from deferring broadly.

**Is deferral equitable?** These are `scripts/subgroup_eval.py`'s new deferral tables for Qwen3-14B prompted DEFER, on CDM val + test (480 cases, one seed). Groups with fewer than 10 cases are hidden.

| Group | Deferred | Accuracy on answered cases | Unflagged errors / 100 |
|---|---|---|---|
| Age 18–29 / 30–44 / 45–64 / 65–79 / 80+ | 29.2 / 38.9 / 31.6 / 32.0 / 36.4 | 88.2 / 86.6 / 94.0 / 84.0 / 92.9 | 8.3 / 8.0 / 3.9 / 10.7 / 4.5 |
| Insurance Medicaid / Medicare / Other | 34.0 / 35.4 / 32.6 | **77.4** / 90.0 / 91.0 | **14.0 (6.0–24.0)** / 6.2 / 6.0 (3.6–8.7) |
| Sex Female / Male | 34.2 / 32.2 | 89.4 / 89.6 | 6.8 / 6.9 |
| Race Black / Hispanic-Latino / White | 22.8 / 32.6 / 35.0 | 88.4 / 89.3 / 89.6 | 8.8 / 7.0 / 6.6 |

- **Deferral is spread fairly evenly across age, insurance and sex** (29–39% of cases; sexes within 2 points). By race, Black patients are deferred least: 22.8% (95% CI 12.3–33.3, 57 cases), with similar accuracy on the cases answered.
- **Deferring at even rates does not even out the errors.** Medicaid patients get the least accurate answers when the model does answer (77.4 against 90–91), with about twice the unflagged errors. The interval is wide: 50 Medicaid cases and one seed.
- This is the analysis to run on the trained models' evaluation output:

  ```
  python scripts/subgroup_eval.py --rollouts outputs/eval/<system>/s*.jsonl --sets eval_cdm_val eval_cdm_test --cases data/cohorts/eval_cdm_val.jsonl data/cohorts/eval_cdm_test.jsonl
  ```

  A learned deferral policy should defer more where it is wrong more, not uniformly.

Cost for this round: about $13.70 of GPU in total.
- One A100 40 GB in us-west-2 for 6.0 h, about $11.94.
- $1.78 for an A6000 (us-south-2) whose network failed before anything ran.

Run times: DiagAgent 35 min; Qwen3-14B 77 min for each seed of the five sets.

## Reproduce
```bash
# on a CUDA 12.8 GPU machine with ~/payload laid out as in the job scripts
bash scripts/lambda/setup.sh && bash scripts/lambda/run_all.sh                                   # round 1
bash scripts/lambda/setup.sh Qwen/Qwen3-8B Qwen/Qwen3-14B Henrychur/DiagAgent-14B openai/gpt-oss-20b \
  && bash scripts/lambda/run_all2.sh                                                             # round 2
# round 3: build the cohorts locally, upload only data/cohorts/eval_*.jsonl as ~/payload/cohorts
deferdx data build-openworld --mimic-dir data/physionet/mimiciv/2.2 --note-dir data/physionet/mimic-iv-note/2.2 \
  --exclude-cases data/cdm/all.jsonl --controls --controls-per-label 3000 --n 2400 --out data/openworld_xl
deferdx data cohorts
bash scripts/lambda/setup.sh Qwen/Qwen3-14B Henrychur/DiagAgent-14B && bash scripts/lambda/run_all3.sh
# locally, after copying ~/outputs back
deferdx evaluate --rollouts outputs/lambda/cdm_qwen8b_all_s0.jsonl
python scripts/subgroup_eval.py --rollouts outputs/lambda2/cdm_da_adapter_all.jsonl --cases data/cdm/all.jsonl
deferdx evaluate --rollouts outputs/lambda2/ow_qwen8b_defer.jsonl
deferdx baseline --method coverage --target 0.8 --val <non-test rollouts> --test <test rollouts>
python scripts/make_report.py --eval-dir outputs/eval --out outputs/report_round3      # round 3 (not docs/)
```
