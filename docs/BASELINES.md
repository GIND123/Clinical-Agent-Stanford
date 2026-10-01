# Inference-only baselines (no training)

Run 2026-10-01. This is step 3 of the evaluation plan in [BENCHMARKS.md](BENCHMARKS.md): run existing models as-is, with no fine-tuning, so the trained agent has numbers to beat. Every number here is an aggregate. Rollout files contain MIMIC text and stay in git-ignored `outputs/`.

## Setup

| | |
|---|---|
| Hardware | Lambda Cloud, 1× A100 40 GB (SXM4), us-east-1. The instance is terminated and its disk deleted. |
| Software | vLLM 0.11.0, torch 2.8.0+cu128, transformers 4.57.6. The pins are needed: the latest vLLM pulls a CUDA 13 torch that Lambda's 570 driver can't run, and transformers 5.x removed a tokenizer attribute vLLM 0.11 reads. |
| Code | This repo at `d32371b`, environment unchanged. Job scripts are in [`scripts/lambda/`](../scripts/lambda/). |
| Data on the VM | Built CDM cases (`all.jsonl`, `test.jsonl`, `splits.json`) and DiagBench only. No raw MIMIC tables. |
| MIMIC-CDM settings | `deferdx rollout --policy vllm --no-defer --closed-world` with the default catalog, `max_steps` 8 and 1,024 new tokens per turn. Qwen3-8B uses its recommended thinking-mode sampling (T 0.6, top-p 0.95, top-k 20); DiagAgent-14B uses greedy decoding, as its authors do. |
| Scoring | `deferdx evaluate`, `deferdx baseline`, `scripts/subgroup_eval.py` |
| Cost | About $9 of a $20 budget. The job took about $5.30; the rest was idle time and two launches that didn't start the job. |

Run log (UTC): Qwen3-8B, all cases 00:37–01:39 · DiagAgent-14B, all cases 01:41–02:35 · DiagBench generation and scoring 02:36–02:46 · Qwen3-8B seeds 1–2 on the test split 02:46–03:01.

## 1. MIMIC-CDM in the DEFER-Dx environment

Forced commit, four classes. "Mean-class" is the unweighted mean of per-class accuracy (LA-CDM's convention); "case" is case-weighted.

| Model | Cases | Seeds | Mean-class acc. | Case acc. | App. / Chol. / Div. / Panc. | Macro-F1 | ECE | Brier | AURC | Invalid | Tests / case | $ / case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B, zero-shot | all 2,400 | 1 | 84.1 | 85.8 (95% CI 84.3–87.1) | 95.1 / 80.9 / 84.8 / 75.5 | 85.0 | 0.067 | 0.085 | 0.037 | 4.0% | 1.95 | 1,221 |
| Qwen3-8B, zero-shot | LA-CDM test (240) | 3 | **87.2 ± 2.4** | 88.3 ± 1.1 | 93.8 / 85.1 / 86.7 ± 12.9 / 83.3 | 87.9 ± 1.8 | 0.102 | 0.070 | 0.021 | 4.3% | 1.99 | 1,192 |
| DiagAgent-14B, zero-shot | all 2,400 | 1 | 0.0 | 0.0 | — | — | — | — | — | **100%** | 4.13 | — |

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
- **DiagAgent-14B never committed: 0 of 2,400 episodes were valid.** It orders tests (4.1 per episode) but answers in its own trained format, not this environment's action format. Comparing it fairly here would need a policy-side format adapter, which hasn't been built.

## 2. Post-hoc deferral on Qwen3-8B (baselines #6 and #7)

These are the "why not just threshold?" baselines from `docs/EXPERIMENTS.md`. Thresholds are calibrated on all 2,160 non-test cases (seed 0); zero-shot runs see no training data, so every non-test case is fair calibration data. They're then applied to the test split.

| Method | Threshold on stated p | Test coverage | Accuracy on committed | Deferral precision / recall |
|---|---|---|---|---|
| Answer everything | — | 100% | 88.3 ± 1.1 | — |
| Threshold, coverage target 0.8 (#6) | ≥ 0.70 | **85.8 ± 2.3%** (3 seeds) | **95.5 ± 0.7** (3 seeds) | 0.36 / 0.30 (seed 0) |
| Threshold, risk target 5% (#6) | ≥ 0.75 | 70.0% | 95.8 | 0.21 / 0.43 |
| SGR, 5% risk at δ 0.05 (#7) | ≥ 0.85 | 60.0% | 97.9 | 0.20 / 0.57 |

**On average, untrained Qwen3-8B plus a confidence cutoff already meets the plan's headline target of "> 95 at 85% coverage".** Learned deferral has to beat this at matched coverage, using the same test cases and seeds.

Calibrated on only the 240 validation cases, SGR certified nothing and deferred everything (0% coverage). Calibrate zero-shot baselines on all non-test cases.

## 3. Checks from the data audit (Qwen3-8B, all 2,400 cases)

**`____` mask cue.** CDM masks the case's own diagnosis in radiology. Accuracy by whether the mask is present, within each class:

| Class | No mask | Mask |
|---|---|---|
| Appendicitis | 95.4 (n 570) | 94.6 (n 387) |
| Cholecystitis | 81.0 (n 485) | 80.4 (n 163) |
| Diverticulitis | 80.9 (n 131) | 88.9 (n 126) |
| Pancreatitis | **71.1** (CI 66.5–75.5, n 367) | **84.8** (CI 78.9–90.1, n 171) |

For pancreatitis the gap is clear (the intervals don't overlap). It is an association, not proof the model reads the token: a masked report usually named the diagnosis, which also means its findings were clear. Only an ablation that replaces the mask can separate the two.

**Groups.** Crude accuracy, then label-adjusted (direct standardisation to the cohort's class mix). Groups with ≥ 50 cases:

| Group | Crude | Label-adjusted |
|---|---|---|
| Age 18–29 / 30–44 / 45–64 / 65–79 / 80+ | 94.5 / 87.8 / 86.3 / 78.2 / 76.1 | 86.9 / 85.8 / 87.2 / 81.6 / 81.2 |
| Insurance Medicaid / Medicare / Other | 84.6 / 75.5 / 88.9 | 85.9 / 80.2 / 87.6 |
| Race Asian / Black / Hispanic-Latino / Other / White | 86.2 / 82.1 / 86.7 / 88.2 / 86.3 | 82.4 / 83.6 / 87.2 / 86.8 / 86.3 |
| Sex Female / Male | 85.1 / 86.5 | 85.5 / 85.8 |

Most of the crude age gap (18 points) is class mix: appendicitis is young and easiest, pancreatitis older and hardest. **About 6 points remain between the youngest and oldest groups, and about 7 between Medicare and other insurance** (these overlap). Race and sex differences are within about 4 points. The audit found that older patients have more tests available and longer histories, even relative to their class (`DATA_AUDIT.md` §7.4).

## 4. DiagBench reproduction (DiagAgent-14B, MIMIC-IV subset, single-turn diagnosis)

Only the MIMIC-IV subset (750 cases) has single-turn tasks; DiagGym's loader skips the other three subsets, which the paper uses for end-to-end and rubric evaluation (not run here). Prompts were built by DiagGym's own `load_dataset` and `_format_messages`.

| Protocol | Diagnosis accuracy (n 750) |
|---|---|
| DiagGym's public script as written | 98.1 |
| Same, with the final answer turn removed | **79.9** |
| Paper, single-turn MIMIC-IV | 87.87 |

- **The public `inference_final_diagnose.py` leaves the case's final assistant turn, which contains "Diagnosis: <answer>", in the prompt.** Run that way, 319 of 750 replies didn't give a final diagnosis at all; they restated a "Current diagnosis" taken from context and asked for another test. With that turn removed, all 750 replies give a final "Diagnosis:".
- The paper's 87.87 falls between the two. From the public code we can't tell which protocol produced it, or how much of the gap is the judge.
- **Judge:** DiagGym's own `accuracy.txt` prompt on a local Qwen3-8B, with thinking off. The paper used GPT-4o, which would send MIMIC-derived text to OpenAI's regular API (PhysioNet DUA).
- **Exam hit ratio is not reported.** The first scoring pass found an exam in only 750 of 3,735 replies, an extractor bug since fixed in `scripts/lambda/diagbench_score.py` (now 3,055 of 3,735; the rest diagnose instead of recommending). The paper's exam ground truth also comes from raw MIMIC events that aren't in the public release. Re-scoring needs the GPU judge again.

## 5. Takeaways for the trained agent
1. **The floor is high.** Zero-shot Qwen3-8B reaches 84–88% mean-class accuracy in this environment, and 95.5% at 86% coverage with a simple threshold. Report DEFER-Dx against these numbers on the same cases and seeds.
2. **Per-class claims should use all 2,400 cases.** Twenty-five test cases moved diverticulitis by 24 points between seeds.
3. **Check the mask cue before claiming reasoning gains,** especially for pancreatitis.
4. **Track the age and Medicare gaps** (about 6–7 points after adjusting for class) through training.
5. **Still open:** OTHER-class metrics (needs the `build_openworld` fixes from `DATA_AUDIT.md`), the mask ablation, a DiagAgent format adapter, and more seeds on all cases.

## Reproduce
```bash
# on a CUDA 12.8 GPU machine with ~/payload laid out as in scripts/lambda/run_all.sh
bash scripts/lambda/setup.sh && bash scripts/lambda/run_all.sh
# locally, after copying ~/outputs back to outputs/lambda/
deferdx evaluate --rollouts outputs/lambda/cdm_qwen8b_all_s0.jsonl
python scripts/subgroup_eval.py --rollouts outputs/lambda/cdm_qwen8b_all_s0.jsonl --cases data/cdm/all.jsonl
deferdx baseline --method coverage --target 0.8 --val <non-test rollouts> --test <test rollouts>
```
