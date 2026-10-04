# DEFER-Dx

An interactive clinical diagnostic agent trained with RL over four actions: **ASK, TEST, COMMIT, DEFER**. Deferring to a clinician is a learned action with its own verifiable reward (the *group-consensus deferral reward*), not a confidence threshold applied afterwards. The label space is open-world: the four MIMIC-CDM abdominal conditions plus **OTHER**.

The research plan is in [stanford_idea_md.md](stanford_idea_md.md). This repo is the base implementation: data pipelines, environment, rewards, metrics, baselines, and the SFT and GRPO trainers. **It contains no results.**

> **PhysioNet DUA.** MIMIC text must never reach a third-party service. The repo has no hosted-API clients: every policy runs on local weights (`transformers` / `vllm`). `.gitignore` excludes `data/`, `outputs/`, `*.jsonl`, `*.csv*` and `*.pkl`. Rollout files contain MIMIC text, so treat them as credentialed data.

## Install

```bash
pip install -e ".[train,data,dev]"   # add ,vllm for fast evaluation rollouts
pytest                               # CPU-only; no downloads, no MIMIC data
```

## Quickstart (synthetic data, no GPU)

```bash
bash scripts/smoke_synthetic.sh
python scripts/check_model.py --model Qwen/Qwen3-0.6B   # real model in the loop, CPU is fine
```

The smoke test generates fabricated cases, runs the oracle and random policies, builds SFT data and evaluates. `check_model.py` runs a real local model on synthetic cases. It reports action-format compliance, whether `<think>` tags survive decoding, and whether SFT prompts are token-identical to generation prompts. Synthetic numbers mean nothing; they only show the pipeline works.

## Real-data pipeline

```bash
# 0. Download (credentialed; resumes and checks PhysioNet's SHA256SUMS). --s3 uses PhysioNet's AWS
#    access points and is far faster than physionet.org; see the script header for setup.
bash scripts/download_data.sh --s3 cdm hosp note     # -> data/physionet/{mimic-iv-ext-cdm/1.1, mimiciv/2.2, mimic-iv-note/2.2}

# 1. MIMIC-IV-Ext-CDM v1.1 -> canonical cases. Confirm file/column names once:
deferdx data inspect-cdm --cdm-dir /path/to/mimic-iv-ext-cdm/1.1
#    Default split = exact LA-CDM 80/10/10 (there is no official split; see docs/RESEARCH.md)
deferdx data build-cdm --cdm-dir /path/to/mimic-iv-ext-cdm/1.1 --out data/cdm

# 2. Check that the test catalog actually reaches your lab names
deferdx data coverage --cases data/cdm/train.jsonl

# 3. Open-world cohort (OTHER) + same-pipeline in-set controls
deferdx data build-openworld --mimic-dir /path/mimiciv/2.2 --note-dir /path/mimic-iv-note/2.2 \
    --exclude-cases data/cdm/all.jsonl --controls --out data/openworld

# 4. Stage 1 SFT data: STaR rejection sampling from the base model + deferral exemplars
deferdx rollout --cases data/cdm/train.jsonl --policy vllm --model Qwen/Qwen3-8B \
    --samples 8 --temperature 1.0 --out outputs/rollouts/base_train.jsonl
deferdx sft-data rollouts --rollouts outputs/rollouts/base_train.jsonl --out data/sft/sft_train.jsonl
#    (cold start without a usable base model: `deferdx sft-data oracle ...`)
deferdx train sft --config configs/sft.yaml

# 5. Stage 2 GRPO
deferdx train grpo --config configs/grpo.yaml

# 6. Evaluate on test + OTHER (Qwen3 sampling defaults; repeat over --seed 0 1 2)
deferdx rollout --cases data/cdm/test.jsonl data/openworld/other.jsonl --policy vllm \
    --model Qwen/Qwen3-8B --adapter outputs/grpo/final --out outputs/eval/deferdx_test.jsonl
deferdx evaluate --rollouts outputs/eval/deferdx_test.jsonl --out outputs/eval/deferdx_test.json
```

The full §6.1 baseline grid is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Data audit, benchmarks and baselines

These docs hold aggregates only. MIMIC data and model outputs stay in git-ignored `data/` and `outputs/`.

| Doc | What it holds | Produced by |
|---|---|---|
| [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md) | Inventory and checksums, column-level schema of every file, label checks, label-leakage checks, bias analysis, and the open-world pool | `python scripts/audit_data.py` (about 2 min) |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Which metrics and benchmarks evaluate each component, the best published result on each, and the evaluation protocol | literature review |
| [docs/BASELINES.md](docs/BASELINES.md) | Inference-only results: models run as-is, plus post-hoc deferral baselines, open-world results, the DiagBench reproduction, and per-group scores | [`scripts/lambda/`](scripts/lambda/), `scripts/subgroup_eval.py` |

## Layout

| Path | What it does | Plan section |
|---|---|---|
| [src/deferdx/data/cdm_loader.py](src/deferdx/data/cdm_loader.py) | MIMIC-IV-Ext-CDM (PhysioNet CSV or Hager-framework pickles) to `Case`; earliest value per repeated test | §4.1 |
| [src/deferdx/data/openworld.py](src/deferdx/data/openworld.py) | MIMIC-CDM-OW: ICD filtering in DuckDB, stratified sampling, same-pipeline controls | §4.2 |
| [src/deferdx/data/parity.py](src/deferdx/data/parity.py) | CDM's own text rules (history blob, PE window, radiology section filter, `____` masking) applied to OTHER cases | §4.2 |
| [src/deferdx/data/splits.py](src/deferdx/data/splits.py) | Exact LA-CDM split; stratified ratio splits | §4.1 |
| [src/deferdx/data/synthetic.py](src/deferdx/data/synthetic.py) | Fabricated cases for tests | — |
| [src/deferdx/env/](src/deferdx/env/) | Reveal-on-request POMDP; 22 tests (a superset of LA-CDM's) matched by MIMIC itemid; action parser | §3.1 |
| [src/deferdx/rewards/scoring.py](src/deferdx/rewards/scoring.py) | Commit reward (accuracy, Brier or log score, severity, cost) and the group-consensus DEFER reward | §3.2 |
| [src/deferdx/rewards/constraint.py](src/deferdx/rewards/constraint.py) | Adaptive-Lagrangian coverage floor | §3.2 |
| [src/deferdx/rollout.py](src/deferdx/rollout.py) | Batched multi-turn rollouts | — |
| [src/deferdx/policy/](src/deferdx/policy/) | Local HF / vLLM policies; oracle and random scripted policies | §6.1 |
| [src/deferdx/training/](src/deferdx/training/) | SFT data (oracle, STaR, deferral exemplars), SFT loop, multi-turn GRPO | §5.2–5.4 |
| [src/deferdx/eval/](src/deferdx/eval/) | Leaderboard metrics, risk-coverage/AURC, ECE/Brier, confident errors, OOD AUROC, cost | §6.2 |
| [src/deferdx/baselines/](src/deferdx/baselines/) | Post-hoc threshold and SGR (conformal-style) | §6.1 #6–7 |
| [configs/](configs/) | Env, reward, catalog, severity matrix, ICD lists, trainer configs | — |

## Where this departs from the plan, and why

1. **The coverage penalty applies to deferring rollouts only.** §3.2 writes `R ← R − ν·max(0, rate − ρ)` for every rollout. Subtracting the same constant from a whole GRPO group cancels out in advantage normalisation, so the penalty would do nothing. ν is charged to each DEFER rollout instead, which is the per-sample gradient of the Lagrangian. ν is updated by projected dual ascent.
2. **The open-world DEFER reward defaults to 1.2, not 1.1.** After the handoff cost μ = 0.1, a value of 1.1 exactly ties a perfect COMMIT(OTHER). The plan wants DEFER slightly higher.
3. **τ is not the effective threshold.** With the §3.2 weights, τ = 0.85 makes the agent defer only when p̂ < ~0.645 (for a calibrated committer). Run `deferdx crossover` to see the τ → threshold table. Report both numbers, and pick γ, μ and τ from this table.
4. **p̂ with no commits in the group** defaults to "neutral" (p̂ = τ), so deferral costs only μ. Setting it to "zero" would reward all-defer groups and speed up collapse.
5. **There are two "Mean Acc" conventions.** LA-CDM (and the LA-CDM and ReAct rows that LDTL copies) average the per-class accuracies. LDTL's own rows are case-weighted (93.4, not 90.25). Both are reported: `accuracy_full_coverage` and `mean_class_accuracy`. The plan's "93.4 vs 81.3" crosses both metrics and splits; see [docs/RESEARCH.md](docs/RESEARCH.md).
6. **Full-coverage accuracy for DEFER** uses the top of the deferral differential, so a deferring agent still gets a leaderboard-comparable number. Deferral precision/recall can use a real no-defer run as the counterfactual (`--counterfactual`).
7. **RL samples are per turn, not per conversation.** Qwen3's chat template strips earlier `<think>` blocks (confirmed on the real tokenizer), so the policy gradient uses the exact prompt and completion tokens of each generation call.
8. **ASK covers the physical exam only by default, and the full history is shown at reset.** CDM stores past, social and family history inside one flattened blob, and LA-CDM and LDTL both show it up front. `env.history_at_reset` / `env.ask_topics` change this.
9. **Evaluation samples; it doesn't decode greedily.** Qwen3's model card warns against greedy decoding in thinking mode, so evaluation uses T=0.6, top-p 0.95, top-k 20. Report mean ± sd over seeds.

## Checked vs. still to verify

Resolved from primary sources ([docs/RESEARCH.md](docs/RESEARCH.md)):
- CDM v1.1 file and column names, the `pathology_ids.json` label file, and the modality/region vocabulary.
- The split: there is no official one. The default reproduces LA-CDM's exactly.
- How CDM's text was processed; the open-world cases now match it.
- How CDM attached data and chose admissions (Hager et al., `fill_nan_hadm` and `check_missing`); `build_openworld` now does the same. That covers rows without a `hadm_id` timed from one day before the first transfer to the last, the exam name from `radiology_detail`, and the inclusion rule (exam of 40+ characters, a lab, an abdominal study). On CDM's own admissions the builder's join finds exactly what CDM holds ([docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)).
- Costs for panels and imaging: the 2025 BIDMC standard charges from LA-CDM.

Checked on the real data (2026-09/10):
- **The CDM columns** ([docs/DATA_AUDIT.md](docs/DATA_AUDIT.md) §2). The CSVs have no `subject_id` and no lab or microbiology `charttime`, although `configs/cdm_format.yaml` maps them. So `build-cdm`'s patient-leakage count is always 0; joined to `admissions`, fewer than 10 patients span splits.
- **Catalog coverage** (`deferdx data coverage`; [docs/BENCHMARKS.md](docs/BENCHMARKS.md) §3.2):
  - The core panels resolve for ≥ 99.8% of cases.
  - `hida_scan` never resolves, because CDM has no HIDA reports.
  - Anion Gap, eGFR, Mg, LDH and troponin are present but no catalog test reaches them.
- **The source-classifier check.** Same-pipeline controls vs label-matched CDM cases score AUROC 0.563 combined. The physical exam alone is still separable (0.673; 51 controls).

Still to verify:
- **Placeholder costs.** Tests marked `cost_source: placeholder` (lipase, CRP, lactate, cultures, HIDA, …) need real prices.
- **The severity matrix** is a time-to-harm proxy. Replace it with the clinician-built version.
- **ICD lists and sanitize terms** ([configs/openworld_icd.yaml](configs/openworld_icd.yaml)) need clinician review.
- **Test availability is informative.** For example, a HIDA scan implies someone already suspected cholecystitis. `charge_unavailable: true` charges for tests that were never performed; LA-CDM doesn't. Report both.

## Not built yet

- A multi-GPU / verl port of the trainer. The reward functions are framework-agnostic; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- A runner for LA-CDM (its repo has code but no trained weights). DiagAgent-14B runs here through [`scripts/lambda/diagagent_adapter.py`](scripts/lambda/diagagent_adapter.py) and on DiagBench through `scripts/lambda/diagbench_*.py`.
- MIMIC-IV-ED triage variant (§4.3), clinician adjudication tooling (§6.3), and the Med-PRM verifier (stretch goal).
