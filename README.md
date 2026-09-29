# DEFER-Dx

An interactive clinical diagnostic agent trained with RL over four actions: **ASK, TEST, COMMIT, DEFER**. Deferring to a clinician is a learned action with its own verifiable reward (the *group-consensus deferral reward*), not a confidence threshold applied afterwards. The label space is open-world: the four MIMIC-CDM abdominal conditions plus **OTHER**.

The research plan is in [stanford_idea_md.md](stanford_idea_md.md). This repo is the base implementation: data pipelines, environment, rewards, metrics, baselines, and the SFT and GRPO trainers. **It contains no results.**

> **PhysioNet DUA.** MIMIC text must never reach a third-party service. The repo has no hosted-API clients: every policy runs on local weights (`transformers` / `vllm`). `.gitignore` excludes `data/`, `outputs/`, `*.jsonl`, `*.csv*` and `*.pkl`. Rollout files contain MIMIC text, so treat them as credentialed data.

## Install

```bash
pip install -e ".[train,data,dev]"   # add ,vllm for fast evaluation rollouts
pytest                               # 47 tests; CPU-only; no downloads
```

## Quickstart (synthetic data, no GPU)

```bash
bash scripts/smoke_synthetic.sh
```

This generates fabricated cases, runs the oracle and random policies, builds SFT data and evaluates. Synthetic numbers mean nothing; they only show the pipeline works.

## Real-data pipeline

```bash
# 1. MIMIC-IV-Ext-CDM -> canonical cases. FIRST check file/column names:
deferdx data inspect-cdm --cdm-dir /path/to/mimic-iv-ext-cdm
#    ...fix configs/cdm_format.yaml if needed, then:
deferdx data build-cdm --cdm-dir /path/to/mimic-iv-ext-cdm --split-file <official_split.csv> --out data/cdm

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

# 6. Evaluate (greedy) on test + OTHER
deferdx rollout --cases data/cdm/test.jsonl data/openworld/other.jsonl --policy vllm \
    --model Qwen/Qwen3-8B --adapter outputs/grpo/final --out outputs/eval/deferdx_test.jsonl
deferdx evaluate --rollouts outputs/eval/deferdx_test.jsonl --out outputs/eval/deferdx_test.json
```

The full §6.1 baseline grid is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Layout

| Path | What it does | Plan section |
|---|---|---|
| [src/deferdx/data/cdm_loader.py](src/deferdx/data/cdm_loader.py) | MIMIC-IV-Ext-CDM (PhysioNet CSV or Hager-framework pickles) to `Case`; earliest value per repeated test | §4.1 |
| [src/deferdx/data/openworld.py](src/deferdx/data/openworld.py) | MIMIC-CDM-OW: ICD filtering in DuckDB, discharge-note section extraction, stratified sampling, leak flags | §4.2 |
| [src/deferdx/data/synthetic.py](src/deferdx/data/synthetic.py) | Fabricated cases for tests | — |
| [src/deferdx/env/](src/deferdx/env/) | Reveal-on-request POMDP, 19 tests and 7 ASK topics with costs, action parser | §3.1 |
| [src/deferdx/rewards/scoring.py](src/deferdx/rewards/scoring.py) | Commit reward (accuracy, Brier, severity, cost) and the group-consensus DEFER reward | §3.2 |
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
5. **"Mean Acc" means case-weighted accuracy.** LDTL's 93.4 matches the case-weighted mean of its per-class numbers; the unweighted mean would be 90.25. `accuracy_full_coverage` reproduces 93.4. Balanced accuracy is reported separately.
6. **Full-coverage accuracy for DEFER** uses the top of the deferral differential, so a deferring agent still gets a leaderboard-comparable number. Deferral precision/recall can use a real no-defer run as the counterfactual (`--counterfactual`).
7. **RL samples are per turn, not per conversation.** Qwen3's chat template strips earlier `<think>` blocks, so the policy gradient uses the exact prompt and completion tokens of each generation call.

## Guesses to verify before trusting any number

- **[configs/cdm_format.yaml](configs/cdm_format.yaml): the CDM file and column names are best guesses.** Run `deferdx data inspect-cdm` first. The loader drops admissions without an unambiguous label and logs how many.
- **Official 70/10/20 split.** Without `--split-file`, a deterministic patient-level split is generated with a warning. That split is *not* LDTL's.
- **Catalog lab names and costs** ([configs/test_catalog.yaml](configs/test_catalog.yaml)): costs are approximate CMS-scale placeholders. `deferdx data coverage` shows which catalog tests resolve on your data and which lab names nothing reaches.
- **The severity matrix** is a time-to-harm proxy. Replace it with the clinician-built version.
- **ICD lists** ([configs/openworld_icd.yaml](configs/openworld_icd.yaml)) need clinician review. Cases flagged `meta.possible_label_leak` should be reviewed by hand.
- **Pipeline-shift leakage in the open-world data.** OTHER cases come from discharge summaries via this repo's pipeline, while CDM cases come from Hager et al.'s pipeline. Build `--controls` and check that a simple classifier cannot tell `openworld_control` from `cdm` cases. If it can, the agent can learn to detect the source instead of the disease.
- **Test availability is informative.** For example, a HIDA scan implies someone already suspected cholecystitis. `charge_unavailable: true` charges for tests that were never performed. Consider reporting an ablation.

## Not built yet

- A multi-GPU / verl port of the trainer. The reward functions are framework-agnostic; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- Runners for external baselines (LA-CDM, DiagAgent-14B / DiagGym), which live in their own repos.
- MIMIC-IV-ED triage variant (§4.3), clinician adjudication tooling (§6.3), and the Med-PRM verifier (stretch goal).
