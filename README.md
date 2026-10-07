# DEFER-Dx

An interactive clinical diagnostic agent trained with reinforcement learning over four actions: **ASK, TEST, COMMIT, DEFER**. Deferring to a clinician is a learned action. Its reward is the **counterfactual escalation value (CEV)**: at the exact state where the agent escalates, forced continuations branched from that state estimate the clinical value of *not* escalating, by deciding now or investigating further. The DEFER turn is credited with the difference. The handoff is a probabilistic differential trained with a strictly proper scoring rule. The label space is open-world: the four MIMIC-CDM abdominal conditions plus **OTHER**.

- Research plan: [stanford_idea_md.md](stanford_idea_md.md) (target: Stanford AI+HEALTH 2026 abstract, due Oct 15, 2026)
- The method as implemented, with every constant: [docs/METHODS.md](docs/METHODS.md)
- Paper draft: [docs/PAPER.md](docs/PAPER.md) · abstract scaffold: [docs/ABSTRACT.md](docs/ABSTRACT.md)
- Results: [docs/RESULTS.md](docs/RESULTS.md) (generated as evaluations finish)

> **PhysioNet DUA.** MIMIC text never reaches a third-party service. Every model runs on local weights (`transformers` / `vllm`), and the repo has no hosted-API clients. `.gitignore` excludes `data/`, `outputs/`, `.env`, `*.jsonl`, `*.csv*` and `*.pkl`. Rollout files contain MIMIC text, so treat them as credentialed data. Committed documents hold aggregates only, and patient-derived counts of 1–9 are suppressed. Weights trained on MIMIC are not pushed to public hubs; PhysioNet's credentialed hosting is the DUA-consistent release route.

---

## 1. Status (2026-10-07)

| Component | State |
|---|---|
| Data: MIMIC-IV-Ext-CDM 1.1, MIMIC-IV 2.2 hosp, MIMIC-IV-Note 2.2 | downloaded; SHA256 and gzip verified |
| Open-world set MIMIC-CDM-OW: 713 OTHER + 648 same-pipeline controls | built; deterministic (byte-identical rebuilds) |
| Patient-disjoint cohorts (`data/cohorts/`) | built |
| Single-GPU colocated GRPO trainer | built, tested and in use |
| Case-level group-consensus arm (`configs/grpo_deferdx.yaml`; the published mechanism, kept as a comparator) | **running**: resumed at step 76 after a reboot; ends around Oct 7, 22:30 IST |
| **DEFER-Dx with the counterfactual escalation value** (`configs/grpo_cev.yaml`, the method) | implemented and unit-tested; queued next (`scripts/queue_v2.sh`): smoke test, then 150 steps, around Oct 8–9 |
| Held-out evaluation of the consensus arm and the zero-shot baselines | queued; first numbers around Oct 8, 01:30 |
| GRPO control (no DEFER), the key comparator | queued; around Oct 9, 01:00 |
| Ablations of what is new: CEV without the scored handoff; a constant deferral reward. Robustness and gpt-oss-20b | queued (`scripts/queue_v2.sh`); through about Oct 11 |
| Tests | 108 pass (`pytest`; CPU only) |

**No held-out result exists yet.** The numbers in §3 are the interim training and dev-set signals and earlier zero-shot baselines. Read them as such.

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

Published MIMIC-CDM numbers use different splits, metrics and environments, so only some comparisons are like-for-like. What exists today (held-out numbers are from earlier zero-shot runs, [docs/BASELINES.md](docs/BASELINES.md)):

| System | Split | Mean-class acc | Case-weighted acc | Diverticulitis | Selective (coverage) |
|---|---|---|---|---|---|
| LA-CDM, trained (ICLR 2026, reported) | LA-CDM test | 81.3 | – | 75.0 | – |
| LDTL (reported; current SOTA) | own unpublished 70/10/20 | – | **93.4** | 78.8 | – |
| Random planner (from LDTL) | own split | – | 84.8 | 90.4 | – |
| DiagAgent-14B (run here) | LA-CDM test | 71.9 | 77.9 | 56.0 | – |
| **Qwen3-8B zero-shot in this environment** (run here, 3 seeds) | LA-CDM test | **87.2 ± 2.4** | 88.3 ± 1.1 | 84.8 (all 2,400 cases) | **95.5 ± 0.7 (85.8%)** with a post-hoc threshold |
| Case-level consensus arm (published mechanism) | LA-CDM val + test | pending | pending | pending | pending |
| **DEFER-Dx (CEV)** | LA-CDM val + test | pending | pending | pending | pending |

**Verdict so far:**
- **Above LA-CDM:** the environment plus an untrained Qwen3-8B already exceeds LA-CDM's 81.3 on its own test split. The environment differs, though: full history versus a summary, 22 tests versus 12, and a different base model.
- **Above LDTL on diverticulitis:** 84.8 versus 78.8, on a different split.
- **Meets the plan's selective target:** "> 95 at 85% coverage" is already met by a post-hoc threshold.
- **Below LDTL's 93.4 at full coverage:** this is the one SOTA number not yet beaten, and LDTL's split is unpublished, so no head-to-head is possible.
- **Undecided:** whether learned deferral beats thresholding is the central claim. It is decided only by the DEFER-Dx versus thresholded-control paired comparison (around Oct 9).

The plan itself (§6.4) says not to claim SOTA on LDTL's full-coverage metric; the pitch is a Pareto improvement on safety axes nobody else reports.

## 3. Interim training results (not held-out)

**Dev set** (212 cases held out of RL: 160 CDM, 32 OTHER, 20 controls; one sample at T = 0.6, so differences under ~3 points are noise):

| Step | Acc (full cov.) | Mean-class | Coverage | Selective acc | Diverticulitis (n = 18) | AURC ↓ | ECE ↓ | Brier ↓ | OOD defer | OOD false commit | Tests/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 77.2 | 73.2 | 75.0 | 89.6 | 61.1 | 0.078 | 0.043 | 0.096 | 53.1 | 9.4 | 1.65 |
| 50 | 77.8 | 74.1 | 73.9 | 87.2 | 66.7 | 0.089 | 0.054 | 0.121 | 59.4 | 12.5 | 2.15 |
| 75 | **84.4** | **82.2** | 76.1 | **95.6** | **83.3** | **0.047** | 0.063 | **0.063** | **71.9** | 9.4 | 2.37 |

**Training batches** (12 cases × 8 rollouts per step, T = 1.0; window means):

| Steps | Commit accuracy | Full-coverage acc | In-set deferral | OTHER said on in-set cases | OOD deferral | OOD false commit | Tests/case | Entropy |
|---|---|---|---|---|---|---|---|---|
| 1–25 | 85.7 | 77.7 | 22.5 | 6.1 | 41.3 | 8.7 | 1.80 | 0.31 |
| 26–50 | 87.0 | 76.9 | 24.0 | 6.4 | 38.9 | 5.1 | 1.87 | 0.31 |
| 51–75 | 91.1 | 81.4 | 23.2 | 3.0 | 65.5 | 9.1 | 2.24 | 0.36 |
| 76–86 | **94.7** | **83.8** | 25.4 | **1.0** | **78.4** | **2.5** | 2.32 | 0.45 |

- **No deferral collapse:** in-set deferral stays at 22–25%, under the 30% cap, and the coverage multiplier ν ≈ 0.
- **Watched:** rising tests per case (cost) and rising entropy.

## 4. Method in brief

- **Actions.** `ASK(physical_exam)`; `TEST(x)` for 22 individual tests (CBC, CMP, lipase, CT abdomen, …) at 2025 BIDMC charges; `COMMIT(d, p)` with d ∈ {4 conditions, OTHER} and stated probability p; `DEFER(differential, reason)`. At most 8 investigations. The full history is shown at reset.
- **Commit reward:** α·1[d = y] − λ(p − 1[d = y])² − κ·C[y, d] − c·Σcost, with α = 1, λ = 1 (Brier), κ = 0.5, C the severity matrix, and c = 1/Σ(catalog prices).
- **Escalation (CEV):** the DEFER turn's advantage is E − V̂(s).
  - V̂(s) is the mean clinical return of K = 3 forced continuations from the same state (no escalation; tests allowed).
  - E = V_τ (OTHER: 1.2) + η·S(handoff) − μ − ν, with S the normalised Brier score of the handed-over differential, η = 0.3, μ = 0.1.
  - τ is annealed 0.95 → 0.85.
- **Comparator arm (published mechanism):** a case-level reward γ(τ − p̂) − μ, with p̂ the group's commit accuracy.
- **Coverage constraint:** in-set deferral ≤ 30% by projected dual ascent; ν is charged to deferring rollouts only.
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
  - gpt-oss-20b;
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
setsid nohup bash scripts/queue_v2.sh >> outputs/logs/queue_v2.log 2>&1 &

# Single pieces:
deferdx train grpo-vllm --config configs/grpo_cev.yaml            # DEFER-Dx, counterfactual escalation (the method)
deferdx train grpo-vllm --config configs/grpo_deferdx.yaml         # case-level consensus comparator arm
deferdx train grpo-vllm --config configs/grpo_nodefer.yaml         # control
deferdx eval-suite --model Qwen/Qwen3-8B --adapter outputs/runs/deferdx/final --name deferdx
python scripts/make_report.py && python scripts/make_figures.py    # docs/RESULTS.md, docs/figures/
python scripts/plot_training.py --runs deferdx=outputs/runs/deferdx nodefer=outputs/runs/nodefer
deferdx crossover                                                   # tau -> effective deferral threshold
```

**After a reboot,** relaunch the queue command above. Steps marked done in `outputs/queue/*.done` are skipped, and training resumes from its newest checkpoint (one is saved every 5 steps).

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
| [tests/](tests/) | 108 CPU tests (`pytest`) |

## 9. Documents

| Doc | What it holds |
|---|---|
| [docs/METHODS.md](docs/METHODS.md) | Environment, reward with constants, cohorts, training, evaluation protocol |
| [docs/RESULTS.md](docs/RESULTS.md) | ML-benchmark and clinical tables with intervals, open world, published context, paired tests (generated) |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | The run grid, queues, ablations and the question each answers |
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
