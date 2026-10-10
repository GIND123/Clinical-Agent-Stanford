# stanford_idea.md

> **Superseded plan (kept for the record).** This is the original research plan. Its method, the group-consensus deferral reward in §3.2 and the draft abstract in §7, turned out to be published (TIAR, KARL, AWA-RL) and is now only a comparator arm. The method is the counterfactual escalation value with a scored handoff ([docs/METHODS.md](docs/METHODS.md)). What actually ran is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); the submission draft is [docs/ABSTRACT.md](docs/ABSTRACT.md).

**Target:** Stanford AI+HEALTH 2026 (6th annual), online, Dec 8–9, 2026
**Deadline:** Thursday, Oct 15, 2026, 11:59pm PT
**Format:** 5-min lightning talk; trainees get complimentary registration
**Author status:** Student, credentialed PhysioNet access, unrestricted GPU, team

---

## 0. THE IDEA IN ONE PARAGRAPH

**DEFER-Dx** — the first interactive clinical diagnostic agent trained with reinforcement learning over a **four-action space {ASK, TEST, COMMIT, DEFER}**, where deferral to a clinician is a *learned action with its own verifiable reward*, not a post-hoc threshold. The core technical contribution is a **group-consensus deferral reward** that exploits GRPO's existing rollout structure to compute, for free, whether the policy *would have been wrong* on a case — and pays the agent to hand that case to a human. We additionally break the closed-world assumption that every prior MIMIC-CDM agent makes, by injecting out-of-label-set abdominal admissions so the agent can face patients whose true diagnosis is *not among the options*. Evaluated on risk-coverage, calibration, minority-class safety, and cost — not accuracy alone.

**The headline result we are hunting:** the current state of the art (LDTL, Apr 2026) raised mean accuracy on MIMIC-CDM to 93.4% but *dropped diverticulitis accuracy to 78.8% — below the 90.4% of a random planner*. Outcome-aligned trajectory optimization bought mean accuracy by sacrificing the minority class. DEFER-Dx should recover that loss by deferring exactly those cases, and demonstrate it on a risk-coverage curve.

---

## 1. WHY THIS IS THE NEED OF THE HOUR

Four independent forces converge on "calibrated deferral" in 2026. An AIMI abstract that names all four is very hard to reject as unmotivated.

### 1.1 Regulatory: the FDA has made human escalation the gating issue
- The FDA has authorized **zero** generative-AI clinical decision devices to date.
- The **Digital Health Advisory Committee meeting of Nov 6, 2025** (docket FDA-2025-N-2338, chaired by Ami Bhatt, MD) on generative-AI mental-health devices repeatedly emphasized the need for reliable mechanisms to **detect and escalate acute safety concerns to ensure timely human intervention**. That is, verbatim, the capability this project trains.
- The FDA's **real-world performance RFC** (docket FDA-2025-N-4203, closed Dec 1, 2025) solicited comment on postmarket monitoring metrics and human-AI interaction.
- **PCCPs** (predetermined change control plans) from the Jan 6, 2025 draft guidance on AI-enabled device software require sponsors to pre-specify monitoring; calibrated abstention rates are exactly the kind of monitorable quantity that fits.
- **ONC HTI-1** requires 31 source attributes ("AI nutrition labels") for predictive DSIs. A deferral policy with a published risk-coverage curve *is* a transparency artifact.

**Framing sentence for the abstract:** *"Regulators are converging on the requirement that clinical AI escalate to humans when uncertain. No published diagnostic agent is trained to do so."*

### 1.2 Clinical: forced-choice agents are unsafe in the only way that matters
Every sequential-diagnosis system published on MIMIC-CDM is a **closed-world forced-choice classifier over four abdominal conditions**. A real ED patient with abdominal pain may have bowel obstruction, perforated ulcer, mesenteric ischemia, ruptured AAA, ectopic pregnancy, nephrolithiasis, or DKA. Every current agent must answer "appendicitis, cholecystitis, diverticulitis, or pancreatitis." There is no escape hatch. This is not a benchmark artifact to be waved away — it is the single largest gap between these systems and deployment.

### 1.3 Technical: the field has optimized the wrong objective, and it shows
The LDTL ablation table is the proof. Below, mean accuracy rises monotonically while the minority class collapses:

| Method | Mean Acc | Diverticulitis Acc |
|---|---|---|
| Random planner | 84.8 | **90.4** |
| w/o latent-path reg. | 88.6 | 80.7 |
| LDTL (SOTA) | **93.4** | **78.8** |

A random planner beats the state of the art on diverticulitis by 11.6 points. This is a reward-design pathology, and it is precisely what a calibration-aware deferral term is designed to catch. **This single table is your motivation slide.**

### 1.4 Commercial: nobody sells this
Ambient documentation (Abridge, Microsoft Dragon Copilot, Suki, Nabla, Ambience) and literature-grounded Q&A (OpenEvidence — ~$12B valuation after its Jan 2026 Series D, >20M consultations/month; UpToDate Expert AI; ChatGPT for Clinicians) are mature markets. Assurance infrastructure (CHAI, Mayo Clinic Platform Validate, Duke Health AI Evaluation, Health AI Partnership) is scaling. **What no vendor ships is a system that knows when to stop and hand over.** Health systems are buying nothing in this category because nothing exists.

---

## 2. COMPETITIVE LANDSCAPE — EXACT NUMBERS AND EXACT GAPS

### 2.1 The MIMIC-CDM sequential leaderboard (from LDTL, arXiv 2604.05116, Apr 6 2026)
Backbone Llama-3-8B, LoRA r=16 α=32, lr 2e-5, ~6 hrs on 2×H100, 70/10/20 patient-level split, max 3 steps, action space = {lab, physical, image}.

| Method | App. | Chol. | Div. | Panc. | **Mean Acc** | **F1** |
|---|---|---|---|---|---|---|
| SFT-all (full info, upper bound) | 97.9 | 93.1 | 90.4 | 90.7 | **94.2** | 95.8 |
| Fixed info: history only (ZS) | 47.9 | 34.6 | 73.1 | 79.6 | 54.1 | 58.8 |
| Fixed info: history + 1 test (ZS) | 50.5 | 47.7 | 76.9 | 72.2 | 58.3 | 60.5 |
| Fixed info: history + 2 tests (ZS) | 60.9 | 60.7 | 78.8 | 80.5 | 67.2 | 68.2 |
| All info (ZS) | 73.9 | 70.0 | 82.7 | 87.9 | 76.9 | 76.7 |
| Random planner | 82.8 | 85.4 | 90.4 | 85.2 | 84.8 | 83.7 |
| ReAct | 90.2 | 79.7 | 66.7 | 62.9 | 74.9 | 79.1 |
| **LA-CDM** (ICLR 2026) | 93.1 | 83.6 | 75.0 | 73.5 | **81.3** | 84.1 |
| **LDTL** (SOTA) | 98.9 | 95.4 | 78.8 | 87.9 | **93.4** | 91.7 |

Frontier APIs in the same loop: gpt-4o 95.6, claude-sonnet-4 95.8, gemini-3-flash 95.2, deepseek-chat 96.6, gpt-4o-mini 90.5.

> ⚠️ **PhysioNet compliance note.** LDTL ran frontier APIs on MIMIC text under an inference-only protocol. Do **not** replicate this casually. PhysioNet's DUA restricts sending MIMIC data to third-party services; approved routes are limited (e.g., Azure OpenAI under a BAA, GCP under the responsible-use agreement). **Default to open-weight models on your own GPUs.** This is also a rhetorical asset: your system is deployable on-prem, theirs is not.

### 2.2 The four systems you must differentiate from

| System | Venue / date | Action space | Trained? | Cost reward | Calibration | **DEFER action** | Open-world | Data |
|---|---|---|---|---|---|---|---|---|
| **MediQ** | NeurIPS 2024 | ask, answer | ✗ prompting | ✗ | abstention module (prompted) | ✗ | ✗ | MedQA-derived |
| **LA-CDM** | **ICLR 2026** (code public) | test, hypothesize, commit | ✓ SFT+RL | ✓ | ✓ uncertainty *expression* | ✗ | ✗ | MIMIC-CDM |
| **DiagAgent / DiagGym** | arXiv 2510.24654 (weights + code public) | exam, commit | ✓ multi-turn GRPO | ✓ info yield | ✗ | ✗ | ✗ | 863 MIMIC-IV cases |
| **LDTL** | arXiv 2604.05116 | 3 coarse test categories, commit | ✓ two-stage KL-to-posterior | ✓ (implicit, via steps) | binary YES/NO only | ✗ | ✗ | MIMIC-CDM |
| **MedExAgent** | arXiv 2605.07058 (May 2026) | ask, exam, diagnose | ✓ SFT + DAPO | ✓ CMS fee schedule + discomfort | ✗ | ✗ | ✗ | DDxPlus, PMC-Patients (synthetic) |
| **Doctor-R1** | ICLR 2026 | ask, diagnose | ✓ GRPO | ✗ | ✗ | ✗ | ✗ | simulated outpatients |
| **MedAbstain** | arXiv 2601.12471 | answer, abstain | ✗ conformal/prompted | ✗ | ✓ | (abstain, but) | ✗ | single-turn MCQA |
| **Safe-Psych** | arXiv 2607.13036 | DIAGNOSE/CLARIFY/ABSTAIN | ✗ benchmark only | ✗ | ✓ | ✓ (evaluated, not trained) | ✗ | psychiatry |
| **DEFER-Dx (ours)** | — | **ASK, TEST, COMMIT, DEFER** | **✓ GRPO** | **✓** | **✓ proper scoring rule in reward** | **✓ learned action** | **✓** | MIMIC-CDM + MIMIC-IV OOD |

### 2.3 The novelty claim, stated precisely
> No published system trains an LLM agent, via reinforcement learning with verifiable rewards in a multi-turn clinical environment, over an action space that includes **deferral-to-clinician as a first-class learned action**, with a reward that jointly optimizes **diagnostic accuracy, information cost, and calibrated abstention**, under an **open-world label space**.

Each of the three italicized pillars exists separately. The combination does not. Say it exactly this way in the abstract — narrow, checkable claims survive review; broad ones invite a reviewer to name a counterexample.

---

## 3. THE METHOD

### 3.1 Formalization
A POMDP ⟨S, A, T, Ω, O, R⟩ where the hidden state is the true diagnosis y ∈ D ∪ {OTHER}.

At step t the policy π_θ emits one action:

| Action | Meaning | Effect |
|---|---|---|
| `ASK(q)` | history/exam question | returns finding from the case record; cost c_ask |
| `TEST(x)` | order a specific lab/imaging study | returns result; cost c_test(x) |
| `COMMIT(d, p)` | final diagnosis **d** with explicit probability **p ∈ [0,1]** | terminal |
| `DEFER(differential, reason)` | escalate to clinician with a ranked differential | terminal |

Two departures from LDTL worth stating in the abstract: the action space is **individual tests, not three coarse categories** (their stated limitation), and commitment carries a **numeric probability**, not a binary confident-YES/NO flag.

### 3.2 The reward (the actual contribution)

> **Superseded:** see the note at the top; the method is the counterfactual escalation value (docs/METHODS.md §2).

**On COMMIT(d, p):**

```
R_commit = α·1[d = y]                    accuracy
         − λ·(p − 1[d = y])²             Brier — proper scoring rule, punishes miscalibration
         − κ·C[y, d]                     clinical severity cost matrix (asymmetric)
         − Σ_t cost(a_t)                 cumulative investigation cost
```

**On DEFER — the group-consensus deferral reward:**

GRPO already samples a group of G rollouts per case. Let **p̂ = fraction of the committing rollouts in that group that were correct**. This is a free, on-policy, label-verified estimate of "how hard is this case *for me right now*." Then:

```
R_defer = γ·(τ − p̂)  −  μ  −  Σ_t cost(a_t)
```

- If the policy's own group accuracy on this case is below the safety threshold τ, deferring earns positive reward.
- If the group reliably gets it right, deferring is penalized.
- μ is a fixed handoff cost (clinician time is not free).
- τ is a **policy dial with clinical meaning** — set it from a target selective-accuracy level (start τ = 0.85; sweep it as an ablation to generate the risk-coverage curve directly).

**Why this is elegant and defensible:** it requires no extra labels, no separate difficulty model, no human annotation. It reuses computation GRPO already performs. It is self-normalizing as the policy improves — the bar rises with competence. And it is a *counterfactual* signal ("would I have been wrong?"), which is the correct target for a deferral policy, rather than a heuristic confidence threshold. Present this as the method slide.

**Anti-collapse constraint.** Abstain-R1 documents reward hacking via over-abstention. Enforce a coverage floor with an adaptive Lagrangian penalty:

```
R ← R − ν_k · max(0, deferral_rate_batch − ρ_max),    ρ_max ≈ 0.30
```
with ν_k increased when the constraint is violated across consecutive batches. Also require the model to emit p on every COMMIT so calibration is always scored.

**Open-world handling.** For out-of-label-set cases, both `COMMIT(OTHER, p)` and `DEFER` earn positive reward, with DEFER weighted slightly higher — clinically, "not one of these" warrants escalation rather than a shrug.

**Severity matrix C[y, d].** Asymmetric: missing a condition with a short time-to-harm window costs more than a benign confusion. Build a 5×5 matrix with a clinician collaborator; if none is available in time, use a defensible published proxy (time-sensitivity of definitive management) and label it a limitation.

### 3.3 Architecture
Single policy, not a two-agent split. LDTL and LA-CDM both use planner/diagnoser pairs and LDTL explicitly notes joint training is unstable — but a single policy with a unified action space is what MedExAgent showed works under DAPO, and it is the only way the deferral decision can be conditioned on the full reasoning trace. Emit reasoning in `<think>` tags, then a structured action block. Optional stretch goal: a **step-level medical process reward model** (Med-PRM style) as an inference-time verifier — keep this as a stretch, not a dependency.

---

## 4. DATA — EXACT SPECIFICATION

### 4.1 Primary environment: MIMIC-IV-Ext-CDM v1.0
- PhysioNet: `physionet.org/content/mimic-iv-ext-cdm/1.0/`
- Source: Hager et al., *Nature Medicine* 30(9):2613–2622, 2024 (doi 10.1038/s41591-024-03097-1); built from **MIMIC-IV v2.2 hosp module**, BIDMC 2008–2019.
- **2,400 patients**: appendicitis 957, cholecystitis 648, diverticulitis 257, pancreatitis 538.
- Per case: patient history (documented symptoms, comorbidities, family history), physical examination notes, imaging reports (CT, X-ray, US, MRI), laboratory records (blood, urine, microbiology). **Not every test exists for every patient** — handle missing actions explicitly.
- Crucially, it ships **standardized test-name mappings across patients**, which is what makes a consistent action space possible at all.
- Use the **official patient-level 70/10/20 split** — same as LDTL — so your numbers are directly comparable. Do not re-split.
- Preprocessing to match LDTL: when a test is repeated within an admission, **keep only the earliest value** (approximates early-stage diagnosis).

### 4.2 The open-world extension (your second novel asset)
Build **MIMIC-CDM-OW** from MIMIC-IV v2.2 `hosp` module:
- Tables: `admissions`, `diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems`, `microbiologyevents`; notes from **MIMIC-IV-Note** (`radiology`, `discharge`).
- Cohort: admissions with an abdominal-pain presentation whose **primary ICD-9/10 diagnosis is *not* one of the four CDM conditions** — e.g. intestinal obstruction, perforated peptic ulcer, mesenteric ischemia, nephrolithiasis, GI bleed, AAA, ectopic pregnancy, DKA, gastroenteritis.
- Target **600–1,000 cases**, label `OTHER`, mixed into the test set only (or a small fraction into training — ablate both).
- Reuse Hager et al.'s public extraction pipeline rather than writing your own. **⚠️ Verify the repo and its licence in week 1; this is your single largest engineering risk.**

### 4.3 Secondary environment (optional, for external validity)
**MIMIC-IV-ED v2.2** — tables `edstays`, `triage`, `vitalsign`, `diagnosis`; ~425,000 ED stays, BIDMC 2011–2019; **ESI acuity 1–5** as a verifiable label. Gives a triage-deferral variant and a fairness axis (build on the EQUITRIAGE demographic-bias findings). Also lets you cite **ED-Copilot** (ICML 2024) as a cost-aware precedent. Treat as stretch — do not let it eat week 3.

### 4.4 Free environment you should exploit
**DiagGym** (`huggingface.co/Henrychur/DiagGym`, `github.com/MAGIC-AI4Med/DiagGym`) is a released EHR world model that generates examination results conditioned on patient profile + requested exam, validated on 863 MIMIC-IV cases, with **DiagAgent-14B** released as a GRPO-trained agent. Use it for (a) unlimited rollout generation beyond the 2,400 real cases, and (b) a strong published baseline you can run yourself rather than merely cite. This materially de-risks the timeline.

---

## 5. MODELS AND INFRASTRUCTURE

### 5.1 Base models
| Model | Params | Licence | Role |
|---|---|---|---|
| **Qwen3-8B** | 8B | Apache 2.0 | **Primary.** Matches LDTL's 8B scale for apples-to-apples; native thinking mode |
| **Qwen3-14B** | 14B | Apache 2.0 | Scale ablation |
| **DiagAgent-14B** | 14B | see model card | Strong published baseline + optional warm start |
| **MedGemma-27B-text-it** | 27B | HAI-DEF | Medical-specialist comparison (note: it underperformed general models in MedExAgent's table — useful, slightly contrarian data point) |
| **Meditron3-8B** | 8B | Llama 3.1 Community | MedExAgent's base; use if replicating them |
| **gpt-oss-20b** | 20B | Apache 2.0 | Reasoning-model variant |

Do **not** make frontier APIs your headline system — PhysioNet DUA friction plus the on-prem deployability argument both point to open weights.

### 5.2 Training configuration (start here, tune later)
- **Framework:** `verl` or TRL GRPO; vLLM for rollouts.
- **Algorithm:** GRPO (LA-CDM, DiagAgent) or DAPO (MedExAgent). Start GRPO; DAPO's dynamic sampling helps if you hit gradient starvation from all-correct groups.
- **Group size G = 16** (MedExAgent's choice; also what makes p̂ a usable estimate — G≥8 minimum).
- LoRA r=16, α=32 for the first pass; full fine-tune if GPUs allow.
- lr 1e-6 (RL) / 2e-5 (SFT), AdamW, cosine with 5% warmup.
- Clip ratio (0.2, 0.3), KL coef 0.0, entropy coef 0.002, rollout temperature 1.0.
- **Stage 1 SFT** on trajectories with reasoning traces (including deferral exemplars — you must seed the DEFER action or RL will never explore it). **Stage 2 GRPO.**
- **Compute reality check:** LDTL trained in ~6 hrs on 2×H100. MedExAgent's RL took ~5.5 days on 4×96GB. Budget 2–5 days per full RL run; plan for ~6–10 runs.

### 5.3 Zero-API operation (no external model spend)

This project requires **no paid API calls**. Two structural reasons: the label space is closed (4 conditions + OTHER), so reward is exact match rather than LLM-judged semantic matching; and the environment is a real-data reveal-on-request lookup, not an LLM patient simulator. Those are the two cost centres in comparable papers.

| Where competitors use an API | Our replacement | Cost |
|---|---|---|
| LLM judge for diagnosis reward (MedExAgent: gpt-4.1-mini, validated on 99 DOID pairs) | Exact label match over 5 classes | $0 |
| Patient simulator per rollout turn (MedExAgent: gpt-4.1-mini / Qwen3-30B-A3B) | MIMIC-CDM reveal-on-request lookup | $0 |
| SFT trace generation (MedExAgent: gpt-4o-mini) | Local Qwen3-32B or gpt-oss-120b; **or** rejection sampling from own base model (STaR-style: roll out at T=1, keep correct trajectories) | GPU only |
| Deferral exemplars | Derived from ground-truth label + control-model errors | $0 |
| Frontier-model baseline rows (LDTL's Table 3) | **Omit.** LA-CDM and DiagAgent-14B are open weights | $0 |
| Open-world cohort construction | Deterministic ICD filtering in SQL | $0 |
| Embeddings, if needed | MedEmbed-base-v0.1 (Apache 2.0), local | $0 |

**Turn the constraint into an argument.** State in the paper that the PhysioNet DUA restricts transmission of MIMIC text to third-party services, so an on-prem open-weight system is the only honestly deployable configuration. Pre-empt the "why not GPT-5?" question: raw closed-world accuracy is not the contribution axis, and frontier models have no calibrated deferral and collapse under the open-world label space regardless.

**Only future need for a judge:** extending to free-text open-ended diagnosis. Then use a local Qwen3-32B judge and validate agreement against a few hundred hand-labeled pairs — MedExAgent's protocol, self-hosted.

### 5.4 Seeding the DEFER action
Cold-start exploration is the most likely silent failure. Mitigations, in order: (1) include explicit DEFER exemplars in SFT data, generated by forcing an oracle to defer on cases the SFT model gets wrong; (2) entropy bonus on the action-type distribution for the first N steps; (3) curriculum — begin with τ high (deferral cheap), anneal down.

---

## 6. EVALUATION PROTOCOL

### 6.1 Baselines you will run yourself (not just cite)
1. Base model, zero-shot, sequential (ReAct-style) — replicates the 74.9 row.
2. Base model + SFT, forced commit, no defer — **your primary control**.
3. GRPO with accuracy + cost only, no calibration, no defer — an LA-CDM/DiagAgent-class reproduction.
4. **LA-CDM** from public code (`github.com/dharouni/LA-CDM`).
5. **DiagAgent-14B** from released weights.
6. Post-hoc thresholding: baseline #3 + a confidence threshold tuned on validation — **the critical ablation.** A reviewer will ask "why not just threshold?" You must show that *learning* deferral beats *thresholding* it. If it does not, that is itself a publishable negative result, but you need to know by week 3.
7. Conformal prediction wrapper (Safe-to-Stop-style) as a non-RL abstention comparator.

### 6.2 Metrics
**Closed-world accuracy (comparable to the leaderboard):** per-class accuracy, mean accuracy, macro-F1.

**Selective prediction (the new axis):**
- Risk-coverage curve; **AURC**; accuracy@{70, 80, 90}% coverage.
- ECE and Brier score on committed cases.
- Deferral precision/recall against the counterfactual "would have been wrong."

**Safety:**
- **Confident-error rate**: fraction of cases committed with p > 0.8 that are wrong. Target: large reduction.
- Severity-weighted error (via C).
- **Minority-class accuracy (diverticulitis)** — the headline. Target: beat LDTL's 78.8 *and* random's 90.4.

**Open-world:** false-commit rate on OTHER cases; OOD AUROC from the deferral score.

**Cost:** mean tests/case; termination-step distribution (LDTL reports 318 one-step terminations — comparable).

### 6.3 Clinical validity
Structured labels (ICD, ESI) are proxies, not truth. Strengthen with a **small clinician adjudication**: have 1–2 clinician collaborators review ~50–100 deferral decisions and rate each as appropriate / over-cautious / missed-escalation. Even n=50 converts "we optimized a metric" into "clinicians agreed with 84% of the handoffs," which is the sentence that gets a lightning talk accepted at AIMI. **Start recruiting the clinician in week 1** — this has the longest lead time of anything in the plan.

### 6.4 Targets
| Metric | Current best | DEFER-Dx target |
|---|---|---|
| Mean accuracy (closed) | 93.4 (LDTL) | ≥ 93 at full coverage; **> 95 at 85% coverage** |
| Macro-F1 | 91.7 (LDTL) | > 92 |
| Diverticulitis acc | 78.8 (LDTL) | **> 90** |
| ECE | not reported by anyone | < 0.05 |
| Confident-error rate | not reported | ≥ 50% reduction vs. no-defer control |
| False-commit on OTHER | ~100% (forced choice) | < 30% |

Note the first row carefully: you may *not* beat LDTL's raw mean accuracy, and **you do not need to**. The pitch is Pareto improvement on a safety axis nobody reports, while remaining competitive on the axis they do. State this honestly — trying to claim SOTA on their metric is the weakest version of this abstract.

---

## 7. DRAFT ABSTRACT

> **Superseded:** see the note at the top; the method is the counterfactual escalation value (docs/METHODS.md §2).

> ⚠️ Open the submission form (linked from the AIMI Call for Abstracts, forms.gle/QxXyep9hx7RymSZh9) in week 1 to confirm word limit and required fields — the public page does not publish a strict template. Draft below is ~280 words; have a 150-word cut ready.

---

**Training Clinical Agents to Ask, Test, Commit, and Defer**

*Alternative titles: "Clinical Agents That Learn When to Hand a Case Back" (memorable, hides method) · "Calibrated Deferral in Interactive Clinical Diagnostic Agents" (conventional) · "Reinforcement Learning for Clinical Agents That Defer Under Uncertainty" (method-forward, suits Foundations track)*

**Background.** Regulators are converging on a requirement that clinical AI reliably escalate to human clinicians when uncertain; the FDA's Digital Health Advisory Committee emphasized timely human intervention as central to generative-AI device safety. Yet every published interactive diagnostic agent is a closed-world forced-choice system: it must name a diagnosis, cannot say "I don't know," and cannot hand a case back. The consequences are measurable — the current state of the art on the MIMIC-IV clinical-decision-making benchmark raises mean accuracy to 93.4% while reducing accuracy on the minority condition to 78.8%, below that of a random test-selection policy.

**Methods.** We introduce DEFER-Dx, an interactive diagnostic agent trained with group-relative policy optimization over a four-action space — ask, test, commit, and defer-to-clinician — in which deferral is a learned action rather than a post-hoc confidence threshold. We introduce a group-consensus deferral reward that uses the policy's own on-policy rollout agreement as a label-verified counterfactual estimate of case difficulty, combined with a proper scoring rule over stated diagnostic probability and an asymmetric clinical severity cost. We evaluate on MIMIC-IV-Ext-CDM (2,400 patients, four abdominal conditions) and on an open-world extension containing out-of-label-set abdominal admissions drawn from MIMIC-IV.

**Results.** [Risk-coverage separation vs. no-deferral control; accuracy@80% coverage; reduction in confident errors; recovery of minority-class accuracy; false-commit rate on out-of-set cases; clinician agreement with deferral decisions.]

**Conclusion.** Training deferral as an explicit action, rather than thresholding confidence after the fact, yields diagnostic agents that are safer at equal cost — and produces the calibration and escalation artifacts that emerging regulatory frameworks require.

---

## 8. RISKS AND FALLBACKS

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| A preprint publishes this first before Oct 15 | **Medium-high** — this subfield moves monthly | Re-check weekly. If scooped on the core, pivot the framing to the **open-world** contribution (nobody has done out-of-label-set MIMIC-CDM) or to the **empirical audit** of minority-class collapse |
| Deferral collapse (defers everything) | High without mitigation; documented in Abstain-R1 | Coverage floor with adaptive Lagrangian; τ curriculum; mandatory probability output |
| Cold-start: DEFER never explored | Medium | Seed SFT with deferral exemplars; action-type entropy bonus |
| Learned deferral doesn't beat post-hoc threshold | Medium | Know by week 3. This is a legitimate negative result — reframe as "when does learning deferral pay?" AIMI's audience respects a rigorous negative |
| Open-world cohort construction eats the schedule | Medium | Assign to a dedicated teammate; cap at 600 cases; drop to a closed-world-only abstract if week 3 arrives without it |
| No clinician collaborator | Medium | Proxy labels only; state as explicit limitation. Do not fabricate clinical validation |
| PhysioNet DUA violation via API calls | Low but **catastrophic** | Open weights on your own GPUs, full stop. No MIMIC text to third-party APIs |
| Single-center, single-organ-system | Certain | State plainly as a limitation. AIMI reviewers respect acknowledged scope far more than overclaiming |

---

## 9. LIGHTNING-TALK NARRATIVE (5 minutes, ~5 slides)

1. **The problem (45s).** A patient arrives with abdominal pain. Every published diagnostic agent must choose among four conditions. What if she has a ruptured ectopic pregnancy? *Nothing in the system can say so.*
2. **The evidence (60s).** The LDTL table. Mean accuracy went up; the minority class went below random. We are optimizing the wrong objective.
3. **The method (90s).** Four actions, one of which is "hand this to a human." The group-consensus reward: GRPO already tells us, for free, whether we'd have been wrong.
4. **The result (90s).** The risk-coverage curve. One figure, annotated with the diverticulitis recovery and the confident-error reduction.
5. **Why now (35s).** The FDA is asking for exactly this capability, and no one ships it. Limitations: single center, four conditions, proxy labels.

---

## 10. WHAT TO VERIFY BEFORE COMMITTING

- [ ] Hager et al.'s extraction pipeline repo — exists, runs, licence permits reuse? **(largest engineering risk)**
- [ ] LA-CDM reproduces its reported 81.3 on your hardware
- [ ] DiagAgent-14B weights load and the DiagGym loop runs
- [ ] Exact submission form fields, word limit, whether results are required
- [ ] Whether arXiv 2607.13036 (Safe-Psych) or 2609.09678 (Safe-to-Stop) has been extended to RL since publication
- [ ] Whether a newer paper cites LDTL and adds abstention — check Semantic Scholar citations weekly
- [ ] Your institution's PhysioNet DUA terms regarding cloud compute

---

## 11. KEY REFERENCES

**Environment / data**
- Hager P, Jungmann F, Holland R, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. *Nature Medicine* 30(9):2613–2622, 2024. doi:10.1038/s41591-024-03097-1
- MIMIC-IV-Ext-CDM v1.0 — physionet.org/content/mimic-iv-ext-cdm/1.0/
- Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data* 10:1, 2023.
- MIMIC-IV-Ext-CDS v1.0.2 — physionet.org/content/mimic-iv-ext-cds/1.0.2/

**Direct competitors**
- Bani-Harouni D, Pellegrini C, Özsoy E, Keicher M, Navab N. Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning. **ICLR 2026.** arXiv:2506.13474. Code: github.com/dharouni/LA-CDM
- Shen X, Liu H, Song D, Min MR. Uncertainty-Guided Latent Diagnostic Trajectory Learning for Sequential Clinical Diagnosis. arXiv:2604.05116, Apr 2026.
- Qiu P, Wu C, Liu J, et al. Evolving Diagnostic Agents in a Virtual Clinical Environment. arXiv:2510.24654. github.com/MAGIC-AI4Med/DiagGym
- Gao Y, Zhou X, Li Y, Zhao Y, Liu R. MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments. arXiv:2605.07058, May 2026. github.com/EndlessCG/medexagent
- Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning. ICLR 2026. github.com/thu-unicorn/Doctor-R1
- Li SS, Balachandran V, Feng S, Ilgen J, Pierson E, Koh PW, Tsvetkov Y. MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning. NeurIPS 2024. arXiv:2406.00922

**Abstention / calibration**
- Machcha S, Yerra S, Gupta S, et al. MedAbstain: Knowing When to Abstain — Medical LLMs Under Clinical Uncertainty. arXiv:2601.12471, 2026.
- Bani-Harouni D, et al. Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of LLMs. arXiv:2503.02623
- TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning. arXiv:2509.25760
- Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL. arXiv:2604.17073
- Safe-Psych. arXiv:2607.13036 · Safe-to-Stop. arXiv:2609.09678
- Geifman Y, El-Yaniv R. SelectiveNet. ICML 2019 · Guo C, et al. On Calibration of Modern Neural Networks. ICML 2017

**Sequential diagnosis, other**
- Nori H, Daswani M, Kelly C, et al. Sequential Diagnosis with Language Models (MAI-DxO). arXiv:2506.22405
- Sun L, Agarwal A, Kornblith A, Yu B, Xiong C. ED-Copilot: Reduce Emergency Department Wait Time with Language Model Diagnostic Assistance. ICML 2024
- Schmidgall S, et al. AgentClinic. *npj Digital Medicine*, 2026
- Jiang Y, Black KC, Geng G, et al. MedAgentBench. *NEJM AI*, 2025

**Algorithms**
- Yu Q, et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476
- Qian C, et al. ToolRL: Reward is All Tool Learning Needs, 2025
- Yang A, et al. Qwen3 Technical Report. arXiv:2505.09388

**Regulatory**
- FDA Digital Health Advisory Committee, Nov 6 2025 — docket FDA-2025-N-2338
- FDA RFC on real-world performance — docket FDA-2025-N-4203
- FDA draft guidance, AI-Enabled Device Software Functions, Jan 6 2025
- ONC/ASTP HTI-1 final rule, DSI transparency (31 source attributes)

---

*Compiled Sept 19, 2026. Re-verify §2 novelty claims within 72 hours of submission.*