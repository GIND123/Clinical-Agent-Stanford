# Research notes: facts checked against primary sources

Checked 2026-09-28/29 before any MIMIC data was available. Each item says what changed in the repo because of it.

## 1. MIMIC-IV-Ext-CDM (the dataset)

Sources: [PhysioNet v1.1 page](https://physionet.org/content/mimic-iv-ext-cdm/1.1/), [Hager et al. dataset code](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset) (MIT), [framework code](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework) (MIT).

- **Use v1.1 (8 Jul 2024).** It adds `pathology_ids.json`, which lists the hadm_ids per pathology and is the authoritative label source. v1.0 has no label file. → the loader reads it first ([configs/cdm_format.yaml](../configs/cdm_format.yaml)).
- **Files and columns** (as read by Hager's `ConvertPhysionet.py`):

  | File | Columns |
  |---|---|
  | `history_of_present_illness.csv` | `hpi` |
  | `physical_examination.csv` | `pe` |
  | `laboratory_tests.csv` | `itemid`, `valuestr`, `ref_range_lower`, `ref_range_upper`, `charttime` |
  | `microbiology.csv` | `test_itemid`, `valuestr`, `spec_itemid` |
  | `radiology_reports.csv` | `note_id`, `modality`, `region`, `exam_name`, `text` |
  | `discharge_diagnosis.csv` | `discharge_diagnosis` |
  | `icd_diagnosis.csv` | `icd_diagnosis` |
  | `lab_test_mapping.csv` | `itemid`, `label`, `fluid`, `category`, `count`, `corresponding_ids` (a list literal) |

- **Class sizes:** 957 / 648 / 257 / 538 (appendicitis / cholecystitis / diverticulitis / pancreatitis), 2,400 in total.
- **No official train/val/test split exists.** → see §3.
- **Only the first value of each lab test** per admission is kept, and radiology keeps findings only.
- **How the dataset was built** (from `CreateDataset.py`, `dataset/discharge.py`, `dataset/radiology.py`, `dataset/dataset.py`):
  - The history is the discharge-summary text from "History of Present Illness:" up to the exam header, **with newlines flattened**. PMH, social and family history are therefore inside one blob; they are not separate fields.
  - The exam runs from the PE header to "Pertinent Results:", and everything from the first "discharge" onwards is cut.
  - Radiology drops sections whose headers start with IMPRESSION, HISTORY, INDICATION, COMPARISON, CONCLUSION, REASON and similar.
  - **Sanitisation.** If the history mentions the target disease, the admission is discarded. In the exam and radiology, each mention is replaced by `____` (four underscores; MIMIC's de-identification uses three). The term lists per pathology are `["acute appendicitis","appendicitis","appendectomy"]`, `["acute cholecystitis","cholecystitis","cholecystostomy"]`, `["acute pancreatitis","pancreatitis","pancreatectomy"]` and `["acute diverticulitis","diverticulitis"]`.
  - → [src/deferdx/data/parity.py](../src/deferdx/data/parity.py) applies the same rules to OTHER cases. Without that, the `____` marker and the missing IMPRESSION sections would let an agent recognise CDM cases by format alone.
  - → The history is shown in full at reset (`history_at_reset`), and ASK defaults to the physical exam.
- **Vocabulary:**
  - Modalities: CT, Ultrasound, Radiograph, MRI, Fluoroscopy, plus special cases (HIDA, MRCP, CTU, EUS, ERCP, …).
  - Regions: Abdomen, Chest, Head, … There is no Pelvis region.
  - Lab panels are itemid lists in the framework's `ADDITIONAL_LAB_TEST_MAPPING`.
  - → The catalog matches labs by itemid, using those lists.

## 2. LA-CDM (ICLR 2026)

Sources: [arXiv 2506.13474](https://arxiv.org/abs/2506.13474), [code](https://github.com/dharouni/LA-CDM).

- **Split:** 80/10/10, stratified, `sklearn.train_test_split(random_state=269)`, over the four per-pathology pickles concatenated as appendicitis, cholecystitis, diverticulitis, pancreatitis. → reproduced exactly (`--split lacdm`, verified against their code in tests).
- **Action space:** 12 tests — Physical Examination; CT, MRI, Radiograph and Ultrasound (by modality only); and the CBC, BMP, CMP, Renal, Liver, Urinalysis and Electrolyte panels. Unavailable tests return "not available" and are **not charged**.
- **Test costs:** 2025 BIDMC standard charges, from LA-CDM's Table 4. For example CT is $1,306, MRI $4,866 and CBC $71. Costs are normalised so that ordering every test costs one correct diagnosis. → `cost_source: bidmc` entries and `cost_scale: auto`.
- **Calibration reward:** a log scoring rule over confidence on a 0–10 scale, rescaled to [−1, 1] (the Rewarding Doubt approach). → `reward.scoring_rule: log`.
- **Backbone and preprocessing:** Qwen2.5-7B-Instruct with LoRA, Adam, lr 1e-5, batch size 2, about 3 days on one A40. The HPI was summarised by an LLM (Mixtral-8x7B in the paper, Qwen2.5-7B in the README).
- **"Mean" is the unweighted mean of per-class accuracy.** 81.3 = (93.1 + 83.6 + 75.0 + 73.5) / 4.

## 3. LDTL (arXiv 2604.05116)

Source: [arXiv 2604.05116](https://arxiv.org/abs/2604.05116).

- The paper claims an "official patient-level split, 70/10/20". **No official split exists** (§1), and LDTL's assignment is unpublished.
- **The LA-CDM and ReAct rows in LDTL's Table 1 are copied from LA-CDM's paper**, which used an 80/10/10 split and the per-class mean. LDTL's own rows don't match a per-class mean: LDTL's 93.4 vs (98.9 + 95.4 + 78.8 + 87.9) / 4 = 90.25, and random's 84.8 vs 85.95. Those rows look case-weighted.
- **So "93.4 vs 81.3" compares different splits and different metrics.** The diverticulitis comparison in the plan (LDTL 78.8 vs random 90.4) is internally consistent, because both rows are LDTL's own. → `evaluate` reports both `accuracy_full_coverage` (case-weighted) and `mean_class_accuracy`, and the default split is LA-CDM's, so every LA-CDM number can be compared like-for-like.
- **Action space:** 3 coarse categories (physical exam, labs, imaging), at most 3 steps, a planner/diagnoser pair, Llama-3-8B with LoRA r=16/α=32 and lr 2e-5, about 6 h on 2×H100.

## 4. Qwen3 (base model)

Sources: the Qwen3-8B model card, and the Qwen3-0.6B tokenizer inspected locally.

- **The chat template drops `<think>` blocks of earlier assistant turns.** EOS is `<|im_end|>`. → RL and SFT samples are built per turn (confirmed on the real tokenizer).
- **Thinking mode:** the recommended sampling is T=0.6, top-p 0.95, top-k 20, and the card says **"DO NOT use greedy decoding"**. → these are now the CLI and GRPO evaluation defaults.
- **Context:** 32,768 tokens natively.

## 5. Real-model loop check (Qwen3-0.6B, CPU, synthetic cases)

`scripts/check_model.py`, run twice:

| | Run 1: 4 cases, 512-token budget, strict parser | Run 2: 2 cases, 1,024-token budget, lenient parser, brevity prompt |
|---|---|---|
| Valid actions | 4/14 (29%) | 4/4 |
| Truncated turns | shown `example_turn` cut off mid-`<action`; truncation not counted | 0/4 |
| `<think>` survives decoding | 13/14 | 4/4 |
| SFT prompt token-identical to generation prompt | 14/14 | 4/4 |
| Mean completion length | 344 tokens | 432 tokens |

In run 2, one of the 4 valid actions was untagged JSON after `</think>`, which only the new fallback parses. Changes made:
- A 1,024-token generation default.
- A "think briefly" instruction in the system prompt (LA-CDM's prompt also demands brevity).
- A more tolerant JSON extractor.
- A truncation counter.

These are small samples on a 0.6B model. They validate the plumbing, not the model's accuracy.

## 6. Consequences for the abstract

- Quote LA-CDM numbers only against results on the LA-CDM split, and say which "mean" is used.
- Quote LDTL numbers as "reported". Their split can't be reproduced.
- The open-world contribution depends on the §1 parity rules. Report the source-classifier check on `--controls`.

## 7. Literature re-check (2026-10-06), and what it changes

The plan was compiled on 2026-09-19. A search on 2026-10-06 (alphaXiv: deferral/abstention × clinical agents × RL) found the following newer or closely related papers. Each was read for its action space, training and reward.

| Paper | What it does | Relation to DEFER-Dx |
|---|---|---|
| Che et al., *Abstention as an Action Can Kill Both the Reward Gradient and the KL Anchor* (arXiv 2608.00301, Jul 2026) | Proves that with an error-penalised reward and a KL-anchored learner, a discrete abstain action drifts to refusing everything (mean reward rises like 1/t while coverage collapses). In the sparse-answer regime, GRPO's group std normalisation replaces the designed penalty with an effective penalty of 1. Recommended repair: train a mandatory confidence report with a strictly proper score and abstain only at deployment by thresholding it. | **The sharpest prior argument against our design, and our control arm is exactly their repair.** DEFER-Dx removes their collapse conditions: (1) deferral is not a constant-0 action; the group-consensus reward makes it negative on cases the group gets right, so blanket deferral is not optimal; (2) advantages are not std-normalised (Dr. GRPO); (3) there is no KL anchor, and an explicit Lagrangian coverage constraint; (4) every COMMIT still carries a proper-scored probability. The DEFER-Dx-vs-thresholded-control comparison is therefore a direct test of their prediction in a multi-turn clinical agent. Report it as such, whichever way it comes out. |
| Zhan et al., *TrustMed-RL* (arXiv 2610.04387, Oct 3 2026) | 8B VLM agent on PubMed rare-disease cases, trained with GiGPO; actions include deferral. Deferral is rewarded only on instances designated by construction (synthetic evidence-corruption probes, an "out-of-knowledge" task category). | **An RL-trained clinical agent with a deferral action now exists, so the "first" claim must be narrowed.** Remaining differences: DEFER-Dx needs no deferral labels (the target comes from the policy's own group consensus on real cases); commits carry calibrated probabilities under a proper scoring rule; real EHR data with an out-of-label-set (open-world) evaluation; head-to-head against post-hoc thresholding. |
| Yang & Wu, *Decide, Ask, or Defer* (arXiv 2610.04542, Oct 3 2026) | DECIDE/ASK/DEFER formulation; evaluation only (Qwen, Gemini, GPT prompted on 200 DDXPlus evidence states). | Not trained. Supports the motivation: separating ASK from DEFER exposes behaviour that binary abstention hides. |
| Peng et al., *CDPR* (arXiv 2608.28599) | Counterfactual process reward for cost-aware sequential diagnosis with GRPO, on DiagGym's MIMIC-IV pipeline (free-text diagnoses). | Cost-aware RL, no deferral, not on MIMIC-CDM; related work only. |
| *Safe to Stop?* (arXiv 2609.09678, Sep 2026) | Risk-constrained (conformal) stopping for sequential diagnosis agents. | Already in the plan: a non-RL abstention comparator, and the reason SGR is in the report. |

**Revised novelty claim (use this wording):** DEFER-Dx is, to our knowledge, the first interactive diagnostic agent trained with RL in which deferral to a clinician is learned from a *label-free* signal (the policy's own group consensus) rather than from cases designated in advance, jointly with calibrated (proper-scored) commitment probabilities and per-test costs. It is evaluated on real EHR admissions including out-of-label-set and never-seen time-critical diagnoses, and compared head-to-head with the threshold-a-calibrated-report approach that theory currently recommends.

## 8. Second re-check (2026-10-07): the group-consensus mechanism has a precedent

Searches on the core mechanism (abstention or deferral rewarded from a model's own parallel rollouts) and on new MIMIC-CDM results:

| Paper | Finding | Consequence |
|---|---|---|
| Gao et al., **KARL** (arXiv 2604.22779, Apr 2026) | GRPO for QA abstention with a "knowledge-boundary-aware reward" estimated online from **within-group response statistics**. If the group contains a correct answer, abstaining scores −1 (like a wrong answer); if not, +1. Two-stage training avoids an "abstention trap" caused by group-normalised advantages. | **The idea of rewarding abstention from within-group statistics is not ours. Cite KARL as the closest mechanism precedent.** DEFER-Dx's reward is continuous, γ(τ − p̂), with a clinical safety dial τ and an analytic crossover; KARL's binary rule is roughly the degenerate case (defer only when p̂ = 0). DEFER-Dx is also multi-turn with costed tests, scores every commitment with a proper scoring rule, and is clinical with an open-world test. KARL's "abstention trap" under group normalisation agrees with Che et al. and supports the Dr. GRPO choice. |
| Wu & Rus, **Safe to Stop?** (arXiv 2609.09678, Sep 2026) | A post-hoc, conformal (LTT) stopping layer for sequential diagnosis agents on a MIMIC-derived abdominal-pain benchmark (1,834 episodes). Reports 16.9% selective error at 78.8% coverage (exploratory, on a split viewed during development). | A non-RL deferral comparator on a different benchmark construction: context only. SGR is our conformal-style baseline. |
| Search for new MIMIC-CDM SOTA | Nothing newer than LDTL (arXiv 2604.05116) on the same task was found. | LDTL's 93.4 (case-weighted, own split) remains the reported SOTA at full coverage. |

**Claim wording (supersedes §7).** "To our knowledge, DEFER-Dx is the first interactive, multi-turn, cost-aware diagnostic agent whose deferral to a clinician is learned from a label-free, continuous group-consensus reward with an explicit clinical operating point. The reward generalises within-group abstention rewards for single-turn QA (KARL), and is learned jointly with proper-scored commitment probabilities. It is evaluated on real EHR admissions with an open-world test that includes never-seen, time-critical diagnoses, and compared head-to-head with thresholding a calibrated confidence report, the repair that recent theory recommends (Che et al.)." Do not claim "first RL-trained deferral" (TrustMed-RL) or "first group-statistics abstention reward" (KARL).

## 9. Third re-check (2026-10-07): the case-level consensus reward is prior art; pivot to a counterfactual escalation value

Targeted searches on abstention rewards from within-group statistics turned up two more exact or near matches:

| Paper | Mechanism | Overlap with the original DEFER-Dx reward γ(τ − p̂) |
|---|---|---|
| Pan et al., **TIAR** (arXiv 2605.25850, May 2026) | p̂ = n_correct / (n_correct + n_wrong) among a GRPO group's non-abstaining trajectories; abstention advantage += λ(1 − 2p̂) | **Identical** for τ = 0.5 (and γ = 2, λ = 1); single-turn QA |
| Zhang et al., **AWA-RL** (arXiv 2607.10738, Jul 2026) | Multi-turn search agents; refusal reward 1 − p_i^γ, with p_i the query's success rate estimated from an earlier checkpoint, plus a batch-level refusal-rate penalty | Same idea (case-level competence sets the abstention reward) in a **multi-turn agent**, with a rate penalty like our coverage dual |
| KARL (§8), TrustMed-RL (§7) | within-group statistics (binary); designated deferral cases | — |

**Conclusion:** rewarding deferral by the policy's estimated case-level success rate is published (TIAR, KARL, AWA-RL). The original reward stays only as a comparator arm (`configs/grpo_deferdx.yaml`).

**New mechanism: the counterfactual escalation value (CEV; docs/METHODS.md §2).**
- For each DEFER, branch K forced continuations from the exact pre-deferral state, with deferral disabled and further tests allowed.
- Estimate the clinical value of not escalating there, V̂(s), as the mean of accuracy, proper score, severity and post-branch test cost.
- Credit the DEFER turn with the state-level counterfactual advantage E − V̂(s).
- E includes a strictly proper score of the handed-over probabilistic differential.

What the searches on 2026-10-07 found nearby, and the differences:
- **Generic tree or branched rollouts for step-level credit** (VinePPO, arXiv 2410.01679, Monte Carlo values from reset intermediate states; TreeRL / Tree-GRPO, SIPO, belief-shift branching, Counterfactual Rollout Replay, ASCT, CIPO): none targets an escalation or abstention action, and none values it against continuation in clinical-value units.
- **CDPR** (2608.28599): counterfactual short rollouts score the chosen investigative action against its alternatives. It has no deferral action and no handoff.
- **Signed Rescue Routing** (2609.07786): escalation value in model cascades, from a second model's correctness. Inference-time routing, not RL training of an agent's escalation.
- **Decide/Ask/Defer** (2610.04542): separates asking from deferring conceptually, in evaluation only.
- **No paper found** rewards the content of a deferral handoff (a probabilistic differential under a proper scoring rule).

**Claim wording (supersedes §7 and §8).** "To our knowledge, DEFER-Dx is the first agent whose escalation to a clinician is trained against a counterfactual of not escalating from the same decision state. Forced continuations branched at each escalation estimate the clinical value of deciding now or investigating further, so the agent learns to separate uncertainty that more testing would resolve from uncertainty that warrants a handoff. The handoff itself is a probabilistic differential trained with a strictly proper scoring rule." Always cite TIAR, KARL, AWA-RL and TrustMed-RL for case-level and designated-case deferral, and the tree-RL work for branching. Novelty is checked against alphaXiv searches and cannot be proven absolute; re-check within 72 hours of submission.
