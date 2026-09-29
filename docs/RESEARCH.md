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
