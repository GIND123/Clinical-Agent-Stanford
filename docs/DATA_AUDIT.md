# Data audit

Generated 2026-10-04 by `scripts/audit_data.py` from the local PhysioNet files. Re-run it after any data change: `python scripts/audit_data.py --root data/physionet --out docs`.

**What this file contains.** Aggregates only: counts, percentages, quantiles and test statistics. It holds no rows, identifiers, dates or note text. Patient-derived counts from 1 to 9 are shown as `<10`, and `·` marks a second cell hidden so the first cannot be recovered from a total. The underlying data stays under the PhysioNet Credentialed Health DUA and is never committed (`.gitignore` excludes `data/`).

## Findings

- `configs/cdm_format.yaml` maps `history` columns that `history_of_present_illness.csv` does not have: subject_id -> subject_id.
- `configs/cdm_format.yaml` maps `labs` columns that `laboratory_tests.csv` does not have: charttime -> charttime.
- `configs/cdm_format.yaml` maps `microbiology` columns that `microbiology.csv` does not have: charttime -> charttime.
- Class imbalance: appendicitis 957 vs diverticulitis 257 cases (3.7x). Report per-class accuracy, not only case-weighted accuracy.
- Fallback label source 'CDM icd_diagnosis' agrees with pathology_ids.json on only 53% of pancreatitis cases (the rest are ambiguous or unmapped); CDM v1.0, which lacks pathology_ids.json, would lose or mislabel these.
- Imaging dedup drops 1,337 radiology reports (412 cases). Without `charttime` the 'earliest' report is simply the first row in the file.
- The `____` mask in radiology text ranges 27–51% across classes; mask presence is itself a label cue (chi-square p <0.001).
- Radiology names other CDM conditions in up to 15% of a class's cases (mostly negations). This is a legitimate but strong exclusion cue; the open-world cases must keep the same kind of mentions, or the agent learns the source instead of the disease.
- Representation vs. ED admissions, age (CDM vs ED): 18-29 18.2% vs 9.7% (SMD +0.25); 30-44 22.2% vs 14.2% (SMD +0.21); 65-79 17.4% vs 23.2% (SMD -0.14); 80+ 9.6% vs 18.3% (SMD -0.25).
- Representation vs. ED admissions, race (CDM vs ED): Asian 5.8% vs 3.2% (SMD +0.12); Black 12.8% vs 18.2% (SMD -0.15); Other 5.7% vs 3.4% (SMD +0.11).
- Representation vs. ED admissions, insurance (CDM vs ED): Medicaid 7.3% vs 11.3% (SMD -0.14); Medicare 21.1% vs 39.2% (SMD -0.40); Other 71.6% vs 49.5% (SMD +0.46).
- Representation vs. ED admissions, marital status (CDM vs ED): MARRIED 43.8% vs 37.3% (SMD +0.13); WIDOWED 6.4% vs 12.1% (SMD -0.20).
- Label depends on age (Cramér's V 0.268, p <0.001).
- Label depends on insurance (Cramér's V 0.23, p <0.001).
- Label depends on marital status (Cramér's V 0.146, p <0.001).
- Demographics alone predict the label with macro-AUROC 0.68 (appendicitis 0.78). An agent can score above chance before ordering any test; compare against this floor.
- CT availability (label-adjusted) differs by age: 65-79 78% vs 18-29 61%.
- Ultrasound availability (label-adjusted) differs by age: 18-29 56% vs 80+ 44%.
- microbiology availability (label-adjusted) differs by age: 80+ 93% vs 18-29 69%.
- HPI length differs by age even within class: 80+ 1.15× vs 18-29 0.88× the class median; the agent sees less history for some groups.
- CT availability (label-adjusted) differs by race: Asian 77% vs Hispanic/Latino 64%.
- Ultrasound availability (label-adjusted) differs by race: Hispanic/Latino 60% vs Asian 46%.
- microbiology availability (label-adjusted) differs by insurance: Medicare 85% vs Medicaid 74%.
- HPI length differs by insurance even within class: Medicare 1.16× vs Medicaid 0.95× the class median; the agent sees less history for some groups.
- Ultrasound availability (label-adjusted) differs by marital status: SINGLE 55% vs WIDOWED 45%.
- microbiology availability (label-adjusted) differs by marital status: WIDOWED 88% vs SINGLE 74%.
- Patient-level leakage in the LA-CDM split: <10 patients have admissions in more than one split. `deferdx data build-cdm` reports 0 because CDM's CSVs carry no `subject_id` (see §2).
- Only 89% of diverticulitis cases have it as the primary (seq 1) ICD diagnosis.
- Only 74% of pancreatitis cases have it as the primary (seq 1) ICD diagnosis.
- Open-world groups with fewer than 100 abdominal-complaint candidates (they cap the stratified sample): ectopic_pregnancy.
- OTHER cases differ demographically from CDM: demographics alone separate them with AUROC 0.63. Match or report this, or DEFER/OTHER can be learned from who the patient is.

## 1. Inventory and provenance

| Dataset | Version | Access | Source |
|---|---|---|---|
| MIMIC-IV-Ext-CDM | 1.1 (2024-07-08) | Credentialed, PhysioNet Credentialed Health DUA 1.5.0 | https://physionet.org/content/mimic-iv-ext-cdm/1.1/ |
| MIMIC-IV (hosp module, 7 tables) | 2.2 | Credentialed, DUA 1.5.0 | https://physionet.org/content/mimiciv/2.2/ |
| MIMIC-IV-Note (discharge, radiology) | 2.2 | Credentialed, DUA 1.5.0 | https://physionet.org/content/mimic-iv-note/2.2/ |

Every file is checked against the SHA256SUMS.txt PhysioNet ships with it. Files were fetched with `aws s3 cp` from PhysioNet's S3 access points (`mimiciv-v2-2-01`, `mimic-iv-note-v2-2-01`) and, for CDM, PhysioNet's ZIP. Version 2.2 is pinned because CDM was built from MIMIC-IV 2.2.

| file | dataset | MB | sha256 |
|---|---|---|---|
| LICENSE.txt | mimic-iv-ext-cdm/1.1 | 0.0 | ok |
| discharge_diagnosis.csv | mimic-iv-ext-cdm/1.1 | 0.1 | ok |
| discharge_procedures.csv | mimic-iv-ext-cdm/1.1 | 0.1 | ok |
| history_of_present_illness.csv | mimic-iv-ext-cdm/1.1 | 2.6 | ok |
| icd_diagnosis.csv | mimic-iv-ext-cdm/1.1 | 0.9 | ok |
| icd_procedures.csv | mimic-iv-ext-cdm/1.1 | 0.2 | ok |
| lab_test_mapping.csv | mimic-iv-ext-cdm/1.1 | 0.1 | ok |
| laboratory_tests.csv | mimic-iv-ext-cdm/1.1 | 5.5 | ok |
| microbiology.csv | mimic-iv-ext-cdm/1.1 | 0.3 | ok |
| pathology_ids.json | mimic-iv-ext-cdm/1.1 | 0.0 | ok |
| physical_examination.csv | mimic-iv-ext-cdm/1.1 | 1.0 | ok |
| radiology_reports.csv | mimic-iv-ext-cdm/1.1 | 7.2 | ok |
| LICENSE.txt | mimic-iv-note/2.2 | 0.0 | ok |
| note/discharge.csv.gz | mimic-iv-note/2.2 | 1,139.2 | ok |
| note/radiology.csv.gz | mimic-iv-note/2.2 | 781.8 | ok |
| note/radiology_detail.csv.gz | mimic-iv-note/2.2 | 39.0 | ok |
| wanted.sha256 | mimic-iv-note/2.2 | 0.0 | not listed |
| hosp/admissions.csv.gz | mimiciv/2.2 | 15.5 | ok |
| hosp/d_icd_diagnoses.csv.gz | mimiciv/2.2 | 0.9 | ok |
| hosp/d_labitems.csv.gz | mimiciv/2.2 | 0.0 | ok |
| hosp/diagnoses_icd.csv.gz | mimiciv/2.2 | 25.1 | ok |
| hosp/labevents.csv.gz | mimiciv/2.2 | 1,939.1 | ok |
| hosp/microbiologyevents.csv.gz | mimiciv/2.2 | 96.7 | ok |
| hosp/patients.csv.gz | mimiciv/2.2 | 2.3 | ok |
| hosp/transfers.csv.gz | mimiciv/2.2 | 36.2 | ok |
| wanted.sha256 | mimiciv/2.2 | 0.0 | not listed |


## 2. MIMIC-IV-Ext-CDM: schema

**discharge_diagnosis.csv**: 2,400 rows; 2,400 admissions, rows per admission p50/p95 = 1/1; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| discharge_diagnosis | str | 0.0 | free text (length only) | chars p5/p50/p95 = 12/22/144 |

**discharge_procedures.csv**: 2,122 rows; 1,810 admissions, rows per admission p50/p95 = 1/2; covers 75.4% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 1,810 distinct |
| discharge_procedure | str | 0.0 | free text (length only) | chars p5/p50/p95 = 9/28/66 |

**history_of_present_illness.csv**: 2,400 rows; 2,400 admissions, rows per admission p50/p95 = 1/1; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| hpi | str | 0.0 | free text (length only) | chars p5/p50/p95 = 365/832/2,586 |

**icd_diagnosis.csv**: 17,357 rows; 2,400 admissions, rows per admission p50/p95 = 5/20; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| icd_diagnosis | str | 0.0 | category | 2,352 distinct; top: Unspecified essential hypertension (507), Acute appendicitis without mention of peritonitis (488), Acute pancreatitis (415), Calculus of gallbladder with acute cholecystitis, without mention of obstruction (295), Other and unspecified hyperlipidemia (291), Esophageal reflux (273), Essential (primary) hypertension (214), Unspecified acute appendicitis (207) |

**icd_procedures.csv**: 2,917 rows; 1,660 admissions, rows per admission p50/p95 = 1/5; covers 69.2% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 1,660 distinct |
| icd_code | str | 0.0 | identifier | 395 distinct |
| icd_title | str | 0.0 | category | 395 distinct; top: Laparoscopic appendectomy (520), Laparoscopic cholecystectomy (343), Endoscopic sphincterotomy and papillotomy (145), Resection of Gallbladder, Percutaneous Endoscopic Approach (137), Endoscopic removal of stone(s) from biliary tract (75), Percutaneous aspiration of gallbladder (73), Venous catheterization, not elsewhere classified (65), Resection of Appendix, Percutaneous Endoscopic Approach (60) |
| icd_version | int64 | 0.0 | category | 2 distinct; top: 9 (2,112), 10 (805) |

**lab_test_mapping.csv**: 1,209 rows

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| itemid | float64 | 9.8 | identifier | 1,091 distinct |
| label | str | 0.0 | other | 983 distinct |
| fluid | str | 22.2 | category | 9 distinct; top: Blood (493), Urine (137), Other Body Fluid (107), Bone Marrow (44), Ascites (40), Pleural (38), Joint Fluid (34), Cerebrospinal Fluid (33) |
| category | str | 24.3 | category | 3 distinct; top: Hematology (462), Chemistry (416), Blood Gas (37) |
| count | float64 | 24.3 | other | 699 distinct |
| corresponding_ids | str | 0.0 | other | 889 distinct |

**laboratory_tests.csv**: 138,788 rows; 2,400 admissions, rows per admission p50/p95 = 56/87; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| itemid | int64 | 0.0 | identifier | 480 distinct |
| valuestr | str | 0.0 | result value | 4.0% parse as numbers |
| ref_range_lower | float64 | 29.0 | other | 98 distinct |
| ref_range_upper | float64 | 29.0 | other | 140 distinct |

**microbiology.csv**: 4,403 rows; 1,856 admissions, rows per admission p50/p95 = 2/7; covers 77.3% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 1,856 distinct |
| test_itemid | int64 | 0.0 | identifier | 74 distinct |
| valuestr | str | 0.0 | result value | 0.0% parse as numbers |
| spec_itemid | int64 | 0.0 | identifier | 40 distinct |

**physical_examination.csv**: 2,400 rows; 2,400 admissions, rows per admission p50/p95 = 1/1; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| pe | str | 0.0 | free text (length only) | chars p5/p50/p95 = 147/347/832 |

**radiology_reports.csv**: 5,960 rows; 2,400 admissions, rows per admission p50/p95 = 2/6; covers 100.0% of labelled cases; admissions without a label: 0

| column | dtype | null_pct | kind | summary |
|---|---|---|---|---|
| hadm_id | int64 | 0.0 | identifier | 2,400 distinct |
| note_id | str | 0.0 | identifier | 5,960 distinct |
| modality | str | 0.0 | category | 14 distinct; top: Radiograph (2,076), CT (2,064), Ultrasound (1,380), MRCP (227), ERCP (98), MRI (67), CTU (21), Drainage (19) |
| region | str | 0.0 | category | 11 distinct; top: Abdomen (3,920), Chest (1,870), Head (87), Venous (45), Spine (16) |
| exam_name | str | 0.0 | category | 129 distinct; top: CT ABD & PELVIS WITH CONTRAST (1,221), LIVER OR GALLBLADDER US (SINGLE ORGAN) (925), CHEST (PORTABLE AP) (907), CHEST (PA & LAT) (526), CT ABDOMEN W/CONTRAST (296), MRCP (MR ABD W&W/OC) (213), CHEST PORT. LINE PLACEMENT (178), ABDOMEN (SUPINE & ERECT) (155) |
| text | str | 0.0 | free text (length only) | chars p5/p50/p95 = 67/769/2,952 |

**pathology_ids.json**: 4 keys (appendicitis 957, cholecystitis 648, pancreatitis 538, diverticulitis 257); ids listed under two classes: 0; duplicate ids within a class: 0.


### Config vs. files (configs/cdm_format.yaml)

| table | file | missing |
|---|---|---|
| history | history_of_present_illness.csv | subject_id -> subject_id |
| labs | laboratory_tests.csv | charttime -> charttime |
| microbiology | microbiology.csv | charttime -> charttime |


## 3. MIMIC-IV-Ext-CDM: labels

| label | cases | share |
|---|---|---|
| appendicitis | 957 | 39.9 |
| cholecystitis | 648 | 27.0 |
| diverticulitis | 257 | 10.7 |
| pancreatitis | 538 | 22.4 |

How often a label re-derived from CDM's own diagnosis fields matches `pathology_ids.json` (the loader's fallback label sources, `label_source` in the config):

| source: class | agrees | names another condition | none or ambiguous |
|---|---|---|---|
| CDM icd_diagnosis: appendicitis | 98.9% | 0.0% | 1.1% |
| CDM icd_diagnosis: cholecystitis | 95.7% | 0.0% | 4.3% |
| CDM icd_diagnosis: diverticulitis | all but <10 | 0.0% | <10 cases |
| CDM icd_diagnosis: pancreatitis | 52.6% | 0.0% | 47.4% |
| CDM discharge_diagnosis text: appendicitis | all but <10 | 0.0% | <10 cases |
| CDM discharge_diagnosis text: cholecystitis | 98.3% | 0.0% | 1.7% |
| CDM discharge_diagnosis text: diverticulitis | all but <10 | 0.0% | <10 cases |
| CDM discharge_diagnosis text: pancreatitis | 93.7% | 0.0% | 6.3% |


## 4. MIMIC-IV-Ext-CDM: what the environment can reveal, by class

| label | n | physical exam | CT | Ultrasound | Radiograph | MRI | any imaging | microbiology | lab items (median) | HPI chars (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| appendicitis | 957 | 100.0% | 93.8% | 23.5% | 18.1% | 2.1% | 100.0% | 76.8% | 52.0 | 603.0 |
| cholecystitis | 648 | 100.0% | 42.0% | 87.2% | 49.7% | 1.5% | 100.0% | 77.5% | 58.0 | 884.0 |
| diverticulitis | 257 | 100.0% | 91.4% | 19.1% | 49.8% | <10 cases | 100.0% | 88.7% | 59.0 | 1,060.0 |
| pancreatitis | 538 | 100.0% | 51.1% | 72.9% | 59.1% | 4.3% | 100.0% | 72.7% | 60.0 | 1,540.0 |

Missing tests are informative: a test that exists for one class much more often than another lets the agent learn from *whether* a result exists, not only from what it says.

`dedup_earliest` keeps one report per (modality, region). CDM has no `charttime`, so it keeps the first in file order: **1,337 reports across 412 cases are never shown to the agent.**


## 5. MIMIC-IV-Ext-CDM: label leakage checks

CDM replaces each mention of the case's own diagnosis with `____`. If the mask is much more common in one class, its presence alone predicts the label.

| label | n | physical exam contains ____ | radiology contains ____ |
|---|---|---|---|
| appendicitis | 957 | <10 cases | 42.1% |
| cholecystitis | 648 | <10 cases | 26.9% |
| diverticulitis | 257 | <10 cases | 51.0% |
| pancreatitis | 538 | <10 cases | 33.5% |

Unmasked mentions of the case's **own** diagnosis (CDM's pipeline should leave none):

| label | HPI | physical exam | radiology |
|---|---|---|---|
| appendicitis | 0 | 0 | 0 |
| cholecystitis | 0 | 0 | 0 |
| diverticulitis | 0 | 0 | 0 |
| pancreatitis | 0 | 0 | 0 |

Radiology reports that name **other** CDM conditions (rows: true label). CDM masks only the case's own diagnosis, so 'no evidence of appendicitis' survives in a cholecystitis case and hints by exclusion:

| label | mentions appendicitis | mentions cholecystitis | mentions diverticulitis | mentions pancreatitis |
|---|---|---|---|---|
| appendicitis | 0.0% | 2.5% | 5.9% | <10 cases |
| cholecystitis | 3.9% | 0.0% | 11.0% | 2.3% |
| diverticulitis | 5.4% | 5.4% | 0.0% | <10 cases |
| pancreatitis | 5.0% | 15.1% | 8.0% | 0.0% |

Procedure tables (`discharge_procedures.csv`, `icd_procedures.csv`) name the treatment, and so the answer. No code under `src/` or `configs/` reads them (checked with grep); keep it that way:

| label | appendectomy | cholecystectomy | colectomy | ERCP |
|---|---|---|---|---|
| appendicitis | 83.9% | <10 cases | <10 cases | 0.0% |
| cholecystitis | <10 cases | 71.1% | 0.0% | 11.1% |
| diverticulitis | <10 cases | 0.0% | 10.5% | <10 cases |
| pancreatitis | 0.0% | 21.0% | 0.0% | 30.5% |


## 6. MIMIC-IV 2.2 and MIMIC-IV-Note 2.2: schema

Types are DuckDB's inference from a sample; counts are exact except distinct ids (HyperLogLog, ~2%). "CDM admissions present" is how many of the 2,400 CDM admissions appear in the table.

**hosp/admissions.csv.gz**: 431,231 rows; ~384,578 admissions; CDM admissions present: 2,400/2,400; ~149,377 patients

| column | type | null_pct |
|---|---|---|
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 0.0 |
| admittime | TIMESTAMP | 0.0 |
| dischtime | TIMESTAMP | 0.0 |
| deathtime | TIMESTAMP | 98.0 |
| admission_type | VARCHAR | 0.0 |
| admit_provider_id | VARCHAR | 0.0 |
| admission_location | VARCHAR | 0.0 |
| discharge_location | VARCHAR | 27.6 |
| insurance | VARCHAR | 0.0 |
| language | VARCHAR | 0.0 |
| marital_status | VARCHAR | 2.1 |
| race | VARCHAR | 0.0 |
| edregtime | TIMESTAMP | 30.6 |
| edouttime | TIMESTAMP | 30.6 |
| hospital_expire_flag | BIGINT | 0.0 |

**hosp/patients.csv.gz**: 299,712 rows; ~266,612 patients

| column | type | null_pct |
|---|---|---|
| subject_id | BIGINT | 0.0 |
| gender | VARCHAR | 0.0 |
| anchor_age | BIGINT | 0.0 |
| anchor_year | BIGINT | 0.0 |
| anchor_year_group | VARCHAR | 0.0 |
| dod | DATE | 90.3 |

**hosp/diagnoses_icd.csv.gz**: 4,756,326 rows; ~384,578 admissions; CDM admissions present: 2,400/2,400; ~149,377 patients

| column | type | null_pct |
|---|---|---|
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 0.0 |
| seq_num | BIGINT | 0.0 |
| icd_code | VARCHAR | 0.0 |
| icd_version | BIGINT | 0.0 |

**hosp/d_icd_diagnoses.csv.gz**: 109,775 rows

| column | type | null_pct |
|---|---|---|
| icd_code | VARCHAR | 0.0 |
| icd_version | BIGINT | 0.0 |
| long_title | VARCHAR | 0.0 |

**hosp/d_labitems.csv.gz**: 1,622 rows

| column | type | null_pct |
|---|---|---|
| itemid | BIGINT | 0.0 |
| label | VARCHAR | 0.2 |
| fluid | VARCHAR | 0.0 |
| category | VARCHAR | 0.0 |

**hosp/microbiologyevents.csv.gz**: 3,228,713 rows; ~184,766 admissions; CDM admissions present: 765/2,400; ~170,712 patients

| column | type | null_pct |
|---|---|---|
| microevent_id | BIGINT | 0.0 |
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 56.7 |
| micro_specimen_id | BIGINT | 0.0 |
| order_provider_id | VARCHAR | 70.4 |
| chartdate | TIMESTAMP | 0.0 |
| charttime | TIMESTAMP | 8.0 |
| spec_itemid | BIGINT | 0.0 |
| spec_type_desc | VARCHAR | 0.0 |
| test_seq | BIGINT | 0.0 |
| storedate | TIMESTAMP | 0.4 |
| storetime | TIMESTAMP | 0.8 |
| test_itemid | BIGINT | 0.0 |
| test_name | VARCHAR | 0.0 |
| org_itemid | BIGINT | 60.1 |
| org_name | VARCHAR | 60.1 |
| isolate_num | BIGINT | 60.1 |
| quantity | VARCHAR | 100.0 |
| ab_itemid | BIGINT | 65.7 |
| ab_name | VARCHAR | 65.7 |
| dilution_text | VARCHAR | 66.6 |
| dilution_comparison | VARCHAR | 66.6 |
| dilution_value | DOUBLE | 66.6 |
| interpretation | VARCHAR | 65.7 |
| comments | VARCHAR | 30.7 |

**hosp/labevents.csv.gz**: 118,171,367 rows; ~313,516 admissions; CDM admissions present: 1,948/2,400; ~227,731 patients

| column | type | null_pct |
|---|---|---|
| labevent_id | BIGINT | 0.0 |
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 48.7 |
| specimen_id | BIGINT | 0.0 |
| itemid | BIGINT | 0.0 |
| order_provider_id | VARCHAR | 71.4 |
| charttime | TIMESTAMP | 0.0 |
| storetime | TIMESTAMP | 1.7 |
| value | VARCHAR | 10.6 |
| valuenum | DOUBLE | 14.4 |
| valueuom | VARCHAR | 14.6 |
| ref_range_lower | DOUBLE | 19.5 |
| ref_range_upper | DOUBLE | 19.5 |
| flag | VARCHAR | 71.6 |
| priority | VARCHAR | 4.8 |
| comments | VARCHAR | 81.6 |

**note/discharge.csv.gz**: 331,793 rows; ~323,669 admissions; CDM admissions present: 2,400/2,400; ~111,716 patients; text chars p5/p50/p95 = 4,771/9,847/18,619

| column | type | null_pct |
|---|---|---|
| note_id | VARCHAR | 0.0 |
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 0.0 |
| note_type | VARCHAR | 0.0 |
| note_seq | BIGINT | 0.0 |
| charttime | TIMESTAMP | 0.0 |
| storetime | TIMESTAMP | 0.0 |
| text | VARCHAR | 0.0 |

**note/radiology.csv.gz**: 2,321,355 rows; ~315,303 admissions; CDM admissions present: 2,179/2,400; ~208,820 patients; text chars p5/p50/p95 = 313/781/3,280

| column | type | null_pct |
|---|---|---|
| note_id | VARCHAR | 0.0 |
| subject_id | BIGINT | 0.0 |
| hadm_id | BIGINT | 50.7 |
| note_type | VARCHAR | 0.0 |
| note_seq | BIGINT | 0.0 |
| charttime | TIMESTAMP | 0.0 |
| storetime | TIMESTAMP | 0.0 |
| text | VARCHAR | 0.0 |

**Can the open-world builder reproduce CDM's data?** Share of CDM's own 2,400 admissions with any data of each kind: as CDM holds it, as the original `build_openworld` found it (`hadm_id` only), and as it finds it now (`hadm_id` plus CDM's window; needs `hosp/transfers`).

| source | CDM's own tables | hadm_id only (old builder) | hadm_id + CDM window (builder now) |
|---|---|---|---|
| labs | 100.0% | 81.2% | 100.0% |
| microbiology | 77.3% | 31.9% | 77.3% |
| radiology | 100.0% | 90.8% | 100.0% |


## 7. Bias analysis

Group definitions: sex from `patients.gender`; age = `anchor_age` + (admission year − `anchor_year`); race collapsed from `admissions.race` (White incl. Portuguese; Hispanic/Latino incl. South American; Unknown = unknown, unable to obtain, declined); `language` in MIMIC-IV 2.2 is `ENGLISH` or `?`. Chi-square p-values are uncorrected; with this many tests, treat p < 0.001 as the bar.


### 7.1 Who is in the cohort

Admission route (`admission_location`) by class:

| admission location | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| AMBULATORY SURGERY TRANSFER | 0 | 0 | 0 | <10 |
| CLINIC REFERRAL | <10 | <10 | <10 | <10 |
| EMERGENCY ROOM | 775 | 507 | 198 | 435 |
| PACU | 18 | 0 | <10 | <10 |
| PHYSICIAN REFERRAL | 72 | 52 | 20 | 20 |
| PROCEDURE SITE | <10 | 0 | 0 | · |
| TRANSFER FROM HOSPITAL | <10 | 46 | · | 41 |
| TRANSFER FROM SKILLED NURSING FACILITY | <10 | <10 | 0 | <10 |
| WALK-IN/SELF REFERRAL | 72 | 34 | 19 | 19 |

| cohort | n | female | age median (IQR) | White | Black | Hispanic/Latino | Asian | Unknown | Medicaid | non-English/unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM (all) | 2,400 | 53.2% | 51 (33–66) | 65.5% | 12.8% | 8.8% | 5.8% | 1.4% | 7.3% | 10.9% |
| appendicitis | 957 | 48.8% | 36 (26–51) | 65.0% | 10.3% | 8.8% | 8.0% | <10 cases | 5.4% | 9.4% |
| cholecystitis | 648 | 55.7% | 59 (44–72) | 64.0% | 15.1% | 9.6% | 5.4% | 1.7% | 9.1% | 13.1% |
| diverticulitis | 257 | 58.8% | 60 (49–72) | 72.8% | 10.5% | 5.8% | 4.7% | <10 cases | 5.4% | 7.8% |
| pancreatitis | 538 | 55.2% | 59 (45–72) | 64.9% | 15.6% | 9.3% | 2.6% | 2.4% | 9.3% | 12.3% |
| Baseline: ED admissions | 232,595 | 50.8% | 60 (45–75) | 66.8% | 18.2% | 6.5% | 3.2% | 1.9% | 11.3% | 10.1% |
| Baseline: all admissions | 431,231 | 52.2% | 60 (45–74) | 67.2% | 16.2% | 6.0% | 3.5% | 3.3% | 9.6% | 9.9% |


### 7.2 Representation: CDM vs. MIMIC-IV ED admissions

SMD = standardised difference of proportions, CDM vs. ED admissions; |SMD| ≥ 0.1 is flagged.

**Sex**

| sex | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| Female | 1,276 | 53.2 | 50.8% | 52.2% | 0.0 |
| Male | 1,124 | 46.8 | 49.2% | 47.8% | -0.0 |

**Age**

| age | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| 18-29 | 438 | 18.2 | 9.7% | 9.0% | 0.2 |
| 30-44 | 534 | 22.2 | 14.2% | 15.4% | 0.2 |
| 45-64 | 780 | 32.5 | 34.5% | 33.9% | -0.0 |
| 65-79 | 418 | 17.4 | 23.2% | 25.5% | -0.1 |
| 80+ | 230 | 9.6 | 18.3% | 16.1% | -0.3 |

**Race**

| race | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| Asian | 138 | 5.8 | 3.2% | 3.5% | 0.1 |
| Black | 308 | 12.8 | 18.2% | 16.2% | -0.1 |
| Hispanic/Latino | 211 | 8.8 | 6.5% | 6.0% | 0.1 |
| Other | 136 | 5.7 | 3.4% | 3.9% | 0.1 |
| White | 1,573 | 65.5 | 66.8% | 67.2% | -0.0 |
| Unknown | 34 | 1.4 | 1.9% | 3.3% | -0.0 |

**Insurance**

| insurance | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| Medicaid | 175 | 7.3 | 11.3% | 9.6% | -0.1 |
| Medicare | 507 | 21.1 | 39.2% | 37.2% | -0.4 |
| Other | 1,718 | 71.6 | 49.5% | 53.2% | 0.5 |

**Language**

| language | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| English | 2,139 | 89.1 | 89.9% | 90.1% | -0.0 |
| Non-English/unknown | 261 | 10.9 | 10.1% | 9.9% | 0.0 |

**Marital status**

| marital status | CDM n | CDM % | ED admissions % | all admissions % | SMD vs ED |
|---|---|---|---|---|---|
| DIVORCED | 125 | 5.2 | 7.5% | 7.3% | -0.1 |
| MARRIED | 1,052 | 43.8 | 37.3% | 42.0% | 0.1 |
| SINGLE | 1,046 | 43.6 | 41.2% | 37.8% | 0.0 |
| WIDOWED | 153 | 6.4 | 12.1% | 10.6% | -0.2 |
| Unknown | 24 | 1.0 | 1.9% | 2.1% | -0.1 |


### 7.3 Does the label depend on demographics?

Row percentages: the class mix within each group. A strong association is not a bug (cholecystitis is more common in women), but it is a shortcut the agent can take instead of reasoning from findings.

**Sex**: chi-square p 0.004, Cramér's V 0.074

| sex | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Female | 467 (36.6%) | 361 (28.3%) | 151 (11.8%) | 297 (23.3%) |
| Male | 490 (43.6%) | 287 (25.5%) | 106 (9.4%) | 241 (21.4%) |

**Age**: chi-square p <0.001, Cramér's V 0.268

| age | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| 18-29 | 339 (77.4%) | 53 (12.1%) | <10 | · |
| 30-44 | 282 (52.8%) | 122 (22.8%) | 44 (8.2%) | 86 (16.1%) |
| 45-64 | 251 (32.2%) | 221 (28.3%) | 99 (12.7%) | 209 (26.8%) |
| 65-79 | 61 (14.6%) | 157 (37.6%) | 72 (17.2%) | 128 (30.6%) |
| 80+ | 24 (10.4%) | 95 (41.3%) | · | · |

**Race**: chi-square p <0.001, Cramér's V 0.082

| race | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Asian | 77 (55.8%) | 35 (25.4%) | · | · |
| Black | 99 (32.1%) | 98 (31.8%) | 27 (8.8%) | 84 (27.3%) |
| Hispanic/Latino | 84 (39.8%) | 62 (29.4%) | 15 (7.1%) | 50 (23.7%) |
| Other | · | 27 (19.9%) | · | 28 (20.6%) |
| White | 622 (39.5%) | 415 (26.4%) | 187 (11.9%) | 349 (22.2%) |
| Unknown | <10 | 11 (32.4%) | <10 | · |

**Insurance**: chi-square p <0.001, Cramér's V 0.23

| insurance | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| Medicaid | 52 (29.7%) | 59 (33.7%) | 14 (8.0%) | 50 (28.6%) |
| Medicare | 59 (11.6%) | 176 (34.7%) | 89 (17.6%) | 183 (36.1%) |
| Other | 846 (49.2%) | 413 (24.0%) | 154 (9.0%) | 305 (17.8%) |

**Language**: chi-square p 0.028, Cramér's V 0.062

| language | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| English | 867 (40.5%) | 563 (26.3%) | 237 (11.1%) | 472 (22.1%) |
| Non-English/unknown | 90 (34.5%) | 85 (32.6%) | 20 (7.7%) | 66 (25.3%) |

**Marital status**: chi-square p <0.001, Cramér's V 0.146

| marital status | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| DIVORCED | 25 (20.0%) | · | · | · |
| MARRIED | 366 (34.8%) | 311 (29.6%) | 137 (13.0%) | 238 (22.6%) |
| SINGLE | 540 (51.6%) | 224 (21.4%) | 76 (7.3%) | 206 (19.7%) |
| WIDOWED | · | 58 (37.9%) | · | 53 (34.6%) |
| Unknown | <10 | <10 | <10 | <10 |

**Year group** (`anchor_year_group`, the patient's anchor period): chi-square p <0.001, Cramér's V 0.129

| year group | appendicitis | cholecystitis | diverticulitis | pancreatitis |
|---|---|---|---|---|
| 2008 - 2010 | 246 (27.7%) | 241 (27.2%) | 128 (14.4%) | 272 (30.7%) |
| 2011 - 2013 | 315 (44.1%) | 210 (29.4%) | 60 (8.4%) | 129 (18.1%) |
| 2014 - 2016 | 248 (49.5%) | 129 (25.7%) | 41 (8.2%) | 83 (16.6%) |
| 2017 - 2019 | 148 (49.7%) | 68 (22.8%) | 28 (9.4%) | 54 (18.1%) |

**Demographics-only classifier** (5-fold CV logistic regression on one-hot groups; no clinical data):

| features | accuracy | macro_f1 | macro_auroc_ovr | majority_accuracy |
|---|---|---|---|---|
| all six attributes | 0.478 | 0.323 | 0.676 | 0.399 |
| sex + age only | 0.456 | 0.276 | 0.664 | 0.399 |

Per-class one-vs-rest AUROC (all attributes): appendicitis 0.779, cholecystitis 0.632, diverticulitis 0.639, pancreatitis 0.655


### 7.4 Does the environment reveal less for some groups?

Share of cases where each source exists, crude and label-adjusted (direct standardisation to the cohort's class mix, so a gap is not just a different class mix). Gaps ≥ 10 points between groups of ≥ 50 cases are flagged. Medians for text length and lab count.

**Sex** (Kruskal-Wallis p: HPI length within class 0.427, lab count 0.132)

| sex | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Female | 1,276 | 64.5% | 65.7 | 55.2% | 54.1 | 3.3% | 3.3 | 100.0% | 100.0 | 75.8% | 75.9 | 57.0 | 843.5 | 345.0 | 1.0 |
| Male | 1,124 | 76.2% | 75.2 | 46.9% | 48.4 | 1.8% | 1.9 | 100.0% | 100.0 | 79.1% | 79.5 | 55.0 | 822.0 | 350.0 | 1.0 |

**Age** (Kruskal-Wallis p: HPI length within class <0.001, lab count <0.001)

| age | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 18-29 | 438 | 78.3% | 61.1 | 48.2% | 55.8 | <10 cases | 1.7 | 100.0% | 100.0 | 72.8% | 68.8 | 52.0 | 593.0 | 319.0 | 0.9 |
| 30-44 | 534 | 68.7% | 62.9 | 45.3% | 51.9 | 3.2% | 2.9 | 100.0% | 100.0 | 74.5% | 73.7 | 54.0 | 731.5 | 329.0 | 0.9 |
| 45-64 | 780 | 67.6% | 70.6 | 50.8% | 46.7 | 2.6% | 2.3 | 100.0% | 100.0 | 74.5% | 74.5 | 55.0 | 879.0 | 360.5 | 1.0 |
| 65-79 | 418 | 71.1% | 78.1 | 57.4% | 44.6 | 2.9% | 2.0 | 100.0% | 100.0 | 84.2% | 83.8 | 61.0 | 1,185.0 | 376.5 | 1.1 |
| 80+ | 230 | 63.5% | 75.6 | 61.7% | 43.9 | <10 cases | 1.8 | 100.0% | 100.0 | 89.6% | 93.0 | 65.0 | 1,238.5 | 412.5 | 1.1 |

**Race** (Kruskal-Wallis p: HPI length within class 0.004, lab count 0.025)

| race | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Asian | 138 | 81.9% | 77.3 | 38.4% | 45.8 | <10 cases | 2.1 | 100.0% | 100.0 | 78.3% | 78.9 | 57.0 | 660.5 | 340.0 | 0.9 |
| Black | 308 | 65.3% | 69.7 | 59.1% | 53.5 | <10 cases | 1.3 | 100.0% | 100.0 | 74.7% | 75.6 | 57.5 | 838.5 | 353.0 | 1.0 |
| Hispanic/Latino | 211 | 62.1% | 63.9 | 63.0% | 60.1 | <10 cases | 4.1 | 100.0% | 100.0 | 73.5% | 73.9 | 54.0 | 786.0 | 345.0 | 0.9 |
| Other | 136 | 74.3% | 69.3 | 48.5% | 54.7 | <10 cases | 3.2 | 100.0% | 100.0 | 80.9% | 80.1 | 58.0 | 798.0 | 347.0 | 1.0 |
| White | 1,573 | 70.7% | 70.3 | 49.7% | 50.2 | 2.7% | 2.7 | 100.0% | 100.0 | 77.6% | 77.4 | 56.0 | 853.0 | 346.0 | 1.0 |
| Unknown | 34 | 64.7% | 72.4 | 47.1% | 39.6 | 0.0% | 0.0 | 100.0% | 100.0 | all but <10 | 98.3 | 58.0 | 1,251.0 | 371.0 | 1.2 |

**Insurance** (Kruskal-Wallis p: HPI length within class <0.001, lab count <0.001)

| insurance | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Medicaid | 175 | 60.6% | 67.4 | 57.1% | 49.2 | <10 cases | 3.8 | 100.0% | 100.0 | 73.7% | 74.5 | 54.0 | 806.0 | 353.0 | 0.9 |
| Medicare | 507 | 65.3% | 75.4 | 59.4% | 46.7 | 3.0% | 2.0 | 100.0% | 100.0 | 83.6% | 84.7 | 61.0 | 1,248.0 | 419.0 | 1.2 |
| Other | 1,718 | 72.4% | 68.6 | 48.3% | 52.4 | 2.3% | 2.4 | 100.0% | 100.0 | 75.8% | 75.7 | 55.0 | 746.0 | 334.5 | 1.0 |

**Language** (Kruskal-Wallis p: HPI length within class 0.092, lab count 0.001)

| language | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| English | 2,139 | 70.2% | 69.7 | 51.1% | 51.8 | 2.8% | 2.8 | 100.0% | 100.0 | 77.4% | 77.3 | 56.0 | 832.0 | 346.0 | 1.0 |
| Non-English/unknown | 261 | 68.6% | 72.8 | 52.5% | 46.4 | <10 cases | 1.0 | 100.0% | 100.0 | 77.0% | 77.1 | 59.0 | 829.0 | 354.0 | 0.9 |

**Marital status** (Kruskal-Wallis p: HPI length within class 0.001, lab count <0.001)

| marital status | n | CT | CT (adj.) | Ultrasound | Ultrasound (adj.) | MRI | MRI (adj.) | any imaging | any imaging (adj.) | microbiology | microbiology (adj.) | lab items (median) | HPI chars (median) | PE chars (median) | HPI ÷ class median (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DIVORCED | 125 | 65.6% | 74.7 | 61.6% | 53.2 | 0.0% | 0.0 | 100.0% | 100.0 | 85.6% | 86.1 | 57.0 | 1,022.0 | 352.0 | 1.1 |
| MARRIED | 1,052 | 68.8% | 70.2 | 48.4% | 46.3 | 2.7% | 2.7 | 100.0% | 100.0 | 78.3% | 78.1 | 57.0 | 872.0 | 347.0 | 1.0 |
| SINGLE | 1,046 | 72.6% | 68.1 | 51.1% | 55.2 | 2.7% | 3.0 | 100.0% | 100.0 | 74.1% | 74.4 | 54.0 | 725.5 | 340.0 | 0.9 |
| WIDOWED | 153 | 63.4% | 73.3 | 62.7% | 44.6 | <10 cases | 1.7 | 100.0% | 100.0 | 84.3% | 87.7 | 63.0 | 1,160.0 | 397.0 | 1.1 |
| Unknown | 24 | all but <10 | 75.8 | 58.3% | 54.8 | <10 cases | 10.8 | 100.0% | 100.0 | all but <10 | 91.6 | 69.5 | 1,106.0 | 390.0 | 1.1 |


### 7.5 The LA-CDM split (80/10/10, seed 269)

| split | n | female | age median (IQR) | White | Black | Hispanic/Latino | Asian | Unknown | Medicaid | non-English/unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 1,920 | 52.0% | 51 (33–66) | 65.4% | 13.1% | 8.8% | 5.6% | 1.4% | 6.5% | 10.3% |
| val | 240 | 55.4% | 46 (34–63) | 64.6% | 11.7% | 10.4% | 7.1% | <10 cases | 12.5% | 15.0% |
| test | 240 | 60.4% | 52 (33–67) | 67.5% | 12.1% | 7.5% | 5.8% | <10 cases | 8.3% | 11.2% |

Chi-square across splits: sex p 0.036; race p 0.966; insurance p 0.009; language p 0.087

Patients with more than one CDM admission: 17; with admissions under different labels: 10; **with admissions in more than one split: <10**.


## 8. MIMIC-IV diagnoses vs. CDM labels

Using the ICD prefixes in `configs/openworld_icd.yaml` (`cdm_conditions`, `other_groups`):

| label | primary dx = this condition | primary dx = another CDM condition | primary dx = an OTHER group | any code for this condition |
|---|---|---|---|---|
| appendicitis | 96.0% | 0.0% | 0.0% | 100.0% |
| cholecystitis | 91.0% | <10 cases | 0.0% | 100.0% |
| diverticulitis | 89.1% | 0.0% | <10 cases | 100.0% |
| pancreatitis | 74.3% | 6.5% | <10 cases | 100.0% |

| label | also coded appendicitis | also coded cholecystitis | also coded diverticulitis | also coded pancreatitis |
|---|---|---|---|---|
| appendicitis | – | 0.0% | <10 cases | 0.0% |
| cholecystitis | <10 cases | – | <10 cases | 3.9% |
| diverticulitis | <10 cases | 0.0% | – | 0.0% |
| pancreatitis | 0.0% | 20.1% | 0.0% | – |


## 9. Open-world (OTHER) candidate pool

Same filters as `build_openworld`: primary diagnosis in an OTHER group, no CDM condition code at any position, not a CDM admission, a discharge note, and a chief complaint matching `complaint_regex`. The builder then drops histories that name their own diagnosis and admissions failing CDM's inclusion rule (exam of 40+ characters, a lab, an abdominal study), so the built set is much smaller.

| OTHER group | icd_candidates | with_discharge_note | abdominal_chief_complaint |
|---|---|---|---|
| abdominal_aortic_aneurysm | 750 | 714 | 377 |
| bowel_obstruction | 3,411 | 3,243 | 2,224 |
| diabetic_ketoacidosis | 1,160 | 1,034 | 137 |
| ectopic_pregnancy | 228 | 157 | 66 |
| gastroenteritis_colitis | 4,211 | 3,915 | 1,385 |
| gi_bleed | 2,810 | 2,661 | 173 |
| mesenteric_ischemia | 248 | 241 | 148 |
| peptic_ulcer_perforation_or_bleed | 162 | 159 | 114 |
| urolithiasis | 1,333 | 1,046 | 465 |

Demographics of the built open-world set (`data/openworld/other.jsonl`, 713 cases), next to CDM:

| cohort | n | female | age median (IQR) | White | Black | Hispanic/Latino | Asian | Unknown | Medicaid | non-English/unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM (in-set) | 2,400 | 53.2% | 51 (33–66) | 65.5% | 12.8% | 8.8% | 5.8% | 1.4% | 7.3% | 10.9% |
| OTHER sample | 713 | 59.5% | 59 (46–74) | 67.5% | 16.5% | 8.0% | 3.4% | 1.7% | 8.7% | 12.3% |
| abdominal_aortic_aneurysm | <10 | – | – | – | – | – | – | – | – | – |
| bowel_obstruction | 146 | 55.5% | 66 (48–75) | 65.1% | 19.9% | <10 cases | <10 cases | <10 cases | 6.8% | 15.8% |
| diabetic_ketoacidosis | 17 | all but <10 | 39 (29–50) | <10 cases | <10 cases | <10 cases | <10 cases | 0.0% | <10 cases | 0.0% |
| ectopic_pregnancy | <10 | – | – | – | – | – | – | – | – | – |
| gastroenteritis_colitis | 219 | 62.6% | 56 (41–69) | 65.8% | 18.3% | 10.0% | <10 cases | <10 cases | 9.6% | 11.4% |
| gi_bleed | 79 | 50.6% | 53 (41–78) | 60.8% | 20.3% | <10 cases | <10 cases | 0.0% | <10 cases | 17.7% |
| mesenteric_ischemia | 94 | 66.0% | 67 (58–80) | 80.9% | <10 cases | <10 cases | <10 cases | <10 cases | <10 cases | <10 cases |
| peptic_ulcer_perforation_or_bleed | 70 | 51.4% | 64 (52–75) | 68.6% | 14.3% | <10 cases | <10 cases | <10 cases | <10 cases | <10 cases |
| urolithiasis | 78 | 60.3% | 58 (44–68) | 71.8% | 12.8% | <10 cases | <10 cases | 0.0% | <10 cases | 15.4% |

Demographics-only classifier, in-set vs. OTHER: AUROC 0.634 (accuracy 0.768, majority 0.771).

`build_openworld` sorts its cohort query by `hadm_id` before the seeded per-group shuffle (added 2026-10-04; unsorted, DuckDB returned a different order on each of 4 runs). Re-running the sorted query 3 times returned rows in the same order.
