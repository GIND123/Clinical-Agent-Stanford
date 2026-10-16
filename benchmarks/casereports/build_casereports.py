#!/usr/bin/env python
"""MIMIC-CDM-CR: published case reports of acute abdominal pain, as environment cases.

Every case is rebuilt from one open-access case report (2020 or later). The facts are paraphrased into the
environment's case format, and the source is cited in the case's `meta.source`:
  * the history of present illness with past history, medications and social history (shown at reset);
  * the physical examination with vital signs (revealed by ASK physical_exam);
  * the laboratory values at presentation that the 22-test catalogue covers, with units converted to the US
    conventional units of MIMIC-IV (creatinine and bilirubin in mg/dL, urea as BUN in mg/dL, glucose in mg/dL,
    CRP in mg/L); tests outside the catalogue (D-dimer, troponin, LDH, blood gas, toxicology, ECG) are left out,
    as for MIMIC cases;
  * every imaging study the report describes at presentation, as a short report of its findings and impression.
Nothing is invented: a finding the report does not give is absent, and an unperformed test answers "not
performed", as in the MIMIC environment. Radiology impressions name diagnoses as real reports do (no "____"
mask), so these cases also test the agent on unmasked text.

Labels: "other" for any principal diagnosis outside the four conditions, with `meta.group`; for the four
conditions, `meta.stage` gives the guideline grade the report supports and `meta.complicated`.
`meta.time_critical` marks diagnoses where hours of delay change outcome. `meta.training_group` says whether the
group appears in training ("seen"), is one of the five held-out time-critical groups ("held_out"), or appears in
no MIMIC cohort here ("new").

    python benchmarks/casereports/build_casereports.py
writes benchmarks/casereports/casereports.jsonl (public, versioned) and a copy at data/casereports/casereports.jsonl,
where the GPU queue evaluates it.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "casereports.jsonl"
QUEUE_COPY = HERE.parents[1] / "data" / "casereports" / "casereports.jsonl"
CITATIONS = json.loads((HERE / "citations.json").read_text(encoding="utf-8"))  # Europe PMC records


def lab(name, value, unit=None, low=None, high=None, fluid="Blood"):
    return {"name": name, "value": str(value), "unit": unit, "ref_low": low, "ref_high": high,
            "flag": None if low is None or high is None else ("abnormal" if not (low <= float(value) <= high) else None),
            "fluid": fluid}


def img(modality, region, exam_name, text):
    return {"modality": modality, "region": region, "exam_name": exam_name, "text": text}


def src(pmcid: str) -> dict:
    return dict(CITATIONS[pmcid])


CASES = [
    # ---- held-out time-critical groups ------------------------------------------------------------------
    dict(
        case_id="CR01", label="other", group="mesenteric_ischemia", time_critical=True, training_group="held_out",
        source=src("PMC11552470"),
        hpi=("83-year-old woman brought in with abdominal pain, nausea and vomiting that began 1 hour before arrival. "
             "Past history: hypertension for more than 20 years on valsartan with poor control; type 2 diabetes for more "
             "than 20 years on metformin with poor control; coronary heart disease and atrial fibrillation for more than "
             "10 years. General condition poor."),
        physical_exam=("Temperature 36.5 C, pulse 113/min, respiratory rate 24/min, blood pressure 149/72 mmHg. Slow to "
                       "respond, in apparent distress, lying in a passive position. A detailed abdominal examination is not "
                       "documented."),
        labs=[lab("White Blood Cells", 12.56, "K/uL", 4, 10), lab("Platelet Count", 162, "K/uL", 100, 300),
              lab("Asparate Aminotransferase (AST)", 21, "IU/L", 7, 50),
              lab("Alanine Aminotransferase (ALT)", 15, "IU/L", 0, 50),
              lab("Creatinine", 0.80, "mg/dL", 0.45, 1.24), lab("Urea Nitrogen", 15, "mg/dL", 5, 23),
              lab("Amylase", 37, "IU/L", 28, 100)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis without and with contrast; CT angiography",
                     "Non-contrast: no bowel wall thickening, clear mesentery, fluid within small bowel loops, no free "
                     "fluid. Contrast phase: reduced mural enhancement of the distal jejunum, the ileum and the right "
                     "colon. CT angiography: intraluminal filling defect in the superior mesenteric artery about 6.5 cm "
                     "from its origin, with non-opacification of its distal branches (middle colic, right colic, "
                     "ileocolic, jejunal and ileal arteries). No free intraperitoneal gas. Appendix normal. Pancreas normal "
                     "without peripancreatic fluid. IMPRESSION: acute superior mesenteric artery embolism with ischaemia of "
                     "the small bowel and right colon.")],
        note="Atrial fibrillation with a fast ventricular rate on ECG (ECG is not in the test catalogue).",
    ),
    dict(
        case_id="CR02", label="other", group="abdominal_aortic_aneurysm", time_critical=True, training_group="held_out",
        source=src("PMC12661095"),
        hpi=("65-year-old man with abdominal pain for about 6 hours, starting in the epigastrium and moving to the right "
             "lower quadrant; continuous, moderate to severe, worse with movement, coughing and walking. One episode of "
             "vomiting. No diarrhoea, constipation or urinary symptoms. Past history: hypertension and dyslipidaemia on "
             "an ACE inhibitor and a statin. Smoker, 23 pack-years."),
        physical_exam=("Alert and oriented. Blood pressure 102/61 mmHg, heart rate 92/min, temperature 36.4 C, SpO2 98% on "
                       "room air. Abdomen mildly distended, tender in the right lower quadrant with a positive McBurney sign "
                       "and mild rebound tenderness; no guarding or rigidity; normal bowel sounds. Rectal examination "
                       "unremarkable."),
        labs=[lab("White Blood Cells", 18.2, "K/uL", 4.3, 10.3), lab("Neutrophils", 81.4, "%", 41, 73),
              lab("Red Blood Cells", 3.91, "m/uL", 4.38, 5.77), lab("Hematocrit", 38.2, "%", 39.5, 51),
              lab("Hemoglobin", 12.5, "g/dL", 13.6, 17.2), lab("Platelet Count", 298, "K/uL", 140, 440),
              lab("PT", 11.7, "sec", 10, 15), lab("INR(PT)", 1.0, None), lab("PTT", 23.7, "sec", 26, 36),
              lab("Asparate Aminotransferase (AST)", 12, "IU/L", 0, 40),
              lab("Alanine Aminotransferase (ALT)", 18, "IU/L", 0, 41),
              lab("Urea Nitrogen", 17, "mg/dL", 6, 23), lab("Creatinine", 1.02, "mg/dL", 0.7, 1.2),
              lab("C-Reactive Protein", 61, "mg/L", 0, 5)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis with intravenous and oral contrast",
                     "Infrarenal abdominal aortic aneurysm, 70 mm in maximum transverse diameter, with a contained "
                     "retroperitoneal rupture and a large retroperitoneal haematoma (about 760 mL). Proximal neck 28 mm "
                     "wide and 19 mm long, angulated about 45 degrees; iliac arteries patent. IMPRESSION: contained rupture "
                     "of an infrarenal abdominal aortic aneurysm.")],
        note="Alvarado score 9/10 at presentation; urea 36 mg/dL in the report converted to BUN.",
    ),
    dict(
        case_id="CR03", label="other", group="ectopic_pregnancy", time_critical=True, training_group="held_out",
        source=src("PMC12060474"),
        hpi=("28-year-old woman, gravida 5 para 4 (four spontaneous vaginal deliveries), with 4 weeks of amenorrhoea and 4 "
             "days of abdominal pain, first on the right and then spreading to the whole abdomen. No vaginal bleeding, no "
             "contraceptive use, no abdominal trauma."),
        physical_exam=("Pulse 120/min, blood pressure 120/68 mmHg, respiratory rate 20/min. Conjunctival pallor. Diffuse "
                       "abdominal tenderness without a mass or organomegaly. Cervical motion tenderness; cervix closed on "
                       "vaginal examination."),
        labs=[lab("Hemoglobin", 8.9, "g/dL", 12, 16), lab("Hematocrit", 19.6, "%", 36, 48),
              lab("HCG, Urine, Qualitative", "POSITIVE", None, fluid="Urine")],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen and pelvis",
                     "Free fluid in the pouch of Douglas, both paracolic gutters and Morison's pouch. The uterus is empty. "
                     "Complex right adnexal mass of 2 x 3 cm. IMPRESSION: haemoperitoneum with an empty uterus and a complex "
                     "right adnexal mass in a patient with a positive pregnancy test.")],
    ),
    dict(
        case_id="CR04", label="other", group="peptic_ulcer_perforation_or_bleed", time_critical=True,
        training_group="held_out", source=src("PMC11264131"),
        hpi=("42-year-old woman with several days of nausea and copious vomiting after food or drink, the vomit dark in "
             "colour, and epigastric pain that was mild until today, when it became severe. Multiple sclerosis since 2015 "
             "on immunomodulatory therapy, with urinary and faecal incontinence for 5 months. Takes metamizole and "
             "diazepam as needed, frequently. Smokes 20 cigarettes a day."),
        physical_exam="Acute abdomen: board-like rigidity with diffuse tenderness of the abdominal wall.",
        labs=[lab("Hemoglobin", 9.1, "g/dL", 12.0, 16.5), lab("White Blood Cells", 23.4, "K/uL", 3.5, 10.0),
              lab("Neutrophils", 91.5, "%", 40, 75), lab("Platelet Count", 1089, "K/uL", 150, 390),
              lab("Urea Nitrogen", 26, "mg/dL", 6, 23), lab("Protein, Total", 5.7, "g/dL", 6.4, 8.3),
              lab("Albumin", 3.0, "g/dL", 3.5, 5.0), lab("C-Reactive Protein", 287.2, "mg/L", 0, 5)],
        imaging=[img("CT", "Abdomen", "CT abdomen",
                     "Large pockets of free intraperitoneal air (pneumoperitoneum). Free fluid of up to 15 HU in all "
                     "abdominal compartments. Discontinuity of the anterior gastric wall. IMPRESSION: hollow viscus "
                     "perforation, likely at the anterior wall of the stomach.")],
        note="Absolute neutrophils 21.4 x10^9/L given as a percentage of WBC; urea 9.4 mmol/L converted to BUN.",
    ),
    dict(
        case_id="CR05", label="other", group="splanchnic_vein_thrombosis", time_critical=True, training_group="new",
        source=src("PMC11526769"),
        hpi=("32-year-old man with generalised abdominal pain for one month, mild at first but severe (9/10) for the past "
             "3 days, keeping him awake. Worse lying flat and after meals; he has been avoiding food to prevent it. No "
             "bowel movement for 3 days. No fever, nausea, vomiting, diarrhoea, chest pain or back pain. No recent "
             "travel. Medications: clomiphene citrate for infertility for 3 months. No other medical or surgical history."),
        physical_exam="Diffuse abdominal tenderness, most marked in the right lower quadrant. Vital signs not documented.",
        labs=[lab("White Blood Cells", 8.4, "K/uL", 4.0, 10.0), lab("Hemoglobin", 15.3, "g/dL", 13, 18),
              lab("Platelet Count", 224, "K/uL", 150, 400), lab("PT", 14.7, "sec", 9.4, 12.5),
              lab("Alanine Aminotransferase (ALT)", 31, "IU/L", 0, 41),
              lab("Asparate Aminotransferase (AST)", 25, "IU/L", 0, 41),
              lab("Alkaline Phosphatase", 63, "IU/L", 40, 129), lab("C-Reactive Protein", 16.4, "mg/L", 0, 5),
              lab("Lactate", 1.3, "mmol/L", 0.5, 2.2), lab("Amylase", 28, "IU/L", 13, 53),
              lab("Creatinine", 1.35, "mg/dL", 0.70, 1.20), lab("Urea Nitrogen", 10, "mg/dL", 7, 22),
              lab("Protein, Total", 6.6, "g/dL", 6.0, 8.0)],
        imaging=[img("CT", "Abdomen", "CT abdomen with contrast",
                     "Filling defects in the right portal vein branch and its tributaries, extending to the confluence of "
                     "the splenic and superior mesenteric veins, with complete non-opacification. No features of cirrhosis. "
                     "No CT evidence of bowel ischaemia. IMPRESSION: acute portal and superior mesenteric vein "
                     "thrombosis.")],
        note="Creatinine 119 umol/L and urea 3.7 mmol/L converted; D-dimer 2.05 mg/L is outside the catalogue.",
    ),
    dict(
        case_id="CR06", label="other", group="peptic_ulcer_perforation_or_bleed", time_critical=True,
        training_group="held_out", source=src("PMC8890457"),
        hpi=("62-year-old woman with paranoid schizophrenia (declines antipsychotics), coronary artery disease and "
             "hypertension, admitted after being found sitting on the floor of her building's hallway complaining of back "
             "pain, pain all over her body, nausea and several episodes of vomiting. History is limited by agitation. "
             "Treated overnight with intravenous fluids for dehydration. On the second hospital day she is found "
             "screaming in pain and clutching her abdomen; she asks to be discharged. Medications: aspirin, atorvastatin, "
             "metoprolol. Urine toxicology positive for cocaine."),
        physical_exam=("On arrival: blood pressure 111/71 mmHg, pulse 69/min, respiratory rate 17/min, temperature 36.4 C; "
                       "examination unremarkable. On the second day, during the abdominal pain: abdomen soft, not "
                       "distended and not tender."),
        labs=[lab("Urea Nitrogen", 47, "mg/dL", 6, 20), lab("Creatinine", 1.54, "mg/dL", 0.5, 1.0),
              lab("Lipase", 60, "IU/L", 12, 53)],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen", "Normal gallbladder without stones."),
                 img("CT", "Abdomen", "CT abdomen",
                     "Pneumoperitoneum. Irregularity of the gastric wall and proximal duodenum at the probable site of "
                     "perforation. IMPRESSION: perforated viscus, likely gastroduodenal.")],
        note="BUN 67 mg/dL and creatinine 2.12 mg/dL on arrival, 47 and 1.54 after fluids (the values given).",
    ),
    dict(
        case_id="CR07", label="other", group="lead_poisoning", time_critical=False, training_group="new",
        source=src("PMC7557016"),
        hpi=("26-year-old man with recurrent abdominal pain for 3 months, worse for the past 2 days: paroxysmal colic "
             "around the umbilicus and lower abdomen, radiating to the lower back. A community hospital diagnosed kidney "
             "stones; antispasmodics and analgesics did not help. Works as a warehouse keeper and has handled new "
             "children's picture books daily for more than 2 years."),
        physical_exam=("Vital signs stable. Pale, anaemic appearance. Slight periumbilical tenderness; no costovertebral "
                       "angle tenderness on percussion. Bluish-grey line along the gingival margins of the upper and "
                       "lower gums."),
        labs=[lab("Hemoglobin", 7.4, "g/dL", 13, 17.5), lab("Bilirubin, Total", 1.8, "mg/dL", 0, 1.2)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis",
                     "Small calculi in both kidneys. Small amount of pelvic free fluid. Mildly enlarged abdominal and "
                     "small retroperitoneal lymph nodes. No other acute abnormality.")],
        note=("Normocytic anaemia with mainly indirect hyperbilirubinaemia; other tests reported as normal (values not "
              "given). Blood lead 52.8 ug/dL and basophilic stippling were found later (not in the catalogue)."),
    ),
    dict(
        case_id="CR08", label="other", group="diabetic_ketoacidosis", time_critical=True, training_group="held_out",
        source=src("PMC6977311"),
        hpi=("71-year-old woman with type 2 diabetes on canagliflozin 100 mg/day, metformin 1000 mg/day and saxagliptin "
             "5 mg/day. General malaise for one month; poor appetite and reduced oral intake for two weeks while taking "
             "all her medications at the usual doses. Since yesterday the malaise has worsened, with nausea and "
             "abdominal pain. Her local doctor found a severe metabolic acidosis and transferred her urgently."),
        physical_exam=("Temperature 35.0 C, pulse 118/min, respiratory rate 28/min, blood pressure 111/75 mmHg. Dry oral "
                       "mucosa; cold peripheries. Chest clear. Mild tenderness in the lower abdomen."),
        labs=[lab("Urea Nitrogen", 18, "mg/dL", 8, 20), lab("Creatinine", 2.02, "mg/dL", 0.65, 1.07),
              lab("Sodium", 143, "mEq/L", 138, 145), lab("Potassium", 4.6, "mEq/L", 3.6, 4.8),
              lab("Chloride", 107, "mEq/L", 101, 108), lab("Bicarbonate", 2, "mEq/L", 21, 29),
              lab("Glucose", 259, "mg/dL", 70, 110), lab("Lactate", 3.3, "mmol/L", 0.44, 2.13),
              lab("C-Reactive Protein", 57.8, "mg/L", 0, 1.4),
              lab("Glucose", "4+", None, fluid="Urine"), lab("Ketone", "3+", None, fluid="Urine")],
        imaging=[],
        note=("Arterial pH 6.83, PaCO2 11.4 mmHg, HCO3 1.9 mmol/L, base excess -31.3 (blood gas is outside the catalogue; "
              "the bicarbonate is shown with the chemistry). Plasma 3-hydroxybutyrate > 10,000 umol/L."),
    ),
    dict(
        case_id="CR09", label="other", group="ectopic_pregnancy", time_critical=True, training_group="held_out",
        source=src("PMC10470474"),
        hpi=("Woman in her late 20s, gravida 4 para 3 (spontaneous vaginal deliveries), with two days of intermittent, "
             "worsening, stitch-like lower abdominal pain, worse on movement, that has moved to the right iliac fossa. No "
             "diarrhoea, nausea, vomiting or urinary symptoms; no relation to food. Loss of appetite and intentional "
             "weight loss over 5 months. No vaginal or rectal bleeding. Irregular periods; does not know her last "
             "menstrual period; last sexually active about 3 months ago; no contraception; does not believe she is "
             "pregnant. No regular medications."),
        physical_exam=("Respiratory rate 21/min, SpO2 100% on room air, blood pressure 119/63 mmHg, pulse 97/min, temperature "
                       "37.3 C. Chest clear. Abdomen not distended, tender in the right iliac fossa with guarding, rebound "
                       "tenderness and a positive McBurney sign; Rovsing sign negative; bowel sounds present."),
        labs=[lab("White Blood Cells", 11.8, "K/uL", 4.0, 11.0), lab("Neutrophils", 66, "%", 40, 75),
              lab("Hemoglobin", 11.9, "g/dL", 11.5, 16.5), lab("Platelet Count", 250, "K/uL", 150, 400),
              lab("C-Reactive Protein", 18, "mg/L", 0, 5), lab("Lactate", 0.8, "mmol/L", 0.5, 2.2),
              lab("Human Chorionic Gonadotropin", 75321, "mIU/mL"),
              lab("HCG, Urine, Qualitative", "POSITIVE", None, fluid="Urine"),
              lab("Leukocytes", "NEG", None, fluid="Urine"), lab("Nitrite", "NEG", None, fluid="Urine"),
              lab("Blood", "NEG", None, fluid="Urine")],
        imaging=[img("Ultrasound", "Abdomen", "Point-of-care pelvic ultrasound",
                     "Viable singleton intrauterine pregnancy. No other findings reported.")],
        note=("Liver tests, urea and electrolytes within reference ranges (values not given). Final diagnosis at "
              "laparoscopy: right tubal ectopic pregnancy with haemoperitoneum alongside the intrauterine pregnancy "
              "(heterotopic); appendix normal."),
    ),
    dict(
        case_id="CR10", label="other", group="acute_porphyria", time_critical=True, training_group="new",
        source=src("PMC7357387"),
        hpi=("24-year-old woman referred with 10 days of severe abdominal pain, vomiting, poor appetite, constipation, "
             "fatigue and weakness. Two days ago another hospital performed an exploratory laparotomy for worsening pain, "
             "which found no intra-abdominal abnormality. Past history: condylomata, anaemia under investigation, allergy "
             "to ibuprofen. Her maternal grandmother had an unspecified rare disease."),
        physical_exam=("Heart rate 113/min (tachycardic throughout the illness), blood pressure 177/90 mmHg, temperature "
                       "36.2 C. Dry mucous membranes. Recent midline laparotomy wound healing well, without dehiscence or "
                       "infection. Marked pain on light palpation of the hypogastrium with voluntary guarding, no signs of "
                       "peritoneal irritation. Drowsy on neurological examination."),
        labs=[lab("Creatinine", 0.62, "mg/dL", 0.6, 1.1), lab("Urea Nitrogen", 10, "mg/dL", 8, 20),
              lab("Asparate Aminotransferase (AST)", 32, "IU/L", 10, 45),
              lab("Alanine Aminotransferase (ALT)", 25, "IU/L", 10, 45),
              lab("Alkaline Phosphatase", 113, "IU/L", 44, 147), lab("Sodium", 118, "mEq/L", 135, 145),
              lab("Potassium", 2.6, "mEq/L", 3.5, 5.3), lab("Chloride", 82, "mEq/L", 98, 109),
              lab("Glucose", 81, "mg/dL", 70, 140), lab("Protein, Total", 9.8, "g/dL", 6.0, 8.3),
              lab("Albumin", 3.83, "g/dL", 3.4, 5.4), lab("White Blood Cells", 19.4, "K/uL", 4.5, 11.0),
              lab("Neutrophils", 83, "%", 40, 85), lab("Lymphocytes", 6, "%", 10, 45),
              lab("Hemoglobin", 15.6, "g/dL", 13, 16), lab("Platelet Count", 356, "K/uL", 150, 450)],
        imaging=[img("CT", "Abdomen", "CT abdomen",
                     "Free fluid in the pelvis. Residual pneumoperitoneum. Distended colon. Recent postoperative changes of "
                     "the anterior abdominal wall. No other acute abnormality described.")],
        note=("Plasma osmolality 244 mOsm/kg and magnesium 0.8 mg/dL (outside the catalogue). Urinary porphobilinogen 5 mg/24 h "
              "and porphyrins markedly raised, found later. Source in Spanish; facts translated."),
    ),
    dict(
        case_id="CR11", label="other", group="toxic_ingestion", time_critical=True, training_group="new",
        source=src("PMC11585366"),
        hpi=("73-year-old man with diffuse, intense abdominal pain for about 24 hours with diarrhoea and vomiting. He ate "
             "wild mushrooms the night before the pain began. Past history: hypertension, dyslipidaemia, obesity (class "
             "1), atrial fibrillation. Medications: lisinopril/amlodipine 20/5 mg daily, atorvastatin 40 mg daily, "
             "apixaban 5 mg twice daily."),
        physical_exam="Examination findings and vital signs at presentation are not reported in the source.",
        labs=[lab("Asparate Aminotransferase (AST)", 36, "IU/L", 12, 40),
              lab("Alanine Aminotransferase (ALT)", 33, "IU/L", 7, 40),
              lab("Gamma Glutamyltransferase", 67, "IU/L", 0, 73), lab("Alkaline Phosphatase", 67, "IU/L", 46, 116),
              lab("Bilirubin, Total", 0.68, "mg/dL", 0.3, 1.2), lab("Bilirubin, Direct", 0.18, "mg/dL", 0, 0.3)],
        imaging=[],
        note=("Emergency-department values. Over the next two days transaminases rose to an ALT peak of 15,365 U/L, with "
              "renal, metabolic and haematological dysfunction: amatoxin hepatitis (Amanita)."),
    ),
    dict(
        case_id="CR12", label="other", group="toxic_ingestion", time_critical=True, training_group="new",
        source=src("PMC10390273"),
        hpi=("51-year-old woman, a lawyer living in a residential area, brought by a relative with one month of muscle "
             "aches and vertigo, and over the past week abdominal pain, daily vomiting (up to three times) and "
             "increasing weakness. Past history: irritable bowel syndrome; hypothyroidism on levothyroxine 50 ug daily "
             "for 2 years. No alcohol, tobacco, herbal remedies, pesticide contact or other medicines."),
        physical_exam=("Vital signs within normal limits. Disoriented in time and place with incoherent speech. Generalised "
                       "jaundice. Asterixis (flapping tremor). On a second examination: alopecia of the scalp, armpits and "
                       "eyebrows; brown macules on the face; dry, rough skin without sweating; generalised hypotonia; "
                       "plantar hyperkeratosis; ulcers on both legs; weakness and reduced reflexes with decreased "
                       "sensation in all four limbs."),
        labs=[lab("Bilirubin, Total", 10.3, "mg/dL", 0.3, 1.2), lab("Bilirubin, Direct", 5.0, "mg/dL", 0, 0.3),
              lab("Asparate Aminotransferase (AST)", 78, "IU/L", 0, 40),
              lab("Alanine Aminotransferase (ALT)", 194, "IU/L", 0, 40), lab("INR(PT)", 1.7, None, 0.8, 1.2),
              lab("Albumin", 2.3, "g/dL", 3.5, 5.0), lab("Platelet Count", 60, "K/uL", 150, 400)],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen", "No pathological findings.")],
        note=("Sodium, potassium, chloride, calcium, phosphate, magnesium and ammonia unremarkable (values not given). "
              "C-reactive protein reported as 69 (unit ambiguous; omitted). Non-contrast head CT normal. Thallium "
              "poisoning confirmed by plasma and urine levels."),
    ),
    dict(
        case_id="CR13", label="other", group="adrenal_crisis", time_critical=True, training_group="new",
        source=src("PMC8105925"),
        hpi=("45-year-old woman with one month of worsening abdominal pain, vomiting and diarrhoea, on a background of "
             "generalised fatigue for 18 years. Eighteen years ago, after the home birth of her seventh child, she had "
             "heavy bleeding from a retained placenta and lost consciousness; afterwards she could not breastfeed, has "
             "had no periods and has not conceived. For 6 years: recurrent abdominal pain, nausea, vomiting and "
             "diarrhoea, treated repeatedly at a clinic as dyspepsia and gastroenteritis without lasting benefit; weight "
             "loss, poor appetite, dizziness, cold intolerance, muscle and joint aches, and progressive loss of axillary "
             "and pubic hair. Over the past month she has become extremely weak even at rest and excessively sleepy, "
             "with mild intermittent headache. No head injury, surgery or irradiation."),
        physical_exam=("Conscious but sleepy. Blood pressure 90/60 mmHg, pulse 86/min, respiratory rate 24/min, temperature "
                       "36.9 C. Mild epigastric and periumbilical tenderness. Dry, coarse skin; sparse axillary and pubic "
                       "hair. Other systems unremarkable."),
        labs=[lab("White Blood Cells", 2.59, "K/uL", 4.0, 11.0), lab("Hematocrit", 26.9, "%", 36, 48),
              lab("MCV", 81.3, "fL", 80, 100), lab("Creatinine", 1.2, "mg/dL", 0.5, 1.1),
              lab("Sodium", 126, "mEq/L", 135, 145), lab("Potassium", 2.52, "mEq/L", 3.5, 5.0),
              lab("Glucose", 113, "mg/dL", 70, 140), lab("Triglycerides", 86, "mg/dL", 0, 150)],
        imaging=[],
        note=("Liver tests normal (values not given); repeat potassium 4.2 mEq/L. 8 a.m. cortisol 0.9 ug/dL and low free T4, "
              "and an empty sella on brain MRI, found later (outside the catalogue): Sheehan's syndrome with impending "
              "adrenal crisis."),
    ),
    dict(
        case_id="CR14", label="other", group="myocardial_infarction", time_critical=True, training_group="new",
        source=src("PMC12824467"),
        hpi=("60-year-old man brought in after a brief loss of consciousness lasting a few seconds; he does not recall the "
             "event. He has had severe abdominal pain before and since, and denies chest pain at any point. Past history: "
             "type 2 diabetes, hypertension, a previous myocardial infarction with 10 days of coma in intensive care, and "
             "heart failure with reduced ejection fraction (old records unavailable). Medications: bisoprolol, "
             "rosuvastatin, aspirin, clopidogrel, metformin, valsartan. Long-term vaping."),
        physical_exam=("Alert and oriented but slow to answer. Ill-looking, sweating, holding his abdomen with both hands. "
                       "Overweight. Heart rate 63/min, blood pressure 76/50 mmHg, SpO2 94% on room air. No focal neurological "
                       "deficit. Good air entry bilaterally without wheeze or crackles. Distant heart sounds; no raised "
                       "jugular venous pressure or leg oedema. Abdomen diffusely tender, most in the epigastrium and upper "
                       "quadrants, without rigidity. Left periorbital swelling and bruising from the fall."),
        labs=[lab("Hemoglobin", 13.8, "g/dL", 13.5, 17.5), lab("Hematocrit", 40.5, "%", 37, 53),
              lab("Red Blood Cells", 4.58, "m/uL", 4.5, 5.9), lab("White Blood Cells", 11.3, "K/uL", 4.5, 11),
              lab("MCV", 88.4, "fL", 80, 100), lab("MCH", 30.1, "pg", 26, 34), lab("MCHC", 34.1, "%", 32, 36),
              lab("Platelet Count", 156, "K/uL", 150, 450), lab("RDW", 14.6, "%", 11.5, 13.1),
              lab("Neutrophils", 68, "%", 35, 66), lab("Lymphocytes", 24, "%", 24, 44), lab("Monocytes", 8, "%", 4, 10),
              lab("Urea Nitrogen", 12, "mg/dL", 6, 20), lab("Creatinine", 0.94, "mg/dL", 0.7, 1.2),
              lab("Sodium", 141, "mEq/L", 135, 148), lab("Potassium", 4.26, "mEq/L", 3.5, 5.0),
              lab("Chloride", 108, "mEq/L", 97, 107), lab("Asparate Aminotransferase (AST)", 17, "IU/L", 0, 40),
              lab("Alanine Aminotransferase (ALT)", 20, "IU/L", 0, 41),
              lab("Alkaline Phosphatase", 90, "IU/L", 40, 115), lab("Bilirubin, Total", 0.45, "mg/dL", 0, 1.2),
              lab("Bilirubin, Direct", 0.17, "mg/dL", 0, 0.4), lab("Amylase", 115, "IU/L", 28, 80)],
        imaging=[],
        note=("ECG: inferior ST elevation with reciprocal changes and ST depression V1-V5 (inferior STEMI); troponin I and "
              "D-dimer initially negative. ECG and troponin are outside the catalogue. Urea 26 mg/dL converted to BUN."),
    ),
    dict(
        case_id="CR15", label="other", group="adnexal_torsion", time_critical=True, training_group="new",
        source=src("PMC8146748"),
        hpi=("25-year-old woman, gravida 4 para 4, with severe right lower quadrant abdominal pain and vomiting. She "
             "delivered vaginally at home 5 days ago, without antenatal care or any ultrasound during pregnancy."),
        physical_exam=("Temperature 37.8 C, blood pressure 116/72 mmHg, heart rate 110/min, respiratory rate 22/min. In severe "
                       "distress from pain. Abdomen moderately distended with guarding and rebound tenderness; bowel sounds "
                       "present but hypoactive."),
        labs=[lab("White Blood Cells", 9.6, "K/uL", 4.0, 11.0), lab("C-Reactive Protein", 34, "mg/L", 0, 5)],
        imaging=[img("Ultrasound", "Abdomen", "Transabdominal ultrasound",
                     "Well-defined hypoechoic lesion with internal echoes and posterior acoustic shadowing in the right "
                     "adnexa, with no internal flow on colour Doppler, containing an echogenic mural nodule (dermoid plug)."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with contrast",
                     "Normal appendix. Enlarged uterus. Heterogeneous, non-enhancing right adnexal lesion with subtle "
                     "peripheral calcification and fat and fluid components. IMPRESSION: right adnexal mature cystic "
                     "teratoma without enhancement; torsion should be considered.")],
        note=("C-reactive protein printed as 34 g/dL in the source (impossible unit), shown as 34 mg/L. At surgery: "
              "torsion with a dark, non-viable right ovary; salpingo-oophorectomy."),
    ),
    dict(
        case_id="CR16", label="other", group="testicular_torsion", time_critical=True, training_group="new",
        source=src("PMC12413555"),
        hpi=("25-year-old man, previously healthy, with sudden left lower abdominal pain and left groin swelling that "
             "started 4 hours ago: colicky, 9/10, localised to the left inguinal area, not radiating. No nausea, vomiting, "
             "change in bowel habit, fever, heavy lifting or trauma. Referred with a presumed incarcerated inguinal hernia."),
        physical_exam=("Vital signs normal; afebrile. Abdomen soft and not distended. Tender, irreducible swelling of about "
                       "3 x 3 cm in the left groin. Right testis normal. Left testis slightly elevated, with a thickened "
                       "spermatic cord palpable at its base, not directly tender."),
        labs=[lab("White Blood Cells", 10.8, "K/uL", 4.0, 11.0), lab("Hemoglobin", 16.2, "g/dL", 13.5, 17.5),
              lab("Platelet Count", 225, "K/uL", 150, 400)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis",
                     "No abdominal wall defect to suggest a hernia. Left non-communicating hydrocele with engorgement of the "
                     "left spermatic cord.")],
        note=("Scrotal Doppler ultrasound (outside the catalogue) showed a hypoechoic, avascular, abnormally oriented left "
              "testis. Exploration: torsion of the left spermatic cord; detorsion and bilateral orchidopexy."),
    ),
    dict(
        case_id="CR17", label="other", group="haemoperitoneum", time_critical=True, training_group="new",
        source=src("PMC10643335"),
        hpi=("Woman in her 30s, first pregnancy, at 32 weeks and 5 days, with sudden continuous dull abdominal pain, worst "
             "in the left lower abdomen, worse with activity and briefly eased by analgesia. No vaginal loss, fever, "
             "nausea, vomiting or urinary symptoms. Past history: severe endometriosis; laparoscopic cystectomy for a "
             "ruptured left endometrioma before this pregnancy, during work-up for in-vitro fertilisation."),
        physical_exam=("Vital signs within normal limits. Diffuse abdominal tenderness with guarding. No palpable uterine "
                       "contractions. Cardiotocography normal."),
        labs=[lab("Hemoglobin", 11.1, "g/dL", 11.5, 15.5), lab("White Blood Cells", 20.1, "K/uL", 4.0, 11.0),
              lab("Neutrophils", 90, "%", 40, 75), lab("C-Reactive Protein", 11, "mg/L", 0, 5),
              lab("Lactate", 1.2, "mmol/L", 0.5, 2.2), lab("Leukocytes", "NEG", None, fluid="Urine"),
              lab("Nitrite", "NEG", None, fluid="Urine"), lab("Blood", "NEG", None, fluid="Urine")],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen and pelvis",
                     "Bilateral hydronephrosis. Small pelvic fluid collection. Appendix not visualised."),
                 img("MRI", "Pelvis", "MRI pelvis",
                     "Complex left tubo-ovarian lesion of about 5.0 x 3.5 cm with features of a haemorrhagic ovarian cyst, "
                     "possibly endometriotic. Severe left hydronephrosis. Appendix normal.")],
        note=("Renal and liver tests normal (values not given). Haemoglobin fell to 9.0 g/dL at 24 h; at 36 h she became "
              "hypotensive (60/40 mmHg): spontaneous haemoperitoneum in pregnancy, managed by laparotomy."),
    ),
    dict(
        case_id="CR18", label="other", group="bowel_obstruction", time_critical=True, training_group="seen",
        source=src("PMC10237051"),
        hpi=("51-year-old woman with three days of worsening nausea, vomiting and abdominal pain, after several earlier "
             "episodes of nausea and abdominal pain. Past history: hyperlipidaemia, obesity, non-alcoholic fatty liver "
             "disease. No previous surgery."),
        physical_exam="Abdomen distended with hypoactive bowel sounds in all four quadrants; no guarding or rigidity.",
        labs=[lab("White Blood Cells", 14.42, "K/uL", 4.0, 11.0), lab("Neutrophils", 86.3, "%", 40, 75),
              lab("Lymphocytes", 7.6, "%", 20, 45), lab("Eosinophils", 0.5, "%", 1, 6),
              lab("Chloride", 88, "mEq/L", 98, 107), lab("Bicarbonate", 29, "mEq/L", 22, 28),
              lab("Glucose", 162, "mg/dL", 70, 100), lab("Urea Nitrogen", 45, "mg/dL", 7, 20),
              lab("Creatinine", 1.5, "mg/dL", 0.5, 1.1), lab("Alanine Aminotransferase (ALT)", 59, "IU/L", 0, 40),
              lab("Asparate Aminotransferase (AST)", 68, "IU/L", 0, 40),
              lab("Alkaline Phosphatase", 141, "IU/L", 40, 120), lab("Protein, Total", 9.0, "g/dL", 6.0, 8.3)],
        imaging=[img("Radiograph", "Abdomen", "Abdominal radiograph, upright",
                     "Moderately dilated small bowel loops with air-fluid levels; no gas in the colon."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with contrast",
                     "Gallbladder wall thickening with gas in the gallbladder lumen. Peripherally calcified stone in the "
                     "terminal ileum with dilated proximal small bowel and air-fluid levels. IMPRESSION: small bowel "
                     "obstruction at the terminal ileum by an impacted gallstone (gallstone ileus).")],
        note="Other CBC and chemistry values within normal limits (not given).",
    ),
    dict(
        case_id="CR19", label="other", group="haemoperitoneum", time_critical=True, training_group="new",
        source=src("PMC7504832"),
        hpi=("36-year-old woman, gravida 1 para 1, with one day of acute abdominal pain and vaginal bleeding. Regular "
             "30-day cycles; last menstrual period about 5 weeks ago."),
        physical_exam=("Blood pressure 110/60 mmHg, pulse 64/min, temperature 37.1 C, SpO2 99% on room air. Lower abdominal "
                       "tenderness. Small amount of vaginal bleeding on speculum examination."),
        labs=[lab("Hemoglobin", 10.1, "g/dL", 12, 16), lab("Hematocrit", 29.0, "%", 36, 48),
              lab("HCG, Urine, Qualitative", "POSITIVE", None, fluid="Urine"),
              lab("Human Chorionic Gonadotropin", 1149, "mIU/mL")],
        imaging=[img("Ultrasound", "Pelvis", "Transvaginal ultrasound",
                     "Large amount of complex fluid and mixed-echogenicity material in the pelvis, compatible with blood "
                     "and clot. Ovaries not well seen. No gestational sac in the endometrial cavity.")],
        note=("She became hypotensive (80/50 mmHg) in hospital; laparoscopy found 1.8 L of haemoperitoneum bleeding from "
              "endometriosis on the pouch of Douglas; tubes and ovaries normal (no ectopic pregnancy)."),
    ),
    # ---- the four in-set conditions, with the guideline grade the report supports ------------------------------
    dict(
        case_id="CR20", label="appendicitis", group="appendicitis", time_critical=True, training_group="seen",
        stage="AAST grade 4 / WSES complicated: perforation with subhepatic and pelvic collections", complicated=True,
        source=src("PMC7488453"),
        hpi=("41-year-old woman with no medical history and no medications; two vaginal deliveries and a caesarean section "
             "one year ago. Abdominal pain for 3 days, starting in the epigastrium, increasing, and now most marked in the "
             "right upper and lower quadrants. One episode of non-bilious vomiting; unable to tolerate oral intake. No "
             "smoking or alcohol."),
        physical_exam=("Blood pressure 90/40 mmHg (100 systolic after a 1 L fluid bolus), heart rate 112/min, temperature "
                       "38.2 C, clinically dehydrated. Alert, GCS 15, breathing unlaboured. Abdomen soft but tender in all "
                       "quadrants, most in the right upper and lower quadrants, with rebound tenderness and marked "
                       "percussion tenderness on the right."),
        labs=[lab("White Blood Cells", 11.8, "K/uL", 4.0, 11.0), lab("Hemoglobin", 12.5, "g/dL", 12, 16),
              lab("Platelet Count", 320, "K/uL", 150, 400), lab("Urea Nitrogen", 26, "mg/dL", 7, 20),
              lab("Creatinine", 0.75, "mg/dL", 0.5, 1.1), lab("Leukocytes", "NEG", None, fluid="Urine"),
              lab("Nitrite", "NEG", None, fluid="Urine"), lab("Blood", "NEG", None, fluid="Urine")],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen",
                     "Subhepatic intraperitoneal fluid collection. The appendix is not visualised."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with oral and intravenous contrast",
                     "Subhepatic appendix with a faecolith, perforated, with subhepatic and pelvic fluid collections. "
                     "IMPRESSION: perforated subhepatic appendicitis with collections.")],
        note="Liver tests and coagulation normal (values not given).",
    ),
    dict(
        case_id="CR21", label="appendicitis", group="appendicitis", time_critical=True, training_group="seen",
        stage="AAST grade 5 / WSES complicated: perforation with abscess and necrotising fasciitis of the abdominal wall",
        complicated=True, source=src("PMC7813934"),
        hpi=("81-year-old woman from a skilled nursing facility with 4 days of abdominal pain, nausea, vomiting (with a "
             "faeculent smell) and diarrhoea; no blood in vomit or stool. Past history: coronary artery disease, active "
             "left femoral deep vein thrombosis on warfarin, hypothyroidism, atrial fibrillation, type 2 diabetes."),
        physical_exam=("Ill-appearing. Heart rate 150/min in atrial fibrillation with rapid ventricular response, respiratory "
                       "rate 22/min, blood pressure 100/60 mmHg, temperature 37.5 C. Peritonitis with guarding in all four "
                       "quadrants, worst in the right lower quadrant. No palpable crepitus or skin changes."),
        labs=[lab("White Blood Cells", 19.6, "K/uL", 4.0, 11.0), lab("Hemoglobin", 9.0, "g/dL", 12, 16),
              lab("Platelet Count", 370, "K/uL", 150, 400), lab("INR(PT)", 2.6, None, 0.9, 1.1),
              lab("Creatinine", 1.56, "mg/dL", 0.5, 1.1), lab("Lactate", 4.0, "mmol/L", 0.5, 2.0)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis with intravenous contrast",
                     "9.0 x 4.8 cm collection of fluid and gas anterior to and encircling the urinary bladder. Gas tracks "
                     "into the anterior abdominal wall with extensive gas in multiple fascial layers and the subcutaneous "
                     "tissue. Thickened caecum and appendix. Bilateral hydronephrosis and hydroureter. IMPRESSION: "
                     "periappendiceal abscess with gas tracking into the abdominal wall, concerning for perforated "
                     "appendicitis with necrotising soft-tissue infection.")],
        note="Creatinine raised from baseline. At laparotomy: perforated appendix, necrotic transversalis fascia, bladder perforation.",
    ),
    dict(
        case_id="CR22", label="pancreatitis", group="pancreatitis", time_critical=True, training_group="seen",
        stage="Revised Atlanta severe: necrotising pancreatitis with respiratory and renal failure", complicated=True,
        source=src("PMC8047970"),
        hpi=("28-year-old man with 5 days of fever, dry cough, muscle aches and increasing breathlessness, and severe "
             "stabbing epigastric pain radiating to the back with frequent nausea and vomiting. No travel in the past month. "
             "No alcohol, drug use or medications; completely well a week ago."),
        physical_exam=("Dehydrated. Heart rate 118/min, blood pressure 135/100 mmHg, temperature 38.8 C, SpO2 77% on room "
                       "air. Abdomen soft and mildly distended with severe epigastric tenderness and rebound tenderness."),
        labs=[lab("White Blood Cells", 19.3, "K/uL", 4, 10), lab("Neutrophils", 85.3, "%", 42, 77),
              lab("Lymphocytes", 10.2, "%", 20, 45), lab("Monocytes", 4.5, "%", 2, 6),
              lab("Red Blood Cells", 6.09, "m/uL", 4.6, 6.2), lab("Hemoglobin", 18.3, "g/dL", 12, 16),
              lab("Hematocrit", 53.4, "%", 36, 52), lab("Platelet Count", 265, "K/uL", 130, 450),
              lab("Asparate Aminotransferase (AST)", 87, "IU/L", 0, 38),
              lab("Alanine Aminotransferase (ALT)", 66, "IU/L", 0, 40),
              lab("Alkaline Phosphatase", 227, "IU/L", 98, 279), lab("Bilirubin, Total", 0.9, "mg/dL", 0.1, 1.2),
              lab("Bilirubin, Direct", 0.2, "mg/dL", 0, 0.25), lab("Albumin", 3.9, "g/dL", 3.5, 5.5),
              lab("Amylase", 1273, "IU/L", 0, 90), lab("Lipase", 758, "IU/L", 0, 60),
              lab("Sodium", 136, "mEq/L", 135, 145), lab("Potassium", 4.2, "mEq/L", 3.5, 5.1),
              lab("Calcium, Total", 6.9, "mg/dL", 8.5, 10), lab("Triglycerides", 122, "mg/dL", 40, 160),
              lab("Urea Nitrogen", 47, "mg/dL", 7, 21), lab("Creatinine", 1.7, "mg/dL", 0.7, 1.4)],
        imaging=[img("CT", "Abdomen", "CT abdomen",
                     "Normal gallbladder and biliary tree. Severe inflammation of the fat around the pancreas and stomach; "
                     "perinephric fat stranding, more on the left. Heterogeneous pancreas with non-enhancing head and body. "
                     "Small amount of free fluid in the splenorenal space. IMPRESSION: acute necrotising pancreatitis.")],
        note=("Concurrent COVID-19 pneumonia: chest CT (outside the catalogue) showed bilateral ground-glass opacities and a "
              "small left pleural effusion."),
    ),
    dict(
        case_id="CR23", label="cholecystitis", group="cholecystitis", time_critical=True, training_group="seen",
        stage="Tokyo 2018 grade III: emphysematous cholecystitis with impaired consciousness and acute kidney injury",
        complicated=True, source=src("PMC8281795"),
        hpi=("63-year-old man brought by ambulance after a few days of stupor. Past history: heavy alcohol and drug use, "
             "previous pancreatitis, hypertension, gout."),
        physical_exam="Stuporous. Blood pressure 167/87 mmHg. Right upper quadrant tenderness.",
        labs=[],
        imaging=[img("Ultrasound", "Abdomen", "Right upper quadrant ultrasound",
                     "Moderately distended gallbladder with sludge and stones; air in the gallbladder fundus. IMPRESSION: "
                     "highly suggestive of emphysematous cholecystitis."),
                 img("CT", "Abdomen", "CT abdomen and pelvis without contrast",
                     "Distended gallbladder with an air-fluid level and multiple stones; pericholecystic fluid. IMPRESSION: "
                     "emphysematous cholecystitis.")],
        note=("Labs reported without values: anion-gap metabolic acidosis, rhabdomyolysis, hyponatraemia, raised liver "
              "enzymes, acute kidney injury with severe uraemia. Blood culture grew Fusobacterium nucleatum."),
    ),
    dict(
        case_id="CR24", label="cholecystitis", group="cholecystitis", time_critical=True, training_group="seen",
        stage="Tokyo 2018 grade II: gangrenous cholecystitis (gallbladder volvulus)", complicated=True,
        source=src("PMC9403365"),
        hpi=("87-year-old woman brought by ambulance with sudden, intense left-sided chest pain radiating to the back, and "
             "nausea without vomiting. Past history: hypertension, congestive heart failure, atrial fibrillation with a "
             "pacemaker, coronary artery disease after stenting of the circumflex artery."),
        physical_exam=("Afebrile, normotensive, not tachycardic, SpO2 97% on room air. Right upper quadrant tenderness with a "
                       "positive Murphy sign."),
        labs=[lab("White Blood Cells", 12.5, "K/uL", 4.0, 11.0), lab("Bilirubin, Total", 1.4, "mg/dL", 0.2, 1.2)],
        imaging=[img("Radiograph", "Chest", "Chest radiograph", "No acute findings."),
                 img("Ultrasound", "Abdomen", "Ultrasound abdomen",
                     "Distended gallbladder without stones, consistent with acalculous cholecystitis. Common bile duct "
                     "dilated to 1 cm."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with intravenous contrast",
                     "Distended, low-lying gallbladder with a thickened wall and pericholecystic fluid. IMPRESSION: acute "
                     "cholecystitis.")],
        note=("Other liver tests normal (values not given). ECG rate-controlled atrial fibrillation; troponin negative three "
              "times (outside the catalogue). At surgery: gangrenous gallbladder twisted 360 degrees (volvulus)."),
    ),
    dict(
        case_id="CR25", label="diverticulitis", group="diverticulitis", time_critical=True, training_group="seen",
        stage="Hinchey III / WSES stage 3-4: perforated sigmoid diverticulitis with purulent peritonitis", complicated=True,
        source=src("PMC11298154"),
        hpi=("31-year-old man with Williams syndrome (mild supravalvular aortic stenosis, hypothyroidism, raised blood "
             "pressure, intellectual disability) with 2 days of abdominal pain, mainly in the left iliac fossa, with "
             "diarrhoea, reduced appetite and fever up to 38.8 C."),
        physical_exam="Tender throughout the abdomen, with rebound tenderness in the left iliac fossa.",
        labs=[lab("White Blood Cells", 26.2, "K/uL", 4.0, 11.0), lab("C-Reactive Protein", 183.3, "mg/L", 0, 6)],
        imaging=[img("Radiograph", "Chest", "Chest radiograph", "Free gas under the diaphragm."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with contrast",
                     "Free intraperitoneal gas. Multiple diverticula of the descending and sigmoid colon with a "
                     "perisigmoid fluid collection of 6.4 x 3.9 x 5.2 cm at the site of rupture. IMPRESSION: perforated "
                     "sigmoid diverticulitis with peritonitis.")],
        note="Laparoscopy: sigmoid perforation with purulent peritonitis (Hinchey III); lavage and primary closure.",
    ),
    dict(
        case_id="CR26", label="diverticulitis", group="diverticulitis", time_critical=False, training_group="seen",
        stage="WSES stage 0 / Hinchey 0: uncomplicated (first visit)", complicated=False, source=src("PMC9495291"),
        hpi=("58-year-old man with obesity and type 2 diabetes, and one episode of uncomplicated diverticulitis two years "
             "ago, with one week of constant left lower quadrant abdominal pain. He has never had a colonoscopy."),
        physical_exam="Examination findings at this visit are not reported in the source.",
        labs=[],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis without contrast",
                     "Wall thickening and pericolic fat stranding of the sigmoid colon. No abscess or free air. "
                     "IMPRESSION: uncomplicated diverticulitis.")],
        note=("First emergency visit of the same patient as CR27. Leucocytosis (value not given). Discharged on oral "
              "ciprofloxacin and metronidazole."),
    ),
    dict(
        case_id="CR27", label="diverticulitis", group="diverticulitis", time_critical=True, training_group="seen",
        stage="WSES stage 1b-2a: diverticular abscess with small bowel obstruction (return visit 2 days later)",
        complicated=True, source=src("PMC9495291"),
        hpi=("58-year-old man with obesity and type 2 diabetes, and uncomplicated diverticulitis two years ago, who was seen "
             "here two days ago with a week of left lower quadrant pain, diagnosed with uncomplicated diverticulitis on CT "
             "and sent home on oral ciprofloxacin and metronidazole. He returns with worsening left lower quadrant pain."),
        physical_exam="Examination findings at this visit are not reported in the source.",
        labs=[lab("White Blood Cells", 23.0, "K/uL", 4.0, 11.0)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis without contrast (two days ago)",
                     "Wall thickening and pericolic fat stranding of the sigmoid colon. IMPRESSION: uncomplicated "
                     "diverticulitis."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with intravenous contrast (today)",
                     "Sigmoid wall thickening and fat stranding with a new encapsulated fluid collection, consistent with "
                     "abscess. Dilated proximal small bowel with decompressed distal small bowel and a transition point "
                     "near the abscess. IMPRESSION: complicated diverticulitis with abscess and small bowel obstruction.")],
        note="Return visit of the patient in CR26. Laparotomy: sigmoid inflammatory mass adherent to small bowel; sigmoid resection.",
    ),
    dict(
        case_id="CR28", label="diverticulitis", group="diverticulitis", time_critical=True, training_group="seen",
        stage="WSES stage 1a: localised perforation with septic thrombophlebitis of the portal system (pylephlebitis)",
        complicated=True, source=src("PMC8082042"),
        hpi=("69-year-old man with 6 days of dizziness, mild right upper quadrant pain and rigors. No urinary, "
             "gastrointestinal or respiratory symptoms. Past history: glaucoma only."),
        physical_exam="Afebrile. Abdomen soft and not distended. Lung fields clear.",
        labs=[],
        microbiology=[{"test_name": "Blood Culture, Routine", "specimen": "Blood",
                       "result": "Streptococcus species and Bacteroides species isolated."}],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen",
                     "Normal gallbladder without stones. No intra- or extrahepatic biliary dilatation. Liver smooth with "
                     "regular echotexture; normal flow in the main portal vein. Appendix, pancreas, spleen and kidneys "
                     "normal."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with contrast (later in the week)",
                     "Normal biliary tree. Acute sigmoid diverticulitis with tiny locules of extramural gas (localised "
                     "perforation) without abscess, with stranding of the peritoneal reflection, sigmoid mesentery and left "
                     "iliac fossa. Linear filling defects of thrombus in the inferior mesenteric vein to its confluence "
                     "with the splenic vein; further thrombus with gas locules in the main portal and splenic veins. "
                     "IMPRESSION: perforated sigmoid diverticulitis complicated by pylephlebitis.")],
        note=("Inflammatory markers raised and liver tests mildly cholestatic (values not given); initially treated as "
              "ascending cholangitis."),
    ),
    dict(
        case_id="CR29", label="diverticulitis", group="diverticulitis", time_critical=True, training_group="seen",
        stage="Perforated jejunal diverticulitis with mesenteric abscess (WSES-equivalent stage 1b)", complicated=True,
        source=src("PMC11460305"),
        hpi=("84-year-old woman with worsening left lower quadrant abdominal pain for 2 days, nausea and reduced appetite, "
             "like her previous episodes of diverticulitis. No fever, chills, diarrhoea, urinary symptoms, chest pain or "
             "breathlessness; last bowel movement yesterday. Past history: hypertension, reflux disease, diverticulitis, "
             "hip osteoarthritis, chronic back pain with an implanted spinal cord stimulator."),
        physical_exam=("Vital signs normal; afebrile, non-toxic, not in distress. Abdomen soft with mild left lower quadrant "
                       "tenderness and no peritonism."),
        labs=[lab("Lactate", 2.6, "mmol/L", 0.5, 2.0)],
        imaging=[img("CT", "Abdomen", "CT abdomen and pelvis with intravenous contrast",
                     "Thickened small bowel loop with surrounding inflammatory change and marked fat stranding, with a focus "
                     "of gas in the adjacent mesentery suggesting focal perforation. IMPRESSION: focal small bowel "
                     "perforation, likely of a diverticulum.")],
        note=("Other labs unremarkable (values not given); an inflammatory marker reported as 38 mm/hr (unit inconsistent; "
              "omitted). Laparotomy: perforated jejunal diverticulum 60 cm from the ligament of Treitz with a small "
              "mesenteric abscess. Labelled diverticulitis, as the CDM label covers diverticular disease of any segment (ICD "
              "K57)."),
    ),
    dict(
        case_id="CR30", label="pancreatitis", group="pancreatitis", time_critical=False, training_group="seen",
        stage="Revised Atlanta mild: interstitial oedematous pancreatitis without organ failure", complicated=False,
        source=src("PMC10241188"),
        hpi=("37-year-old man with insulin-dependent diabetes, congenital clubfoot and several past operations, with "
             "epigastric pain for a few days, now radiating to the back: dull, constant, worse with eating and drinking, "
             "not relieved by over-the-counter analgesics; nausea and vomiting. No fever; normal bowel movements. "
             "Medications: atorvastatin, pantoprazole, insulin glargine and lispro, and weekly dulaglutide, increased from "
             "0.75 to 1.5 mg two weeks ago, after which he had severe nausea and vomiting. Non-smoker, occasional alcohol, "
             "no drug use."),
        physical_exam="Examination findings are not reported in the source.",
        labs=[lab("Lipase", 1508, "IU/L", 0, 60), lab("Triglycerides", 69, "mg/dL", 0, 150)],
        imaging=[img("CT", "Abdomen", "CT abdomen without contrast",
                     "Peripancreatic fat stranding. IMPRESSION: acute pancreatitis."),
                 img("Ultrasound", "Abdomen", "Right upper quadrant ultrasound",
                     "No intrahepatic duct dilatation and no hepatic mass. Common bile duct 2 mm without stones. Gallbladder "
                     "distended without evidence of acute cholecystitis.")],
        note="Serum alcohol < 3 (outside the catalogue). Lipase fell to 439 U/L by day 5.",
    ),
    dict(
        case_id="CR31", label="cholecystitis", group="cholecystitis", time_critical=False, training_group="seen",
        stage="Tokyo 2018 grade I: acalculous cholecystitis without organ dysfunction or marked local inflammation",
        complicated=False, source=src("PMC7306929"),
        hpi=("68-year-old previously healthy woman in hospital for 10 days with confirmed COVID-19 pneumonia (admitted "
             "after 6 days of fever, chest tightness and diarrhoea), who now develops constant right upper quadrant "
             "abdominal pain and a new fever of 38.3 C."),
        physical_exam="Temperature 38.3 C. Right upper quadrant tenderness with a positive Murphy sign. Coarse breath sounds in both lungs.",
        labs=[lab("C-Reactive Protein", 90.1, "mg/L", 0, 10)],
        imaging=[img("CT", "Abdomen", "CT abdomen without contrast",
                     "Distended gallbladder with wall thickening and biliary sludge; no gallstones. IMPRESSION: acute "
                     "acalculous cholecystitis.")],
        note=("On admission (10 days earlier): WBC 3.64 x10^9/L, neutrophils 70.7%, ALT 19, AST 27 U/L, bilirubin 19.9 umol/L, "
              "CRP 33.7 mg/L. Treated with percutaneous transhepatic gallbladder drainage."),
    ),
    dict(
        case_id="CR32", label="appendicitis", group="appendicitis", time_critical=False, training_group="seen",
        stage="AAST grade 1 / WSES uncomplicated: left-sided appendicitis with intestinal malrotation", complicated=False,
        source=src("PMC7026876"),
        hpi=("32-year-old man without known illness with 2 days of insidious, progressive, constant pain in the left iliac "
             "fossa, not radiating, with nausea, vomiting and loss of appetite. No rectal bleeding, melaena or fever; "
             "normal bowel and bladder habits. No previous surgery. Another centre suspected diverticulitis."),
        physical_exam=("Tachycardic. Tenderness, rebound tenderness and guarding in the left iliac fossa. Normal digital rectal "
                       "examination."),
        labs=[],
        imaging=[img("Ultrasound", "Abdomen", "Ultrasound abdomen and pelvis",
                     "Non-compressible, blind-ending tubular structure in the left iliac fossa with surrounding echogenic "
                     "mesentery and probe tenderness; no free fluid."),
                 img("CT", "Abdomen", "CT abdomen and pelvis with contrast",
                     "Large bowel mainly on the left and small bowel mainly on the right; superior mesenteric artery to the "
                     "right of the vein; no duodenum between the SMA and aorta (complete midgut non-rotation). Appendix "
                     "arising from the caecum in the left iliac fossa, with a thickened wall, periappendiceal fat "
                     "stranding and two appendicoliths (9 x 7 and 6 x 5 mm). IMPRESSION: acute appendicitis with intestinal "
                     "malrotation.")],
    ),
]


def to_case(d: dict) -> dict:
    meta = {k: d[k] for k in ("group", "time_critical", "training_group", "source") if k in d}
    for k in ("stage", "complicated", "note"):
        if k in d:
            meta[k] = d[k]
    return {"case_id": d["case_id"], "label": d["label"], "hpi": d["hpi"], "history": {},
            "physical_exam": d["physical_exam"], "labs": d.get("labs", []), "microbiology": d.get("microbiology", []),
            "imaging": d.get("imaging", []), "subject_id": None, "source": "casereport", "meta": meta}


def main() -> None:
    rows = [to_case(d) for d in CASES]
    ids = [r["case_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    OUT.write_text(text, encoding="utf-8")
    QUEUE_COPY.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_COPY.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} and {QUEUE_COPY} ({len(rows)} cases)")


if __name__ == "__main__":
    main()
