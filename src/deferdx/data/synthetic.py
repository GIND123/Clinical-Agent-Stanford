"""Synthetic cases for pipeline development and tests.

Entirely fabricated: no MIMIC content. Findings are label-conditioned with noise and a
configurable fraction of deliberately ambiguous cases, so that accuracy, calibration and
deferral metrics have something non-trivial to measure. Never report results on this.
"""

from __future__ import annotations

import random

from ..labels import APPENDICITIS, CHOLECYSTITIS, DIVERTICULITIS, IN_SET_LABELS, OTHER, PANCREATITIS
from .schema import Case, ImagingReport, LabResult

_PRESENTATION = {
    APPENDICITIS: ("periumbilical pain migrating to the right lower quadrant", "anorexia and nausea"),
    CHOLECYSTITIS: ("right upper quadrant pain radiating to the right shoulder after a fatty meal", "nausea and vomiting"),
    DIVERTICULITIS: ("left lower quadrant pain for three days", "low-grade fever and change in bowel habit"),
    PANCREATITIS: ("severe epigastric pain radiating to the back", "persistent vomiting"),
}
_OTHER_PRESENTATIONS = [
    ("small bowel obstruction", "crampy diffuse abdominal pain with distension", "no flatus for two days",
     "dilated small bowel loops with a transition point"),
    ("nephrolithiasis", "colicky left flank pain radiating to the groin", "hematuria",
     "7 mm obstructing stone at the left ureterovesical junction with hydronephrosis"),
    ("perforated peptic ulcer", "sudden severe epigastric pain", "rigid abdomen",
     "free intraperitoneal air"),
    ("mesenteric ischemia", "pain out of proportion to examination", "history of atrial fibrillation",
     "occlusion of the superior mesenteric artery"),
    ("gastroenteritis", "diffuse crampy abdominal pain", "watery diarrhea",
     "fluid-filled bowel without focal abnormality"),
]
_CT_FINDING = {
    APPENDICITIS: "dilated appendix measuring 11 mm with periappendiceal fat stranding",
    CHOLECYSTITIS: "distended gallbladder with wall thickening and pericholecystic fluid",
    DIVERTICULITIS: "sigmoid diverticula with adjacent fat stranding and bowel wall thickening",
    PANCREATITIS: "peripancreatic fat stranding and edema of the pancreas",
}
_US_FINDING = {
    CHOLECYSTITIS: "gallstones, gallbladder wall thickening to 5 mm, positive sonographic Murphy sign",
    APPENDICITIS: "non-compressible blind-ending tubular structure in the right lower quadrant",
}
_EXAM = {
    APPENDICITIS: "Tender at McBurney's point with guarding. Rovsing sign positive.",
    CHOLECYSTITIS: "Right upper quadrant tenderness, Murphy sign positive.",
    DIVERTICULITIS: "Left lower quadrant tenderness without peritoneal signs.",
    PANCREATITIS: "Epigastric tenderness, mild distension, voluntary guarding.",
}


def _lab(name, value, unit, lo, hi, fluid="Blood"):
    flag = "abnormal" if (lo is not None and value < lo) or (hi is not None and value > hi) else None
    return LabResult(name=name, value=f"{value:g}", unit=unit, ref_low=lo, ref_high=hi, flag=flag, fluid=fluid)


def _labs(label: str, rng: random.Random, ambiguous: bool) -> list[LabResult]:
    wbc = rng.gauss(14 if label in (APPENDICITIS, CHOLECYSTITIS, DIVERTICULITIS) else 11, 3)
    lipase = rng.gauss(900, 300) if label == PANCREATITIS and not ambiguous else rng.gauss(40, 15)
    alp = rng.gauss(180, 40) if label == CHOLECYSTITIS and not ambiguous else rng.gauss(85, 20)
    tbili = rng.gauss(2.1, 0.6) if label == CHOLECYSTITIS and not ambiguous else rng.gauss(0.6, 0.2)
    crp = rng.gauss(90, 30) if label != OTHER else rng.gauss(30, 20)
    labs = [
        _lab("White Blood Cells", round(max(wbc, 2.0), 1), "K/uL", 4.0, 11.0),
        _lab("Hemoglobin", round(rng.gauss(13.5, 1.2), 1), "g/dL", 12.0, 16.0),
        _lab("Platelet Count", round(rng.gauss(250, 50)), "K/uL", 150, 440),
        _lab("Sodium", round(rng.gauss(139, 2)), "mEq/L", 133, 145),
        _lab("Potassium", round(rng.gauss(4.1, 0.3), 1), "mEq/L", 3.3, 5.1),
        _lab("Creatinine", round(max(rng.gauss(0.9, 0.2), 0.3), 2), "mg/dL", 0.5, 1.1),
        _lab("Glucose", round(rng.gauss(110, 20)), "mg/dL", 70, 100),
        _lab("Lipase", round(max(lipase, 5)), "IU/L", 0, 60),
        _lab("Alkaline Phosphatase", round(max(alp, 20)), "IU/L", 35, 105),
        _lab("Alanine Aminotransferase (ALT)", round(max(rng.gauss(30, 10), 5)), "IU/L", 0, 40),
        _lab("Bilirubin, Total", round(max(tbili, 0.1), 1), "mg/dL", 0.0, 1.5),
        _lab("C-Reactive Protein", round(max(crp, 1)), "mg/L", 0, 5),
        _lab("Lactate", round(max(rng.gauss(1.4, 0.5), 0.4), 1), "mmol/L", 0.5, 2.0),
    ]
    if rng.random() < 0.6:
        labs.append(_lab("Leukocytes", 0, "", None, None, fluid="Urine"))
    # Missingness: not every test exists for every patient.
    return [x for x in labs if rng.random() > 0.08]


def synth_case(idx: int, label: str, rng: random.Random, ambiguous_rate: float = 0.2) -> Case:
    ambiguous = rng.random() < ambiguous_rate
    age = rng.randint(19, 88)
    sex = rng.choice(["male", "female"])
    imaging: list[ImagingReport] = []
    if label == OTHER:
        other_dx, pain, extra, ct = rng.choice(_OTHER_PRESENTATIONS)
        hpi = f"{age}-year-old {sex} presenting with {pain} and {extra}."
        exam = "Diffusely tender abdomen." if rng.random() < 0.5 else "Mild tenderness, no rebound."
        ct_text = f"FINDINGS: {ct}. Appendix normal. Gallbladder unremarkable. IMPRESSION: {ct}."
        meta = {"other_group": other_dx, "synthetic": True}
    else:
        pain, extra = _PRESENTATION[label]
        if ambiguous:
            decoy = rng.choice([x for x in IN_SET_LABELS if x != label])
            pain = _PRESENTATION[decoy][0]
        hpi = f"{age}-year-old {sex} presenting with {pain}, associated with {extra}."
        exam = _EXAM[label] if not ambiguous else "Diffuse abdominal tenderness, difficult to localise."
        finding = _CT_FINDING[label]
        if ambiguous:
            finding = "nonspecific inflammatory changes; limited evaluation due to motion artifact"
        ct_text = f"FINDINGS: {finding}. IMPRESSION: {finding}."
        meta = {"ambiguous": ambiguous, "synthetic": True}
    if rng.random() < 0.85:
        imaging.append(ImagingReport("CT", "Abdomen", "CT ABD & PELVIS WITH CONTRAST", ct_text))
    if label in _US_FINDING and rng.random() < 0.6:
        us = _US_FINDING[label] if not ambiguous else "limited study; no definite abnormality"
        imaging.append(ImagingReport("Ultrasound", "Abdomen", "US ABDOMEN", f"FINDINGS: {us}."))
    if rng.random() < 0.3:
        imaging.append(ImagingReport("Radiograph", "Chest", "CHEST (PA AND LAT)", "No acute cardiopulmonary process."))
    return Case(
        case_id=f"syn{idx:05d}",
        subject_id=f"sub{idx:05d}",
        label=label,
        hpi=hpi,
        history={
            "past_medical_history": rng.choice(["Hypertension.", "Type 2 diabetes.", "None.", "Hyperlipidemia."]),
            "medications": rng.choice(["Lisinopril.", "Metformin.", "None."]),
            "social_history": rng.choice(["Social alcohol use.", "Heavy alcohol use.", "Non-smoker."]),
            "family_history": "Non-contributory.",
        },
        physical_exam=f"Vitals: T {rng.gauss(37.8, 0.5):.1f} C, HR {rng.randint(70, 115)}. {exam}",
        labs=_labs(label, rng, ambiguous),
        imaging=imaging,
        source="synthetic",
        meta=meta,
    )


# Class balance mirrors MIMIC-IV-Ext-CDM (957/648/257/538).
CDM_PRIORS = {APPENDICITIS: 957, CHOLECYSTITIS: 648, DIVERTICULITIS: 257, PANCREATITIS: 538}


def synth_cases(n: int = 400, n_other: int = 80, seed: int = 0, ambiguous_rate: float = 0.2) -> list[Case]:
    rng = random.Random(seed)
    labels = rng.choices(list(CDM_PRIORS), weights=list(CDM_PRIORS.values()), k=n) + [OTHER] * n_other
    rng.shuffle(labels)
    return [synth_case(i, lab, rng, ambiguous_rate) for i, lab in enumerate(labels)]
