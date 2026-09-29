from deferdx.data.parity import MASK, cdm_history, cdm_physical_exam, cdm_radiology, mask, mentions

NOTE = (
    "Chief Complaint:\nRUQ pain\nHistory of Present Illness:\n55F with RUQ pain\nafter meals.\n"
    "Past Medical History:\nDM2\nPhysical Exam:\nVS: afebrile\nRUQ tender\nDISCHARGE PHYSICAL EXAM: benign\n"
    "Pertinent Results:\nWBC 14\nBrief Hospital Course:\n..."
)


def test_history_is_flattened_blob_up_to_exam():
    assert cdm_history(NOTE) == "55F with RUQ pain after meals. Past Medical History: DM2"
    assert cdm_history("no headers here") == ""


def test_physical_exam_window_and_discharge_cut():
    assert cdm_physical_exam(NOTE) == "VS: afebrile RUQ tender"


def test_radiology_drops_conclusion_sections():
    rep = "US ABDOMEN\nINDICATION: ?cholecystitis\nFINDINGS: gallstones, wall thickening\nIMPRESSION: acute cholecystitis"
    out = cdm_radiology(rep)
    assert "FINDINGS:\ngallstones, wall thickening" in out
    assert "IMPRESSION" not in out and "INDICATION" not in out
    assert cdm_radiology("IMPRESSION: normal") == ""


def test_masking_matches_cdm_marker():
    terms = ["acute cholecystitis", "cholecystitis", "cholecystostomy"]
    assert mask("Acute cholecystitis; s/p cholecystostomy", terms) == f"{MASK}; s/p {MASK}"
    assert mentions("h/o Cholecystitis", terms) and not mentions("RUQ pain", terms)
