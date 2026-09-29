import pytest

pd = pytest.importorskip("pandas")
sklearn = pytest.importorskip("sklearn")

from deferdx.data.splits import lacdm_split, split_counts, stratified_split, subject_leakage  # noqa: E402

# CDM v1.1 class sizes
SIZES = {"appendicitis": 957, "cholecystitis": 648, "diverticulitis": 257, "pancreatitis": 538}


def fake_pathology_ids():
    ids, n = {}, 20000000
    for cond, k in SIZES.items():
        ids[cond] = [str(n + i) for i in range(k)]
        n += 100000
    return ids


def lacdm_reference(pathology_ids):
    """LA-CDM scripts/split_dataset.py, restated on DataFrames exactly as they do it."""
    from sklearn.model_selection import train_test_split

    dfs = []
    for cond in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
        info = {i: {"Label": cond} for i in pathology_ids[cond]}
        dfs.append(pd.DataFrame.from_dict(info, orient="index"))
    df_all = pd.concat(dfs).reset_index().rename(columns={"index": "Patient ID"})
    train_df, rem = train_test_split(df_all, test_size=0.2, stratify=df_all["Label"], random_state=269)
    val_df, test_df = train_test_split(rem, test_size=0.5, stratify=rem["Label"], random_state=269)
    return {str(i): s for s, d in (("train", train_df), ("val", val_df), ("test", test_df)) for i in d["Patient ID"]}


def test_lacdm_split_matches_reference():
    pids = fake_pathology_ids()
    assert lacdm_split(pids) == lacdm_reference(pids)


def test_lacdm_split_sizes():
    pids = fake_pathology_ids()
    splits = lacdm_split(pids)
    label_of = {i: c for c, ids in pids.items() for i in ids}
    counts = split_counts(splits, label_of)
    assert sum(counts["train"].values()) == 1920
    assert sum(counts["val"].values()) == 240 and sum(counts["test"].values()) == 240
    assert abs(counts["test"]["diverticulitis"] - 257 * 0.1) <= 1


def test_stratified_70_10_20_and_leakage():
    ids = [f"h{i}" for i in range(1000)]
    labels = ["appendicitis" if i % 4 else "diverticulitis" for i in range(1000)]
    splits = stratified_split(ids, labels, (0.7, 0.1, 0.2), seed=1)
    assert sorted(list(splits.values()).count(s) for s in ("train", "val", "test")) == [100, 200, 700]
    subject_of = {i: f"s{int(i[1:]) // 2}" for i in ids}  # two admissions per subject
    assert subject_leakage(splits, subject_of) > 0
    assert subject_leakage(splits, {}) == 0


def test_build_cdm_cli_lacdm(tmp_path):
    import json

    from deferdx.cli import main
    from deferdx.data.io import load_cases

    cdm = tmp_path / "cdm"
    cdm.mkdir()
    conds = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]
    hadm = list(range(100, 140))
    pd.DataFrame({"hadm_id": hadm, "hpi": ["abdominal pain"] * 40}).to_csv(cdm / "history_of_present_illness.csv", index=False)
    (cdm / "pathology_ids.json").write_text(json.dumps({c: hadm[i::4] for i, c in enumerate(conds)}))
    main(["data", "build-cdm", "--cdm-dir", str(cdm), "--out", str(tmp_path / "out")])
    sizes = {s: len(load_cases(tmp_path / "out" / f"{s}.jsonl")) for s in ("train", "val", "test")}
    assert sizes == {"train": 32, "val": 4, "test": 4}
    ref = lacdm_split({c: [str(x) for x in hadm[i::4]] for i, c in enumerate(conds)})
    assert {c.case_id for c in load_cases(tmp_path / "out" / "test.jsonl")} == {k for k, v in ref.items() if v == "test"}
