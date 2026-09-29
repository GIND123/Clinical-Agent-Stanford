"""JSONL I/O and patient-level splitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import Case


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def save_cases(path: str | Path, cases: Iterable[Case]) -> int:
    return write_jsonl(path, (c.to_dict() for c in cases))


def load_cases(path: str | Path) -> list[Case]:
    return [Case.from_dict(d) for d in iter_jsonl(path)]


def _bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def assign_splits(
    cases: list[Case],
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> dict[str, str]:
    """Deterministic patient-level split: every admission of a subject lands in the
    same split. Returns {case_id: "train"|"val"|"test"}.

    Only use this when no official split file is available — for leaderboard
    comparability load the official split with `load_split_file` instead.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fractions}")
    train_cut = fractions[0]
    val_cut = fractions[0] + fractions[1]
    out = {}
    for case in cases:
        u = _bucket(case.subject_id or case.case_id, seed)
        out[case.case_id] = "train" if u < train_cut else ("val" if u < val_cut else "test")
    return out


def load_split_file(path: str | Path) -> dict[str, str]:
    """Read a split file: CSV/TSV with columns (case_id|hadm_id, split) or a JSON dict."""
    path = Path(path)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return {str(k): v for k, v in json.load(f).items()}
    import pandas as pd

    df = pd.read_csv(path, sep=None, engine="python")
    id_col = "case_id" if "case_id" in df.columns else "hadm_id"
    return {str(k): str(v) for k, v in zip(df[id_col], df["split"])}


def write_splits(out_dir: str | Path, cases: list[Case], splits: dict[str, str]) -> dict[str, int]:
    out_dir = Path(out_dir)
    buckets: dict[str, list[Case]] = {"train": [], "val": [], "test": []}
    for case in cases:
        split = splits.get(case.case_id)
        if split in buckets:
            buckets[split].append(case)
    counts = {name: save_cases(out_dir / f"{name}.jsonl", items) for name, items in buckets.items()}
    with (out_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=0)
    return counts
