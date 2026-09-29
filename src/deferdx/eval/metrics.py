"""Metric primitives: classification, selective prediction, calibration, OOD."""

from __future__ import annotations

import numpy as np


def per_class_accuracy(y_true: list[str], y_pred: list[str | None], labels: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for lab in labels:
        idx = [i for i, y in enumerate(y_true) if y == lab]
        if idx:
            out[lab] = float(np.mean([y_pred[i] == lab for i in idx]))
    return out


def macro_f1(y_true: list[str], y_pred: list[str | None], labels: tuple[str, ...]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        if tp + fp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def micro_f1(y_true: list[str], y_pred: list[str | None], labels: tuple[str, ...]) -> float:
    """Micro-F1 over `labels` (predictions outside `labels`, e.g. None/OTHER, count as FN)."""
    tp = sum(t == p and t in labels for t, p in zip(y_true, y_pred))
    fp = sum(p in labels and p != t for t, p in zip(y_true, y_pred))
    fn = sum(t in labels and p != t for t, p in zip(y_true, y_pred))
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending confidence; point k = accept the top-k cases.

    Ties are broken by original order (stable sort), which is the convention of
    Geifman & El-Yaniv; report ties honestly rather than sorting them favourably.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(scores)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-scores, kind="stable")
    errs = np.cumsum(1.0 - correct[order])
    k = np.arange(1, n + 1)
    return k / n, errs / k


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over coverage levels)."""
    _, risk = risk_coverage_curve(scores, correct)
    return float(np.mean(risk)) if len(risk) else float("nan")


def accuracy_at_coverage(scores: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    cov, risk = risk_coverage_curve(scores, correct)
    if not len(cov):
        return float("nan")
    k = max(1, int(np.ceil(coverage * len(cov))))
    return float(1.0 - risk[k - 1])


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - np.asarray(correct, dtype=float)) ** 2)) if len(probs) else float("nan")


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties. Positives should score HIGHER."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = (pred & target).sum()
    precision = tp / pred.sum() if pred.sum() else float("nan")
    recall = tp / target.sum() if target.sum() else float("nan")
    return float(precision), float(recall)
