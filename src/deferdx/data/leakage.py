"""Source-classifier check: can a simple text model tell two case sets apart?

Run it on same-label cases from two pipelines, e.g. CDM appendicitis vs open-world
`--controls` appendicitis. An AUROC well above 0.5 means the pipelines leave
fingerprints (formatting, masks, section lengths) that an agent could exploit instead
of the disease, which would invalidate open-world results. The top features say what
the fingerprint is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Case


def case_text(case: Case, fields: tuple[str, ...] = ("hpi", "physical_exam", "imaging")) -> str:
    parts = []
    if "hpi" in fields:
        parts.append(case.hpi)
        parts.extend(case.history.values())
    if "physical_exam" in fields:
        parts.append(case.physical_exam)
    if "imaging" in fields:
        parts.extend(im.text for im in case.imaging)
    return "\n".join(p for p in parts if p)


@dataclass
class LeakageReport:
    auroc_mean: float
    auroc_std: float
    n_a: int
    n_b: int
    top_features_a: list[tuple[str, float]]
    top_features_b: list[tuple[str, float]]

    def verdict(self, threshold: float = 0.65) -> str:
        return ("LEAK: sources are separable" if self.auroc_mean > threshold
                else "ok: sources not easily separable")


def source_classifier(a: list[Case], b: list[Case], fields: tuple[str, ...] = ("hpi", "physical_exam", "imaging"),
                      folds: int = 5, seed: int = 0, top_k: int = 15) -> LeakageReport:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    texts = [case_text(c, fields) for c in a + b]
    y = np.array([1] * len(a) + [0] * len(b))
    # token_pattern keeps punctuation runs such as "____" as features
    vec_kw = dict(lowercase=True, token_pattern=r"[A-Za-z]{2,}|_{2,}|\d+", ngram_range=(1, 2), min_df=2,
                  sublinear_tf=True)
    aucs = []
    n_splits = max(2, min(folds, int(y.sum()), int((1 - y).sum())))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(texts, y):
        vec = TfidfVectorizer(**vec_kw)
        x_tr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]))
    vec = TfidfVectorizer(**vec_kw)
    x = vec.fit_transform(texts)
    coef = LogisticRegression(max_iter=2000, C=1.0).fit(x, y).coef_[0]
    names = vec.get_feature_names_out()
    order = np.argsort(coef)
    return LeakageReport(
        auroc_mean=float(np.mean(aucs)), auroc_std=float(np.std(aucs)), n_a=len(a), n_b=len(b),
        top_features_a=[(names[i], round(float(coef[i]), 3)) for i in order[::-1][:top_k]],
        top_features_b=[(names[i], round(float(coef[i]), 3)) for i in order[:top_k]],
    )
