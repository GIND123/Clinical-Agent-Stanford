import pytest

pytest.importorskip("sklearn")

from deferdx.data.leakage import source_classifier  # noqa: E402
from deferdx.data.parity import MASK  # noqa: E402
from deferdx.data.synthetic import synth_cases  # noqa: E402


def test_detects_pipeline_fingerprint_and_passes_clean_split():
    cases = [c for c in synth_cases(n=240, n_other=0, seed=3) if c.label == "appendicitis"]
    half = len(cases) // 2
    a, b = cases[:half], cases[half:]
    clean = source_classifier(a, b, folds=3)
    assert clean.auroc_mean < 0.7

    for c in a:  # simulate CDM-style masking present in one source only
        c.physical_exam += f" concern for {MASK}"
    leaky = source_classifier(a, b, folds=3)
    assert leaky.auroc_mean > 0.95
    assert leaky.top_features_a[0][0].startswith("__") or any("__" in f for f, _ in leaky.top_features_a[:3])
    assert "LEAK" in leaky.verdict()
