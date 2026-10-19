from collections import deque

from deferdx.training.grpo_vllm import register_adapter


def test_resume_does_not_delete_the_adapter_just_saved(tmp_path):
    # an interrupted run left adapters for steps 36 and 37 (keep = 2); the resume restarts at step 36
    ad = tmp_path / "adapters"
    for s in (36, 37):
        (ad / f"step_{s:04d}").mkdir(parents=True)
    adapters = deque(sorted(ad.glob("step_*")))
    for s in (36, 37, 38):
        path = ad / f"step_{s:04d}"
        path.mkdir(exist_ok=True)  # save_pretrained
        register_adapter(adapters, path, 2)
        assert path.exists(), f"adapter for step {s} was deleted right after being saved"
    assert [p.name for p in adapters] == ["step_0037", "step_0038"]
    assert sorted(p.name for p in ad.iterdir()) == ["step_0037", "step_0038"]


def test_prunes_oldest_beyond_keep(tmp_path):
    adapters = deque()
    for s in range(1, 5):
        path = tmp_path / f"step_{s:04d}"
        path.mkdir()
        register_adapter(adapters, path, 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step_0003", "step_0004"]
