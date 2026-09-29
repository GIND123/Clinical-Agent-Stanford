import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deferdx.env import EnvConfig, TestCatalog  # noqa: E402
from deferdx.rewards import RewardConfig, SeverityMatrix  # noqa: E402


@pytest.fixture(scope="session")
def catalog() -> TestCatalog:
    return TestCatalog.from_yaml(ROOT / "configs" / "test_catalog.yaml")


@pytest.fixture(scope="session")
def severity() -> SeverityMatrix:
    return SeverityMatrix.from_yaml(ROOT / "configs" / "severity_matrix.yaml")


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(max_steps=4, max_invalid=1)


@pytest.fixture
def reward_cfg(severity) -> RewardConfig:
    return RewardConfig(severity=severity)


@pytest.fixture(scope="session")
def synth():
    from deferdx.data.synthetic import synth_cases

    return synth_cases(n=60, n_other=12, seed=1)
