"""DEFER-Dx reward: commit reward with a proper scoring rule, and the group-consensus
deferral reward computed from GRPO's own rollout group.

COMMIT(d, p):
    R = alpha * 1[d = y] + lambda * S(p, 1[d = y]) - kappa * C[y, d] - cost_scale * sum cost(a_t)
    S = -(p - 1[d = y])^2                     (scoring_rule: brier, default)
    S = log(q) / -log(eps), q = p or 1 - p    (scoring_rule: log; LA-CDM / Rewarding Doubt,
                                               affinely rescaled to [-1, 0])
    Both are strictly proper: the expected reward is maximised by the true P(correct).

DEFER, in-set case:
    p_hat = fraction of the group's COMMIT rollouts on this case that were correct
    R = gamma * (tau - p_hat) - mu - cost_scale * sum cost(a_t)

DEFER, OTHER case:  R = openworld_defer_reward - mu - cost   (slightly above a correct
    COMMIT(OTHER), because "none of these" warrants escalation, stanford_idea §3.2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import load_yaml
from ..env.environment import EpisodeResult
from ..labels import ALL_LABELS, LABEL_INDEX, OTHER


class SeverityMatrix:
    def __init__(self, matrix: np.ndarray, labels: tuple[str, ...] = ALL_LABELS):
        if matrix.shape != (len(labels), len(labels)):
            raise ValueError(f"severity matrix must be {len(labels)}x{len(labels)}")
        if list(labels) != list(ALL_LABELS):
            order = [list(labels).index(x) for x in ALL_LABELS]
            matrix = matrix[np.ix_(order, order)]
        self.m = matrix.astype(float)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SeverityMatrix":
        d = load_yaml(path)
        labels = tuple(d["labels"])
        return cls(np.array([d["matrix"][lab] for lab in labels], dtype=float), labels)

    @classmethod
    def uniform(cls) -> "SeverityMatrix":
        n = len(ALL_LABELS)
        return cls(np.ones((n, n)) - np.eye(n))

    def __call__(self, y: str, d: str) -> float:
        return float(self.m[LABEL_INDEX[y], LABEL_INDEX[d]])


@dataclass
class RewardConfig:
    alpha: float = 1.0
    calib_lambda: float = 1.0
    scoring_rule: str = "brier"  # brier | log
    log_eps: float = 0.001  # probability floor for the log rule
    severity_kappa: float = 0.5
    # reward units per $ of investigation cost. Config value "auto" = alpha / (sum of all
    # catalog test costs): ordering every test costs one correct diagnosis (LA-CDM, App. C).
    cost_scale: float = 0.0005
    gamma: float = 2.0
    tau: float = 0.85
    handoff_mu: float = 0.1
    openworld_defer_reward: float = 1.2  # net of handoff_mu must exceed alpha
    invalid_penalty: float = 1.0
    timeout_penalty: float = 1.0
    # p_hat when no rollout in the group committed: "neutral" sets p_hat = tau so the
    # consensus term is 0 (deferral then costs mu); "zero" treats the case as hard.
    empty_group_phat: str = "neutral"
    # Group-level terminal-action entropy bonus (cold-start exploration, §5.4). 0 = off.
    action_entropy_coef: float = 0.0
    severity: SeverityMatrix = field(default_factory=SeverityMatrix.uniform, repr=False)

    @classmethod
    def from_dict(cls, d: dict | None, severity: SeverityMatrix | None = None,
                  total_test_cost: float | None = None) -> "RewardConfig":
        d = dict(d or {})
        d.pop("severity_matrix", None)
        if "brier_lambda" in d:  # backwards-compatible name
            d.setdefault("calib_lambda", d.pop("brier_lambda"))
        if d.get("cost_scale") == "auto":
            if not total_test_cost:
                raise ValueError("cost_scale: auto needs the catalog's total test cost")
            d["cost_scale"] = float(d.get("alpha", cls.alpha)) / total_test_cost
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "severity"})
        if severity is not None:
            cfg.severity = severity
        return cfg


@dataclass
class RewardBreakdown:
    total: float
    accuracy: float = 0.0
    calibration: float = 0.0
    severity: float = 0.0
    cost: float = 0.0
    consensus: float = 0.0
    handoff: float = 0.0
    penalty: float = 0.0
    constraint: float = 0.0
    exploration: float = 0.0
    p_hat: float | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def calibration_score(p: float, correct: float, cfg: RewardConfig) -> float:
    """Proper scoring rule in [-1, 0] (0 = perfect)."""
    if cfg.scoring_rule == "brier":
        return -((p - correct) ** 2)
    if cfg.scoring_rule == "log":
        q = p if correct else 1.0 - p
        return math.log(max(q, cfg.log_eps)) / -math.log(cfg.log_eps)
    raise ValueError(f"unknown scoring_rule {cfg.scoring_rule!r}")


def commit_reward(res: EpisodeResult, cfg: RewardConfig) -> RewardBreakdown:
    correct = float(res.diagnosis == res.label)
    p = float(res.probability if res.probability is not None else 0.0)
    acc = cfg.alpha * correct
    calib = cfg.calib_lambda * calibration_score(p, correct, cfg)
    sev = 0.0 if correct else -cfg.severity_kappa * cfg.severity(res.label, res.diagnosis)
    cost = -cfg.cost_scale * res.total_cost
    return RewardBreakdown(total=acc + calib + sev + cost, accuracy=acc, calibration=calib, severity=sev, cost=cost)


def defer_reward(res: EpisodeResult, p_hat: float, cfg: RewardConfig) -> RewardBreakdown:
    cost = -cfg.cost_scale * res.total_cost
    if res.label == OTHER:
        consensus = cfg.openworld_defer_reward
    else:
        consensus = cfg.gamma * (cfg.tau - p_hat)
    return RewardBreakdown(
        total=consensus - cfg.handoff_mu + cost, consensus=consensus, handoff=-cfg.handoff_mu, cost=cost, p_hat=p_hat
    )


def group_p_hat(group: list[EpisodeResult], cfg: RewardConfig) -> float:
    commits = [r for r in group if r.terminal == "commit"]
    if not commits:
        return cfg.tau if cfg.empty_group_phat == "neutral" else 0.0
    return sum(r.diagnosis == r.label for r in commits) / len(commits)


def group_rewards(group: list[EpisodeResult], cfg: RewardConfig, defer_penalty: float = 0.0) -> list[RewardBreakdown]:
    """Rewards for the G rollouts of ONE case.

    defer_penalty: the coverage-constraint multiplier (nu_k) from `CoverageConstraint`;
    it is charged to deferring rollouts only. (Subtracting a constant from every
    rollout would be a no-op after GRPO's per-group advantage normalisation.)
    """
    p_hat = group_p_hat(group, cfg)
    out = []
    for res in group:
        if res.terminal == "commit":
            rb = commit_reward(res, cfg)
            rb.p_hat = p_hat
        elif res.terminal == "defer":
            rb = defer_reward(res, p_hat, cfg)
            if defer_penalty:
                rb.constraint = -defer_penalty
                rb.total -= defer_penalty
        else:
            pen = cfg.timeout_penalty if res.terminal == "timeout" else cfg.invalid_penalty
            cost = -cfg.cost_scale * res.total_cost
            rb = RewardBreakdown(total=-pen + cost, penalty=-pen, cost=cost, p_hat=p_hat)
        out.append(rb)
    if cfg.action_entropy_coef > 0 and group:
        freq: dict[str, int] = {}
        for res in group:
            freq[res.terminal] = freq.get(res.terminal, 0) + 1
        for res, rb in zip(group, out):
            bonus = cfg.action_entropy_coef * -math.log(freq[res.terminal] / len(group))
            rb.exploration = bonus
            rb.total += bonus
    return out


def crossover_p_hat(cfg: RewardConfig, severity_cost: float | None = None, grid: int = 1001) -> float | None:
    """The case difficulty p_hat below which DEFER beats a calibrated COMMIT in expectation.

    Assumes a calibrated committer (p = p_hat), zero investigation cost, and a mean
    wrong-answer severity (default: mean off-diagonal in-set entry). tau is only the
    *nominal* dial; this is the operating threshold the reward actually induces.
    """
    if severity_cost is None:
        m = cfg.severity.m[:4, :4]
        severity_cost = float(m[~np.eye(4, dtype=bool)].mean())
    for q in np.linspace(0, 1, grid):
        commit = q * (cfg.alpha + cfg.calib_lambda * calibration_score(q, 1.0, cfg)) + (1 - q) * (
            cfg.calib_lambda * calibration_score(q, 0.0, cfg) - cfg.severity_kappa * severity_cost
        )
        defer = cfg.gamma * (cfg.tau - q) - cfg.handoff_mu
        if commit >= defer:
            return float(q)
    return None
