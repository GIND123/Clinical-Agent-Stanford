"""Coverage floor against deferral collapse (Abstain-R1-style over-abstention).

Constraint: batch deferral rate <= rho_max. Lagrangian relaxation with projected dual
ascent on the multiplier nu:

    nu_{k+1} = max(0, nu_k + lr * (defer_rate_k - rho_max))       (after `patience`
                                                                    consecutive violations)

The penalty nu_k is charged per DEFERRING rollout (see `group_rewards`), which is the
per-sample gradient of nu * (mean 1[defer] - rho_max). While the constraint is satisfied
nu decays back towards 0, so the constraint stops distorting the reward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageConstraint:
    rho_max: float = 0.30
    lr: float = 2.0
    patience: int = 2  # consecutive violating batches before nu starts rising
    nu: float = 0.0
    nu_max: float = 5.0
    _violations: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "CoverageConstraint":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in ("rho_max", "lr", "patience", "nu", "nu_max")})

    def update(self, defer_rate: float) -> float:
        gap = defer_rate - self.rho_max
        if gap > 0:
            self._violations += 1
            if self._violations >= self.patience:
                self.nu = min(self.nu_max, self.nu + self.lr * gap)
        else:
            self._violations = 0
            self.nu = max(0.0, self.nu + self.lr * gap)
        return self.nu

    def state(self) -> dict:
        return {"nu": self.nu, "violations": self._violations, "rho_max": self.rho_max}
