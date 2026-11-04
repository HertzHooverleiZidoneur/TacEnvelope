from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch


SIDES = ("left", "right")


def _recent_valid(values: Iterable[float], valid: Iterable[bool], count: int) -> list[float]:
    selected = [float(value) for value, keep in zip(values, valid, strict=True) if bool(keep)]
    return selected[-count:]


def stable_contact_from_samples(
    magnitudes: Iterable[float], valid: Iterable[bool], *, tau_on: float = 5.0,
    recent_samples: int = 4, required_samples: int = 3,
) -> bool:
    """Return an entry gate using only samples that precede the action anchor."""
    recent = _recent_valid(magnitudes, valid, recent_samples)
    if len(recent) < recent_samples:
        return False
    return sum(value > tau_on for value in recent) >= required_samples


def stable_contact_mask(
    history: torch.Tensor | np.ndarray,
    history_mask: torch.Tensor | np.ndarray,
    *,
    magnitude_index: int = 3,
    tau_on: float = 5.0,
    recent_samples: int = 4,
    required_samples: int = 3,
) -> np.ndarray:
    """Compute [N, 2] causal stable-contact gates from [N, 2, H, F] history."""
    values = np.asarray(history, dtype=np.float32)
    mask = np.asarray(history_mask, dtype=bool)
    if values.ndim == 3:
        values = values[None]
        mask = mask[None]
        squeeze = True
    elif values.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"history must be [2,H,F] or [N,2,H,F], got {values.shape}")
    if values.shape[:3] != mask.shape or values.shape[1] != 2:
        raise ValueError(f"history/mask shape mismatch: {values.shape} vs {mask.shape}")
    result = np.zeros((values.shape[0], 2), dtype=bool)
    for row in range(values.shape[0]):
        for side in range(2):
            result[row, side] = stable_contact_from_samples(
                values[row, side, :, magnitude_index], mask[row, side],
                tau_on=tau_on, recent_samples=recent_samples,
                required_samples=required_samples,
            )
    return result[0] if squeeze else result


@dataclass
class StableContactHysteresis:
    """Stateful online gate driven exclusively by past/current measured tactile samples."""

    tau_on: float = 5.0
    tau_off: float = 3.0
    recent_samples: int = 4
    required_samples: int = 3
    _history: dict[str, deque[float]] = field(init=False, repr=False)
    _active: dict[str, bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tau_off >= self.tau_on:
            raise ValueError("tau_off must be lower than tau_on for hysteresis")
        self.reset()

    def reset(self) -> None:
        self._history = {
            side: deque(maxlen=self.recent_samples) for side in SIDES
        }
        self._active = {side: False for side in SIDES}

    def update(self, side: str, force_magnitude: float, *, valid: bool = True) -> bool:
        if side not in self._history:
            raise KeyError(f"Unknown tactile side {side!r}")
        if valid and np.isfinite(force_magnitude):
            self._history[side].append(float(force_magnitude))
        recent = list(self._history[side])
        if len(recent) < self.recent_samples:
            return self._active[side]
        if self._active[side]:
            if sum(value < self.tau_off for value in recent) >= self.required_samples:
                self._active[side] = False
        elif sum(value > self.tau_on for value in recent) >= self.required_samples:
            self._active[side] = True
        return self._active[side]

    def update_pair(
        self, left_force: float, right_force: float, *,
        left_valid: bool = True, right_valid: bool = True,
    ) -> dict[str, bool]:
        return {
            "left": self.update("left", left_force, valid=left_valid),
            "right": self.update("right", right_force, valid=right_valid),
        }

    @property
    def active(self) -> dict[str, bool]:
        return dict(self._active)


def normalized_soft_cost(
    observed_force: float, upper_force: float, gate_active: bool, *,
    force_scale: float, cost_clip: float,
) -> float:
    """Return a gated convex penalty for force above the learned envelope."""
    if not gate_active:
        return 0.0
    if force_scale <= 0.0 or cost_clip < 0.0:
        raise ValueError("force_scale must be positive and cost_clip must be non-negative")
    exceedance = max(0.0, float(observed_force) - float(upper_force))
    normalized = float(np.clip(exceedance / force_scale, 0.0, cost_clip))
    return normalized**2


def normalized_band_cost(
    observed_force: float,
    lower_force: float,
    upper_force: float,
    gate_active: bool,
    *,
    force_scale: float,
    cost_clip: float,
) -> tuple[float, float]:
    """Return separate convex costs below and above a calibrated force band."""
    if lower_force > upper_force:
        raise ValueError("lower_force must not exceed upper_force")
    if not gate_active:
        return 0.0, 0.0
    if force_scale <= 0.0 or cost_clip < 0.0:
        raise ValueError("force_scale must be positive and cost_clip must be non-negative")
    low_deviation = max(0.0, float(lower_force) - float(observed_force))
    high_deviation = max(0.0, float(observed_force) - float(upper_force))
    low_normalized = float(np.clip(low_deviation / force_scale, 0.0, cost_clip))
    high_normalized = float(np.clip(high_deviation / force_scale, 0.0, cost_clip))
    return low_normalized**2, high_normalized**2


def hard_force_limit_exceeded(raw_force_max: float, hard_force_limit: float) -> bool:
    """Hard safety is deliberately independent of the learned model and gate."""
    return bool(np.isfinite(raw_force_max) and raw_force_max > hard_force_limit)
