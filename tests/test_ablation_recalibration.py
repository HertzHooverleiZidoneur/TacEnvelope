from __future__ import annotations

import numpy as np

from tactile_force_envelope.ablation_recalibration import (
    _interval_metrics,
    _selected_window_fingerprint,
)


def test_interval_width_quantiles_are_reported() -> None:
    target = np.asarray([1.0, 2.0, 3.0])
    mu = np.asarray([1.0, 2.5, 2.0])
    lower = np.asarray([0.0, 1.0, 1.0])
    upper = np.asarray([2.0, 4.0, 5.0])
    metrics = _interval_metrics(target, mu, lower, upper)
    assert metrics["mae_N"] == 0.5
    assert metrics["coverage"] == 1.0
    assert metrics["width_p05_N"] == np.quantile([2.0, 3.0, 4.0], 0.05)
    assert metrics["width_p50_N"] == 3.0
    assert metrics["width_p95_N"] == np.quantile([2.0, 3.0, 4.0], 0.95)


def test_selected_window_fingerprint_depends_only_on_selected_identity() -> None:
    rows = [
        {"batch_id": "a", "episode_index": 1, "window_index": 10},
        {"batch_id": "a", "episode_index": 2, "window_index": 20},
    ]
    first = _selected_window_fingerprint(rows, np.asarray([True, False]))
    rows[1]["unrelated"] = 123
    second = _selected_window_fingerprint(rows, np.asarray([True, False]))
    assert first == second

