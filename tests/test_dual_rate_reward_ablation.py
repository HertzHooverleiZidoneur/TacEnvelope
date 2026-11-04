from __future__ import annotations

import pytest

from tactile_force_envelope.dual_rate_reward_ablation import (
    analyze_paired_h9_windows,
    interval_score_scalar,
    normalized_band_cost,
)


def test_normalized_band_cost_is_gated_and_quadratic() -> None:
    assert normalized_band_cost(12.0, 5.0, 10.0, False, force_scale=5.0) == 0.0
    assert normalized_band_cost(12.0, 5.0, 10.0, True, force_scale=5.0) == pytest.approx(0.16)
    assert normalized_band_cost(3.0, 5.0, 10.0, True, force_scale=5.0) == pytest.approx(0.16)


def test_normalized_band_cost_clips_each_one_sided_violation() -> None:
    assert normalized_band_cost(30.0, 5.0, 10.0, True, force_scale=5.0) == 1.0


def test_interval_score_penalizes_both_tails() -> None:
    assert interval_score_scalar(4.0, 0.0, 3.0) == pytest.approx(23.0)
    assert interval_score_scalar(-1.0, 0.0, 3.0) == pytest.approx(23.0)


def test_paired_h9_counterfactual_counts_missed_and_false_penalties() -> None:
    rows = [
        {"episode": 1, "window_index": 10, "timestamp": 0.0, "observed": 5.0,
         "lower": 0.0, "upper": 10.0, "gate": True},
        {"episode": 1, "window_index": 11, "timestamp": 0.3, "observed": 25.0,
         "lower": 20.0, "upper": 30.0, "gate": True},
        {"episode": 1, "window_index": 12, "timestamp": 0.6, "observed": 15.0,
         "lower": 10.0, "upper": 20.0, "gate": True},
        {"episode": 1, "window_index": 13, "timestamp": 0.9, "observed": 15.0,
         "lower": 0.0, "upper": 10.0, "gate": True},
    ]

    report, transitions, episodes = analyze_paired_h9_windows(
        rows,
        task="test",
        source_file="memory",
        force_scale=5.0,
    )

    material = report["reward"]["single_vs_dual_window_2_material"]
    assert report["accepted_transitions"] == 2
    assert material["missed_penalty_windows"] == 1
    assert material["false_penalty_windows"] == 1
    assert report["window_2_envelope"]["fresh_coverage"] == pytest.approx(0.5)
    assert report["window_2_envelope"]["held_band_coverage"] == pytest.approx(0.5)
    assert len(transitions) == 2
    assert len(episodes) == 1
