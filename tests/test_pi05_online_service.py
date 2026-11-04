from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tactile_force_envelope.pi05_online_service import Pi05EnvelopeCostService


FEATURE_NAMES = [
    "sample_valid",
    "force_valid",
    "force_fx",
    "force_fy",
    "force_fz",
]


def sample(timestamp_ns: int, force_x: float) -> dict[str, Any]:
    return {
        "pc_time_ns": timestamp_ns,
        "feature_names": FEATURE_NAMES,
        "values": [1.0, 1.0, force_x, 0.0, 0.0],
    }


class FakeSource:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows

    def status(self) -> dict[str, Any]:
        return {"ok": True}

    def recent(self, start_ns: int, end_ns: int) -> dict[str, list[dict[str, Any]]]:
        return {
            side: [
                row
                for row in rows
                if start_ns <= int(row["pc_time_ns"]) < end_ns
            ]
            for side, rows in self.rows.items()
        }


class FakePredictor:
    condition_steps = 1
    condition_dim = 512
    action_steps = 9
    action_dim = 8
    history_steps = 6
    history_dim = 5
    control_window_ms = 300
    production_ready = False
    feature_metadata: dict[str, Any] = {}

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.last_actions = np.asarray(args[3], dtype=np.float32)
        self.last_tactile_history = np.asarray(
            kwargs["tactile_history"], dtype=np.float32
        )
        return {
            "mu_left": 6.0,
            "sigma_left": self.sigma,
            "upper95_left": 8.0,
            "mu_right": 6.0,
            "sigma_right": self.sigma,
            "upper95_right": 8.0,
        }


def request(anchor_ns: int, reward_count: int = 50) -> dict[str, Any]:
    return {
        "type": "evaluate_chunk",
        "policy_feature_source": "pi05_jax_z_rl",
        "policy_feature": np.zeros(512, dtype=np.float32).tolist(),
        "robot_state": np.zeros(8, dtype=np.float32).tolist(),
        "proposed_action_chunk": np.zeros((50, 8), dtype=np.float32).tolist(),
        "executed_action_chunk": np.ones((50, 8), dtype=np.float32).tolist(),
        "started_unix_ns": anchor_ns,
        "finished_unix_ns": anchor_ns + int(1.67e9),
        "reward_count": reward_count,
        "steps_executed": reward_count,
    }


def make_service(
    tmp_path: Path,
    rows: dict[str, list[dict[str, Any]]],
    *,
    sigma: float = 1.0,
) -> Pi05EnvelopeCostService:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    return Pi05EnvelopeCostService(
        FakePredictor(sigma),
        FakeSource(rows),
        {"left": np.zeros(3), "right": np.zeros(3)},
        checkpoint_path=checkpoint,
        action_fps=30.0,
        bilateral_lower_alpha=-2.0,
        bilateral_lower_coverage=0.05,
        bilateral_upper_alpha=2.0,
        bilateral_upper_coverage=0.95,
        bilateral_calibration={
            "lower_alpha_source": "calibration split",
            "upper_alpha_source": "calibration split",
            "calibration_samples": 580,
            "raw_peak_threshold": {"source": "train split maximum plus 10 N"},
        },
        raw_peak_threshold_N=100.0,
        force_scale=2.0,
        cost_clip=2.0,
    )


def test_cost_only_covers_first_300_ms_and_requires_prior_gate(tmp_path: Path) -> None:
    anchor = 1_000_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    future_offsets_ms = (10, 40, 70, 100)
    rows = {
        "left": prior + [
            sample(anchor + offset * 1_000_000, 10.0)
            for offset in future_offsets_ms
        ],
        "right": prior + [
            sample(anchor + offset * 1_000_000, 8.0)
            for offset in future_offsets_ms
        ],
    }
    result = make_service(tmp_path, rows).evaluate_chunk(request(anchor))
    costs = np.asarray(result["costs"])

    assert costs.shape == (50,)
    assert costs[8] == pytest.approx(0.25)
    assert np.count_nonzero(costs[:8]) == 0
    assert np.count_nonzero(costs[9:]) == 0
    assert result["metrics"]["gate_left"] == 1.0
    assert result["metrics"]["gate_right"] == 1.0
    assert result["metrics"]["gate_bilateral"] == 1.0
    assert result["metrics"]["observed_avg_N"] == pytest.approx(9.0)
    assert result["metrics"]["lower_avg_N"] == pytest.approx(4.0)
    assert result["metrics"]["upper_avg_N"] == pytest.approx(8.0)
    assert result["metrics"]["low_deviation_N"] == pytest.approx(0.0)
    assert result["metrics"]["high_deviation_N"] == pytest.approx(1.0)
    assert result["metrics"]["low_normalized_cost"] == pytest.approx(0.0)
    assert result["metrics"]["high_normalized_cost"] == pytest.approx(0.25)
    assert result["metrics"]["cost_index"] == 8.0


def test_contact_onset_does_not_penalize_current_window(tmp_path: Path) -> None:
    anchor = 2_000_000_000
    low_prior = [
        sample(anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    onset = [
        sample(anchor + offset_ms * 1_000_000, 12.0)
        for offset_ms in (10, 40, 70, 100)
    ]
    rows = {"left": low_prior + onset, "right": low_prior + onset}
    service = make_service(tmp_path, rows)
    result = service.evaluate_chunk(request(anchor))

    assert np.count_nonzero(result["costs"]) == 0
    assert result["metrics"]["gate_left"] == 0.0
    assert result["metrics"]["gate_right"] == 0.0
    assert service.gate.active == {"left": True, "right": True}


def test_pre_enqueue_prediction_uses_only_pre_anchor_history(tmp_path: Path) -> None:
    anchor = 3_000_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    future_contact = [
        sample(anchor + offset_ms * 1_000_000, 20.0)
        for offset_ms in (10, 40, 70, 100)
    ]
    service = make_service(
        tmp_path,
        {
            "left": prior + future_contact,
            "right": prior + future_contact,
        },
    )
    result = service.predict_chunk(
        {
            "type": "predict_chunk",
            "policy_feature_source": "pi05_jax_z_rl",
            "policy_feature": np.zeros(512, dtype=np.float32).tolist(),
            "robot_state": np.zeros(8, dtype=np.float32).tolist(),
            "proposed_action_chunk": np.zeros((50, 8), dtype=np.float32).tolist(),
            "anchor_unix_ns": anchor,
        }
    )

    assert result["prediction"]["stable_contact_left"] is False
    assert result["prediction"]["stable_contact_right"] is False
    assert result["history_report"]["left"]["history_samples"] == 4


def test_bilateral_gate_requires_both_sides(tmp_path: Path) -> None:
    anchor = 4_000_000_000
    high_prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    low_prior = [
        sample(anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 20.0)
        for offset_ms in (10, 40, 70, 100)
    ]
    result = make_service(
        tmp_path,
        {"left": high_prior + future, "right": low_prior + future},
    ).evaluate_chunk(request(anchor))

    assert np.count_nonzero(result["costs"]) == 0
    assert result["metrics"]["gate_left"] == 1.0
    assert result["metrics"]["gate_right"] == 0.0
    assert result["metrics"]["gate_bilateral"] == 0.0


def test_active_gate_marks_low_quality_future_window_without_soft_penalty(
    tmp_path: Path,
) -> None:
    anchor = 5_000_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    left_future = [
        sample(anchor + offset_ms * 1_000_000, 10.0)
        for offset_ms in (10, 80, 150)
    ]
    right_future = [
        sample(anchor + offset_ms * 1_000_000, 10.0)
        for offset_ms in (10, 70, 130, 190)
    ]
    rows = {"left": prior + left_future, "right": prior + right_future}

    result = make_service(tmp_path, rows).evaluate_chunk(request(anchor))

    assert result["metrics"]["gate_bilateral"] == 1.0
    assert result["metrics"]["quality_valid"] == 0.0
    assert result["metrics"]["low_quality_window"] == 1.0
    assert result["metrics"]["observed_samples_left"] == 3.0
    assert result["metrics"]["observed_samples_right"] == 4.0
    assert np.count_nonzero(result["average_costs"]) == 0


def test_envelope_uses_executed_not_proposed_actions(tmp_path: Path) -> None:
    anchor = 6_000_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 0.5)
        for offset_ms in (10, 40, 70, 100)
    ]
    service = make_service(
        tmp_path,
        {"left": prior + future, "right": prior + future},
    )

    service.evaluate_chunk(request(anchor))

    assert np.all(service.predictor.last_actions == 1.0)


def test_metadata_exposes_bilateral_reward_contract(tmp_path: Path) -> None:
    service = make_service(tmp_path, {"left": [], "right": []})

    metadata = service.metadata()["reward_envelope"]

    assert metadata["type"] == "bilateral_average_uncertainty_scaled_band"
    assert metadata["uncertainty_scale_method"] == "mean"
    assert metadata["scale_is_standard_deviation"] is False
    assert metadata["calibration_method"] == "standardized_residual_quantiles"
    assert metadata["lower_coverage"] == 0.05
    assert metadata["lower_alpha_label"] == "alpha05"
    assert metadata["lower_alpha"] == -2.0
    assert metadata["alpha05"] == -2.0
    assert metadata["upper_coverage"] == 0.95
    assert metadata["upper_alpha_label"] == "alpha95"
    assert metadata["upper_alpha"] == 2.0
    assert metadata["alpha95"] == 2.0
    assert metadata["calibration_samples"] == 580
    assert metadata["cost_power"] == 2
    assert metadata["cost_clip"] == 2.0
    peak = service.metadata()["raw_peak_penalty"]
    assert peak["type"] == "ungated_binary_threshold"
    assert peak["threshold_N"] == 100.0
    assert peak["stable_contact_gate_required"] is False
    assert service.metadata()["supported_reward_criteria"] == [
        "adaptive_envelope",
        "fixed_upper",
    ]


def test_low_force_below_l05_receives_quadratic_penalty(tmp_path: Path) -> None:
    anchor = 6_500_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 2.0)
        for offset_ms in (10, 40, 70, 100)
    ]

    result = make_service(
        tmp_path,
        {"left": prior + future, "right": prior + future},
    ).evaluate_chunk(request(anchor))

    assert result["metrics"]["lower_avg_N"] == pytest.approx(4.0)
    assert result["metrics"]["observed_avg_N"] == pytest.approx(2.0)
    assert result["metrics"]["low_deviation_N"] == pytest.approx(2.0)
    assert result["metrics"]["high_deviation_N"] == pytest.approx(0.0)
    assert result["metrics"]["low_normalized_cost"] == pytest.approx(1.0)
    assert result["metrics"]["high_normalized_cost"] == pytest.approx(0.0)
    assert result["average_costs"][8] == pytest.approx(1.0)
    assert result["metrics"]["low_cost_sum"] == pytest.approx(1.0)


def test_force_inside_calibrated_band_has_no_average_penalty(tmp_path: Path) -> None:
    anchor = 6_750_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 6.0)
        for offset_ms in (10, 40, 70, 100)
    ]

    result = make_service(
        tmp_path,
        {"left": prior + future, "right": prior + future},
    ).evaluate_chunk(request(anchor))

    assert result["metrics"]["lower_avg_N"] == pytest.approx(4.0)
    assert result["metrics"]["upper_avg_N"] == pytest.approx(8.0)
    assert result["metrics"]["observed_avg_N"] == pytest.approx(6.0)
    assert result["metrics"]["low_normalized_cost"] == pytest.approx(0.0)
    assert result["metrics"]["high_normalized_cost"] == pytest.approx(0.0)
    assert np.count_nonzero(result["average_costs"]) == 0


def test_fixed_upper_criterion_ignores_adaptive_mu_and_uses_train_threshold(
    tmp_path: Path,
) -> None:
    anchor = 6_900_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 66.0)
        for offset_ms in (10, 40, 70, 100)
    ]
    payload = request(anchor)
    payload["reward_criterion"] = "fixed_upper"
    payload["fixed_force_threshold_N"] = 65.0

    result = make_service(
        tmp_path,
        {"left": prior + future, "right": prior + future},
    ).evaluate_chunk(payload)

    assert result["metrics"]["reward_criterion_fixed_upper"] == 1.0
    assert result["metrics"]["fixed_force_threshold_N"] == 65.0
    assert result["metrics"]["lower_avg_N"] == 0.0
    assert result["metrics"]["upper_avg_N"] == 65.0
    assert result["metrics"]["low_normalized_cost"] == 0.0
    assert result["metrics"]["high_deviation_N"] == pytest.approx(1.0)
    assert result["metrics"]["high_normalized_cost"] == pytest.approx(0.25)
    assert result["average_costs"][8] == pytest.approx(0.25)


def test_incomplete_terminal_window_is_logged_without_penalty(tmp_path: Path) -> None:
    anchor = 7_000_000_000
    prior = [
        sample(anchor - offset_ms * 1_000_000, 6.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    future = [
        sample(anchor + offset_ms * 1_000_000, 20.0)
        for offset_ms in (10, 40)
    ]
    payload = request(anchor)
    payload["finished_unix_ns"] = anchor + 70_000_000
    payload["steps_executed"] = 2

    result = make_service(
        tmp_path,
        {"left": prior + future, "right": prior + future},
    ).evaluate_chunk(payload)

    assert np.count_nonzero(result["costs"]) == 0
    assert result["metrics"]["gate_bilateral"] == 1.0
    assert result["metrics"]["window_complete"] == 0.0
    assert result["metrics"]["quality_valid"] == 0.0


def test_raw_peak_penalty_is_ungated_and_keeps_single_sample_spike(
    tmp_path: Path,
) -> None:
    anchor = 8_000_000_000
    low_prior = [
        sample(anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    offsets = (10, 40, 70, 100)
    left_values = (1.0, 1.0, 160.0, 1.0)
    left_future = [
        sample(anchor + offset * 1_000_000, value)
        for offset, value in zip(offsets, left_values, strict=True)
    ]
    right_future = [
        sample(anchor + offset * 1_000_000, 1.0) for offset in offsets
    ]

    result = make_service(
        tmp_path,
        {
            "left": low_prior + left_future,
            "right": low_prior + right_future,
        },
    ).evaluate_chunk(request(anchor))

    assert result["metrics"]["gate_bilateral"] == 0.0
    assert result["metrics"]["raw_peak_N"] == pytest.approx(160.0)
    assert result["metrics"]["peak_triggered"] == 1.0
    assert np.flatnonzero(result["average_costs"]).tolist() == []
    assert np.flatnonzero(result["peak_costs"]).tolist() == [8]
    assert result["peak_costs"][8] == 1.0


def test_history_and_gate_advance_to_each_new_chunk_anchor(tmp_path: Path) -> None:
    first_anchor = 9_000_000_000
    second_anchor = first_anchor + 1_000_000_000
    first_prior = [
        sample(first_anchor - offset_ms * 1_000_000, 0.5)
        for offset_ms in (120, 90, 60, 30)
    ]
    first_future = [
        sample(first_anchor + offset_ms * 1_000_000, 0.5)
        for offset_ms in (10, 40, 70, 100)
    ]
    second_prior = [
        sample(second_anchor - offset_ms * 1_000_000, 7.0)
        for offset_ms in (120, 90, 60, 30)
    ]
    second_future = [
        sample(second_anchor + offset_ms * 1_000_000, 7.0)
        for offset_ms in (10, 40, 70, 100)
    ]
    rows = first_prior + first_future + second_prior + second_future
    service = make_service(tmp_path, {"left": rows, "right": rows})

    first = service.evaluate_chunk(request(first_anchor))
    second = service.evaluate_chunk(request(second_anchor))

    assert first["metrics"]["gate_bilateral"] == 0.0
    assert second["metrics"]["gate_bilateral"] == 1.0
    assert second["metrics"]["history_samples_left"] == 4.0
    assert second["metrics"]["history_samples_right"] == 4.0
    history = service.predictor.last_tactile_history
    np.testing.assert_allclose(history[:, -4:, 3], 7.0)
