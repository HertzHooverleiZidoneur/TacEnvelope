from __future__ import annotations

import numpy as np
import pytest

from tactile_force_envelope.online_rl import (
    PI05_JAX_FEATURE_SOURCE,
    SocketTactileCostEvaluator,
    combine_tactile_penalties,
    validate_server_metadata,
)
from rlt_openpi.rollout.reward_provider import ChunkExecutionContext


def test_pi05_metadata_is_accepted() -> None:
    validate_server_metadata(
        {
            "schema_version": 2,
            "policy_feature_source": PI05_JAX_FEATURE_SOURCE,
            "policy_feature_dim": 512,
            "reward_envelope": {
                "type": "bilateral_average_uncertainty_scaled_band",
                "calibration_method": "standardized_residual_quantiles",
                "uncertainty_scale_method": "mean",
                "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
                "scale_is_standard_deviation": False,
                "lower_coverage": 0.05,
                "lower_alpha_label": "alpha05",
                "lower_alpha": -1.7,
                "upper_coverage": 0.95,
                "upper_alpha_label": "alpha95",
                "upper_alpha": 1.6,
                "cost_power": 2,
                "cost_clip": 1.0,
            },
            "raw_peak_penalty": {"type": "ungated_binary_threshold"},
        },
        expected_feature_source=PI05_JAX_FEATURE_SOURCE,
        expected_feature_dim=512,
    )


def test_smolvla_metadata_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="SmolVLA"):
        validate_server_metadata(
            {
                "schema_version": 2,
                "policy_feature_source": "smolvla_condition_tokens",
                "policy_feature_dim": 960,
                "reward_envelope": {
                    "type": "bilateral_average_uncertainty_scaled_band",
                    "cost_power": 2,
                    "cost_clip": 2.0,
                },
                "raw_peak_penalty": {"type": "ungated_binary_threshold"},
            },
            expected_feature_source=PI05_JAX_FEATURE_SOURCE,
            expected_feature_dim=512,
        )


def test_fixed_criterion_requires_updated_sidecar_contract() -> None:
    with pytest.raises(RuntimeError, match="does not support reward criterion"):
        validate_server_metadata(
            {
                "schema_version": 2,
                "policy_feature_source": PI05_JAX_FEATURE_SOURCE,
                "policy_feature_dim": 512,
                "reward_envelope": {
                    "type": "bilateral_average_uncertainty_scaled_band",
                    "calibration_method": "standardized_residual_quantiles",
                    "uncertainty_scale_method": "mean",
                    "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
                    "scale_is_standard_deviation": False,
                    "lower_coverage": 0.05,
                    "lower_alpha_label": "alpha05",
                    "lower_alpha": -1.7,
                    "upper_coverage": 0.95,
                    "upper_alpha_label": "alpha95",
                    "upper_alpha": 1.6,
                    "cost_power": 2,
                    "cost_clip": 1.0,
                },
                "raw_peak_penalty": {"type": "ungated_binary_threshold"},
            },
            expected_feature_source=PI05_JAX_FEATURE_SOURCE,
            expected_feature_dim=512,
            required_reward_criterion="fixed_upper",
        )


def test_combined_reward_deduction_keeps_two_terms_separate() -> None:
    result = combine_tactile_penalties(
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        average_penalty_weight=0.4,
        peak_penalty=1.0,
        total_penalty_cap=1.0,
    )

    assert result.tolist() == pytest.approx([0.0, 0.2, 1.0])


def test_combined_reward_deduction_caps_soft_and_peak_sum() -> None:
    result = combine_tactile_penalties(
        [1.0, 1.0],
        [0.0, 1.0],
        average_penalty_weight=0.4,
        peak_penalty=1.0,
        total_penalty_cap=1.0,
    )

    assert result.tolist() == pytest.approx([0.4, 1.0])


def test_component_budget_stops_at_its_episode_cap() -> None:
    applied, cap_hit = SocketTactileCostEvaluator._limit_component(
        0.35,
        used=0.4,
        cap=0.6,
    )

    assert applied == pytest.approx(0.2)
    assert cap_hit == 1.0


def test_dual_rate_h14_uses_one_h9_penalty() -> None:
    evaluator = SocketTactileCostEvaluator.__new__(SocketTactileCostEvaluator)
    evaluator.feature_dim = 4
    evaluator.average_penalty_weight = 1.0
    evaluator.peak_penalty = 1.0
    evaluator.soft_episode_cap = 1.0
    evaluator.peak_episode_cap = 1.0
    evaluator._soft_episode_penalty = 0.0
    evaluator._peak_episode_penalty = 0.0
    evaluator.total_penalty_cap = 1.0
    evaluator.tactile_window_steps = 9
    evaluator.action_fps = 30.0
    evaluator._prepared_windows = {}
    requests: list[dict] = []

    def request(payload: dict) -> dict:
        requests.append(payload)
        if payload["type"] == "predict_chunk":
            index = len([item for item in requests if item["type"] == "predict_chunk"]) - 1
            return {
                "prediction": {
                    "mu_left_N": float(index),
                    "sigma_left_N": 1.0,
                    "upper_left_N": 2.0,
                    "stable_contact_left": True,
                    "mu_right_N": float(index),
                    "sigma_right_N": 1.0,
                    "upper_right_N": 2.0,
                    "stable_contact_right": True,
                },
                "history_report": {"valid": 4},
            }
        if payload["type"] == "evaluate_chunk":
            window = len([item for item in requests if item["type"] == "evaluate_chunk"])
            penalty = 0.2 if window == 1 else 0.6
            return {
                "average_costs": [penalty] + [0.0] * 8,
                "peak_costs": [0.0] * 9,
                "metrics": {"cached_prediction": 1.0},
            }
        raise AssertionError(payload)

    evaluator._request = request
    initial_observation = {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
    }
    executed = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    observer = evaluator.prepare_execution(
        observation=initial_observation,
        rl_state=np.arange(4, dtype=np.float32),
        reference_action_chunk=executed,
        proposed_action_chunk=executed,
    )
    observer(
        window_index=0,
        anchor_unix_ns=1_000_000_000,
        observation=None,
        executed_action_chunk=executed,
        action_fps=30.0,
    )
    zeros = np.zeros(14, dtype=np.float32)
    result = evaluator.evaluate_chunk(
        ChunkExecutionContext(
            observation=initial_observation,
            next_observation=initial_observation,
            rl_state=np.arange(4, dtype=np.float32),
            reference_action_chunk=executed,
            proposed_action_chunk=executed,
            executed_action_chunk=executed,
            task_rewards=zeros,
            done=False,
            info={"steps_executed": 14, "tactile_windows_started": 1},
            started_unix_ns=1_000_000_000,
            finished_unix_ns=1_466_666_667,
        )
    )

    assert result.costs.sum() == pytest.approx(0.2)
    assert result.costs[8] == pytest.approx(0.2)
    assert result.metrics["tactile_window_1_penalty"] == pytest.approx(0.2)
    assert "tactile_window_2_penalty" not in result.metrics
    assert result.metrics["tactile_window_penalty_mean"] == pytest.approx(0.2)
    predictions = [item for item in requests if item["type"] == "predict_chunk"]
    np.testing.assert_array_equal(
        np.asarray(predictions[0]["proposed_action_chunk"]), executed[:9]
    )
    assert predictions[0]["robot_state"] == pytest.approx([0, 1, 2, 3, 4, 5, 6, 0.5])


def test_dual_rate_h18_averages_two_h9_penalties() -> None:
    evaluator = SocketTactileCostEvaluator.__new__(SocketTactileCostEvaluator)
    evaluator.feature_dim = 4
    evaluator.average_penalty_weight = 1.0
    evaluator.peak_penalty = 1.0
    evaluator.soft_episode_cap = 1.0
    evaluator.peak_episode_cap = 1.0
    evaluator._soft_episode_penalty = 0.0
    evaluator._peak_episode_penalty = 0.0
    evaluator.total_penalty_cap = 1.0
    evaluator.tactile_window_steps = 9
    evaluator.action_fps = 30.0
    evaluator._prepared_windows = {}
    requests: list[dict] = []

    def request(payload: dict) -> dict:
        requests.append(payload)
        if payload["type"] == "predict_chunk":
            return {
                "prediction": {
                    "mu_left_N": 1.0,
                    "sigma_left_N": 1.0,
                    "upper_left_N": 2.0,
                    "stable_contact_left": True,
                    "mu_right_N": 1.0,
                    "sigma_right_N": 1.0,
                    "upper_right_N": 2.0,
                    "stable_contact_right": True,
                },
                "history_report": {"valid": 4},
            }
        if payload["type"] == "evaluate_chunk":
            index = len([item for item in requests if item["type"] == "evaluate_chunk"])
            penalty = 0.2 if index == 1 else 0.6
            return {
                "average_costs": [penalty] + [0.0] * 8,
                "peak_costs": [0.0] * 9,
                "metrics": {"cached_prediction": 1.0},
            }
        raise AssertionError(payload)

    evaluator._request = request
    initial_observation = {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
    }
    midpoint_observation = {
        "observation/joint_position": np.arange(7, dtype=np.float32) + 10,
        "observation/gripper_position": np.array([0.25], dtype=np.float32),
    }
    executed = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    observer = evaluator.prepare_execution(
        observation=initial_observation,
        rl_state=np.arange(4, dtype=np.float32),
        reference_action_chunk=executed,
        proposed_action_chunk=executed,
    )
    observer(
        window_index=0,
        anchor_unix_ns=1_000_000_000,
        observation=None,
        executed_action_chunk=executed,
        action_fps=30.0,
    )
    observer(
        window_index=1,
        anchor_unix_ns=1_300_000_000,
        observation=midpoint_observation,
        executed_action_chunk=executed,
        action_fps=30.0,
    )
    result = evaluator.evaluate_chunk(
        ChunkExecutionContext(
            observation=initial_observation,
            next_observation=midpoint_observation,
            rl_state=np.arange(4, dtype=np.float32),
            reference_action_chunk=executed,
            proposed_action_chunk=executed,
            executed_action_chunk=executed,
            task_rewards=np.zeros(50, dtype=np.float32),
            done=False,
            info={"steps_executed": 18, "tactile_windows_started": 2},
            started_unix_ns=1_000_000_000,
            finished_unix_ns=1_600_000_000,
        )
    )

    predictions = [item for item in requests if item["type"] == "predict_chunk"]
    assert len(predictions) == 2
    np.testing.assert_array_equal(
        np.asarray(predictions[0]["proposed_action_chunk"]), executed[:9]
    )
    np.testing.assert_array_equal(
        np.asarray(predictions[1]["proposed_action_chunk"]), executed[9:18]
    )
    assert predictions[1]["robot_state"] == pytest.approx(
        [10, 11, 12, 13, 14, 15, 16, 0.25]
    )
    assert result.metrics["tactile_subwindows"] == 2.0
    assert result.metrics["tactile_window_1_penalty"] == pytest.approx(0.2)
    assert result.metrics["tactile_window_2_penalty"] == pytest.approx(0.6)
    assert result.metrics["tactile_window_penalty_mean"] == pytest.approx(0.4)
    assert result.costs.sum() == pytest.approx(0.4)
    assert result.costs[17] == pytest.approx(0.4)


def test_dual_rate_uses_only_windows_that_really_started() -> None:
    evaluator = SocketTactileCostEvaluator.__new__(SocketTactileCostEvaluator)
    evaluator.tactile_window_steps = 9
    evaluator._prepared_windows = {}
    context = ChunkExecutionContext(
        observation={},
        next_observation={},
        rl_state=np.zeros(1, dtype=np.float32),
        reference_action_chunk=np.zeros((50, 8), dtype=np.float32),
        proposed_action_chunk=np.zeros((50, 8), dtype=np.float32),
        executed_action_chunk=np.zeros((50, 8), dtype=np.float32),
        task_rewards=np.zeros(18, dtype=np.float32),
        done=True,
        info={"steps_executed": 18, "tactile_windows_started": 0},
        started_unix_ns=0,
        finished_unix_ns=0,
    )

    result = evaluator.evaluate_chunk(context)

    assert result.metrics["tactile_subwindows"] == 0.0
    assert result.costs.sum() == 0.0


def test_dual_rate_reports_realtime_window_failures() -> None:
    evaluator = SocketTactileCostEvaluator.__new__(SocketTactileCostEvaluator)
    evaluator.tactile_window_steps = 9
    evaluator._prepared_windows = {}
    context = ChunkExecutionContext(
        observation={},
        next_observation={},
        rl_state=np.zeros(1, dtype=np.float32),
        reference_action_chunk=np.zeros((50, 8), dtype=np.float32),
        proposed_action_chunk=np.zeros((50, 8), dtype=np.float32),
        executed_action_chunk=np.zeros((50, 8), dtype=np.float32),
        task_rewards=np.zeros(14, dtype=np.float32),
        done=False,
        info={
            "steps_executed": 14,
            "tactile_windows_started": 0,
            "tactile_window_errors": ["RuntimeError: no fresh tactile samples"],
        },
        started_unix_ns=0,
        finished_unix_ns=0,
    )

    with pytest.raises(RuntimeError, match="no fresh tactile samples"):
        evaluator.evaluate_chunk(context)


def test_posthoc_mode_disables_fast_callback_and_evaluates_once() -> None:
    evaluator = SocketTactileCostEvaluator.__new__(SocketTactileCostEvaluator)
    evaluator.feature_dim = 4
    evaluator.average_penalty_weight = 0.6
    evaluator.peak_penalty = 0.4
    evaluator.soft_episode_cap = 0.6
    evaluator.peak_episode_cap = 0.4
    evaluator.total_penalty_cap = 1.0
    evaluator._soft_episode_penalty = 0.0
    evaluator._peak_episode_penalty = 0.0
    evaluator.execution_mode = "posthoc_single_rate"
    evaluator.reward_criterion = "adaptive_envelope"
    evaluator.fixed_force_threshold_N = None
    requests: list[dict] = []

    def request_once(payload: dict) -> dict:
        requests.append(payload)
        return {
            "average_costs": [0.0] * 8 + [0.5] + [0.0] * 5,
            "peak_costs": [0.0] * 14,
            "metrics": {
                "low_cost_sum": 0.0,
                "high_cost_sum": 0.5,
                "low_quality_window": 0.0,
            },
        }

    evaluator._request = request_once
    observation = {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
    }
    actions = np.zeros((50, 8), dtype=np.float32)

    assert evaluator.prepare_execution(
        observation=observation,
        rl_state=np.arange(4, dtype=np.float32),
        reference_action_chunk=actions,
        proposed_action_chunk=actions,
    ) is None

    result = evaluator.evaluate_chunk(
        ChunkExecutionContext(
            observation=observation,
            next_observation=observation,
            rl_state=np.arange(4, dtype=np.float32),
            reference_action_chunk=actions,
            proposed_action_chunk=actions,
            executed_action_chunk=actions,
            task_rewards=np.zeros(14, dtype=np.float32),
            done=False,
            info={"steps_executed": 14},
            started_unix_ns=1_000_000_000,
            finished_unix_ns=1_466_666_667,
        )
    )

    assert len(requests) == 1
    assert requests[0]["reward_criterion"] == "adaptive_envelope"
    assert result.costs[8] == pytest.approx(0.3)
    assert result.metrics["tactile_execution_mode_posthoc"] == 1.0
    assert result.metrics["tactile_subwindows"] == 1.0
