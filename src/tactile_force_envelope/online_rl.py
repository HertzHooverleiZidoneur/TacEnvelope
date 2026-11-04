"""Online RL adapter for an out-of-process tactile envelope service.

The sidecar owns sensor capture and envelope inference.  This adapter only
sends the policy context for an executed chunk and receives non-negative
per-step costs.  A metadata handshake prevents mixing SmolVLA condition-token
experts with Pi0.5 JAX ``z_rl`` features.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any

import numpy as np

from rlt_openpi.rollout.reward_provider import (
    ChunkCostResult,
    ChunkExecutionContext,
    ChunkRewardProvider,
    RewardProviderMode,
)

SCHEMA_VERSION = 2
PI05_JAX_FEATURE_SOURCE = "pi05_jax_z_rl"
ADAPTIVE_ENVELOPE = "adaptive_envelope"
FIXED_UPPER = "fixed_upper"
DUAL_RATE = "dual_rate"
POSTHOC_SINGLE_RATE = "posthoc_single_rate"


def validate_server_metadata(
    metadata: dict[str, Any],
    *,
    expected_feature_source: str,
    expected_feature_dim: int,
    required_reward_criterion: str = ADAPTIVE_ENVELOPE,
) -> None:
    source = str(metadata.get("policy_feature_source", ""))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("Tactile sidecar schema mismatch; expected schema_version=2")
    dimension = int(metadata.get("policy_feature_dim", -1))
    if source != expected_feature_source:
        raise RuntimeError(
            "Tactile expert feature mismatch: "
            f"server={source or '(missing)'}, policy={expected_feature_source}. "
            "The current SmolVLA condition-token expert cannot be attached to "
            "Pi0.5 JAX Online RL; train or serve a Pi0.5 z_rl-compatible expert."
        )
    if dimension != expected_feature_dim:
        raise RuntimeError(
            "Tactile expert feature dimension mismatch: "
            f"server={dimension}, policy={expected_feature_dim}"
        )
    reward = metadata.get("reward_envelope") or {}
    peak = metadata.get("raw_peak_penalty") or {}
    if reward.get("type") != "bilateral_average_uncertainty_scaled_band":
        raise RuntimeError("Tactile sidecar is missing the uncertainty-scaled bilateral force band")
    if reward.get("calibration_method") != "standardized_residual_quantiles":
        raise RuntimeError("Tactile sidecar must use standardized-residual calibration")
    if reward.get("uncertainty_scale_method") != "mean":
        raise RuntimeError("Tactile sidecar has an unexpected uncertainty-scale method")
    if reward.get("scale_definition") != "u_mean = 0.5 * (sigma_left + sigma_right)":
        raise RuntimeError("Tactile sidecar has an unexpected uncertainty-scale definition")
    if reward.get("scale_is_standard_deviation") is not False:
        raise RuntimeError("Bilateral uncertainty scale must not be labeled a standard deviation")
    if float(reward.get("lower_coverage", -1.0)) != 0.05:
        raise RuntimeError("Tactile sidecar must use the calibrated L05 envelope")
    if reward.get("lower_alpha_label") != "alpha05":
        raise RuntimeError("Tactile sidecar is missing calibration-only alpha05")
    if reward.get("lower_alpha") is None:
        raise RuntimeError("Tactile sidecar is missing the alpha05 value")
    if float(reward.get("upper_coverage", -1.0)) != 0.95:
        raise RuntimeError("Tactile sidecar must use the calibrated U95 envelope")
    if reward.get("upper_alpha_label") != "alpha95":
        raise RuntimeError("Tactile sidecar is missing calibration-only alpha95")
    if reward.get("upper_alpha") is None:
        raise RuntimeError("Tactile sidecar is missing the alpha95 value")
    if int(reward.get("cost_power", -1)) != 2:
        raise RuntimeError("Tactile sidecar is not using the quadratic envelope cost")
    if float(reward.get("cost_clip", -1.0)) != 1.0:
        raise RuntimeError("Tactile sidecar envelope cost clip must be 1.0")
    supported_criteria = metadata.get(
        "supported_reward_criteria", [ADAPTIVE_ENVELOPE]
    )
    if required_reward_criterion not in supported_criteria:
        raise RuntimeError(
            "Tactile sidecar does not support reward criterion "
            f"{required_reward_criterion!r}; supported={supported_criteria}"
        )
    if peak.get("type") != "ungated_binary_threshold":
        raise RuntimeError("Tactile sidecar is missing the ungated raw-peak reward")


def combine_tactile_penalties(
    average_costs: Any,
    peak_costs: Any,
    *,
    average_penalty_weight: float,
    peak_penalty: float,
    total_penalty_cap: float = 1.0,
) -> np.ndarray:
    """Return the actual per-step reward deduction for the tactile-only path."""
    average = np.asarray(average_costs, dtype=np.float32).reshape(-1)
    peak = np.asarray(peak_costs, dtype=np.float32).reshape(-1)
    if average.shape != peak.shape:
        raise ValueError(
            f"average/peak tactile cost shape mismatch: {average.shape} vs {peak.shape}"
        )
    if average_penalty_weight < 0.0 or peak_penalty < 0.0:
        raise ValueError("tactile penalty weights must be non-negative")
    if total_penalty_cap <= 0.0:
        raise ValueError("total_penalty_cap must be positive")
    if not np.isfinite(average).all() or not np.isfinite(peak).all():
        raise ValueError("tactile costs must be finite")
    if np.any(average < 0.0) or np.any(peak < 0.0):
        raise ValueError("tactile costs must be non-negative")
    uncapped = (
        float(average_penalty_weight) * average
        + float(peak_penalty) * peak
    )
    return np.minimum(uncapped, float(total_penalty_cap)).astype(np.float32)


class SocketTactileCostEvaluator:
    """Evaluate one causal H9 tactile window inside one H14 transition."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        feature_dim: int,
        average_penalty_weight: float,
        peak_penalty: float,
        soft_episode_cap: float,
        peak_episode_cap: float,
        total_penalty_cap: float,
        timeout_s: float = 2.0,
        tactile_window_steps: int = 9,
        action_fps: float = 30.0,
        reward_criterion: str = ADAPTIVE_ENVELOPE,
        fixed_force_threshold_N: float | None = None,
        execution_mode: str = DUAL_RATE,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.feature_dim = int(feature_dim)
        self.average_penalty_weight = float(average_penalty_weight)
        self.peak_penalty = float(peak_penalty)
        self.soft_episode_cap = float(soft_episode_cap)
        self.peak_episode_cap = float(peak_episode_cap)
        self.total_penalty_cap = float(total_penalty_cap)
        if min(self.soft_episode_cap, self.peak_episode_cap) < 0.0:
            raise ValueError("tactile episode component caps must be non-negative")
        if self.soft_episode_cap + self.peak_episode_cap > self.total_penalty_cap:
            raise ValueError("soft and peak episode caps exceed total_penalty_cap")
        self._soft_episode_penalty = 0.0
        self._peak_episode_penalty = 0.0
        self.timeout_s = float(timeout_s)
        self.tactile_window_steps = int(tactile_window_steps)
        self.action_fps = float(action_fps)
        if self.tactile_window_steps != 9:
            raise ValueError("dual-rate tactile windows must contain exactly 9 steps")
        if self.action_fps <= 0.0:
            raise ValueError("action_fps must be positive")
        self.reward_criterion = str(reward_criterion)
        if self.reward_criterion not in {ADAPTIVE_ENVELOPE, FIXED_UPPER}:
            raise ValueError(
                "reward_criterion must be adaptive_envelope or fixed_upper"
            )
        self.fixed_force_threshold_N = (
            None
            if fixed_force_threshold_N is None
            else float(fixed_force_threshold_N)
        )
        if (
            self.reward_criterion == FIXED_UPPER
            and (
                self.fixed_force_threshold_N is None
                or self.fixed_force_threshold_N <= 0.0
            )
        ):
            raise ValueError(
                "fixed_upper reward requires a positive fixed_force_threshold_N"
            )
        self.execution_mode = str(execution_mode)
        if self.execution_mode not in {DUAL_RATE, POSTHOC_SINGLE_RATE}:
            raise ValueError(
                "execution_mode must be dual_rate or posthoc_single_rate"
            )
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._prepared_windows: dict[int, dict[str, Any]] = {}
        metadata = self._request(
            {
                "type": "metadata",
                "schema_version": SCHEMA_VERSION,
            }
        )
        validate_server_metadata(
            metadata,
            expected_feature_source=PI05_JAX_FEATURE_SOURCE,
            expected_feature_dim=self.feature_dim,
            required_reward_criterion=self.reward_criterion,
        )

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("tactile sidecar closed the socket")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _connect(self) -> socket.socket:
        if self._socket is None:
            self._socket = socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout_s,
            )
            self._socket.settimeout(self.timeout_s)
        return self._socket

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        with self._lock:
            sock = self._connect()
            try:
                sock.sendall(struct.pack("!I", len(body)) + body)
                response_size = struct.unpack("!I", self._recv_exact(sock, 4))[0]
                response = json.loads(self._recv_exact(sock, response_size))
            except Exception:
                self.close()
                raise
        if not isinstance(response, dict):
            raise RuntimeError("tactile sidecar response must be an object")
        if not response.get("ok", True):
            raise RuntimeError(str(response.get("error", "tactile sidecar rejected request")))
        return response

    def reset_episode(self) -> None:
        self._prepared_windows.clear()
        self._soft_episode_penalty = 0.0
        self._peak_episode_penalty = 0.0
        self._request({"type": "reset_episode", "schema_version": SCHEMA_VERSION})

    @staticmethod
    def _limit_component(
        value: float,
        *,
        used: float,
        cap: float,
    ) -> tuple[float, float]:
        remaining = max(0.0, float(cap) - float(used))
        return min(float(value), remaining), float(float(value) > remaining)

    @staticmethod
    def _robot_state(observation: dict[str, Any]) -> np.ndarray:
        state = np.concatenate(
            (
                np.asarray(
                    observation["observation/joint_position"], dtype=np.float32
                ).reshape(-1)[:7],
                np.asarray(
                    observation["observation/gripper_position"], dtype=np.float32
                ).reshape(-1)[:1],
            )
        )
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"expected a finite 8D robot state, got {state.shape}")
        return state

    def prepare_execution(
        self,
        *,
        observation: dict[str, Any],
        rl_state: np.ndarray,
        reference_action_chunk: np.ndarray,
        proposed_action_chunk: np.ndarray,
    ):
        """Return the non-actuating fast-channel callback used by the environment."""
        if getattr(self, "execution_mode", DUAL_RATE) == POSTHOC_SINGLE_RATE:
            return None
        del reference_action_chunk, proposed_action_chunk
        rl_state = np.asarray(rl_state, dtype=np.float32).reshape(-1)
        if rl_state.size < self.feature_dim:
            raise ValueError(
                f"RL state has {rl_state.size} values, expected at least {self.feature_dim}"
            )
        policy_feature = rl_state[: self.feature_dim].copy()
        initial_robot_state = self._robot_state(observation)
        self._prepared_windows.clear()

        def observe_window(
            *,
            window_index: int,
            anchor_unix_ns: int,
            observation: dict[str, Any] | None,
            executed_action_chunk: np.ndarray,
            action_fps: float,
        ) -> None:
            if int(window_index) < 0:
                raise ValueError(f"tactile window index must be non-negative, got {window_index}")
            if not np.isclose(float(action_fps), self.action_fps):
                raise ValueError(
                    f"tactile action FPS mismatch: runtime={action_fps}, expected={self.action_fps}"
                )
            actions = np.asarray(executed_action_chunk, dtype=np.float32)
            start = int(window_index) * self.tactile_window_steps
            stop = start + self.tactile_window_steps
            if actions.ndim != 2 or actions.shape[0] < stop:
                raise ValueError(
                    f"executed action chunk cannot provide window {window_index}: {actions.shape}"
                )
            robot_state = (
                initial_robot_state
                if observation is None
                else self._robot_state(observation)
            )
            action_window = np.ascontiguousarray(actions[start:stop])
            response = self._request(
                {
                    "type": "predict_chunk",
                    "schema_version": SCHEMA_VERSION,
                    "policy_feature_source": PI05_JAX_FEATURE_SOURCE,
                    "policy_feature": policy_feature.tolist(),
                    "robot_state": robot_state.tolist(),
                    "proposed_action_chunk": action_window.tolist(),
                    "anchor_unix_ns": int(anchor_unix_ns),
                }
            )
            self._prepared_windows[int(window_index)] = {
                "anchor_unix_ns": int(anchor_unix_ns),
                "policy_feature": policy_feature.copy(),
                "robot_state": robot_state,
                "actions": action_window,
                "prediction": dict(response["prediction"]),
                "history_report": dict(response.get("history_report", {})),
            }

        return observe_window

    def _evaluate_posthoc(self, context: ChunkExecutionContext) -> ChunkCostResult:
        """Synchronously evaluate one H9 window after the RTC transition."""
        rl_state = np.asarray(context.rl_state, dtype=np.float32).reshape(-1)
        if rl_state.size < self.feature_dim:
            raise ValueError(
                f"RL state has {rl_state.size} values, expected at least {self.feature_dim}"
            )
        response = self._request(
            {
                "type": "evaluate_chunk",
                "schema_version": SCHEMA_VERSION,
                "policy_feature_source": PI05_JAX_FEATURE_SOURCE,
                "policy_feature": rl_state[: self.feature_dim].tolist(),
                "robot_state": self._robot_state(context.observation).tolist(),
                "executed_action_chunk": np.asarray(
                    context.executed_action_chunk, dtype=np.float32
                ).tolist(),
                "started_unix_ns": int(context.started_unix_ns),
                "finished_unix_ns": int(context.finished_unix_ns),
                "reward_count": int(np.asarray(context.task_rewards).size),
                "steps_executed": int(
                    context.info.get(
                        "steps_executed",
                        np.asarray(context.executed_action_chunk).shape[0],
                    )
                ),
                "reward_criterion": self.reward_criterion,
                "fixed_force_threshold_N": self.fixed_force_threshold_N,
            }
        )
        average = np.asarray(response["average_costs"], dtype=np.float32).reshape(-1)
        peak = np.asarray(response["peak_costs"], dtype=np.float32).reshape(-1)
        if average.shape != peak.shape:
            raise RuntimeError(
                "Tactile sidecar average/peak shape mismatch: "
                f"{average.shape} vs {peak.shape}"
            )

        raw_soft_costs = self.average_penalty_weight * average
        raw_peak_costs = self.peak_penalty * peak
        raw_chunk_soft = float(raw_soft_costs.sum())
        raw_chunk_peak = float(raw_peak_costs.sum())
        chunk_soft, soft_cap_hit = self._limit_component(
            raw_chunk_soft,
            used=self._soft_episode_penalty,
            cap=self.soft_episode_cap,
        )
        chunk_peak, peak_cap_hit = self._limit_component(
            raw_chunk_peak,
            used=self._peak_episode_penalty,
            cap=self.peak_episode_cap,
        )
        soft_scale = chunk_soft / raw_chunk_soft if raw_chunk_soft > 0.0 else 0.0
        peak_scale = chunk_peak / raw_chunk_peak if raw_chunk_peak > 0.0 else 0.0
        costs = raw_soft_costs * soft_scale + raw_peak_costs * peak_scale
        self._soft_episode_penalty += chunk_soft
        self._peak_episode_penalty += chunk_peak

        details = {
            str(key): float(value)
            for key, value in response.get("metrics", {}).items()
        }
        metrics = {
            **details,
            "tactile_subwindows": 1.0,
            "tactile_execution_mode_dual_rate": 0.0,
            "tactile_execution_mode_posthoc": 1.0,
            "reward_criterion_fixed_upper": float(
                self.reward_criterion == FIXED_UPPER
            ),
            "fixed_force_threshold_N": float(
                self.fixed_force_threshold_N
                if self.fixed_force_threshold_N is not None
                else -1.0
            ),
            "tactile_window_1_penalty": float(costs.sum()),
            "tactile_window_penalty_mean": float(costs.sum()),
            "tactile_reward_deduction_sum": float(costs.sum()),
            "soft_force_reward_deduction_sum": chunk_soft,
            "peak_force_reward_deduction_sum": chunk_peak,
            "uncapped_soft_force_reward_deduction_sum": raw_chunk_soft,
            "uncapped_peak_force_reward_deduction_sum": raw_chunk_peak,
            "soft_episode_penalty_total": self._soft_episode_penalty,
            "peak_episode_penalty_total": self._peak_episode_penalty,
            "soft_episode_cap": self.soft_episode_cap,
            "peak_episode_cap": self.peak_episode_cap,
            "soft_episode_cap_hit": soft_cap_hit,
            "peak_episode_cap_hit": peak_cap_hit,
            "uncapped_tactile_reward_deduction_sum": (
                raw_chunk_soft + raw_chunk_peak
            ),
            "tactile_penalty_cap_hits": float(
                np.count_nonzero(raw_soft_costs + raw_peak_costs > self.total_penalty_cap)
            ),
            "tactile_low_quality_windows": details.get(
                "low_quality_window", 0.0
            ),
            "average_penalty_weight": self.average_penalty_weight,
            "peak_penalty": self.peak_penalty,
            "total_penalty_cap": self.total_penalty_cap,
            "low_force_reward_deduction_sum": (
                self.average_penalty_weight
                * details.get("low_cost_sum", 0.0)
                * soft_scale
            ),
            "high_force_reward_deduction_sum": (
                self.average_penalty_weight
                * details.get("high_cost_sum", 0.0)
                * soft_scale
            ),
        }
        return ChunkCostResult(costs=costs.astype(np.float32), metrics=metrics)

    def evaluate_chunk(self, context: ChunkExecutionContext) -> ChunkCostResult:
        if getattr(self, "execution_mode", DUAL_RATE) == POSTHOC_SINGLE_RATE:
            return self._evaluate_posthoc(context)
        steps_executed = int(context.info.get("steps_executed", 0))
        started_windows = int(context.info.get("tactile_windows_started", -1))
        window_errors = context.info.get("tactile_window_errors") or []
        if window_errors:
            details = "; ".join(str(error) for error in window_errors)
            raise RuntimeError(f"real-time tactile window failed: {details}")
        complete_windows = max(0, steps_executed // self.tactile_window_steps)
        if started_windows >= 0:
            expected_windows = min(started_windows, complete_windows)
        else:
            expected_windows = complete_windows
        if expected_windows <= 0:
            return ChunkCostResult(
                costs=np.zeros_like(context.task_rewards, dtype=np.float32),
                metrics={"tactile_subwindows": 0.0},
            )
        missing = [index for index in range(expected_windows) if index not in self._prepared_windows]
        if missing:
            raise RuntimeError(
                f"missing real-time tactile prediction(s) for H9 window(s): {missing}"
            )

        window_penalties: list[float] = []
        window_soft: list[float] = []
        window_peak: list[float] = []
        window_uncapped: list[float] = []
        window_metrics: list[dict[str, float]] = []
        for index in range(expected_windows):
            prepared = self._prepared_windows[index]
            anchor_ns = int(prepared["anchor_unix_ns"])
            response = self._request(
                {
                    "type": "evaluate_chunk",
                    "schema_version": SCHEMA_VERSION,
                    "policy_feature_source": PI05_JAX_FEATURE_SOURCE,
                    "policy_feature": prepared["policy_feature"].tolist(),
                    "robot_state": prepared["robot_state"].tolist(),
                    "executed_action_chunk": prepared["actions"].tolist(),
                    "cached_prediction": prepared["prediction"],
                    "cached_history_report": prepared["history_report"],
                    "started_unix_ns": anchor_ns,
                    "finished_unix_ns": anchor_ns
                    + int(self.tactile_window_steps / self.action_fps * 1e9),
                    "reward_count": self.tactile_window_steps,
                    "steps_executed": self.tactile_window_steps,
                    "reward_criterion": getattr(
                        self, "reward_criterion", ADAPTIVE_ENVELOPE
                    ),
                    "fixed_force_threshold_N": getattr(
                        self, "fixed_force_threshold_N", None
                    ),
                }
            )
            average = np.asarray(response["average_costs"], dtype=np.float32).reshape(-1)
            peak = np.asarray(response["peak_costs"], dtype=np.float32).reshape(-1)
            costs = combine_tactile_penalties(
                average,
                peak,
                average_penalty_weight=self.average_penalty_weight,
                peak_penalty=self.peak_penalty,
                total_penalty_cap=self.total_penalty_cap,
            )
            uncapped = self.average_penalty_weight * average + self.peak_penalty * peak
            window_penalties.append(float(costs.sum()))
            window_soft.append(float((self.average_penalty_weight * average).sum()))
            window_peak.append(float((self.peak_penalty * peak).sum()))
            window_uncapped.append(float(uncapped.sum()))
            window_metrics.append(
                {str(key): float(value) for key, value in response.get("metrics", {}).items()}
            )

        raw_chunk_soft = float(np.mean(window_soft))
        raw_chunk_peak = float(np.mean(window_peak))
        chunk_soft, soft_cap_hit = self._limit_component(
            raw_chunk_soft,
            used=self._soft_episode_penalty,
            cap=self.soft_episode_cap,
        )
        chunk_peak, peak_cap_hit = self._limit_component(
            raw_chunk_peak,
            used=self._peak_episode_penalty,
            cap=self.peak_episode_cap,
        )
        self._soft_episode_penalty += chunk_soft
        self._peak_episode_penalty += chunk_peak
        chunk_penalty = chunk_soft + chunk_peak
        result_costs = np.zeros_like(context.task_rewards, dtype=np.float32)
        cost_index = min(
            result_costs.size - 1,
            expected_windows * self.tactile_window_steps - 1,
        )
        if cost_index >= 0:
            result_costs[cost_index] = chunk_penalty

        metrics: dict[str, float] = {
            "tactile_subwindows": float(expected_windows),
            "tactile_execution_mode_dual_rate": 1.0,
            "tactile_execution_mode_posthoc": 0.0,
            "reward_criterion_fixed_upper": float(
                getattr(self, "reward_criterion", ADAPTIVE_ENVELOPE)
                == FIXED_UPPER
            ),
            "fixed_force_threshold_N": float(
                getattr(self, "fixed_force_threshold_N", None)
                if getattr(self, "fixed_force_threshold_N", None) is not None
                else -1.0
            ),
            "tactile_window_penalty_mean": chunk_penalty,
            "tactile_reward_deduction_sum": chunk_penalty,
            "soft_force_reward_deduction_sum": chunk_soft,
            "peak_force_reward_deduction_sum": chunk_peak,
            "uncapped_soft_force_reward_deduction_sum": raw_chunk_soft,
            "uncapped_peak_force_reward_deduction_sum": raw_chunk_peak,
            "soft_episode_penalty_total": self._soft_episode_penalty,
            "peak_episode_penalty_total": self._peak_episode_penalty,
            "soft_episode_cap": self.soft_episode_cap,
            "peak_episode_cap": self.peak_episode_cap,
            "soft_episode_cap_hit": soft_cap_hit,
            "peak_episode_cap_hit": peak_cap_hit,
            "uncapped_tactile_reward_deduction_sum": float(np.mean(window_uncapped)),
            "tactile_penalty_cap_hits": float(
                sum(value > self.total_penalty_cap for value in window_uncapped)
            ),
            "tactile_low_quality_windows": float(
                sum(details.get("low_quality_window", 0.0) for details in window_metrics)
            ),
            "average_penalty_weight": self.average_penalty_weight,
            "peak_penalty": self.peak_penalty,
            "total_penalty_cap": self.total_penalty_cap,
        }
        for index, (penalty, details) in enumerate(
            zip(window_penalties, window_metrics, strict=True), start=1
        ):
            metrics[f"tactile_window_{index}_penalty"] = penalty
            for key in (
                "mu_avg_N",
                "lower_avg_N",
                "upper_avg_N",
                "observed_avg_N",
                "low_deviation_N",
                "high_deviation_N",
                "raw_peak_N",
                "peak_triggered",
                "gate_bilateral",
                "quality_valid",
                "low_quality_window",
                "observed_samples_left",
                "observed_samples_right",
                "cached_prediction",
                "reward_criterion_adaptive",
                "reward_criterion_fixed_upper",
                "fixed_force_threshold_N",
            ):
                if key in details:
                    metrics[f"window_{index}_{key}"] = details[key]
        for key in ("low_cost_sum", "high_cost_sum", "average_cost_sum", "peak_cost_sum"):
            metrics[key] = float(np.mean([details.get(key, 0.0) for details in window_metrics]))
        metrics["low_force_reward_deduction_sum"] = (
            self.average_penalty_weight * metrics["low_cost_sum"]
        )
        metrics["high_force_reward_deduction_sum"] = (
            self.average_penalty_weight * metrics["high_cost_sum"]
        )
        self._prepared_windows.clear()
        return ChunkCostResult(costs=result_costs, metrics=metrics)

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def build_pi05_reward_provider(
    *,
    mode: RewardProviderMode,
    policy_feature_dim: int,
    host: str = "127.0.0.1",
    port: int = 8021,
    penalty_weight: float = 0.6,
    peak_penalty: float = 0.4,
    cost_clip: float = 1.0,
    soft_episode_cap: float = 0.6,
    peak_episode_cap: float = 0.4,
    total_penalty_cap: float = 1.0,
    strict: bool = True,
    action_fps: float = 30.0,
    reward_criterion: str = ADAPTIVE_ENVELOPE,
    fixed_force_threshold_N: float | None = None,
    execution_mode: str = DUAL_RATE,
) -> ChunkRewardProvider | None:
    """Build the isolated Pi0.5 tactile provider.

    ``off`` returns ``None`` so the ordinary rollout path is byte-for-byte the
    same.  Other modes require a compatible sidecar to be running.
    """
    if mode == "off":
        return None
    evaluator = SocketTactileCostEvaluator(
        host,
        port,
        feature_dim=policy_feature_dim,
        average_penalty_weight=penalty_weight,
        peak_penalty=peak_penalty,
        soft_episode_cap=soft_episode_cap,
        peak_episode_cap=peak_episode_cap,
        total_penalty_cap=total_penalty_cap,
        tactile_window_steps=9,
        action_fps=action_fps,
        reward_criterion=reward_criterion,
        fixed_force_threshold_N=fixed_force_threshold_N,
        execution_mode=execution_mode,
    )
    return ChunkRewardProvider(
        mode=mode,
        evaluator=evaluator,
        penalty_weight=1.0,
        cost_clip=total_penalty_cap,
        episode_penalty_cap=total_penalty_cap,
        strict=strict,
        name=(
            "tactile_fixed_force"
            if reward_criterion == FIXED_UPPER
            else "tactile_force_envelope"
        ),
    )
