"""Pi0.5 JAX z_rl force-envelope sidecar implementation."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .data import median_filter
from .inference import TransformerForceEnvelopePredictor
from .stable_contact import StableContactHysteresis, normalized_band_cost

SIDES = ("left", "right")
FEATURE_SOURCE = "pi05_jax_z_rl"
SCHEMA_VERSION = 2
ADAPTIVE_ENVELOPE = "adaptive_envelope"
FIXED_UPPER = "fixed_upper"
SUPPORTED_REWARD_CRITERIA = (ADAPTIVE_ENVELOPE, FIXED_UPPER)


class TactileSampleSource(Protocol):
    def status(self) -> dict[str, Any]:
        """Return sensor status."""

    def recent(self, start_ns: int, end_ns: int) -> dict[str, list[dict[str, Any]]]:
        """Return feature rows in [start_ns, end_ns)."""


def unix_request(
    socket_path: str | Path,
    request: dict[str, Any],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout_s)
    client.connect(str(Path(socket_path).expanduser()))
    client.sendall((json.dumps(request, separators=(",", ":")) + "\n").encode())
    response = b""
    while not response.endswith(b"\n"):
        chunk = client.recv(65536)
        if not chunk:
            break
        response += chunk
    client.close()
    result = json.loads(response.decode("utf-8"))
    if not result.get("ok", False):
        raise RuntimeError(str(result.get("error", "Xense daemon request failed")))
    return result


class XenseRingClient:
    """Read non-destructive force rows from the persistent Xense daemon."""

    def __init__(self, socket_path: str | Path, timeout_s: float = 5.0) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout_s = float(timeout_s)
        status = self.status()
        names = [str(sensor["name"]) for sensor in status.get("sensors", [])]
        if len(names) < 2:
            raise RuntimeError(f"Envelope sidecar requires two Xense sensors; detected={names}")
        self.side_names = _resolve_side_names(names)

    def status(self) -> dict[str, Any]:
        return unix_request(
            self.socket_path,
            {"command": "status"},
            timeout_s=self.timeout_s,
        )

    def recent(self, start_ns: int, end_ns: int) -> dict[str, list[dict[str, Any]]]:
        response = unix_request(
            self.socket_path,
            {
                "command": "recent",
                "start_ns": int(start_ns),
                "end_ns": int(end_ns),
                "max_samples": 500,
            },
            timeout_s=self.timeout_s,
        )
        sensors = response.get("sensors", {})
        return {
            side: list(sensors.get(name, []))
            for side, name in self.side_names.items()
        }


def _resolve_side_names(names: list[str]) -> dict[str, str]:
    lowered = {name: name.lower() for name in names}
    left = next((name for name, value in lowered.items() if "left" in value), None)
    right = next((name for name, value in lowered.items() if "right" in value), None)
    if left is None or right is None or left == right:
        left, right = names[:2]
    return {"left": left, "right": right}


def row_mapping(row: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in zip(
            row.get("feature_names", []),
            row.get("values", []),
            strict=False,
        )
    }


def row_valid(row: dict[str, Any]) -> bool:
    values = row_mapping(row)
    return bool(values.get("sample_valid", 1.0) > 0.5 and values.get("force_valid", 1.0) > 0.5)


def row_xyz(row: dict[str, Any]) -> np.ndarray:
    values = row_mapping(row)
    return np.asarray(
        [
            values.get("force_fx", np.nan),
            values.get("force_fy", np.nan),
            values.get("force_fz", np.nan),
        ],
        dtype=np.float32,
    )


def corrected_force(row: dict[str, Any], baseline: np.ndarray) -> float:
    if not row_valid(row):
        return float("nan")
    xyz = row_xyz(row) - np.asarray(baseline, dtype=np.float32)
    return float(np.linalg.norm(xyz))


def calculate_baselines(
    rows: dict[str, list[dict[str, Any]]],
    *,
    min_samples: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    baselines: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for side in SIDES:
        valid_xyz = [
            row_xyz(row)
            for row in rows.get(side, [])
            if row_valid(row) and np.isfinite(row_xyz(row)).all()
        ]
        if len(valid_xyz) < min_samples:
            raise RuntimeError(
                f"{side}: only {len(valid_xyz)} valid samples during no-contact baseline"
            )
        xyz = np.stack(valid_xyz)
        baseline = np.median(xyz, axis=0).astype(np.float32)
        residual = np.linalg.norm(xyz - baseline, axis=1)
        baselines[side] = baseline
        report[side] = {
            "sample_count": len(valid_xyz),
            "baseline_xyz": baseline.tolist(),
            "residual_median_N": float(np.median(residual)),
            "residual_p95_N": float(np.quantile(residual, 0.95)),
            "residual_max_N": float(np.max(residual)),
        }
    return baselines, report


def build_tactile_history(
    rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    *,
    anchor_ns: int,
    history_ms: int,
    history_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    history = np.zeros((2, history_steps, 5), dtype=np.float32)
    mask = np.zeros((2, history_steps), dtype=bool)
    report: dict[str, Any] = {}
    cutoff_ns = int(anchor_ns - history_ms * 1e6)
    for side_index, side in enumerate(SIDES):
        selected = [
            row
            for row in rows.get(side, [])
            if cutoff_ns <= int(row["pc_time_ns"]) < anchor_ns and row_valid(row)
        ][-history_steps:]
        offset = history_steps - len(selected)
        for index, row in enumerate(selected):
            xyz = row_xyz(row) - baselines[side]
            position = offset + index
            history[side_index, position] = [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                float(np.linalg.norm(xyz)),
                (int(row["pc_time_ns"]) - anchor_ns) / (history_ms * 1e6),
            ]
            mask[side_index, position] = True
        report[side] = {
            "history_samples": len(selected),
            "latest_age_ms": (
                (anchor_ns - int(selected[-1]["pc_time_ns"])) / 1e6
                if selected
                else None
            ),
        }
    return history, mask, report


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Pi05EnvelopeCostService:
    """Evaluate executed Pi0.5 chunks against measured tactile force."""

    def __init__(
        self,
        predictor: TransformerForceEnvelopePredictor,
        sample_source: TactileSampleSource,
        baselines: dict[str, np.ndarray],
        *,
        checkpoint_path: str | Path,
        baseline_report: dict[str, Any] | None = None,
        action_fps: float = 30.0,
        history_ms: int = 200,
        bilateral_lower_alpha: float,
        bilateral_lower_coverage: float,
        bilateral_upper_alpha: float,
        bilateral_upper_coverage: float,
        bilateral_calibration: dict[str, Any] | None = None,
        raw_peak_threshold_N: float,
        min_observed_samples_per_side: int = 4,
        force_scale: float = 5.0,
        cost_clip: float = 1.0,
        tau_on: float = 5.0,
        tau_off: float = 3.0,
        log_path: str | Path | None = None,
    ) -> None:
        if predictor.condition_steps != 1 or predictor.condition_dim != 512:
            raise ValueError(
                "Pi0.5 z_rl checkpoint must expect condition tokens [1,512], got "
                f"[{predictor.condition_steps},{predictor.condition_dim}]"
            )
        if predictor.control_window_ms != 300:
            raise ValueError(
                f"Online integration requires a 300 ms checkpoint, got {predictor.control_window_ms}"
            )
        self.predictor = predictor
        self.sample_source = sample_source
        self.baselines = {
            side: np.asarray(baselines[side], dtype=np.float32)
            for side in SIDES
        }
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.checkpoint_hash = checkpoint_sha256(checkpoint_path)
        self.baseline_report = baseline_report or {}
        self.action_fps = float(action_fps)
        self.history_ms = int(history_ms)
        if bilateral_lower_alpha >= bilateral_upper_alpha:
            raise ValueError(
                "bilateral_lower_alpha must be below bilateral_upper_alpha"
            )
        if not 0.0 < bilateral_lower_coverage < bilateral_upper_coverage:
            raise ValueError(
                "bilateral_lower_coverage must be between zero and upper coverage"
            )
        if not 0.0 < bilateral_upper_coverage < 1.0:
            raise ValueError("bilateral_upper_coverage must be between zero and one")
        if min_observed_samples_per_side < 2:
            raise ValueError("min_observed_samples_per_side must be at least 2")
        self.bilateral_lower_alpha = float(bilateral_lower_alpha)
        self.bilateral_lower_coverage = float(bilateral_lower_coverage)
        self.bilateral_lower_alpha_label = (
            f"alpha{int(round(100.0 * self.bilateral_lower_coverage)):02d}"
        )
        self.bilateral_upper_alpha = float(bilateral_upper_alpha)
        self.bilateral_upper_coverage = float(bilateral_upper_coverage)
        self.bilateral_upper_alpha_label = (
            f"alpha{int(round(100.0 * self.bilateral_upper_coverage))}"
        )
        self.bilateral_calibration = bilateral_calibration or {}
        if raw_peak_threshold_N <= 0.0:
            raise ValueError("raw_peak_threshold_N must be positive")
        self.raw_peak_threshold_N = float(raw_peak_threshold_N)
        self.min_observed_samples_per_side = int(min_observed_samples_per_side)
        self.force_scale = float(force_scale)
        self.cost_clip = float(cost_clip)
        self.gate = StableContactHysteresis(tau_on=tau_on, tau_off=tau_off)
        self.last_gate_sample_ns = {side: 0 for side in SIDES}
        self.lock = threading.RLock()
        self.log_path = Path(log_path).expanduser() if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "policy_feature_source": FEATURE_SOURCE,
            "policy_feature_dim": 512,
            "checkpoint": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_hash,
            "expected_policy_checkpoint": self.predictor.feature_metadata.get("checkpoint"),
            "expected_policy_checkpoint_sha256": self.predictor.feature_metadata.get(
                "checkpoint_sha256"
            ),
            "control_window_ms": self.predictor.control_window_ms,
            "action_steps": self.predictor.action_steps,
            "action_dim": self.predictor.action_dim,
            "history_ms": self.history_ms,
            "history_steps": self.predictor.history_steps,
            "supported_reward_criteria": list(SUPPORTED_REWARD_CRITERIA),
            "reward_envelope": {
                "type": "bilateral_average_uncertainty_scaled_band",
                "formula": (
                    "u_mean = 0.5 * (sigma_left + sigma_right); "
                    "lower_avg = 0.5*(mu_left+mu_right) + alpha05*u_mean; "
                    "upper_avg = 0.5*(mu_left+mu_right) + alpha95*u_mean"
                ),
                "calibration_method": "standardized_residual_quantiles",
                "standardized_residual": "(y_avg-mu_avg)/max(u_mean,1e-6)",
                "uncertainty_scale_method": "mean",
                "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
                "scale_is_standard_deviation": False,
                "lower_coverage": self.bilateral_lower_coverage,
                "lower_alpha_label": self.bilateral_lower_alpha_label,
                "lower_alpha": self.bilateral_lower_alpha,
                self.bilateral_lower_alpha_label: self.bilateral_lower_alpha,
                "lower_alpha_source": self.bilateral_calibration.get(
                    "lower_alpha_source"
                ),
                "upper_coverage": self.bilateral_upper_coverage,
                "upper_alpha_label": self.bilateral_upper_alpha_label,
                "upper_alpha": self.bilateral_upper_alpha,
                self.bilateral_upper_alpha_label: self.bilateral_upper_alpha,
                "upper_alpha_source": self.bilateral_calibration.get(
                    "upper_alpha_source"
                ),
                "calibration_samples": self.bilateral_calibration.get(
                    "calibration_samples"
                ),
                "cost_formula": (
                    "both_gates * ("
                    "clip(max(0, lower_avg-observed_avg)/force_scale, 0, cost_clip)^2"
                    " + "
                    "clip(max(0, observed_avg-upper_avg)/force_scale, 0, cost_clip)^2"
                    ")"
                ),
                "cost_components": ["low_force", "high_force"],
                "force_scale_N": self.force_scale,
                "cost_clip": self.cost_clip,
                "cost_power": 2,
                "cost_assignment": "one cost at the final step of the 300 ms window",
                "min_observed_samples_per_side": self.min_observed_samples_per_side,
            },
            "raw_peak_penalty": {
                "type": "ungated_binary_threshold",
                "force_source": "zero-biased unfiltered bilateral raw force magnitude",
                "threshold_N": self.raw_peak_threshold_N,
                "threshold_source": self.bilateral_calibration.get(
                    "raw_peak_threshold", {}
                ).get("source"),
                "cost_assignment": "one event at the final executed step of the window",
                "stable_contact_gate_required": False,
            },
            "stable_contact": {
                "tau_on": self.gate.tau_on,
                "tau_off": self.gate.tau_off,
                "recent_samples": self.gate.recent_samples,
                "required_samples": self.gate.required_samples,
            },
            "baseline": self.baseline_report,
            "production_ready": self.predictor.production_ready,
            "recommended_mode": "shadow",
            "hard_force_guard_implemented_here": False,
        }

    def reset_episode(self) -> dict[str, Any]:
        with self.lock:
            self.gate.reset()
            self.last_gate_sample_ns = {side: 0 for side in SIDES}
        return {"ok": True}

    def _update_gate_until(
        self,
        rows: dict[str, list[dict[str, Any]]],
        end_ns: int,
    ) -> None:
        for side in SIDES:
            selected = sorted(rows.get(side, []), key=lambda row: int(row["pc_time_ns"]))
            for row in selected:
                timestamp = int(row["pc_time_ns"])
                if timestamp >= end_ns or timestamp <= self.last_gate_sample_ns[side]:
                    continue
                force = corrected_force(row, self.baselines[side])
                self.gate.update(side, force, valid=np.isfinite(force))
                self.last_gate_sample_ns[side] = timestamp

    def _observed_forces(
        self,
        rows: dict[str, list[dict[str, Any]]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[
        dict[str, list[tuple[int, float]]],
        dict[str, list[tuple[int, float]]],
    ]:
        filtered_result: dict[str, list[tuple[int, float]]] = {}
        raw_result: dict[str, list[tuple[int, float]]] = {}
        for side in SIDES:
            selected = [
                row
                for row in rows.get(side, [])
                if start_ns <= int(row["pc_time_ns"]) < end_ns and row_valid(row)
            ]
            timestamps = [int(row["pc_time_ns"]) for row in selected]
            raw = np.asarray(
                [corrected_force(row, self.baselines[side]) for row in selected],
                dtype=np.float32,
            )
            filtered = median_filter(raw, 3) if raw.size else raw
            raw_result[side] = list(zip(timestamps, raw.tolist(), strict=True))
            filtered_result[side] = list(
                zip(timestamps, filtered.tolist(), strict=True)
            )
        return filtered_result, raw_result

    def _costs(
        self,
        observed: dict[str, list[tuple[int, float]]],
        raw_observed: dict[str, list[tuple[int, float]]],
        *,
        reward_count: int,
        steps_executed: int,
        window_complete: bool,
        prediction: dict[str, Any],
        gate_at_anchor: dict[str, bool],
        reward_criterion: str = ADAPTIVE_ENVELOPE,
        fixed_force_threshold_N: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if reward_criterion not in SUPPORTED_REWARD_CRITERIA:
            raise ValueError(f"unsupported tactile reward criterion: {reward_criterion}")
        if reward_criterion == FIXED_UPPER:
            if fixed_force_threshold_N is None or fixed_force_threshold_N <= 0.0:
                raise ValueError(
                    "fixed_upper reward requires a positive fixed_force_threshold_N"
                )
        average_costs = np.zeros(reward_count, dtype=np.float32)
        peak_costs = np.zeros(reward_count, dtype=np.float32)
        bilateral_gate = bool(gate_at_anchor["left"] and gate_at_anchor["right"])
        sample_counts = {side: len(observed[side]) for side in SIDES}
        enough_samples = all(
            sample_counts[side] >= self.min_observed_samples_per_side
            for side in SIDES
        )
        quality_valid = bool(window_complete and enough_samples)
        target_by_side: dict[str, float | None] = {"left": None, "right": None}
        target_avg: float | None = None
        mu_avg = 0.5 * (
            float(prediction["mu_left"]) + float(prediction["mu_right"])
        )
        uncertainty_scale = max(1e-6, 0.5 * (
            float(prediction["sigma_left"]) + float(prediction["sigma_right"])
        ))
        lower_offset = self.bilateral_lower_alpha * uncertainty_scale
        upper_offset = self.bilateral_upper_alpha * uncertainty_scale
        if reward_criterion == FIXED_UPPER:
            lower_avg = 0.0
            upper_avg = float(fixed_force_threshold_N)
        else:
            lower_avg = mu_avg + lower_offset
            upper_avg = mu_avg + upper_offset
        low_deviation = 0.0
        high_deviation = 0.0
        low_cost = 0.0
        high_cost = 0.0
        cost = 0.0
        cost_index = min(self.predictor.action_steps, reward_count, steps_executed) - 1
        raw_peak_by_side = {
            side: (
                max(force for _, force in raw_observed[side])
                if raw_observed[side]
                else None
            )
            for side in SIDES
        }
        finite_raw_peaks = [
            float(value) for value in raw_peak_by_side.values() if value is not None
        ]
        raw_peak_N = max(finite_raw_peaks) if finite_raw_peaks else None
        peak_exceedance_N = (
            max(0.0, raw_peak_N - self.raw_peak_threshold_N)
            if raw_peak_N is not None
            else 0.0
        )
        peak_triggered = bool(peak_exceedance_N > 0.0)

        if quality_valid:
            for side in SIDES:
                values = np.asarray(
                    [force for _, force in observed[side]], dtype=np.float32
                )
                two_largest = np.partition(values, values.size - 2)[-2:]
                target_by_side[side] = float(np.mean(two_largest))
            target_avg = 0.5 * (
                float(target_by_side["left"]) + float(target_by_side["right"])
            )
            if reward_criterion == FIXED_UPPER:
                high_deviation = max(0.0, target_avg - upper_avg)
                if bilateral_gate:
                    high_cost = min(
                        high_deviation / self.force_scale,
                        self.cost_clip,
                    ) ** 2
            else:
                low_deviation = max(0.0, lower_avg - target_avg)
                high_deviation = max(0.0, target_avg - upper_avg)
                low_cost, high_cost = normalized_band_cost(
                    target_avg,
                    lower_avg,
                    upper_avg,
                    bilateral_gate,
                    force_scale=self.force_scale,
                    cost_clip=self.cost_clip,
                )
            cost = low_cost + high_cost
            if cost_index >= 0:
                average_costs[cost_index] = cost
        if peak_triggered and cost_index >= 0:
            peak_costs[cost_index] = 1.0

        costs = average_costs + peak_costs

        return costs, {
            "bilateral_gate": bilateral_gate,
            "reward_criterion": reward_criterion,
            "fixed_force_threshold_N": fixed_force_threshold_N,
            "window_complete": window_complete,
            "quality_valid": quality_valid,
            "low_quality_window": bool(window_complete and not enough_samples),
            "sample_counts": sample_counts,
            "target_top2_mean_N": target_by_side,
            "target_avg_N": target_avg,
            "mu_avg_N": mu_avg,
            "uncertainty_scale_N": uncertainty_scale,
            "sigma_avg_N": uncertainty_scale,
            "lower_coverage": self.bilateral_lower_coverage,
            "lower_alpha_label": self.bilateral_lower_alpha_label,
            "lower_alpha": self.bilateral_lower_alpha,
            "lower_offset_N": lower_offset,
            "lower_avg_N": lower_avg,
            "upper_coverage": self.bilateral_upper_coverage,
            "upper_alpha_label": self.bilateral_upper_alpha_label,
            "upper_alpha": self.bilateral_upper_alpha,
            "upper_offset_N": upper_offset,
            "upper_avg_N": upper_avg,
            "low_deviation_N": low_deviation,
            "high_deviation_N": high_deviation,
            "low_normalized_cost": low_cost,
            "high_normalized_cost": high_cost,
            "normalized_cost": cost,
            "cost_index": cost_index,
            "average_costs": average_costs.tolist(),
            "raw_peak_by_side_N": raw_peak_by_side,
            "raw_peak_N": raw_peak_N,
            "raw_peak_threshold_N": self.raw_peak_threshold_N,
            "peak_exceedance_N": peak_exceedance_N,
            "peak_triggered": peak_triggered,
            "peak_costs": peak_costs.tolist(),
        }

    def predict_chunk(self, request: dict[str, Any]) -> dict[str, Any]:
        """Predict before enqueueing, using only tactile samples before the anchor."""
        with self.lock:
            if request.get("policy_feature_source") != FEATURE_SOURCE:
                raise ValueError("predict_chunk requires Pi0.5 JAX z_rl features")
            feature = np.asarray(request["policy_feature"], dtype=np.float32).reshape(-1)
            state = np.asarray(request["robot_state"], dtype=np.float32).reshape(-1)
            actions = np.asarray(request["proposed_action_chunk"], dtype=np.float32)
            if feature.shape != (512,):
                raise ValueError(f"policy_feature must be [512], got {feature.shape}")
            if state.shape != (8,):
                raise ValueError(f"robot_state must be [8], got {state.shape}")
            expected_action_shape = (
                self.predictor.action_steps,
                self.predictor.action_dim,
            )
            if actions.ndim != 2 or actions.shape[1] != self.predictor.action_dim:
                raise ValueError(
                    f"proposed_action_chunk must be [N,{self.predictor.action_dim}], "
                    f"got {actions.shape}"
                )
            if actions.shape[0] < self.predictor.action_steps:
                raise ValueError(
                    f"proposed_action_chunk must contain at least {expected_action_shape[0]} steps"
                )
            anchor_ns = int(request.get("anchor_unix_ns", time.time_ns()))
            history_start_ns = anchor_ns - int(self.history_ms * 1e6)
            rows = self.sample_source.recent(history_start_ns, anchor_ns)
            self._update_gate_until(rows, anchor_ns)
            gate_at_anchor = self.gate.active
            history, history_mask, history_report = build_tactile_history(
                rows,
                self.baselines,
                anchor_ns=anchor_ns,
                history_ms=self.history_ms,
                history_steps=self.predictor.history_steps,
            )
            prediction = self.predictor.predict(
                feature[None],
                np.ones(1, dtype=bool),
                state,
                actions[: self.predictor.action_steps],
                tactile_history=history,
                tactile_history_mask=history_mask,
            )
            for side in SIDES:
                prediction[f"stable_contact_{side}"] = bool(gate_at_anchor[side])
            mu_avg = 0.5 * (
                float(prediction["mu_left"]) + float(prediction["mu_right"])
            )
            uncertainty_scale = max(1e-6, 0.5 * (
                float(prediction["sigma_left"]) + float(prediction["sigma_right"])
            ))
            prediction.update({
                "mu_avg": mu_avg,
                "uncertainty_scale": uncertainty_scale,
                "sigma_avg": uncertainty_scale,
                "lower_avg": mu_avg + self.bilateral_lower_alpha * uncertainty_scale,
                "upper_avg": mu_avg + self.bilateral_upper_alpha * uncertainty_scale,
                "lower_alpha": self.bilateral_lower_alpha,
                "upper_alpha": self.bilateral_upper_alpha,
            })
            return {
                "ok": True,
                "anchor_unix_ns": anchor_ns,
                "prediction": prediction,
                "history_report": history_report,
            }

    def evaluate_chunk(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if request.get("policy_feature_source") != FEATURE_SOURCE:
                raise ValueError("evaluate_chunk requires Pi0.5 JAX z_rl features")
            feature = np.asarray(request["policy_feature"], dtype=np.float32).reshape(-1)
            state = np.asarray(request["robot_state"], dtype=np.float32).reshape(-1)
            action_key = (
                "executed_action_chunk"
                if "executed_action_chunk" in request
                else "proposed_action_chunk"
            )
            actions = np.asarray(request[action_key], dtype=np.float32)
            if feature.shape != (512,):
                raise ValueError(f"policy_feature must be [512], got {feature.shape}")
            if state.shape != (8,):
                raise ValueError(f"robot_state must be [8], got {state.shape}")
            if actions.ndim != 2 or actions.shape[1] != self.predictor.action_dim:
                raise ValueError(
                    f"{action_key} must be [N,{self.predictor.action_dim}], got {actions.shape}"
                )
            if actions.shape[0] < self.predictor.action_steps:
                raise ValueError(
                    f"{action_key} has {actions.shape[0]} steps; "
                    f"Envelope requires {self.predictor.action_steps}"
                )
            reward_count = int(request.get("reward_count", actions.shape[0]))
            steps_executed = int(request.get("steps_executed", reward_count))
            start_ns = int(request["started_unix_ns"])
            requested_end_ns = start_ns + int(self.predictor.control_window_ms * 1e6)
            finished_ns = int(request.get("finished_unix_ns", requested_end_ns))
            window_end_ns = min(requested_end_ns, finished_ns)
            if window_end_ns <= start_ns:
                raise ValueError("execution interval does not contain a positive tactile window")
            window_complete = bool(
                steps_executed >= self.predictor.action_steps
                and window_end_ns - start_ns
                >= int(0.95 * self.predictor.control_window_ms * 1e6)
            )
            query_start_ns = start_ns - int(self.history_ms * 1e6)
            rows = self.sample_source.recent(query_start_ns, window_end_ns)

            cached_prediction = request.get("cached_prediction")
            if cached_prediction is None:
                self._update_gate_until(rows, start_ns)
                gate_at_anchor = self.gate.active
                history, history_mask, history_report = build_tactile_history(
                    rows,
                    self.baselines,
                    anchor_ns=start_ns,
                    history_ms=self.history_ms,
                    history_steps=self.predictor.history_steps,
                )
                prediction = self.predictor.predict(
                    feature[None],
                    np.ones(1, dtype=bool),
                    state,
                    actions[: self.predictor.action_steps],
                    tactile_history=history,
                    tactile_history_mask=history_mask,
                )
            else:
                prediction = dict(cached_prediction)
                required = {
                    "mu_left",
                    "sigma_left",
                    "upper95_left",
                    "mu_right",
                    "sigma_right",
                    "upper95_right",
                    "stable_contact_left",
                    "stable_contact_right",
                }
                missing = sorted(required - prediction.keys())
                if missing:
                    raise ValueError(
                        f"cached_prediction is missing fields: {missing}"
                    )
                gate_at_anchor = {
                    "left": bool(prediction["stable_contact_left"]),
                    "right": bool(prediction["stable_contact_right"]),
                }
                history_report = dict(request.get("cached_history_report", {}))
                if not all(side in history_report for side in SIDES):
                    _, _, history_report = build_tactile_history(
                        rows,
                        self.baselines,
                        anchor_ns=start_ns,
                        history_ms=self.history_ms,
                        history_steps=self.predictor.history_steps,
                    )
            observed, raw_observed = self._observed_forces(
                rows, start_ns, window_end_ns
            )
            reward_criterion = str(
                request.get("reward_criterion", ADAPTIVE_ENVELOPE)
            )
            fixed_force_threshold_N = request.get("fixed_force_threshold_N")
            if fixed_force_threshold_N is not None:
                fixed_force_threshold_N = float(fixed_force_threshold_N)
            costs, cost_report = self._costs(
                observed,
                raw_observed,
                reward_count=reward_count,
                steps_executed=steps_executed,
                window_complete=window_complete,
                prediction=prediction,
                gate_at_anchor=gate_at_anchor,
                reward_criterion=reward_criterion,
                fixed_force_threshold_N=fixed_force_threshold_N,
            )
            self._update_gate_until(rows, window_end_ns)

            latest_tactile_ns = max(
                (
                    int(row["pc_time_ns"])
                    for side in SIDES
                    for row in rows.get(side, [])
                ),
                default=0,
            )
            metrics = {
                "gate_left": float(gate_at_anchor["left"]),
                "gate_right": float(gate_at_anchor["right"]),
                "gate_bilateral": float(cost_report["bilateral_gate"]),
                "reward_criterion_adaptive": float(
                    reward_criterion == ADAPTIVE_ENVELOPE
                ),
                "reward_criterion_fixed_upper": float(
                    reward_criterion == FIXED_UPPER
                ),
                "fixed_force_threshold_N": float(
                    fixed_force_threshold_N
                    if fixed_force_threshold_N is not None
                    else -1.0
                ),
                "window_complete": float(cost_report["window_complete"]),
                "mu_left_N": float(prediction["mu_left"]),
                "mu_right_N": float(prediction["mu_right"]),
                "sigma_left_N": float(prediction["sigma_left"]),
                "sigma_right_N": float(prediction["sigma_right"]),
                "upper95_left_N": float(prediction["upper95_left"]),
                "upper95_right_N": float(prediction["upper95_right"]),
                "mu_avg_N": float(cost_report["mu_avg_N"]),
                "uncertainty_scale_N": float(cost_report["uncertainty_scale_N"]),
                "sigma_avg_N": float(cost_report["sigma_avg_N"]),
                "lower_envelope_coverage": float(cost_report["lower_coverage"]),
                "lower_alpha": float(cost_report["lower_alpha"]),
                "lower_offset_avg_N": float(cost_report["lower_offset_N"]),
                "lower_avg_N": float(cost_report["lower_avg_N"]),
                "upper_envelope_coverage": float(cost_report["upper_coverage"]),
                "upper_alpha": float(cost_report["upper_alpha"]),
                "upper_offset_avg_N": float(cost_report["upper_offset_N"]),
                "upper_avg_N": float(cost_report["upper_avg_N"]),
                "observed_avg_N": (
                    float(cost_report["target_avg_N"])
                    if cost_report["target_avg_N"] is not None
                    else -1.0
                ),
                "observed_left_top2_mean_N": (
                    float(cost_report["target_top2_mean_N"]["left"])
                    if cost_report["target_top2_mean_N"]["left"] is not None
                    else -1.0
                ),
                "observed_right_top2_mean_N": (
                    float(cost_report["target_top2_mean_N"]["right"])
                    if cost_report["target_top2_mean_N"]["right"] is not None
                    else -1.0
                ),
                "low_deviation_N": float(cost_report["low_deviation_N"]),
                "high_deviation_N": float(cost_report["high_deviation_N"]),
                "low_normalized_cost": float(cost_report["low_normalized_cost"]),
                "high_normalized_cost": float(cost_report["high_normalized_cost"]),
                "raw_peak_N": (
                    float(cost_report["raw_peak_N"])
                    if cost_report["raw_peak_N"] is not None
                    else -1.0
                ),
                "raw_peak_threshold_N": float(cost_report["raw_peak_threshold_N"]),
                "peak_exceedance_N": float(cost_report["peak_exceedance_N"]),
                "peak_triggered": float(cost_report["peak_triggered"]),
                "quality_valid": float(cost_report["quality_valid"]),
                "low_quality_window": float(cost_report["low_quality_window"]),
                "cost_index": float(cost_report["cost_index"]),
                "history_samples_left": float(history_report["left"]["history_samples"]),
                "history_samples_right": float(history_report["right"]["history_samples"]),
                "observed_samples_left": float(len(observed["left"])),
                "observed_samples_right": float(len(observed["right"])),
                "window_tail_gap_ms": (
                    float((window_end_ns - latest_tactile_ns) / 1e6)
                    if latest_tactile_ns
                    else -1.0
                ),
                "envelope_reward_steps": 1.0,
                "cached_prediction": float(cached_prediction is not None),
                "cost_sum": float(costs.sum()),
                "cost_max": float(costs.max(initial=0.0)),
                "average_cost_sum": float(np.sum(cost_report["average_costs"])),
                "low_cost_sum": float(cost_report["low_normalized_cost"]),
                "high_cost_sum": float(cost_report["high_normalized_cost"]),
                "peak_cost_sum": float(np.sum(cost_report["peak_costs"])),
            }
            event = {
                "timestamp_ns": time.time_ns(),
                "request_start_ns": start_ns,
                "request_finish_ns": int(request.get("finished_unix_ns", 0)),
                "gate_at_anchor": gate_at_anchor,
                "gate_after_window": self.gate.active,
                "history": history_report,
                "prediction": prediction,
                "costs": costs.tolist(),
                **cost_report,
            }
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            return {
                "ok": True,
                "costs": costs.tolist(),
                "average_costs": cost_report["average_costs"],
                "peak_costs": cost_report["peak_costs"],
                "metrics": metrics,
            }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = str(request.get("type", ""))
        if request_type == "metadata":
            return self.metadata()
        if request_type == "reset_episode":
            return self.reset_episode()
        if request_type == "predict_chunk":
            return self.predict_chunk(request)
        if request_type == "evaluate_chunk":
            return self.evaluate_chunk(request)
        return {"ok": False, "error": f"unknown request type: {request_type}"}
