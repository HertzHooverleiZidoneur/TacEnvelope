#!/usr/bin/env python3
"""Record a bounded Franka SmolVLA + force-envelope execution episode.

This is intentionally separate from the established policy clients.  It uses
their public helpers, but does not modify their execution behavior or files.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from franka_smolvla_tcp_client import (
    IMAGE_KEYS,
    load_policy,
    observation_to_batch,
    request,
    reset_remote,
    select_device,
    wait_for_queue_execution,
)
from tactile_force_envelope.data import median_filter
from tactile_force_envelope.inference import ForceEnvelopePredictor
from tactile_force_envelope.smolvla_condition import (
    predict_action_chunk_with_condition_tokens,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = Path(
    "/path/to/lerobot/outputs/train/"
    "franka_tactile310_smolvla_b64_e10_20260717_152937/"
    "checkpoints/022123/pretrained_model"
)
DEFAULT_ENVELOPE = Path(
    PROJECT_ROOT / "checkpoints/production/current_stable_contact.pt"
)
DEFAULT_BILATERAL_EVALUATION = Path(
    PROJECT_ROOT / "checkpoints/production/current_bilateral_calibration.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/live_tests/franka_live_envelope_tests"
DEFAULT_TASK = "Insert the red USB flash drive into the USB port on the black power strip"
DEFAULT_XENSE_PYTHON = Path("/home/user/.venvs/xense/bin/python")
DEFAULT_XENSE_SCRIPT = Path(
    PROJECT_ROOT / "integrations/xense/pc_xense_capture.py"
)
SIDES = ("left_tip", "right_tip")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def load_bilateral_calibration(path: Path, envelope_checkpoint: Path) -> dict[str, Any]:
    """Load the calibration-only additive q95 matching this exact checkpoint."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = envelope_checkpoint.resolve()
    for run in payload.get("runs", []):
        checkpoint = Path(run["checkpoint"]).expanduser().resolve()
        if checkpoint == expected:
            return {
                "method": payload.get("method"),
                "selection": payload.get("selection"),
                "q95_N": float(run["q95"]),
                "seed": int(run["seed"]),
                "calibration_samples": int(run["calibration_samples"]),
                "source": str(path),
                "checkpoint": str(checkpoint),
                "test_used_for_q95": False,
            }
    raise ValueError(
        f"No bilateral calibration in {path} matches checkpoint {envelope_checkpoint}"
    )


def unix_request(socket_path: Path, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    client.connect(str(socket_path))
    client.sendall((json.dumps(payload, separators=(",", ":")) + "\n").encode())
    response = b""
    while not response.endswith(b"\n"):
        chunk = client.recv(65536)
        if not chunk:
            break
        response += chunk
    client.close()
    return json.loads(response.decode())


def xense_environment() -> dict[str, str]:
    environment = dict(os.environ)
    nvidia_root = Path("/path/to/lerobot/.venv/lib/python3.13/site-packages/nvidia")
    libraries = sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
    if libraries:
        old = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(libraries + ([old] if old else []))
    return environment


class XenseSession:
    def __init__(
        self,
        run_dir: Path,
        python: Path,
        script: Path,
        serials: str,
        task: str,
    ) -> None:
        self.run_dir = run_dir
        self.python = python
        self.script = script
        self.serials = serials
        self.task = task
        self.socket_path = Path(f"/tmp/franka_envelope_xense_{os.getpid()}.sock")
        self.process: subprocess.Popen[bytes] | None = None
        self.session_path: Path | None = None
        self.log_handle: Any = None

    def start(self) -> None:
        tactile_root = self.run_dir / "tactile_raw"
        tactile_root.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.run_dir / "xense_daemon.log").open("ab", buffering=0)
        command = [
            str(self.python), str(self.script), "daemon",
            "--socket", str(self.socket_path),
            "--root", str(tactile_root),
            "--serials", self.serials,
            "--names", "left_tip,right_tip",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=xense_environment(),
        )
        deadline = time.monotonic() + 90.0
        status: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Xense daemon exited with code {self.process.returncode}")
            try:
                status = unix_request(self.socket_path, {"command": "status"}, timeout=2.0)
                if status.get("ok") and len(status.get("sensors", [])) == 2:
                    break
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError(f"Xense daemon did not become ready; see {self.run_dir/'xense_daemon.log'}")
        write_json(self.run_dir / "xense_start_status.json", status)
        response = unix_request(
            self.socket_path,
            {
                "command": "start",
                "episode_index": 0,
                "dataset_name": "live_envelope_test",
                "task": self.task,
                "pre_roll_sec": 0.0,
            },
        )
        if not response.get("ok"):
            raise RuntimeError(f"Cannot start Xense session: {response}")
        self.session_path = Path(response["path"])

    def rows(self, side: str) -> list[dict[str, Any]]:
        if self.session_path is None:
            return []
        path = self.session_path / side / "samples.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # The daemon may still be appending the final line.
                    continue
        return rows

    def wait_for_samples(self, count: int, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            counts = [len(self.rows(side)) for side in SIDES]
            if min(counts, default=0) >= count:
                return
            time.sleep(0.1)
        raise TimeoutError(f"Xense sample warmup timed out: {counts}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                result = unix_request(self.socket_path, {"command": "stop"})
                write_json(self.run_dir / "xense_stop_status.json", result)
            except Exception as exc:
                write_json(self.run_dir / "xense_stop_error.json", {"error": str(exc)})
            try:
                unix_request(self.socket_path, {"command": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        self.socket_path.unlink(missing_ok=True)
        if self.log_handle is not None:
            self.log_handle.close()


def row_xyz(row: dict[str, Any]) -> np.ndarray:
    mapping = dict(zip(row["feature_names"], row["values"], strict=False))
    return np.asarray(
        [mapping["force_fx"], mapping["force_fy"], mapping["force_fz"]],
        dtype=np.float32,
    )


def row_valid(row: dict[str, Any]) -> bool:
    mapping = dict(zip(row["feature_names"], row["values"], strict=False))
    return bool(float(mapping.get("force_valid", 1.0)) > 0.5)


def calculate_baselines(
    tactile: XenseSession, baseline_started_ns: int, baseline_ended_ns: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    baselines: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    for side in SIDES:
        selected = [
            row for row in tactile.rows(side)
            if baseline_started_ns <= int(row["pc_time_ns"]) < baseline_ended_ns and row_valid(row)
        ]
        if len(selected) < 8:
            raise RuntimeError(f"{side}: only {len(selected)} valid samples in baseline interval")
        xyz = np.stack([row_xyz(row) for row in selected])
        baseline = np.median(xyz, axis=0).astype(np.float32)
        residual = np.linalg.norm(xyz - baseline, axis=1)
        baselines[side] = baseline
        report[side] = {
            "sample_count": len(selected),
            "baseline_xyz": baseline,
            "residual_median_N": float(np.median(residual)),
            "residual_p95_N": float(np.quantile(residual, 0.95)),
            "residual_max_N": float(np.max(residual)),
        }
    return baselines, report


def tactile_history(
    tactile: XenseSession,
    baselines: dict[str, np.ndarray],
    anchor_ns: int,
    history_steps: int,
    history_ms: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    history = np.zeros((2, history_steps, 5), dtype=np.float32)
    mask = np.zeros((2, history_steps), dtype=bool)
    report: dict[str, Any] = {}
    cutoff_ns = anchor_ns - int(history_ms * 1e6)
    for side_index, side in enumerate(SIDES):
        selected = [
            row for row in tactile.rows(side)
            if cutoff_ns <= int(row["pc_time_ns"]) < anchor_ns and row_valid(row)
        ][-history_steps:]
        offset = history_steps - len(selected)
        for index, row in enumerate(selected):
            xyz = row_xyz(row) - baselines[side]
            history[side_index, offset + index] = [
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                float(np.linalg.norm(xyz)),
                # Match the offline training cache: normalized causal time in [-1, 0).
                (int(row["pc_time_ns"]) - anchor_ns) / (history_ms * 1e6),
            ]
            mask[side_index, offset + index] = True
        report[side] = {
            "valid_samples": len(selected),
            "latest_age_ms": (
                (anchor_ns - int(selected[-1]["pc_time_ns"])) / 1e6 if selected else None
            ),
        }
    return history, mask, report


def save_anchor_images(observation: dict[str, Any], output: Path, step: int) -> dict[str, str]:
    image_dir = output / "anchor_images"
    image_dir.mkdir(exist_ok=True)
    paths: dict[str, str] = {}
    for camera_index, key in enumerate(IMAGE_KEYS, start=1):
        payload = observation["images"][key]
        path = image_dir / f"chunk_{step:04d}_camera_{camera_index}.jpg"
        path.write_bytes(base64.b64decode(payload["data"]))
        paths[key] = str(path.relative_to(output))
    return paths


def infer_action_and_condition(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, Any],
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], float]:
    batch = observation_to_batch(observation, task)
    state = np.asarray(batch["observation.state"], dtype=np.float32)
    processed = preprocessor(batch)
    autocast = (
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and bool(getattr(policy.config, "use_amp", False))
        else nullcontext()
    )
    started = time.perf_counter()
    with torch.inference_mode(), autocast:
        actions, condition, condition_mask, metadata = (
            predict_action_chunk_with_condition_tokens(policy, processed)
        )
    actions = postprocessor(actions).squeeze(0).detach().float().cpu().numpy()
    return (
        actions,
        condition.squeeze(0).detach().float().cpu().numpy(),
        condition_mask.squeeze(0).detach().cpu().numpy(),
        state,
        metadata,
        time.perf_counter() - started,
    )


def window_label(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    start_ns: int,
    window_ms: int,
    median_size: int = 3,
) -> dict[str, Any]:
    end_ns = start_ns + int(window_ms * 1e6)
    selected = [
        row for row in rows
        if start_ns <= int(row["pc_time_ns"]) < end_ns and row_valid(row)
    ]
    timestamps = np.asarray([int(row["pc_time_ns"]) for row in selected], dtype=np.int64)
    if selected:
        xyz = np.stack([row_xyz(row) for row in selected]) - baseline
        raw = np.linalg.norm(xyz, axis=1)
        filtered = median_filter(raw, median_size)
        ordered = np.sort(filtered)[::-1]
        top2 = float(np.mean(ordered[:2]))
        raw_max = float(np.max(raw))
    else:
        raw = np.asarray([], dtype=np.float32)
        filtered = raw.copy()
        top2 = float("nan")
        raw_max = float("nan")
    return {
        "sample_count": len(selected),
        "quality_ok": len(selected) >= 4,
        "max_gap_ms": (
            float(np.max(np.diff(timestamps)) / 1e6) if timestamps.size >= 2 else None
        ),
        "filtered_top2_mean_N": top2,
        "raw_max_N": raw_max,
        "sample_timestamps_ns": timestamps.tolist(),
        "raw_magnitude_N": raw.tolist(),
        "filtered_magnitude_N": filtered.tolist(),
    }


def finish_events(
    events: list[dict[str, Any]],
    tactile_rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    window_ms: int,
    bilateral_q95: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        result = dict(event)
        for side_name, short in (("left_tip", "left"), ("right_tip", "right")):
            label = window_label(
                tactile_rows[side_name], baselines[side_name],
                int(event["action_anchor_pc_ns"]), window_ms,
            )
            target = float(label["filtered_top2_mean_N"])
            upper = float(event[f"upper95_{short}"])
            mu = float(event[f"mu_{short}"])
            result[f"target_{short}_top2_N"] = target
            result[f"raw_max_{short}_N"] = float(label["raw_max_N"])
            result[f"sample_count_{short}"] = int(label["sample_count"])
            result[f"max_gap_{short}_ms"] = label["max_gap_ms"]
            result[f"quality_{short}"] = bool(label["quality_ok"])
            result[f"deviation_{short}_N"] = target - upper
            result[f"absolute_error_{short}_N"] = abs(target - mu)
            result[f"covered_{short}"] = bool(target <= upper) if np.isfinite(target) else False
            result[f"label_detail_{short}"] = label
        target_avg = 0.5 * (
            float(result["target_left_top2_N"]) + float(result["target_right_top2_N"])
        )
        mu_avg = 0.5 * (float(result["mu_left"]) + float(result["mu_right"]))
        upper_avg = mu_avg + bilateral_q95
        result.update({
            "quality_bilateral": bool(result["quality_left"] and result["quality_right"]),
            "stable_contact_bilateral": bool(
                result["stable_contact_left"] and result["stable_contact_right"]
            ),
            "target_bilateral_top2_N": target_avg,
            "mu_bilateral": mu_avg,
            # Diagnostic only; the production additive envelope uses calibration q95.
            "sigma_bilateral": 0.5 * float(np.hypot(result["sigma_left"], result["sigma_right"])),
            "q95_bilateral_N": bilateral_q95,
            "upper95_bilateral": upper_avg,
            "deviation_bilateral_N": target_avg - upper_avg,
            "absolute_error_bilateral_N": abs(target_avg - mu_avg),
            "covered_bilateral": bool(target_avg <= upper_avg) if np.isfinite(target_avg) else False,
        })
        rows.append(result)
    return rows


def finite(values: list[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def metric_summary(rows: list[dict[str, Any]], side: str, gated_only: bool) -> dict[str, Any]:
    if side == "bilateral":
        target_key = "target_bilateral_top2_N"
        gate_key = "stable_contact_bilateral"
    else:
        target_key = f"target_{side}_top2_N"
        gate_key = f"stable_contact_{side}"
    selected = [
        row for row in rows
        if row.get(f"quality_{side}")
        and np.isfinite(float(row[target_key]))
        and (not gated_only or row.get(gate_key))
    ]
    if not selected:
        return {"windows": 0, "coverage": None, "mae_N": None, "deviation_mean_N": None,
                "deviation_p95_N": None, "deviation_max_N": None, "exceedance_windows": 0}
    deviations = np.asarray([row[f"deviation_{side}_N"] for row in selected], dtype=np.float64)
    errors = np.asarray([row[f"absolute_error_{side}_N"] for row in selected], dtype=np.float64)
    return {
        "windows": len(selected),
        "coverage": float(np.mean(deviations <= 0.0)),
        "mae_N": float(np.mean(errors)),
        "deviation_mean_N": float(np.mean(deviations)),
        "deviation_p95_N": float(np.quantile(deviations, 0.95)),
        "deviation_max_N": float(np.max(deviations)),
        "exceedance_windows": int(np.sum(deviations > 0.0)),
    }


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_reports(
    rows: list[dict[str, Any]],
    tactile_rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    output: Path,
) -> None:
    if not rows:
        return
    origin_ns = int(rows[0]["action_anchor_pc_ns"])
    times = np.asarray([(int(row["action_anchor_pc_ns"]) - origin_ns) / 1e9 for row in rows])
    colors = {"left": "#2b6cb0", "right": "#d9485f"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, ("left", "right"), strict=True):
        target = np.asarray([row[f"target_{side}_top2_N"] for row in rows])
        mu = np.asarray([row[f"mu_{side}"] for row in rows])
        upper = np.asarray([row[f"upper95_{side}"] for row in rows])
        gate = np.asarray([row[f"stable_contact_{side}"] for row in rows], dtype=bool)
        axis.fill_between(times, mu, upper, color=colors[side], alpha=0.16, label="mu to upper95")
        axis.plot(times, mu, "--", color=colors[side], label="predicted mu")
        axis.plot(times, upper, "-", color=colors[side], label="upper95")
        axis.plot(times, target, "o-", color="#222222", label="actual filtered top2 mean")
        axis.scatter(times[gate], target[gate], s=70, facecolors="none", edgecolors="#16a34a",
                     linewidths=1.6, label="stable gate on")
        axis.set_ylabel(f"{side} force (N)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("time t from first executed chunk (s)")
    fig.suptitle("Actual force and calibrated one-sided force envelope")
    save_figure(fig, output / "force_and_envelope")

    fig, axis = plt.subplots(figsize=(12, 5.5))
    for side in ("left", "right"):
        deviation = np.asarray([row[f"deviation_{side}_N"] for row in rows])
        axis.plot(times, deviation, "o-", color=colors[side], label=f"{side}: actual - upper95")
        exceeded = deviation > 0.0
        axis.scatter(times[exceeded], deviation[exceeded], s=75, color="#dc2626", marker="x")
    axis.axhline(0.0, color="#111827", linewidth=1.2)
    axis.fill_between([times.min(), times.max()], 0, axis.get_ylim()[1], color="#dc2626", alpha=0.06)
    axis.set_xlabel("time t from first executed chunk (s)")
    axis.set_ylabel("deviation: actual top2 - upper95 (N)")
    axis.set_title("Envelope deviation (positive means envelope exceedance)")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, output / "envelope_deviation")

    target_avg = np.asarray([row["target_bilateral_top2_N"] for row in rows])
    mu_avg = np.asarray([row["mu_bilateral"] for row in rows])
    upper_avg = np.asarray([row["upper95_bilateral"] for row in rows])
    gate_avg = np.asarray([row["stable_contact_bilateral"] for row in rows], dtype=bool)
    deviation_avg = target_avg - upper_avg

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.fill_between(times, mu_avg, upper_avg, color="#0f766e", alpha=0.17,
                      label="mu_avg to upper95_avg")
    axis.plot(times, mu_avg, "--", color="#0f766e", label="predicted mu_avg")
    axis.plot(times, upper_avg, "-", color="#0f766e", label="mu_avg + calibration q95")
    axis.plot(times, target_avg, "o-", color="#202938", label="observed y_avg")
    axis.scatter(times[gate_avg], target_avg[gate_avg], s=75, facecolors="none",
                 edgecolors="#16a34a", linewidths=1.8, label="both stable gates on")
    exceeded = gate_avg & (deviation_avg > 0.0)
    axis.scatter(times[exceeded], target_avg[exceeded], s=90, color="#dc2626", marker="x",
                 label="stable-gated exceedance")
    axis.set_xlabel("time t from first executed chunk (s)")
    axis.set_ylabel("bilateral average force (N)")
    axis.set_title("Bilateral average force and calibration-only additive envelope")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    save_figure(fig, output / "bilateral_average_force_and_envelope")

    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(times, deviation_avg, "o-", color="#0f766e",
              label="y_avg - (mu_avg + q95)")
    axis.scatter(times[gate_avg], deviation_avg[gate_avg], s=75, facecolors="none",
                 edgecolors="#16a34a", linewidths=1.8, label="both stable gates on")
    axis.scatter(times[exceeded], deviation_avg[exceeded], s=90, color="#dc2626", marker="x",
                 label="stable-gated exceedance")
    axis.axhline(0.0, color="#111827", linewidth=1.2)
    axis.set_xlabel("time t from first executed chunk (s)")
    axis.set_ylabel("y_avg - upper95_avg (N)")
    axis.set_title("Bilateral average envelope deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, output / "bilateral_average_deviation")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, (side_name, short) in zip(axes, (("left_tip", "left"), ("right_tip", "right")), strict=True):
        samples = tactile_rows[side_name]
        sample_t = np.asarray([(int(row["pc_time_ns"]) - origin_ns) / 1e9 for row in samples])
        sample_f = np.asarray([
            float(np.linalg.norm(row_xyz(row) - baselines[side_name]))
            if row_valid(row) else np.nan
            for row in samples
        ])
        axis.plot(
            sample_t, sample_f, ".", markersize=2.5, color="#6b7280",
            label="baseline-corrected raw magnitude",
        )
        for index, row in enumerate(rows):
            t0 = times[index]
            t1 = t0 + float(row["window_ms"]) / 1000.0
            axis.hlines(float(row[f"upper95_{short}"]), t0, t1, colors=colors[short], linewidth=3)
        axis.set_ylabel(f"{short} force (N)")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel("time t from first executed chunk (s)")
    fig.suptitle("Raw tactile timeline; colored segments are 300 ms upper95 windows")
    save_figure(fig, output / "raw_force_timeline")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "chunk_index", "time_t_s", "infer_s", "observation_to_action_ms", "window_ms",
        "stable_contact_left", "mu_left", "sigma_left", "upper95_left",
        "target_left_top2_N", "raw_max_left_N", "deviation_left_N", "covered_left",
        "sample_count_left", "max_gap_left_ms", "quality_left",
        "stable_contact_right", "mu_right", "sigma_right", "upper95_right",
        "target_right_top2_N", "raw_max_right_N", "deviation_right_N", "covered_right",
        "sample_count_right", "max_gap_right_ms", "quality_right",
        "stable_contact_bilateral", "mu_bilateral", "sigma_bilateral",
        "q95_bilateral_N", "upper95_bilateral", "target_bilateral_top2_N",
        "deviation_bilateral_N", "absolute_error_bilateral_N", "covered_bilateral",
        "quality_bilateral",
    ]
    origin_ns = int(rows[0]["action_anchor_pc_ns"]) if rows else 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compact = {key: row.get(key) for key in fields}
            compact["time_t_s"] = (int(row["action_anchor_pc_ns"]) - origin_ns) / 1e9
            writer.writerow(compact)


def render_html(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    table_rows = []
    origin_ns = int(rows[0]["action_anchor_pc_ns"]) if rows else 0
    for row in rows:
        t = (int(row["action_anchor_pc_ns"]) - origin_ns) / 1e9
        table_rows.append(
            "<tr>"
            f"<td>{row['chunk_index']}</td><td>{t:.2f}</td>"
            f"<td>{row['target_left_top2_N']:.3f}</td><td>{row['upper95_left']:.3f}</td>"
            f"<td class={'bad' if row['deviation_left_N'] > 0 else 'good'}>{row['deviation_left_N']:.3f}</td>"
            f"<td>{'ON' if row['stable_contact_left'] else 'off'}</td>"
            f"<td>{row['target_right_top2_N']:.3f}</td><td>{row['upper95_right']:.3f}</td>"
            f"<td class={'bad' if row['deviation_right_N'] > 0 else 'good'}>{row['deviation_right_N']:.3f}</td>"
            f"<td>{'ON' if row['stable_contact_right'] else 'off'}</td>"
            f"<td>{row['target_bilateral_top2_N']:.3f}</td>"
            f"<td>{row['mu_bilateral']:.3f}</td><td>{row['upper95_bilateral']:.3f}</td>"
            f"<td class={'bad' if row['deviation_bilateral_N'] > 0 else 'good'}>"
            f"{row['deviation_bilateral_N']:.3f}</td>"
            f"<td>{'ON' if row['stable_contact_bilateral'] else 'off'}</td>"
            "</tr>"
        )
    metrics = []
    for group in ("all_windows", "stable_gate_windows"):
        for side in ("left", "right", "bilateral"):
            metric = summary["metrics"][group][side]
            coverage = "-" if metric["coverage"] is None else f"{metric['coverage']*100:.1f}%"
            mae = "-" if metric["mae_N"] is None else f"{metric['mae_N']:.3f} N"
            metrics.append(
                f"<tr><td>{html.escape(group)}</td><td>{side}</td><td>{metric['windows']}</td>"
                f"<td>{coverage}</td><td>{mae}</td><td>{metric['exceedance_windows']}</td></tr>"
            )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Franka 实机力包络测试</title>
<style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:32px;color:#172033;max-width:1280px}}
h1,h2{{color:#102a43}} .note{{background:#edf6ff;border-left:4px solid #2b6cb0;padding:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #cbd5e1;padding:7px;text-align:right}}
th{{background:#eef2f7}}td:first-child,th:first-child{{text-align:left}}.bad{{color:#b91c1c;font-weight:700}}.good{{color:#166534}}
img{{max-width:100%;border:1px solid #d6dde8;margin:8px 0 24px}}
</style></head><body>
<h1>Franka SmolVLA + 触觉力包络实机测试</h1>
<div class="note">每次实际 action chunk 接受时刻作为 t；真实标签使用未来 {summary['window_ms']} ms 内、首秒零偏后的三轴合力，经三点中值滤波后的 top-2 均值。偏差定义为 actual - upper95，正值表示超出包络。</div>
<h2>汇总指标</h2><table><tr><th>统计范围</th><th>侧</th><th>窗口</th><th>Coverage</th><th>mu MAE</th><th>超界窗口</th></tr>{''.join(metrics)}</table>
<h2>实际力与包络</h2><img src="force_and_envelope.png">
<h2>偏差曲线</h2><img src="envelope_deviation.png">
<h2>左右平均力与包络</h2><img src="bilateral_average_force_and_envelope.png">
<h2>左右平均力偏差</h2><img src="bilateral_average_deviation.png">
<h2>原始触觉时间线</h2><img src="raw_force_timeline.png">
<h2>逐窗口数据</h2><table><tr><th>Chunk</th><th>t(s)</th><th>L actual</th><th>L upper</th><th>L deviation</th><th>L gate</th><th>R actual</th><th>R upper</th><th>R deviation</th><th>R gate</th><th>Avg actual</th><th>Avg mu</th><th>Avg upper</th><th>Avg deviation</th><th>双侧 gate</th></tr>{''.join(table_rows)}</table>
<p>平均力定义：y_avg=(y_left+y_right)/2；平均预测：mu_avg=(mu_left+mu_right)/2；平均力上界：mu_avg + calibration q95。q95={summary['bilateral_calibration']['q95_N']:.6f} N，仅由 calibration split 得到，本次实机目标未参与校准。异步原始触觉不做逐点插值平均。</p>
<p>模型：{html.escape(summary['policy'])}<br>Envelope：{html.escape(summary['envelope_checkpoint'])}</p>
</body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def generate_report(
    events: list[dict[str, Any]],
    tactile_rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    output: Path,
    policy: Path,
    envelope: Path,
    window_ms: int,
    bilateral_calibration: dict[str, Any],
) -> dict[str, Any]:
    rows = finish_events(
        events, tactile_rows, baselines, window_ms,
        float(bilateral_calibration["q95_N"]),
    )
    write_csv(rows, output / "prediction_windows.csv")
    write_json(output / "prediction_windows.json", rows)
    plot_reports(rows, tactile_rows, baselines, output)
    summary = {
        "policy": str(policy),
        "envelope_checkpoint": str(envelope),
        "window_ms": window_ms,
        "executed_chunks": len(rows),
        "metrics": {
            "all_windows": {
                side: metric_summary(rows, side, gated_only=False)
                for side in ("left", "right", "bilateral")
            },
            "stable_gate_windows": {
                side: metric_summary(rows, side, gated_only=True)
                for side in ("left", "right", "bilateral")
            },
        },
        "bilateral_calibration": bilateral_calibration,
        "deviation_definition": "actual filtered_top2_mean - calibrated upper95; positive is exceedance",
        "label_definition": "first-second XYZ baseline, magnitude, median filter size 3, top-2 mean",
        "action_anchor": "midpoint of local enqueue request send and acknowledgement receive",
        "action_anchor_uncertainty": "plus at most one remote 30 Hz policy publish period",
    }
    write_json(output / "summary.json", summary)
    render_html(rows, summary, output)
    return summary


def run_live(args: argparse.Namespace) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output = args.output_root.expanduser() / f"run_{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "command_config.json", vars(args))
    tactile = XenseSession(
        output, args.xense_python, args.xense_script, args.xense_serials, args.task
    )
    bilateral_calibration = load_bilateral_calibration(
        args.bilateral_evaluation, args.envelope_checkpoint
    )
    sock: socket.socket | None = None
    events: list[dict[str, Any]] = []
    baselines: dict[str, np.ndarray] = {}
    try:
        tactile.start()
        tactile.wait_for_samples(8)
        baseline_started_ns = time.time_ns()
        time.sleep(args.baseline_seconds)
        baseline_ended_ns = time.time_ns()
        baselines, baseline_report = calculate_baselines(
            tactile, baseline_started_ns, baseline_ended_ns
        )
        write_json(output / "tactile_baseline.json", baseline_report)

        sock = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
        sock.settimeout(args.socket_timeout)
        status = request(sock, {"type": "status"})
        write_json(output / "remote_start_status.json", status)
        if not status.get("ok"):
            raise RuntimeError(f"Remote runtime is not ready: {status.get('missing')}")
        if args.send_actions and not status.get("execute"):
            raise RuntimeError("--send-actions requires remote execute=True; start the execute runtime")
        if args.reset_before_run:
            reset_remote(sock, args.reset_seconds, args.reset_timeout)

        device = select_device(args.device)
        loaded_at = time.perf_counter()
        policy, preprocessor, postprocessor = load_policy(args.policy, device)
        predictor = ForceEnvelopePredictor(args.envelope_checkpoint, device=str(device))
        if not hasattr(predictor, "history_steps") or predictor.history_steps <= 0:
            raise RuntimeError("Selected Envelope checkpoint has no causal tactile history")
        expected = predictor.feature_metadata.get("checkpoint")
        if expected and Path(expected).resolve() != args.policy.resolve():
            raise RuntimeError(f"Envelope expects policy {expected}, got {args.policy}")
        write_json(output / "model_load.json", {
            "seconds": time.perf_counter() - loaded_at,
            "device": str(device),
            "condition_steps": predictor.condition_steps,
            "condition_dim": predictor.condition_dim,
            "history_steps": predictor.history_steps,
            "action_steps": predictor.action_steps,
            "control_window_ms": predictor.control_window_ms,
            "production_ready": predictor.production_ready,
        })

        previous_chunk_size = 0
        for chunk_index in range(args.max_chunks):
            if previous_chunk_size:
                wait_for_queue_execution(
                    sock,
                    sent_chunk_size=previous_chunk_size,
                    min_execute=args.queue_min_execute,
                    poll_interval=args.queue_poll_interval,
                    timeout=args.queue_wait_timeout,
                )
            observation_request_ns = time.time_ns()
            observation = request(sock, {"type": "get_observation", "client_sent_at": time.time()})
            if not observation.get("ok"):
                raise RuntimeError(f"Observation became unavailable: {observation.get('missing')}")
            image_paths = save_anchor_images(observation, output, chunk_index)

            (
                actions,
                condition,
                condition_mask,
                model_state,
                condition_metadata,
                infer_s,
            ) = infer_action_and_condition(
                policy, preprocessor, postprocessor, observation, args.task, device,
            )
            chunk = actions[: args.send_chunk_size, :8]
            # Refresh tactile history after the comparatively slow SmolVLA forward,
            # immediately before envelope inference and actual action enqueue.
            history_anchor_ns = time.time_ns()
            history, history_mask, history_report = tactile_history(
                tactile, baselines, history_anchor_ns,
                predictor.history_steps, args.history_ms,
            )
            envelope_result = predictor.predict(
                condition,
                condition_mask,
                model_state,
                chunk[: predictor.action_steps, : predictor.action_dim],
                args.task,
                tactile_history=history,
                tactile_history_mask=history_mask,
            )
            send_ns = time.time_ns()
            if args.send_actions:
                acknowledgement = request(
                    sock,
                    {"type": "enqueue_actions", "actions": chunk.tolist(), "replace": True},
                )
                if not acknowledgement.get("ok"):
                    raise RuntimeError(f"Remote rejected action chunk: {acknowledgement}")
            else:
                acknowledgement = {"ok": True, "execute": False, "queued": 0, "queue_size": 0}
            acknowledgement_ns = time.time_ns()
            action_anchor_ns = (send_ns + acknowledgement_ns) // 2
            event = {
                "chunk_index": chunk_index,
                "action_anchor_pc_ns": action_anchor_ns,
                "observation_request_pc_ns": observation_request_ns,
                "history_anchor_pc_ns": history_anchor_ns,
                "observation_to_action_ms": (action_anchor_ns - observation_request_ns) / 1e6,
                "enqueue_rtt_ms": (acknowledgement_ns - send_ns) / 1e6,
                "window_ms": predictor.control_window_ms,
                "infer_s": infer_s,
                "state": observation["state"],
                "action_chunk": chunk,
                "envelope_action_prefix": chunk[: predictor.action_steps],
                "first_action": chunk[0],
                "last_action": chunk[-1],
                "image_paths": image_paths,
                "image_age_s": {
                    key: float(value["age"]) for key, value in observation["images"].items()
                },
                "history_report": history_report,
                "history": history,
                "history_mask": history_mask,
                "condition_metadata": condition_metadata,
                "acknowledgement": acknowledgement,
                **envelope_result,
            }
            events.append(event)
            append_jsonl(output / "inference_events.jsonl", event)
            previous_chunk_size = len(chunk) if args.send_actions else 0
            print(
                f"[live_envelope] chunk={chunk_index}/{args.max_chunks} "
                f"send={args.send_actions} infer={infer_s:.3f}s "
                f"L(mu={event['mu_left']:.2f},upper={event['upper95_left']:.2f},gate={event['stable_contact_left']}) "
                f"R(mu={event['mu_right']:.2f},upper={event['upper95_right']:.2f},gate={event['stable_contact_right']})"
            )
        if previous_chunk_size:
            wait_for_queue_execution(
                sock,
                sent_chunk_size=previous_chunk_size,
                min_execute=args.queue_min_execute,
                poll_interval=args.queue_poll_interval,
                timeout=args.queue_wait_timeout,
            )
        time.sleep(max(0.4, args.window_ms / 1000.0 + 0.1))
    except KeyboardInterrupt:
        print("[live_envelope] interrupted; holding current pose", file=sys.stderr)
    finally:
        if sock is not None:
            try:
                request(sock, {"type": "hold"})
            except Exception:
                pass
            sock.close()
        tactile.stop()

    if not events:
        raise RuntimeError(f"No inference events were recorded; inspect {output}")
    tactile_rows = {side: tactile.rows(side) for side in SIDES}
    # The stopped session remains on disk, so rows are still available here.
    if not all(tactile_rows.values()) and tactile.session_path is not None:
        for side in SIDES:
            path = tactile.session_path / side / "samples.jsonl"
            if path.exists():
                tactile_rows[side] = [json.loads(line) for line in path.read_text().splitlines() if line]
    summary = generate_report(
        events, tactile_rows, baselines, output,
        args.policy, args.envelope_checkpoint, args.window_ms, bilateral_calibration,
    )
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"[live_envelope] report: {output/'report.html'}")
    return output


def self_test(output_root: Path) -> Path:
    output = output_root / f"self_test_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    origin = time.time_ns()
    tactile_rows: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    feature_names = ["force_valid", "force_fx", "force_fy", "force_fz"]
    for sample_index in range(240):
        stamp = origin + int(sample_index / 22.0 * 1e9)
        for side_index, side in enumerate(SIDES):
            force = 4.0 + side_index + 2.0 * np.sin(sample_index / 14.0)
            tactile_rows[side].append({
                "pc_time_ns": stamp,
                "feature_names": feature_names,
                "values": [1.0, 0.0, 0.0, float(force)],
            })
    events = []
    for index in range(6):
        anchor = origin + int((2.0 + index * 1.5) * 1e9)
        events.append({
            "chunk_index": index, "action_anchor_pc_ns": anchor, "window_ms": 300,
            "infer_s": 0.42, "observation_to_action_ms": 450.0,
            "mu_left": 4.5, "sigma_left": 1.0, "upper95_left": 6.0,
            "stable_contact_left": index >= 2,
            "mu_right": 5.0, "sigma_right": 1.2, "upper95_right": 6.8,
            "stable_contact_right": index >= 2,
        })
    generate_report(
        events, tactile_rows, {side: np.zeros(3, dtype=np.float32) for side in SIDES},
        output, Path("synthetic_policy"), Path("synthetic_envelope"), 300,
        {
            "method": "synthetic_additive_envelope", "selection": "both stable gates",
            "q95_N": 1.5, "seed": 0, "calibration_samples": 100,
            "source": "self-test", "checkpoint": "synthetic_envelope",
            "test_used_for_q95": False,
        },
    )
    print(f"[self-test] report: {output/'report.html'}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--envelope-checkpoint", type=Path, default=DEFAULT_ENVELOPE)
    parser.add_argument(
        "--bilateral-evaluation", type=Path, default=DEFAULT_BILATERAL_EVALUATION,
        help="Calibration-only bilateral evaluation JSON containing q95 for this checkpoint",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--send-chunk-size", type=int, default=50)
    parser.add_argument("--queue-min-execute", type=int, default=50)
    parser.add_argument("--queue-poll-interval", type=float, default=0.03)
    parser.add_argument("--queue-wait-timeout", type=float, default=8.0)
    parser.add_argument("--window-ms", type=int, default=300)
    parser.add_argument("--history-ms", type=int, default=200)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--reset-before-run", action="store_true")
    parser.add_argument("--reset-seconds", type=float, default=8.0)
    parser.add_argument("--reset-timeout", type=float, default=90.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--xense-python", type=Path, default=DEFAULT_XENSE_PYTHON)
    parser.add_argument("--xense-script", type=Path, default=DEFAULT_XENSE_SCRIPT)
    parser.add_argument("--xense-serials", default="OG000836,OG000837")
    parser.add_argument("--self-test-report", action="store_true")
    args = parser.parse_args()
    args.policy = args.policy.expanduser().resolve()
    args.envelope_checkpoint = args.envelope_checkpoint.expanduser().resolve()
    args.bilateral_evaluation = args.bilateral_evaluation.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    # Keep the virtualenv launcher path. Resolving this symlink to /usr/bin/python
    # bypasses the venv's site-packages and makes xensesdk unavailable.
    args.xense_python = args.xense_python.expanduser().absolute()
    args.xense_script = args.xense_script.expanduser().resolve()
    if args.max_chunks <= 0:
        parser.error("--max-chunks must be positive")
    if args.send_chunk_size <= 0:
        parser.error("--send-chunk-size must be positive")
    if not 1 <= args.queue_min_execute <= args.send_chunk_size:
        parser.error("--queue-min-execute must be in 1..send-chunk-size")
    if args.window_ms != 300:
        parser.error("This checkpoint was trained for a fixed 300 ms control window")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test_report:
        self_test(args.output_root)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
