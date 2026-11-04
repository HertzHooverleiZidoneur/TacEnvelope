from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import iter_jsonl, safe_float
from .discovery import DatasetSchema


@dataclass
class TactileStream:
    name: str
    timestamps: np.ndarray
    forces_xyz: np.ndarray
    valid: np.ndarray
    feature_names: list[str]
    timestamp_duplicates: int
    timestamp_rollbacks: int


def load_frames(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_tactile_streams(
    sidecar: Path, force_axes: list[str]
) -> tuple[list[TactileStream], dict[str, Any]]:
    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    offset_ns = int(manifest.get("nuc_minus_pc_offset_ns", 0))
    streams: list[TactileStream] = []
    for sensor in manifest.get("sensors", []):
        name = str(sensor["name"])
        rows = list(iter_jsonl(sidecar / name / "samples.jsonl"))
        if not rows:
            continue
        names = list(rows[0].get("feature_names") or manifest.get("feature_names") or [])
        missing = [axis for axis in force_axes if axis not in names]
        if missing:
            raise ValueError(f"{sidecar}/{name}: force fields not found: {missing}; have {names}")
        indices = [names.index(axis) for axis in force_axes]
        valid_index = names.index("force_valid") if "force_valid" in names else None
        timestamps: list[float] = []
        forces: list[list[float]] = []
        valid: list[bool] = []
        for row in rows:
            pc_ns = row.get("pc_time_ns")
            if pc_ns is None:
                raise ValueError(f"Raw tactile sample has no pc_time_ns: {sidecar}/{name}")
            values = row.get("values")
            if not isinstance(values, list) or max(indices) >= len(values):
                continue
            vector = [safe_float(values[index]) for index in indices]
            finite = all(value is not None for value in vector)
            is_valid = finite and (
                valid_index is None
                or valid_index >= len(values)
                or float(values[valid_index]) > 0.5
            )
            timestamps.append((int(pc_ns) + offset_ns) / 1e9)
            forces.append([float(value or 0.0) for value in vector])
            valid.append(bool(is_valid))
        raw_timestamps = np.asarray(timestamps, dtype=np.float64)
        raw_deltas = np.diff(raw_timestamps)
        order = np.argsort(raw_timestamps, kind="stable")
        streams.append(TactileStream(
            name=name,
            timestamps=np.asarray(timestamps, dtype=np.float64)[order],
            forces_xyz=np.asarray(forces, dtype=np.float32)[order],
            valid=np.asarray(valid, dtype=bool)[order],
            feature_names=names,
            timestamp_duplicates=int(np.sum(raw_deltas == 0)),
            timestamp_rollbacks=int(np.sum(raw_deltas < 0)),
        ))
    return streams, manifest


def anchor_indices(
    frames: list[dict[str, Any]], schema: DatasetSchema, mode: str, stride_s: float
) -> tuple[list[int], str]:
    if mode not in {"auto", "chunk", "stride", "every_frame"}:
        raise ValueError(f"Unknown anchor mode: {mode}")
    field = schema.action_chunk_start_field
    if mode in {"auto", "chunk"} and field:
        anchors: list[int] = []
        previous: Any = object()
        for index, frame in enumerate(frames):
            value = frame.get(field)
            if isinstance(value, bool):
                if value:
                    anchors.append(index)
            elif index == 0 or value != previous:
                anchors.append(index)
            previous = value
        return anchors, f"chunk_field:{field}"
    if mode == "chunk":
        raise ValueError("anchor_mode=chunk but no action chunk field was discovered")
    if mode == "every_frame":
        return list(range(len(frames))), "every_frame_fallback"
    timestamps = [float(row[schema.timestamp_field]) for row in frames]
    anchors = []
    next_time = -math.inf
    for index, timestamp in enumerate(timestamps):
        if timestamp >= next_time:
            anchors.append(index)
            next_time = timestamp + stride_s
    return anchors, "fixed_stride_fallback"


def preceding_image(
    frames: list[dict[str, Any]], anchor_index: int, field: str, stamp_field: str | None,
    timestamp_field: str,
) -> tuple[str | None, float | None]:
    anchor_time = float(frames[anchor_index][timestamp_field])
    for index in range(anchor_index, -1, -1):
        path = frames[index].get(field)
        if not path:
            continue
        stamp = safe_float(frames[index].get(stamp_field)) if stamp_field else None
        if stamp is None:
            stamp = safe_float(frames[index].get(timestamp_field))
        if stamp is not None and stamp <= anchor_time + 1e-6:
            return str(path), stamp
    return None, None


def window_slice(timestamps: np.ndarray, start: float, end: float) -> slice:
    left = int(np.searchsorted(timestamps, start, side="left"))
    right = int(np.searchsorted(timestamps, end, side="left"))
    return slice(left, right)


def median_filter(values: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or values.size < 2:
        return values.copy()
    if size % 2 == 0:
        raise ValueError("median_filter_size must be odd")
    radius = size // 2
    return np.asarray([
        np.median(values[max(0, i - radius):min(values.size, i + radius + 1)])
        for i in range(values.size)
    ], dtype=np.float32)


def baseline_for_stream(stream: TactileStream, episode_start: float, baseline_s: float) -> np.ndarray:
    selection = window_slice(stream.timestamps, episode_start, episode_start + baseline_s)
    values = stream.forces_xyz[selection]
    valid = stream.valid[selection]
    if values.size == 0 or not np.any(valid):
        return np.zeros(3, dtype=np.float32)
    return np.median(values[valid], axis=0).astype(np.float32)


def baseline_diagnostics(
    stream: TactileStream, baseline: np.ndarray, episode_start: float, baseline_s: float
) -> dict[str, Any]:
    selection = window_slice(stream.timestamps, episode_start, episode_start + baseline_s)
    values = stream.forces_xyz[selection]
    valid = stream.valid[selection]
    residual = values[valid] - baseline if values.size else np.empty((0, 3), dtype=np.float32)
    magnitude = np.linalg.norm(residual, axis=1) if residual.size else np.asarray([], dtype=np.float32)
    return {
        "sample_count": int(magnitude.size),
        "residual_median": float(np.median(magnitude)) if magnitude.size else None,
        "residual_p95": float(np.quantile(magnitude, 0.95)) if magnitude.size else None,
        "residual_max": float(np.max(magnitude)) if magnitude.size else None,
    }


def force_window(
    streams: list[TactileStream], baselines: dict[str, np.ndarray], start: float, end: float,
    median_size: int, contact_threshold: float | None = None,
) -> dict[str, Any]:
    all_times: list[float] = []
    sensor_counts: dict[str, int] = {}
    sensor_max_gaps_ms: dict[str, float | None] = {}
    sensor_timestamps: dict[str, list[float]] = {}
    per_sensor: dict[str, dict[str, Any]] = {}
    for stream in streams:
        selection = window_slice(stream.timestamps, start, end)
        timestamps = stream.timestamps[selection]
        xyz = stream.forces_xyz[selection] - baselines[stream.name]
        valid = stream.valid[selection]
        timestamps = timestamps[valid]
        xyz = xyz[valid]
        raw_magnitude = np.linalg.norm(xyz, axis=1) if xyz.size else np.asarray([], dtype=np.float32)
        filtered_magnitude = median_filter(raw_magnitude, median_size)
        all_times.extend(float(x) for x in timestamps)
        sensor_counts[stream.name] = int(raw_magnitude.size)
        sensor_timestamps[stream.name] = [float(x) for x in timestamps]
        if timestamps.size >= 2:
            sensor_max_gaps_ms[stream.name] = float(np.max(np.diff(timestamps)) * 1000.0)
        else:
            sensor_max_gaps_ms[stream.name] = None
        raw_ordered = sorted((float(x) for x in raw_magnitude), reverse=True)
        filtered_ordered = sorted((float(x) for x in filtered_magnitude), reverse=True)
        per_sensor[stream.name] = {
            "sample_count": int(raw_magnitude.size),
            "raw_max": raw_ordered[0] if raw_ordered else None,
            "filtered_max": filtered_ordered[0] if filtered_ordered else None,
            "filtered_top2_mean": (
                sum(filtered_ordered[:2]) / len(filtered_ordered[:2])
                if filtered_ordered else None
            ),
            "first_contact_stamp": (
                float(timestamps[np.flatnonzero(raw_magnitude >= contact_threshold)[0]])
                if contact_threshold is not None
                and np.any(raw_magnitude >= contact_threshold)
                else None
            ),
        }
    raw_values = [value["raw_max"] for value in per_sensor.values() if value["raw_max"] is not None]
    filtered_max_values = [
        value["filtered_max"] for value in per_sensor.values()
        if value["filtered_max"] is not None
    ]
    filtered_top2_values = [
        value["filtered_top2_mean"] for value in per_sensor.values()
        if value["filtered_top2_mean"] is not None
    ]
    y_raw_max = max(raw_values) if raw_values else None
    y_filtered_max = max(filtered_max_values) if filtered_max_values else None
    y_filtered_top2 = max(filtered_top2_values) if filtered_top2_values else None
    contact_stamps = [
        value["first_contact_stamp"] for value in per_sensor.values()
        if value["first_contact_stamp"] is not None
    ]
    sorted_times = sorted(all_times)
    available_gaps = [gap for gap in sensor_max_gaps_ms.values() if gap is not None]
    max_gap_ms = max(available_gaps) if available_gaps else None
    return {
        "sample_count": sum(sensor_counts.values()),
        "sensor_counts": sensor_counts,
        "per_sensor": per_sensor,
        "max_gap_ms": float(max_gap_ms) if max_gap_ms is not None else None,
        "sensor_max_gap_ms": sensor_max_gaps_ms,
        "sensor_timestamps": sensor_timestamps,
        "first_stamp": sorted_times[0] if sorted_times else None,
        "last_stamp": sorted_times[-1] if sorted_times else None,
        "y_raw_max": y_raw_max,
        "y_filtered_max": y_filtered_max,
        "y_filtered_top2_mean": y_filtered_top2,
        "first_contact_stamp": min(contact_stamps) if contact_stamps else None,
        # Compatibility aliases for older readers.
        "y_max": y_raw_max,
        "y_top2_mean": y_filtered_top2,
    }
