from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import distribution, read_jsonl, write_json
from .data import load_tactile_streams, window_slice
from .discovery import discover_tactile_sidecars


HISTORY_FEATURES = ("force_fx", "force_fy", "force_fz", "force_magnitude", "relative_time")


def cache_tactile_history(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    rows_path = output / "windows.jsonl"
    rows = read_jsonl(rows_path)
    history_cfg = config.get("causal_tactile_history", {})
    history_ms = int(history_cfg.get("history_ms", 200))
    history_steps = int(history_cfg.get("steps", 6))
    sensor_names = config["label"].get(
        "sensor_names", {"left": "left_tip", "right": "right_tip"}
    )
    sensor_order = [str(sensor_names["left"]), str(sensor_names["right"])]
    force_axes = list(config["label"]["force_axes"])
    if history_ms <= 0 or history_steps <= 0:
        raise ValueError("Causal tactile history_ms and steps must be positive")

    history = torch.zeros((len(rows), 2, history_steps, len(HISTORY_FEATURES)), dtype=torch.float32)
    mask = torch.zeros((len(rows), 2, history_steps), dtype=torch.bool)
    timestamps = torch.zeros((len(rows), 2, history_steps), dtype=torch.float64)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    default_tactile_root = str(
        Path(config["data"]["tactile_root"]).expanduser().resolve()
    )
    for row in rows:
        grouped[(
            str(row.get("tactile_root") or default_tactile_root),
            int(row.get("source_episode_index", row["episode_index"])),
        )].append(row)

    sidecar_maps: dict[str, dict[int, Path]] = {}
    counts: list[int] = []
    causal_violations = 0
    missing_sensors: dict[str, int] = defaultdict(int)
    history_s = history_ms / 1000.0
    for (root_value, episode_index), episode_rows in grouped.items():
        if root_value not in sidecar_maps:
            sidecar_maps[root_value] = discover_tactile_sidecars(Path(root_value))
        sidecar = sidecar_maps[root_value].get(episode_index)
        if sidecar is None:
            raise FileNotFoundError(
                f"No tactile sidecar for episode {episode_index} under {root_value}"
            )
        streams, _ = load_tactile_streams(sidecar, force_axes)
        streams_by_name = {stream.name: stream for stream in streams}
        baseline_map = episode_rows[0]["quality"]["force_baseline_xyz"]
        for row in episode_rows:
            anchor = float(row["start_timestamp"])
            window_index = int(row["window_index"])
            for side_index, sensor in enumerate(sensor_order):
                stream = streams_by_name.get(sensor)
                if stream is None:
                    missing_sensors[sensor] += 1
                    continue
                selection = window_slice(stream.timestamps, anchor - history_s, anchor)
                sample_times = stream.timestamps[selection]
                sample_values = stream.forces_xyz[selection]
                sample_valid = stream.valid[selection]
                sample_times = sample_times[sample_valid]
                sample_values = sample_values[sample_valid]
                if sample_times.size:
                    keep = min(history_steps, int(sample_times.size))
                    sample_times = sample_times[-keep:]
                    sample_values = sample_values[-keep:]
                    baseline = np.asarray(baseline_map[sensor], dtype=np.float32)
                    force = sample_values - baseline
                    magnitude = np.linalg.norm(force, axis=1, keepdims=True)
                    relative_time = ((sample_times - anchor) / history_s)[:, None]
                    features = np.concatenate((force, magnitude, relative_time), axis=1)
                    start = history_steps - keep
                    history[window_index, side_index, start:] = torch.from_numpy(
                        features.astype(np.float32)
                    )
                    mask[window_index, side_index, start:] = True
                    timestamps[window_index, side_index, start:] = torch.from_numpy(sample_times)
                    causal_violations += int(np.sum(sample_times >= anchor))
                    counts.append(keep)
                else:
                    counts.append(0)

    windows_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    report = {
        "encoder": "causal_tactile_history_tokens",
        "history_ms": history_ms,
        "history_steps": history_steps,
        "sensor_order": sensor_order,
        "features": list(HISTORY_FEATURES),
        "sample_selection": "last real valid samples in [t_k-history_ms,t_k), left padded",
        "interpolation": False,
        "baseline": "episode contact-free force_baseline_xyz from windows.jsonl",
        "causal_violations": causal_violations,
        "missing_sensor_windows": dict(missing_sensors),
        "sample_count_per_side_window": distribution([float(value) for value in counts]),
        "windows": len(rows),
        "windows_sha256": windows_hash,
    }
    cache_path = output / "causal_tactile_history.pt"
    torch.save({
        "history": history,
        "mask": mask,
        "timestamps": timestamps,
        "windows_sha256": windows_hash,
        "report": report,
    }, cache_path)
    write_json(output / "causal_tactile_history_report.json", report)
    return {"cache": str(cache_path), **report}
