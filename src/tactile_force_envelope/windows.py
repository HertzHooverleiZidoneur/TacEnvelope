from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    distribution,
    episode_split,
    episode_split_four,
    read_jsonl,
    write_json,
    write_jsonl,
)


def _episode_splits(episode_ids: list[int], training: dict[str, Any]) -> dict[str, list[int]]:
    validation_ratio = float(training.get("validation_ratio", 0.0))
    if validation_ratio > 0.0:
        return episode_split_four(
            episode_ids,
            float(training["train_ratio"]),
            validation_ratio,
            float(training["calibration_ratio"]),
            int(training["seed"]),
        )
    return episode_split(
        episode_ids,
        float(training["train_ratio"]),
        float(training["calibration_ratio"]),
        int(training["seed"]),
    )
from .data import (
    anchor_indices,
    baseline_diagnostics,
    baseline_for_stream,
    force_window,
    load_frames,
    load_tactile_streams,
    preceding_image,
)
from .discovery import (
    discover_schema,
    discover_tactile_sidecars,
    episode_paths,
    episode_tasks,
)


def _action_steps(frame_paths: dict[int, Path], timestamp_field: str, window_s: float) -> int:
    intervals: list[float] = []
    for path in list(frame_paths.values())[:20]:
        frames = load_frames(path)
        timestamps = np.asarray([float(row[timestamp_field]) for row in frames])
        intervals.extend(float(x) for x in np.diff(timestamps) if x > 0)
    if not intervals:
        raise ValueError("Cannot infer action_steps without positive frame intervals")
    return max(1, int(round(window_s / float(np.median(intervals)))))


def _build_single_windows(config: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    tactile_root = Path(config["data"]["tactile_root"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    schema = discover_schema(dataset, config)
    frame_paths = episode_paths(dataset)
    sidecars = discover_tactile_sidecars(tactile_root)
    matched_episode_ids = sorted(set(frame_paths) & set(sidecars))
    excluded_episode_ids = {
        int(value) for value in config["data"].get("exclude_episode_ids", [])
    }
    episode_ids = [
        episode_id for episode_id in matched_episode_ids
        if episode_id not in excluded_episode_ids
    ]
    tasks = episode_tasks(dataset, str(config["data"].get("task_field") or "task"))
    window_s = float(config["alignment"]["window_ms"]) / 1000.0
    horizon_ms = sorted(set(int(value) for value in config["alignment"].get(
        "horizons_ms", [int(config["alignment"]["window_ms"])]
    )))
    if not horizon_ms or horizon_ms[-1] != int(config["alignment"]["window_ms"]):
        raise ValueError("horizons_ms must include window_ms as its largest horizon")
    stride_s = float(config["alignment"]["anchor_stride_ms"]) / 1000.0
    action_steps = config["alignment"].get("action_steps")
    action_steps = int(action_steps) if action_steps else _action_steps(
        frame_paths, schema.timestamp_field, window_s
    )
    min_samples = int(config["alignment"]["min_tactile_samples"])
    min_per_sensor = int(config["alignment"].get("min_samples_per_sensor", min_samples))
    expected_sensors = set(config["alignment"].get("expected_sensors", []))
    max_image_age_ms = float(config["alignment"]["max_image_age_ms"])
    median_size = int(config["label"]["median_filter_size"])
    baseline_s = float(config["label"]["baseline_ms"]) / 1000.0
    force_axes = list(config["label"]["force_axes"])
    drop_low_quality = bool(config["alignment"].get("drop_low_quality", True))
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    anchor_sources: Counter[str] = Counter()
    for episode_id in episode_ids:
        frames = load_frames(frame_paths[episode_id])
        if not frames:
            rejected["empty_episode"] += 1
            continue
        timestamps = np.asarray(
            [float(row[schema.timestamp_field]) for row in frames], dtype=np.float64
        )
        streams, manifest = load_tactile_streams(sidecars[episode_id], force_axes)
        if not streams:
            rejected["no_tactile_stream"] += 1
            continue
        baselines = {
            stream.name: baseline_for_stream(stream, float(timestamps[0]), baseline_s)
            for stream in streams
        }
        baseline_quality = {
            stream.name: baseline_diagnostics(
                stream, baselines[stream.name], float(timestamps[0]), baseline_s
            )
            for stream in streams
        }
        baseline_frame_indices = np.flatnonzero(
            (timestamps >= float(timestamps[0]))
            & (timestamps < float(timestamps[0]) + baseline_s)
        ).tolist()
        gripper_index = int(config["label"].get("gripper_index", -1))
        baseline_min_open = float(config["label"].get("baseline_min_open", 0.95))
        baseline_max_residual_p95 = float(
            config["label"].get("baseline_max_residual_p95", 1.0)
        )
        baseline_gripper_values = [
            float(frames[index][schema.action_field][gripper_index])
            for index in baseline_frame_indices
        ]
        baseline_open = bool(baseline_gripper_values) and min(baseline_gripper_values) >= baseline_min_open
        baseline_stable = all(
            quality["sample_count"] >= min_per_sensor
            and quality["residual_p95"] is not None
            and quality["residual_p95"] <= baseline_max_residual_p95
            for quality in baseline_quality.values()
        )
        baseline_contact_free = baseline_open and baseline_stable
        if bool(config["label"].get("require_contact_free_baseline", True)) and not baseline_contact_free:
            rejected["baseline_contact_free_not_verified"] += 1
            continue
        anchors, anchor_source = anchor_indices(
            frames, schema, str(config["alignment"]["anchor_mode"]), stride_s
        )
        anchor_sources[anchor_source] += len(anchors)
        for anchor_index in anchors:
            start = float(timestamps[anchor_index])
            end = start + window_s
            if end > float(timestamps[-1]):
                rejected["incomplete_future_window"] += 1
                continue
            action_indices = np.flatnonzero((timestamps >= start) & (timestamps < end)).tolist()
            if not action_indices:
                rejected["no_executed_action"] += 1
                continue
            if len(action_indices) > action_steps:
                selection_positions = np.linspace(
                    0, len(action_indices) - 1, action_steps, dtype=np.int64
                )
                selected = [action_indices[int(position)] for position in selection_positions]
            else:
                selected = action_indices
            actions = [[float(x) for x in frames[index][schema.action_field]] for index in selected]
            action_times = [float(timestamps[index]) for index in selected]
            action_mask = [1.0] * len(actions)
            while len(actions) < action_steps:
                actions.append(list(actions[-1]))
                action_times.append(action_times[-1])
                action_mask.append(0.0)
            labels_by_horizon = {
                str(horizon): force_window(
                    streams, baselines, start, start + horizon / 1000.0, median_size,
                    contact_threshold=float(config["label"].get("contact_threshold", 5.0)),
                )
                for horizon in horizon_ms
            }
            labels = labels_by_horizon[str(horizon_ms[-1])]
            missing_sensors = expected_sensors - set(labels["sensor_counts"])
            minimum_by_horizon = {
                int(key): int(value) for key, value in config["alignment"].get(
                    "min_samples_per_sensor_by_horizon", {}
                ).items()
            }
            low_quality = (
                labels["sample_count"] < min_samples
                or bool(missing_sensors)
                or any(
                    labels["sensor_counts"].get(sensor, 0) < min_per_sensor
                    for sensor in (expected_sensors or set(labels["sensor_counts"]))
                )
                or any(
                    labels_by_horizon[str(horizon)]["sensor_counts"].get(sensor, 0)
                    < minimum_by_horizon.get(horizon, min_per_sensor)
                    for horizon in horizon_ms
                    for sensor in (expected_sensors or set(labels["sensor_counts"]))
                )
            )
            if low_quality and drop_low_quality:
                rejected["insufficient_tactile_samples"] += 1
                continue
            images: dict[str, dict[str, Any]] = {}
            image_missing = False
            stale_fields: list[str] = []
            for field in schema.image_fields:
                image_path, image_stamp = preceding_image(
                    frames, anchor_index, field, schema.image_stamp_fields[field],
                    schema.timestamp_field,
                )
                if image_path is None or image_stamp is None:
                    image_missing = True
                images[field] = {
                    "path": image_path,
                    "stamp": image_stamp,
                    "age_ms": (start - image_stamp) * 1000.0 if image_stamp is not None else None,
                }
                if image_stamp is not None and (start - image_stamp) * 1000.0 > max_image_age_ms:
                    stale_fields.append(field)
            if image_missing:
                rejected["missing_preceding_image"] += 1
                continue
            if stale_fields:
                for field in stale_fields:
                    rejected[f"stale_image:{field}"] += 1
                continue
            task_override = config["data"].get("task_override")
            task = str(
                task_override
                if task_override is not None
                else frames[anchor_index].get(config["data"].get("task_field"))
                or tasks.get(episode_id, "")
            )
            previous_index = max(0, anchor_index - 1)
            previous_state = np.asarray(
                frames[previous_index][schema.state_field], dtype=np.float64
            )
            current_state = np.asarray(
                frames[anchor_index][schema.state_field], dtype=np.float64
            )
            state_dt = float(timestamps[anchor_index] - timestamps[previous_index])
            qdot = (
                ((current_state - previous_state) / state_dt).tolist()
                if anchor_index > 0 and state_dt > 0.0
                else [0.0] * len(current_state)
            )
            previous_action = [
                float(x) for x in frames[previous_index][schema.action_field]
            ]
            relative_actions = [
                (np.asarray(action, dtype=np.float64) - current_state).tolist()
                for action in actions
            ]
            rows.append({
                "window_index": len(rows),
                "episode_index": episode_id,
                "anchor_frame_index": anchor_index,
                "anchor_source": anchor_source,
                "start_timestamp": start,
                "end_timestamp": end,
                "state": [float(x) for x in frames[anchor_index][schema.state_field]],
                "qdot": [float(x) for x in qdot],
                "qdot_dt": state_dt if anchor_index > 0 else None,
                "previous_action": previous_action,
                "executed_actions": actions,
                "relative_actions": relative_actions,
                "executed_action_timestamps": action_times,
                "action_window_source_count": len(action_indices),
                "action_sequence_available_at_anchor": bool(
                    config["data"].get("action_sequence_available_at_anchor", False)
                ),
                "control_window_ms": int(config["alignment"]["window_ms"]),
                "action_mask": action_mask,
                "task": task,
                "images": images,
                "tactile": labels,
                "tactile_horizons": labels_by_horizon,
                "quality": {
                    "low_quality": low_quality,
                    "missing_sensors": sorted(missing_sensors),
                    "clock_rtt_ms": int(manifest.get("clock_rtt_ns", 0)) / 1e6,
                    "force_baseline_xyz": {
                        name: [float(x) for x in value]
                        for name, value in baselines.items()
                    },
                    "baseline_open_gripper_verified": baseline_open,
                    "baseline_stable_verified": baseline_stable,
                    "baseline_contact_free_proxy_verified": baseline_contact_free,
                    "baseline_gripper_min": min(baseline_gripper_values),
                    "baseline_tactile": baseline_quality,
                },
            })
    valid_episode_ids = sorted({int(row["episode_index"]) for row in rows})
    train_cfg = config["training"]
    splits = _episode_splits(valid_episode_ids, train_cfg)
    split_lookup = {episode: name for name, ids in splits.items() for episode in ids}
    for row in rows:
        row["split"] = split_lookup[int(row["episode_index"])]
    task_names = sorted({str(row["task"]) for row in rows})
    summary = {
        "schema": schema.to_dict(),
        "window_ms": config["alignment"]["window_ms"],
        "action_steps": action_steps,
        "action_dim": schema.action_dim,
        "state_dim": schema.state_dim,
        "image_fields": schema.image_fields,
        "force_axes": force_axes,
        "target_types": [
            "left_raw_max", "right_raw_max",
            "left_filtered_top2_mean", "right_filtered_top2_mean",
            "raw_max", "filtered_top2_mean",
        ],
        "windows": len(rows),
        "episodes": len(valid_episode_ids),
        "matched_episode_ids": matched_episode_ids,
        "excluded_episode_ids": sorted(excluded_episode_ids & set(matched_episode_ids)),
        "splits": splits,
        "split_window_counts": dict(Counter(row["split"] for row in rows)),
        "anchor_sources": dict(anchor_sources),
        "rejected": dict(rejected),
        "sample_count": distribution([float(row["tactile"]["sample_count"]) for row in rows]),
        "max_gap_ms": distribution([
            float(row["tactile"]["max_gap_ms"]) for row in rows
            if row["tactile"]["max_gap_ms"] is not None
        ]),
        "tasks": task_names,
        "target_distribution": {
            split: {
                "raw_max": distribution([
                    float(row["tactile"]["y_raw_max"])
                    for row in rows if row["split"] == split
                ]),
                "filtered_top2_mean": distribution([
                    float(row["tactile"]["y_filtered_top2_mean"])
                    for row in rows if row["split"] == split
                ]),
            }
            for split in splits
        },
    }
    write_jsonl(output / "windows.jsonl", rows)
    write_json(output / "windows_summary.json", summary)
    write_json(output / "episode_splits.json", splits)
    return summary


def _collection_summary(
    rows: list[dict[str, Any]], source_summaries: list[dict[str, Any]],
    config: dict[str, Any], rejected: Counter[str], anchor_sources: Counter[str],
) -> dict[str, Any]:
    first = source_summaries[0]
    split_names = ("train", "validation", "calibration", "test") if any(
        row["split"] == "validation" for row in rows
    ) else ("train", "calibration", "test")
    splits = {
        name: sorted({int(row["episode_index"]) for row in rows if row["split"] == name})
        for name in split_names
    }
    return {
        "schema": first["schema"],
        "window_semantics": "fixed-horizon control window [t_k, t_k + T)",
        "window_ms": int(config["alignment"]["window_ms"]),
        "action_steps": int(first["action_steps"]),
        "action_dim": int(first["action_dim"]),
        "state_dim": int(first["state_dim"]),
        "image_fields": first["image_fields"],
        "force_axes": first["force_axes"],
        "target_types": [
            "left_raw_max", "right_raw_max",
            "left_filtered_top2_mean", "right_filtered_top2_mean",
            "raw_max", "filtered_top2_mean",
        ],
        "windows": len(rows),
        "episodes": len({int(row["episode_index"]) for row in rows}),
        "sources": len(source_summaries),
        "splits": splits,
        "split_window_counts": dict(Counter(row["split"] for row in rows)),
        "anchor_sources": dict(anchor_sources),
        "rejected": dict(rejected),
        "sample_count": distribution([float(row["tactile"]["sample_count"]) for row in rows]),
        "per_sensor_sample_count": {
            sensor: distribution([
                float(row["tactile"]["sensor_counts"].get(sensor, 0)) for row in rows
            ])
            for sensor in sorted({
                sensor for row in rows for sensor in row["tactile"]["sensor_counts"]
            })
        },
        "max_gap_ms": distribution([
            float(row["tactile"]["max_gap_ms"]) for row in rows
            if row["tactile"]["max_gap_ms"] is not None
        ]),
        "tasks": sorted({str(row["task"]) for row in rows}),
        "target_distribution": {
            split: {
                target: distribution([
                    float(row["tactile"][field]) for row in rows if row["split"] == split
                ])
                for target, field in (
                    ("raw_max", "y_raw_max"),
                    ("filtered_top2_mean", "y_filtered_top2_mean"),
                )
            }
            for split in split_names
        },
    }


def _build_collection_windows(config: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    collection_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    anchor_sources: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="force-window-build-", dir=output) as temporary:
        temporary_root = Path(temporary)
        for source_index, batch in enumerate(manifest.get("batches", [])):
            source_config = copy.deepcopy(config)
            source_config["data"].pop("collection_manifest", None)
            source_config["data"]["robot_dataset"] = str(collection_root / batch["source_frames"])
            source_config["data"]["tactile_root"] = str(collection_root / batch["tactile_raw"])
            source_config["data"]["output_dir"] = str(temporary_root / f"source_{source_index:03d}")
            summary = _build_single_windows(source_config)
            source_summaries.append(summary)
            rejected.update(summary.get("rejected", {}))
            anchor_sources.update(summary.get("anchor_sources", {}))
            source_rows = read_jsonl(Path(source_config["data"]["output_dir"]) / "windows.jsonl")
            global_start = int(batch["global_episode_start"])
            for row in source_rows:
                local_episode = int(row["episode_index"])
                row["source_episode_index"] = local_episode
                row["episode_index"] = global_start + local_episode
                row["batch_id"] = str(batch["batch_id"])
                row["robot_dataset"] = source_config["data"]["robot_dataset"]
                row["tactile_root"] = source_config["data"]["tactile_root"]
                all_rows.append(row)
    if not all_rows:
        raise ValueError(f"No valid force windows were built from {manifest_path}")
    valid_episode_ids = sorted({int(row["episode_index"]) for row in all_rows})
    train_cfg = config["training"]
    splits = _episode_splits(valid_episode_ids, train_cfg)
    split_lookup = {episode: name for name, ids in splits.items() for episode in ids}
    for index, row in enumerate(all_rows):
        row["window_index"] = index
        row["split"] = split_lookup[int(row["episode_index"])]
    summary = _collection_summary(all_rows, source_summaries, config, rejected, anchor_sources)
    write_jsonl(output / "windows.jsonl", all_rows)
    write_json(output / "windows_summary.json", summary)
    write_json(output / "episode_splits.json", splits)
    return summary


def build_windows(config: dict[str, Any]) -> dict[str, Any]:
    collection_manifest = config["data"].get("collection_manifest")
    if collection_manifest:
        return _build_collection_windows(config, Path(collection_manifest))
    return _build_single_windows(config)
