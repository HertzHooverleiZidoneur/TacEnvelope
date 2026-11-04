from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .common import distribution, rate_from_timestamps, safe_float, write_json
from .data import (
    anchor_indices,
    baseline_diagnostics,
    baseline_for_stream,
    force_window,
    load_frames,
    load_tactile_streams,
)
from .discovery import discover_schema, discover_tactile_sidecars, episode_paths


def _duplicate_ratio(values: list[Any]) -> float:
    if len(values) < 2:
        return 0.0
    duplicates = sum(a == b for a, b in zip(values, values[1:]))
    return duplicates / (len(values) - 1)


def _finalize_report_warnings(
    report: dict[str, Any], has_action_chunk: bool, robot_timestamp_rollbacks: int
) -> None:
    if not has_action_chunk:
        report["warnings"].append(
            "No action chunk boundary field exists; fixed-stride anchors are a documented fallback."
        )
    if robot_timestamp_rollbacks:
        report["warnings"].append(
            f"Robot timestamps rolled back {robot_timestamp_rollbacks} times."
        )
    for field, camera in report["cameras"].items():
        if camera["obvious_delay_count"]:
            report["warnings"].append(
                f"{field}: {camera['obvious_delay_count']} image samples exceeded the age threshold."
            )
        if camera["timestamp_rollbacks"]:
            report["warnings"].append(
                f"{field}: timestamps rolled back {camera['timestamp_rollbacks']} times."
            )
    for key in ("start", "end"):
        value = report["timestamp_range"][key]
        report["timestamp_range"][f"{key}_iso_utc"] = (
            datetime.fromtimestamp(value, tz=timezone.utc).isoformat() if value is not None else None
        )
    for name, tactile in report["tactile"].items():
        if tactile["timestamp_rollbacks"]:
            report["warnings"].append(
                f"Tactile {name}: source timestamps rolled back {tactile['timestamp_rollbacks']} times."
            )
        if tactile["gap_over_150ms_count"]:
            report["warnings"].append(
                f"Tactile {name}: {tactile['gap_over_150ms_count']} gaps exceeded 150 ms."
            )


def _run_single_audit(config: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    tactile_root = Path(config["data"]["tactile_root"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    schema = discover_schema(dataset, config)
    frame_paths = episode_paths(dataset)
    sidecars = discover_tactile_sidecars(tactile_root)
    matched_ids = sorted(set(frame_paths) & set(sidecars))
    excluded_ids = {
        int(value) for value in config["data"].get("exclude_episode_ids", [])
    }
    common_ids = [episode_id for episode_id in matched_ids if episode_id not in excluded_ids]
    force_axes = list(config["label"]["force_axes"])
    window_s = float(config["alignment"]["window_ms"]) / 1000.0
    stride_s = float(config["alignment"]["anchor_stride_ms"]) / 1000.0
    image_intervals: dict[str, list[float]] = {key: [] for key in schema.image_fields}
    image_ages: dict[str, list[float]] = {key: [] for key in schema.image_fields}
    image_timestamp_duplicates: Counter[str] = Counter()
    image_timestamp_rollbacks: Counter[str] = Counter()
    action_intervals: list[float] = []
    tactile_intervals: dict[str, list[float]] = {}
    tactile_value_duplicates: dict[str, list[float]] = {}
    tactile_ranges: dict[str, list[float]] = {}
    tactile_timestamp_duplicates: Counter[str] = Counter()
    tactile_timestamp_rollbacks: Counter[str] = Counter()
    window_counts: list[float] = []
    window_sensor_counts: dict[str, list[float]] = {}
    window_gaps: list[float] = []
    action_duplicate_ratios: list[float] = []
    state_duplicate_ratios: list[float] = []
    timestamp_rollbacks = 0
    timestamp_duplicates = 0
    anchor_sources: Counter[str] = Counter()
    episode_reports: list[dict[str, Any]] = []
    action_semantics = Counter()
    future_action_windows = 0
    future_action_changed = 0
    contact_windows = 0
    post_contact_action_changed = 0
    baseline_verified_episodes = 0
    baseline_unverified_episodes: list[int] = []
    for episode_id in common_ids:
        frames = load_frames(frame_paths[episode_id])
        if not frames:
            continue
        frame_times = [float(row[schema.timestamp_field]) for row in frames]
        diffs = np.diff(frame_times)
        action_intervals.extend(float(x) for x in diffs if x > 0)
        timestamp_rollbacks += int(np.sum(diffs < 0))
        timestamp_duplicates += int(np.sum(diffs == 0))
        action_duplicate_ratios.append(_duplicate_ratio([row[schema.action_field] for row in frames]))
        state_duplicate_ratios.append(_duplicate_ratio([row[schema.state_field] for row in frames]))
        for row in frames:
            action = np.asarray(row[schema.action_field], dtype=np.float32)
            state = np.asarray(row[schema.state_field], dtype=np.float32)
            gello = row.get("gello.q")
            robot = row.get("robot.q")
            gripper = row.get("gripper.target")
            if isinstance(gello, list) and action.size >= len(gello):
                action_semantics["gello_arm_checked"] += 1
                action_semantics["gello_arm_matched"] += int(
                    np.allclose(action[:len(gello)], np.asarray(gello), atol=1e-6)
                )
            if isinstance(robot, list) and state.size >= len(robot):
                action_semantics["robot_state_checked"] += 1
                action_semantics["robot_state_matched"] += int(
                    np.allclose(state[:len(robot)], np.asarray(robot), atol=1e-6)
                )
            if gripper is not None and action.size:
                action_semantics["gripper_checked"] += 1
                action_semantics["gripper_matched"] += int(
                    np.isclose(action[-1], float(gripper), atol=1e-6)
                )
        for field in schema.image_fields:
            stamp_field = schema.image_stamp_fields[field]
            if not stamp_field:
                continue
            stamps = [safe_float(row.get(stamp_field)) for row in frames]
            valid_stamps = [stamp for stamp in stamps if stamp is not None]
            stamp_diffs = np.diff(valid_stamps)
            image_timestamp_duplicates[field] += int(np.sum(stamp_diffs == 0))
            image_timestamp_rollbacks[field] += int(np.sum(stamp_diffs < 0))
            unique_stamps = [stamp for i, stamp in enumerate(valid_stamps)
                             if i == 0 or stamp != valid_stamps[i - 1]]
            image_intervals[field].extend(float(x) for x in np.diff(unique_stamps) if x > 0)
            image_ages[field].extend(
                (frame_time - stamp) * 1000.0
                for frame_time, stamp in zip(frame_times, stamps) if stamp is not None
            )
        streams, manifest = load_tactile_streams(sidecars[episode_id], force_axes)
        baseline_s = float(config["label"]["baseline_ms"]) / 1000.0
        baselines = {
            stream.name: baseline_for_stream(stream, frame_times[0], baseline_s)
            for stream in streams
        }
        baseline_quality = {
            stream.name: baseline_diagnostics(stream, baselines[stream.name], frame_times[0], baseline_s)
            for stream in streams
        }
        baseline_indices = np.flatnonzero(
            (np.asarray(frame_times) >= frame_times[0])
            & (np.asarray(frame_times) < frame_times[0] + baseline_s)
        )
        gripper_index = int(config["label"].get("gripper_index", -1))
        baseline_min_open = float(config["label"].get("baseline_min_open", 0.95))
        baseline_max_residual_p95 = float(
            config["label"].get("baseline_max_residual_p95", 1.0)
        )
        baseline_gripper = [
            float(frames[int(index)][schema.action_field][gripper_index])
            for index in baseline_indices
        ]
        baseline_open = bool(baseline_gripper) and min(baseline_gripper) >= baseline_min_open
        baseline_stable = all(
            quality["sample_count"] >= int(config["alignment"].get("min_samples_per_sensor", 4))
            and quality["residual_p95"] is not None
            and quality["residual_p95"] <= baseline_max_residual_p95
            for quality in baseline_quality.values()
        )
        baseline_contact_free = baseline_open and baseline_stable
        if baseline_contact_free:
            baseline_verified_episodes += 1
        else:
            baseline_unverified_episodes.append(episode_id)
        for stream in streams:
            key = stream.name
            tactile_intervals.setdefault(key, []).extend(
                float(x) for x in np.diff(stream.timestamps) if x > 0
            )
            if stream.timestamps.size:
                tactile_ranges.setdefault(key, []).extend(
                    [float(stream.timestamps[0]), float(stream.timestamps[-1])]
                )
            vectors = [tuple(float(v) for v in row) for row in stream.forces_xyz]
            tactile_value_duplicates.setdefault(key, []).append(_duplicate_ratio(vectors))
            tactile_timestamp_duplicates[key] += stream.timestamp_duplicates
            tactile_timestamp_rollbacks[key] += stream.timestamp_rollbacks
        anchors, source = anchor_indices(
            frames, schema, str(config["alignment"]["anchor_mode"]), stride_s
        )
        anchor_sources[source] += len(anchors)
        episode_window_counts = []
        for index in anchors:
            start = frame_times[index]
            if start + window_s > frame_times[-1]:
                continue
            action_indices = np.flatnonzero(
                (np.asarray(frame_times) >= start) & (np.asarray(frame_times) < start + window_s)
            )
            if action_indices.size:
                action_values = np.asarray([
                    frames[int(row_index)][schema.action_field] for row_index in action_indices
                ], dtype=np.float32)
                future_action_windows += 1
                future_action_changed += int(
                    np.max(np.abs(action_values - action_values[0])) > 1e-4
                )
            label = force_window(
                streams, baselines, start, start + window_s, 1,
                contact_threshold=float(config["label"].get("contact_threshold", 5.0)),
            )
            contact_stamp = label["first_contact_stamp"]
            if contact_stamp is not None and action_indices.size:
                contact_windows += 1
                after_contact = action_indices[np.asarray(frame_times)[action_indices] >= contact_stamp]
                if after_contact.size:
                    post_values = np.asarray([
                        frames[int(row_index)][schema.action_field] for row_index in after_contact
                    ], dtype=np.float32)
                    post_contact_action_changed += int(
                        np.max(np.abs(post_values - action_values[0])) > 1e-4
                    )
            window_counts.append(float(label["sample_count"]))
            for name, count in label["sensor_counts"].items():
                window_sensor_counts.setdefault(name, []).append(float(count))
            episode_window_counts.append(float(label["sample_count"]))
            if label["max_gap_ms"] is not None:
                window_gaps.append(float(label["max_gap_ms"]))
        episode_reports.append({
            "episode_index": episode_id,
            "frames": len(frames),
            "start": frame_times[0],
            "end": frame_times[-1],
            "duration_s": frame_times[-1] - frame_times[0],
            "anchors": len(anchors),
            "anchor_source": source,
            "tactile_window_samples": distribution(episode_window_counts),
            "clock_rtt_ms": int(manifest.get("clock_rtt_ns", 0)) / 1e6,
            "baseline": {
                "duration_ms": int(round(baseline_s * 1000.0)),
                "open_gripper_verified": baseline_open,
                "stable_tactile_verified": baseline_stable,
                "contact_free_proxy_verified": baseline_contact_free,
                "gripper_min": min(baseline_gripper) if baseline_gripper else None,
                "per_sensor": baseline_quality,
            },
        })
    action_rate = {
        "hz_median": float(1.0 / np.median(action_intervals)) if action_intervals else None,
        "hz_mean": float(1.0 / np.mean(action_intervals)) if action_intervals else None,
        "interval_ms": distribution([x * 1000.0 for x in action_intervals]),
    }
    report = {
        "dataset": str(dataset),
        "tactile_root": str(tactile_root),
        "schema": schema.to_dict(),
        "episodes": {
            "robot": len(frame_paths), "tactile": len(sidecars), "matched": len(matched_ids),
            "selected": len(common_ids),
            "excluded": sorted(excluded_ids & set(matched_ids)),
            "robot_only": sorted(set(frame_paths) - set(sidecars)),
            "tactile_only": sorted(set(sidecars) - set(frame_paths)),
        },
        "timestamp_range": {
            "start": min((row["start"] for row in episode_reports), default=None),
            "end": max((row["end"] for row in episode_reports), default=None),
        },
        "action": {
            **action_rate,
            "consecutive_duplicate_ratio": distribution(action_duplicate_ratios),
            "timestamp_duplicates": timestamp_duplicates,
            "timestamp_rollbacks": timestamp_rollbacks,
            "chunk_start_field": schema.action_chunk_start_field,
            "anchor_sources": dict(anchor_sources),
            "semantics": {
                "field": schema.action_field,
                "gello_arm_match": {
                    "checked": action_semantics["gello_arm_checked"],
                    "matched": action_semantics["gello_arm_matched"],
                },
                "gripper_target_match": {
                    "checked": action_semantics["gripper_checked"],
                    "matched": action_semantics["gripper_matched"],
                },
                "interpretation": (
                    "actual teleoperation command target sent to the controller"
                    if action_semantics["gello_arm_checked"]
                    and action_semantics["gello_arm_checked"] == action_semantics["gello_arm_matched"]
                    else "not conclusively identified"
                ),
            },
            "future_window_feedback_corrections": {
                "windows_checked": future_action_windows,
                "windows_with_command_change": future_action_changed,
                "ratio": future_action_changed / future_action_windows if future_action_windows else None,
                "causality_warning": (
                    "Continuous teleoperation commands inside the future window can include human "
                    "feedback after contact. They are valid retrospective executed actions but are not "
                    "available at t_k to an Online RL policy unless a planned chunk is recorded separately."
                ),
                "contact_windows": contact_windows,
                "contact_windows_with_post_contact_command_change": post_contact_action_changed,
                "post_contact_change_ratio": (
                    post_contact_action_changed / contact_windows if contact_windows else None
                ),
            },
        },
        "state": {
            "consecutive_duplicate_ratio": distribution(state_duplicate_ratios),
            "robot_q_match": {
                "checked": action_semantics["robot_state_checked"],
                "matched": action_semantics["robot_state_matched"],
            },
        },
        "cameras": {
            field: {
                "hz_median": float(1.0 / np.median(intervals)) if intervals else None,
                "interval_ms": distribution([x * 1000.0 for x in intervals]),
                "age_ms": distribution(image_ages[field]),
                "obvious_delay_count": sum(
                    age > float(config["alignment"]["max_image_age_ms"])
                    for age in image_ages[field]
                ),
                "timestamp_duplicates": image_timestamp_duplicates[field],
                "timestamp_rollbacks": image_timestamp_rollbacks[field],
            }
            for field, intervals in image_intervals.items()
        },
        "tactile": {
            name: {
                "hz_median": float(1.0 / np.median(intervals)) if intervals else None,
                "interval_ms": distribution([x * 1000.0 for x in intervals]),
                "consecutive_duplicate_ratio": distribution(tactile_value_duplicates[name]),
                "timestamp_duplicates": tactile_timestamp_duplicates[name],
                "timestamp_rollbacks": tactile_timestamp_rollbacks[name],
                "gap_over_150ms_count": sum(interval > 0.150 for interval in intervals),
                "timestamp_range": {
                    "start": min(tactile_ranges.get(name, []), default=None),
                    "end": max(tactile_ranges.get(name, []), default=None),
                },
            }
            for name, intervals in tactile_intervals.items()
        },
        "windows": {
            "semantics": "fixed-horizon control window [t_k, t_k + T)",
            "window_ms": config["alignment"]["window_ms"],
            "sample_count": distribution(window_counts),
            "per_sensor_sample_count": {
                name: distribution(values) for name, values in window_sensor_counts.items()
            },
            "max_sample_gap_ms": distribution(window_gaps),
            "below_minimum": sum(
                value < int(config["alignment"]["min_tactile_samples"])
                for value in window_counts
            ),
        },
        "baseline_validation": {
            "method": (
                "first-second robust median; requires open gripper and per-sensor baseline residual "
                f"p95 <= {float(config['label'].get('baseline_max_residual_p95', 1.0))}"
            ),
            "limitation": "Open gripper does not prove absence of external contact; inspect high residual episodes.",
            "episodes_verified_open": baseline_verified_episodes,
            "episodes_unverified": baseline_unverified_episodes,
        },
        "episode_details": episode_reports,
        "warnings": [],
    }
    _finalize_report_warnings(
        report,
        has_action_chunk=schema.action_chunk_start_field is not None,
        robot_timestamp_rollbacks=timestamp_rollbacks,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", report)
    return report


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    collection_manifest = config["data"].get("collection_manifest")
    if not collection_manifest:
        return _run_single_audit(config)
    manifest_path = Path(collection_manifest).expanduser().resolve()
    collection_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    source_reports: list[dict[str, Any]] = []
    for index, batch in enumerate(manifest.get("batches", [])):
        source_config = copy.deepcopy(config)
        source_config["data"].pop("collection_manifest", None)
        source_config["data"]["robot_dataset"] = str(collection_root / batch["source_frames"])
        source_config["data"]["tactile_root"] = str(collection_root / batch["tactile_raw"])
        source_config["data"]["output_dir"] = str(
            output / "audits" / f"{index:03d}_{batch['batch_id']}"
        )
        report = _run_single_audit(source_config)
        source_reports.append({"batch": batch, "report": report})
    result = {
        "collection_manifest": str(manifest_path),
        "fixed_horizon_control_window_ms": int(config["alignment"]["window_ms"]),
        "source_count": len(source_reports),
        "episode_count": sum(item["report"]["episodes"]["matched"] for item in source_reports),
        "sources": source_reports,
        "warnings": [
            f"{item['batch']['batch_id']}: {warning}"
            for item in source_reports for warning in item["report"]["warnings"]
        ],
        "deployment_contract": (
            "Online RL must use the same fixed horizon and provide the planned/actually queued command "
            "sequence available at t_k. Retrospective human corrections are not causal policy inputs."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", result)
    return result
