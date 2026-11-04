from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def normalized_band_cost(
    observed: float,
    lower: float,
    upper: float,
    gate: bool,
    *,
    force_scale: float,
    cost_clip: float = 1.0,
) -> float:
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    if cost_clip < 0.0:
        raise ValueError("cost_clip must be non-negative")
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if not gate:
        return 0.0
    low = max(0.0, float(lower) - float(observed))
    high = max(0.0, float(observed) - float(upper))
    low_normalized = float(np.clip(low / force_scale, 0.0, cost_clip))
    high_normalized = float(np.clip(high / force_scale, 0.0, cost_clip))
    return low_normalized**2 + high_normalized**2


def interval_score_scalar(
    observed: float,
    lower: float,
    upper: float,
    *,
    alpha: float = 0.1,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    return float(
        upper
        - lower
        + (2.0 / alpha) * max(lower - observed, 0.0)
        + (2.0 / alpha) * max(observed - upper, 0.0)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _required_window(row: dict[str, Any], index: int) -> bool:
    required = (
        f"window_{index}_lower_avg_N",
        f"window_{index}_upper_avg_N",
        f"window_{index}_observed_avg_N",
        f"window_{index}_gate_bilateral",
        f"window_{index}_quality_valid",
        f"window_{index}_peak_triggered",
        f"tactile_window_{index}_penalty",
    )
    return all(key in row and row[key] is not None for key in required)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(values, probability)) if values else float("nan")


def _reward_summary(
    rows: list[dict[str, Any]], *, material_threshold: float
) -> dict[str, Any]:
    dual = np.asarray([row["window_2_dual_soft_reward"] for row in rows])
    held_band = np.asarray([row["window_2_held_band_fresh_gate_soft_reward"] for row in rows])
    single = np.asarray([row["window_2_single_rate_soft_reward"] for row in rows])
    dual_transition = np.asarray([row["dual_transition_soft_reward"] for row in rows])
    single_transition = np.asarray([row["single_transition_soft_reward"] for row in rows])

    def events(values: np.ndarray, threshold: float) -> np.ndarray:
        return values >= threshold

    def confusion(candidate: np.ndarray, threshold: float) -> dict[str, Any]:
        reference_event = events(dual, threshold)
        candidate_event = events(candidate, threshold)
        missed = reference_event & ~candidate_event
        false = ~reference_event & candidate_event
        return {
            "threshold": float(threshold),
            "dual_events": int(reference_event.sum()),
            "candidate_events": int(candidate_event.sum()),
            "missed_penalty_windows": int(missed.sum()),
            "false_penalty_windows": int(false.sum()),
            "miss_rate_among_dual_events": (
                float(missed.sum() / reference_event.sum())
                if reference_event.any()
                else 0.0
            ),
            "false_rate_among_dual_nonevents": (
                float(false.sum() / (~reference_event).sum())
                if (~reference_event).any()
                else 0.0
            ),
            "false_fraction_among_candidate_events": (
                float(false.sum() / candidate_event.sum())
                if candidate_event.any()
                else 0.0
            ),
        }

    difference = single_transition - dual_transition
    return {
        "transitions": int(len(rows)),
        "dual_window_2_soft_reward_sum": float(dual.sum()),
        "held_band_fresh_gate_window_2_soft_reward_sum": float(held_band.sum()),
        "single_rate_window_2_soft_reward_sum": float(single.sum()),
        "single_vs_dual_window_2_any": confusion(single, 1e-12),
        "single_vs_dual_window_2_material": confusion(single, material_threshold),
        "held_band_fresh_gate_vs_dual_material": confusion(
            held_band, material_threshold
        ),
        "transition_soft_reward_mae": float(np.mean(np.abs(difference))),
        "transition_soft_reward_error_p90": float(
            np.quantile(np.abs(difference), 0.90)
        ),
        "transition_soft_reward_error_p95": float(
            np.quantile(np.abs(difference), 0.95)
        ),
        "transition_soft_reward_error_max": float(np.max(np.abs(difference))),
        "single_under_penalty_amount": float(np.maximum(-difference, 0.0).sum()),
        "single_over_penalty_amount": float(np.maximum(difference, 0.0).sum()),
        "dual_transition_soft_reward_sum": float(dual_transition.sum()),
        "single_transition_soft_reward_sum": float(single_transition.sum()),
    }


def _summarize_transitions(
    transitions: list[dict[str, Any]],
    *,
    task: str,
    source_file: str,
    source_rows: int,
    rejected: dict[str, int],
    force_scale: float,
    cost_clip: float,
    material_threshold: float,
    source_protocol: str,
    fresh_reward_reproduction_max_abs_error: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not transitions:
        raise RuntimeError(f"No complete dual-rate transitions found in {source_file}")

    gate_2 = [row for row in transitions if row["window_2_fresh_gate"]]
    fresh_covered = [
        row["window_2_fresh_lower_N"]
        <= row["window_2_observed_N"]
        <= row["window_2_fresh_upper_N"]
        for row in gate_2
    ]
    held_covered = [
        row["window_2_held_lower_N"]
        <= row["window_2_observed_N"]
        <= row["window_2_held_upper_N"]
        for row in gate_2
    ]
    fresh_scores = [
        interval_score_scalar(
            row["window_2_observed_N"],
            row["window_2_fresh_lower_N"],
            row["window_2_fresh_upper_N"],
        )
        for row in gate_2
    ]
    held_scores = [
        interval_score_scalar(
            row["window_2_observed_N"],
            row["window_2_held_lower_N"],
            row["window_2_held_upper_N"],
        )
        for row in gate_2
    ]

    episode_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in transitions:
        episode_groups[row["episode"]].append(row)
    episodes: list[dict[str, Any]] = []
    for episode, episode_rows in sorted(episode_groups.items()):
        dual_raw = float(sum(row["dual_transition_soft_reward"] for row in episode_rows))
        single_raw = float(
            sum(row["single_transition_soft_reward"] for row in episode_rows)
        )
        cap = float(episode_rows[0]["soft_episode_cap"])
        episodes.append(
            {
                "task": task,
                "episode": episode,
                "success": int(episode_rows[-1]["success"]),
                "transitions": len(episode_rows),
                "dual_uncapped_soft_reward": dual_raw,
                "single_uncapped_soft_reward": single_raw,
                "dual_capped_soft_reward": min(dual_raw, cap),
                "single_capped_soft_reward": min(single_raw, cap),
                "soft_episode_cap": cap,
                "material_missed_penalty_windows": int(
                    sum(row["material_missed_penalty"] for row in episode_rows)
                ),
                "material_false_penalty_windows": int(
                    sum(row["material_false_penalty"] for row in episode_rows)
                ),
            }
        )

    reward = _reward_summary(transitions, material_threshold=material_threshold)
    outcome = {}
    for success, label in ((1, "success"), (0, "failure")):
        selected = [row for row in transitions if row["success"] == success]
        if selected:
            outcome[label] = _reward_summary(
                selected, material_threshold=material_threshold
            )

    report = {
        "task": task,
        "source_file": source_file,
        "source_protocol": source_protocol,
        "source_rows": source_rows,
        "accepted_transitions": len(transitions),
        "rejected": dict(rejected),
        "episodes": len(episodes),
        "force_scale_N": float(force_scale),
        "cost_clip": float(cost_clip),
        "material_reward_threshold": float(material_threshold),
        "fresh_reward_reproduction_max_abs_error": (
            None
            if fresh_reward_reproduction_max_abs_error is None
            else float(fresh_reward_reproduction_max_abs_error)
        ),
        "gate": {
            "window_1_active": int(sum(row["window_1_gate"] for row in transitions)),
            "window_2_active": len(gate_2),
            "changed_between_windows": int(
                sum(
                    row["window_1_gate"] != row["window_2_fresh_gate"]
                    for row in transitions
                )
            ),
        },
        "window_2_envelope": {
            "gate_active_samples": len(gate_2),
            "fresh_coverage": _mean([float(value) for value in fresh_covered]),
            "held_band_coverage": _mean([float(value) for value in held_covered]),
            "fresh_interval_score_mean_N": _mean(fresh_scores),
            "held_band_interval_score_mean_N": _mean(held_scores),
        },
        "reward": reward,
        "reward_by_episode_outcome": outcome,
        "episode_caps": {
            "dual_uncapped_total": float(
                sum(row["dual_uncapped_soft_reward"] for row in episodes)
            ),
            "single_uncapped_total": float(
                sum(row["single_uncapped_soft_reward"] for row in episodes)
            ),
            "dual_capped_total": float(
                sum(row["dual_capped_soft_reward"] for row in episodes)
            ),
            "single_capped_total": float(
                sum(row["single_capped_soft_reward"] for row in episodes)
            ),
            "episodes_with_different_capped_reward": int(
                sum(
                    not np.isclose(
                        row["dual_capped_soft_reward"],
                        row["single_capped_soft_reward"],
                    )
                    for row in episodes
                )
            ),
        },
    }
    return report, transitions, episodes


def analyze_paired_h9_windows(
    rows: list[dict[str, Any]],
    *,
    task: str,
    source_file: str,
    force_scale: float,
    average_penalty_weight: float = 0.6,
    soft_episode_cap: float = 0.6,
    material_threshold: float = 0.01,
    cost_clip: float = 1.0,
    minimum_gap_s: float = 0.25,
    maximum_gap_s: float = 0.38,
    source_protocol: str = "offline episode split paired H9 counterfactual",
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair consecutive offline H9 windows into non-overlapping E18 transitions."""
    if minimum_gap_s <= 0.0 or maximum_gap_s < minimum_gap_s:
        raise ValueError("Invalid H9 pairing gap")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)
    for row in rows:
        values = [float(row[key]) for key in ("timestamp", "observed", "lower", "upper")]
        if not np.isfinite(values).all() or float(row["observed"]) < 0.0:
            rejected["invalid_window"] += 1
            continue
        grouped[int(row["episode"])].append(row)

    transitions: list[dict[str, Any]] = []
    for episode, episode_rows in sorted(grouped.items()):
        ordered = sorted(episode_rows, key=lambda row: float(row["timestamp"]))
        index = 0
        chunk = 1
        while index + 1 < len(ordered):
            first = ordered[index]
            second = ordered[index + 1]
            gap_s = float(second["timestamp"]) - float(first["timestamp"])
            consecutive_index = int(second["window_index"]) == int(first["window_index"]) + 1
            if not consecutive_index or not minimum_gap_s <= gap_s <= maximum_gap_s:
                rejected["nonconsecutive_h9_gap"] += 1
                index += 1
                continue

            window = {}
            for window_index, source in ((1, first), (2, second)):
                observed = float(source["observed"])
                lower = float(source["lower"])
                upper = float(source["upper"])
                gate = bool(source["gate"])
                cost = normalized_band_cost(
                    observed,
                    lower,
                    upper,
                    gate,
                    force_scale=force_scale,
                    cost_clip=cost_clip,
                )
                window[window_index] = {
                    "observed": observed,
                    "lower": lower,
                    "upper": upper,
                    "gate": gate,
                    "soft_reward": average_penalty_weight * cost,
                }

            held_fresh_gate_reward = average_penalty_weight * normalized_band_cost(
                window[2]["observed"],
                window[1]["lower"],
                window[1]["upper"],
                window[2]["gate"],
                force_scale=force_scale,
                cost_clip=cost_clip,
            )
            single_window_2_reward = average_penalty_weight * normalized_band_cost(
                window[2]["observed"],
                window[1]["lower"],
                window[1]["upper"],
                window[1]["gate"],
                force_scale=force_scale,
                cost_clip=cost_clip,
            )
            dual_transition_reward = 0.5 * (
                window[1]["soft_reward"] + window[2]["soft_reward"]
            )
            single_transition_reward = 0.5 * (
                window[1]["soft_reward"] + single_window_2_reward
            )
            transitions.append(
                {
                    "task": task,
                    "source_file": source_file,
                    "source_row": int(first["window_index"]),
                    "episode": episode,
                    "chunk": chunk,
                    "success": int(second.get("success", first.get("success", 1))),
                    "window_1_timestamp_s": float(first["timestamp"]),
                    "window_2_timestamp_s": float(second["timestamp"]),
                    "h9_gap_s": gap_s,
                    "window_1_observed_N": window[1]["observed"],
                    "window_1_lower_N": window[1]["lower"],
                    "window_1_upper_N": window[1]["upper"],
                    "window_1_gate": int(window[1]["gate"]),
                    "window_1_dual_soft_reward": window[1]["soft_reward"],
                    "window_2_observed_N": window[2]["observed"],
                    "window_2_fresh_lower_N": window[2]["lower"],
                    "window_2_fresh_upper_N": window[2]["upper"],
                    "window_2_held_lower_N": window[1]["lower"],
                    "window_2_held_upper_N": window[1]["upper"],
                    "window_2_fresh_gate": int(window[2]["gate"]),
                    "window_2_held_gate": int(window[1]["gate"]),
                    "window_2_dual_soft_reward": window[2]["soft_reward"],
                    "window_2_held_band_fresh_gate_soft_reward": held_fresh_gate_reward,
                    "window_2_single_rate_soft_reward": single_window_2_reward,
                    "dual_transition_soft_reward": dual_transition_reward,
                    "single_transition_soft_reward": single_transition_reward,
                    "transition_soft_reward_error": (
                        single_transition_reward - dual_transition_reward
                    ),
                    "material_dual_event": int(
                        window[2]["soft_reward"] >= material_threshold
                    ),
                    "material_single_event": int(
                        single_window_2_reward >= material_threshold
                    ),
                    "material_missed_penalty": int(
                        window[2]["soft_reward"] >= material_threshold
                        and single_window_2_reward < material_threshold
                    ),
                    "material_false_penalty": int(
                        window[2]["soft_reward"] < material_threshold
                        and single_window_2_reward >= material_threshold
                    ),
                    "average_penalty_weight": float(average_penalty_weight),
                    "soft_episode_cap": float(soft_episode_cap),
                }
            )
            chunk += 1
            index += 2
        if index < len(ordered):
            rejected["unpaired_final_h9_window"] += 1

    return _summarize_transitions(
        transitions,
        task=task,
        source_file=source_file,
        source_rows=len(rows),
        rejected=dict(rejected),
        force_scale=force_scale,
        cost_clip=cost_clip,
        material_threshold=material_threshold,
        source_protocol=source_protocol,
        fresh_reward_reproduction_max_abs_error=None,
    )


def analyze_dual_rate_log(
    path: Path,
    *,
    task: str,
    force_scale: float,
    material_threshold: float = 0.01,
    cost_clip: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = load_jsonl(path)
    transitions: list[dict[str, Any]] = []
    rejected = defaultdict(int)
    validation_errors: list[float] = []

    for source_index, row in enumerate(source_rows):
        if float(row.get("tactile_subwindows", 0.0)) < 2.0:
            rejected["fewer_than_two_subwindows"] += 1
            continue
        if not (_required_window(row, 1) and _required_window(row, 2)):
            rejected["missing_window_fields"] += 1
            continue
        if not (
            bool(row["window_1_quality_valid"])
            and bool(row["window_2_quality_valid"])
        ):
            rejected["low_quality_window"] += 1
            continue
        numeric_values = [
            float(row[f"window_{index}_{field}_N"])
            for index in (1, 2)
            for field in ("observed_avg", "lower_avg", "upper_avg")
        ]
        if not np.isfinite(numeric_values).all() or any(
            float(row[f"window_{index}_observed_avg_N"]) < 0.0
            for index in (1, 2)
        ):
            rejected["invalid_observed_force_sentinel"] += 1
            continue

        weight = float(row["average_penalty_weight"])
        peak_weight = float(row["peak_penalty"])
        window: dict[int, dict[str, Any]] = {}
        for index in (1, 2):
            observed = float(row[f"window_{index}_observed_avg_N"])
            lower = float(row[f"window_{index}_lower_avg_N"])
            upper = float(row[f"window_{index}_upper_avg_N"])
            gate = bool(row[f"window_{index}_gate_bilateral"])
            peak = bool(row[f"window_{index}_peak_triggered"])
            normalized = normalized_band_cost(
                observed,
                lower,
                upper,
                gate,
                force_scale=force_scale,
                cost_clip=cost_clip,
            )
            soft_reward = weight * normalized
            reported_total = float(row[f"tactile_window_{index}_penalty"])
            reported_soft = max(0.0, reported_total - peak_weight * float(peak))
            validation_errors.append(abs(reported_soft - soft_reward))
            window[index] = {
                "observed": observed,
                "lower": lower,
                "upper": upper,
                "gate": gate,
                "soft_reward": soft_reward,
            }

        held_band_fresh_gate_cost = normalized_band_cost(
            window[2]["observed"],
            window[1]["lower"],
            window[1]["upper"],
            window[2]["gate"],
            force_scale=force_scale,
            cost_clip=cost_clip,
        )
        held_band_held_gate_cost = normalized_band_cost(
            window[2]["observed"],
            window[1]["lower"],
            window[1]["upper"],
            window[1]["gate"],
            force_scale=force_scale,
            cost_clip=cost_clip,
        )
        held_band_fresh_gate_reward = weight * held_band_fresh_gate_cost
        single_window_2_reward = weight * held_band_held_gate_cost
        dual_transition_reward = 0.5 * (
            window[1]["soft_reward"] + window[2]["soft_reward"]
        )
        single_transition_reward = 0.5 * (
            window[1]["soft_reward"] + single_window_2_reward
        )
        transitions.append(
            {
                "task": task,
                "source_file": str(path),
                "source_row": source_index + 1,
                "episode": int(row["episode"]),
                "chunk": int(row["chunk"]),
                "success": int(row.get("success", 0)),
                "window_1_observed_N": window[1]["observed"],
                "window_1_lower_N": window[1]["lower"],
                "window_1_upper_N": window[1]["upper"],
                "window_1_gate": int(window[1]["gate"]),
                "window_1_dual_soft_reward": window[1]["soft_reward"],
                "window_2_observed_N": window[2]["observed"],
                "window_2_fresh_lower_N": window[2]["lower"],
                "window_2_fresh_upper_N": window[2]["upper"],
                "window_2_held_lower_N": window[1]["lower"],
                "window_2_held_upper_N": window[1]["upper"],
                "window_2_fresh_gate": int(window[2]["gate"]),
                "window_2_held_gate": int(window[1]["gate"]),
                "window_2_dual_soft_reward": window[2]["soft_reward"],
                "window_2_held_band_fresh_gate_soft_reward": held_band_fresh_gate_reward,
                "window_2_single_rate_soft_reward": single_window_2_reward,
                "dual_transition_soft_reward": dual_transition_reward,
                "single_transition_soft_reward": single_transition_reward,
                "transition_soft_reward_error": (
                    single_transition_reward - dual_transition_reward
                ),
                "material_dual_event": int(
                    window[2]["soft_reward"] >= material_threshold
                ),
                "material_single_event": int(
                    single_window_2_reward >= material_threshold
                ),
                "material_missed_penalty": int(
                    window[2]["soft_reward"] >= material_threshold
                    and single_window_2_reward < material_threshold
                ),
                "material_false_penalty": int(
                    window[2]["soft_reward"] < material_threshold
                    and single_window_2_reward >= material_threshold
                ),
                "average_penalty_weight": weight,
                "soft_episode_cap": float(row["soft_episode_cap"]),
            }
        )

    if not transitions:
        raise RuntimeError(f"No complete dual-rate transitions found in {path}")
    max_validation_error = max(validation_errors)
    if max_validation_error > 1e-5:
        raise RuntimeError(
            "Configured force scale does not reproduce logged fresh-window rewards: "
            f"max_abs_error={max_validation_error:.6g}, path={path}"
        )

    return _summarize_transitions(
        transitions,
        task=task,
        source_file=str(path),
        source_rows=len(source_rows),
        rejected=dict(rejected),
        force_scale=force_scale,
        cost_clip=cost_clip,
        material_threshold=material_threshold,
        source_protocol=(
            "online E18 dual-rate rollout with paired single-rate counterfactual"
        ),
        fresh_reward_reproduction_max_abs_error=max_validation_error,
    )
