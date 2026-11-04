from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import write_json
from .stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


def _mean_std(values: list[float | int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _episode_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("batch_id", "")), int(row["episode_index"])


def _evaluate_seed(
    config: dict[str, Any], checkpoint_path: Path, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    checkpoint, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path)
    )
    calibration_gates = _gates_for_rows(
        grouped["calibration"], history, history_mask, config
    )
    test_gates = _gates_for_rows(grouped["test"], history, history_mask, config)
    calibration_mask = calibration_gates[:, 0] & calibration_gates[:, 1]
    test_mask = test_gates[:, 0] & test_gates[:, 1]
    if not np.any(calibration_mask) or not np.any(test_mask):
        raise ValueError("Bilateral stable-contact calibration/test selection is empty")

    cal_left = predictions["calibration"]["left"]
    cal_right = predictions["calibration"]["right"]
    cal_y_avg = 0.5 * (cal_left["target"] + cal_right["target"])
    cal_mu_avg = 0.5 * (cal_left["mu"] + cal_right["mu"])
    calibration_residual = cal_y_avg[calibration_mask] - cal_mu_avg[calibration_mask]
    q95 = float(np.quantile(calibration_residual, 0.95))

    test_left = predictions["test"]["left"]
    test_right = predictions["test"]["right"]
    y_avg_all = 0.5 * (test_left["target"] + test_right["target"])
    mu_avg_all = 0.5 * (test_left["mu"] + test_right["mu"])
    upper_avg_all = mu_avg_all + q95
    y_avg = y_avg_all[test_mask]
    mu_avg = mu_avg_all[test_mask]
    upper_avg = upper_avg_all[test_mask]
    penalty = y_avg > upper_avg

    # Compare against the currently calibrated independent additive envelopes
    # on exactly the same bilateral-gate test windows.
    q95_left = float(checkpoint["calibration"]["q95_left"])
    q95_right = float(checkpoint["calibration"]["q95_right"])
    independent_penalty_all = (
        (test_left["target"] > test_left["mu"] + q95_left)
        | (test_right["target"] > test_right["mu"] + q95_right)
    )
    independent_penalty = independent_penalty_all[test_mask]

    selected_positions = np.flatnonzero(test_mask)
    episode_positions: dict[tuple[str, int], list[int]] = {}
    for compact_index, test_position in enumerate(selected_positions):
        episode_positions.setdefault(
            _episode_key(grouped["test"][int(test_position)]), []
        ).append(compact_index)

    per_episode: list[dict[str, Any]] = []
    false_episode_count = 0
    independent_false_episode_count = 0
    for (batch_id, episode), positions in sorted(episode_positions.items()):
        index = np.asarray(positions, dtype=np.int64)
        episode_false = int(np.sum(penalty[index]))
        independent_episode_false = int(np.sum(independent_penalty[index]))
        false_episode_count += int(episode_false > 0)
        independent_false_episode_count += int(independent_episode_false > 0)
        per_episode.append({
            "seed": seed,
            "batch_id": batch_id,
            "episode": episode,
            "bilateral_stable_windows": int(index.size),
            "average_coverage": float(np.mean(~penalty[index])),
            "average_false_penalty_windows": episode_false,
            "independent_joint_coverage": float(np.mean(~independent_penalty[index])),
            "independent_false_penalty_windows": independent_episode_false,
            "y_avg_mae": float(np.mean(np.abs(y_avg[index] - mu_avg[index]))),
        })

    report = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "selection": "stable_prior_left AND stable_prior_right",
        "selection_uses_future_target": False,
        "calibration_samples": int(calibration_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "q95": q95,
        "q95_source": "calibration split empirical 0.95 quantile of y_avg - mu_avg",
        "test_not_used_for_q95": True,
        "mae": float(np.mean(np.abs(y_avg - mu_avg))),
        "coverage": float(np.mean(~penalty)),
        "mean_envelope_width": q95,
        "false_penalty_windows": int(np.sum(penalty)),
        "episodes": len(episode_positions),
        "episodes_with_false_penalty": false_episode_count,
        "independent_additive_comparison": {
            "q95_left": q95_left,
            "q95_right": q95_right,
            "joint_coverage": float(np.mean(~independent_penalty)),
            "false_penalty_windows": int(np.sum(independent_penalty)),
            "episodes_with_false_penalty": independent_false_episode_count,
            "definition": "penalty if either independently calibrated side is exceeded",
        },
    }
    curves = {
        "rows": grouped["test"],
        "calibration_residual": calibration_residual,
        "calibration_mask": calibration_mask,
        "calibration_left": cal_left,
        "calibration_right": cal_right,
        "q95": q95,
        "test_gates": test_gates,
        "test_mask": test_mask,
        "selected_positions": selected_positions,
        "y_avg_all": y_avg_all,
        "mu_avg_all": mu_avg_all,
        "upper_avg_all": upper_avg_all,
        "penalty_all": test_mask & (y_avg_all > upper_avg_all),
        "independent_penalty_all": test_mask & independent_penalty_all,
        "test_left": test_left,
        "test_right": test_right,
        "y_avg": y_avg,
        "mu_avg": mu_avg,
        "upper_avg": upper_avg,
        "episode_positions": episode_positions,
    }
    return report, per_episode, curves


def _plot_episode_curves(curves: dict[str, Any], output: Path, seed: int) -> list[int]:
    ranked = sorted(
        curves["episode_positions"].items(), key=lambda item: len(item[1]), reverse=True
    )[:6]
    figure, axes = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    plotted: list[int] = []
    for axis, ((batch_id, episode), positions) in zip(axes.flat, ranked, strict=False):
        compact = np.asarray(positions, dtype=np.int64)
        test_positions = curves["selected_positions"][compact]
        timestamps = np.asarray([
            float(curves["rows"][int(position)]["start_timestamp"])
            for position in test_positions
        ])
        order = np.argsort(timestamps)
        elapsed = timestamps[order] - timestamps[order][0]
        axis.plot(elapsed, curves["y_avg"][compact][order], label="observed y_avg", linewidth=2)
        axis.plot(elapsed, curves["mu_avg"][compact][order], label="predicted mu_avg", linewidth=1.7)
        axis.plot(elapsed, curves["upper_avg"][compact][order], label="upper_avg", linewidth=1.7)
        axis.set_title(f"batch={batch_id} episode={episode} n={compact.size}")
        axis.set_xlabel("time from first bilateral-stable anchor (s)")
        axis.set_ylabel("average force")
        axis.grid(alpha=0.25)
        plotted.append(episode)
    for axis in axes.flat[len(ranked):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.suptitle(f"Variant B seed {seed}: bilateral stable-contact average envelope")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return plotted


def evaluate_bilateral_average(config: dict[str, Any]) -> dict[str, Any]:
    stable = config["stable_contact"]
    seeds = [int(seed) for seed in stable.get("seeds", [42, 43, 44])]
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    root = workspace / str(stable.get("artifact_subdir", "stable_contact_ab"))
    output = root / "bilateral_average_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    selected_curves = None
    selected_seed = int(stable.get("reuse_variant_a_seed", 44))
    for seed in seeds:
        checkpoint = (
            root / f"B_seed_{seed}" / "stable_evaluation"
            / "stable_contact_calibrated_checkpoint.pt"
        )
        report, per_episode, curves = _evaluate_seed(config, checkpoint, seed)
        reports.append(report)
        episode_rows.extend(per_episode)
        if seed == selected_seed:
            selected_curves = curves

    keys = (
        "mae", "coverage", "mean_envelope_width", "false_penalty_windows",
        "episodes_with_false_penalty",
    )
    aggregate = {key: _mean_std([report[key] for report in reports]) for key in keys}
    independent_aggregate = {
        key: _mean_std([
            report["independent_additive_comparison"][key] for report in reports
        ])
        for key in (
            "joint_coverage", "false_penalty_windows", "episodes_with_false_penalty"
        )
    }
    coverage_mean = aggregate["coverage"]["mean"]
    fewer_false_windows = (
        aggregate["false_penalty_windows"]["mean"]
        < independent_aggregate["false_penalty_windows"]["mean"]
    )
    fewer_false_episodes = (
        aggregate["episodes_with_false_penalty"]["mean"]
        < independent_aggregate["episodes_with_false_penalty"]["mean"]
    )
    plot_path = output / f"bilateral_average_episode_curves_seed_{selected_seed}.png"
    plotted_episodes = _plot_episode_curves(selected_curves, plot_path, selected_seed)
    summary = {
        "method": "bilateral_stable_contact_average_additive_envelope",
        "formula": {
            "y_avg": "0.5 * (y_left + y_right)",
            "mu_avg": "0.5 * (mu_left + mu_right)",
            "upper_avg": "mu_avg + q95",
        },
        "selection": "both causal stable-contact gates are true",
        "future_target_used_for_selection": False,
        "false_penalty_definition": "observed target exceeds its calibrated upper envelope",
        "q95_uses_calibration_only": True,
        "development_test_used_for_tuning": False,
        "seeds": seeds,
        "runs": reports,
        "aggregate_mean_std": aggregate,
        "std_definition": "population standard deviation across the three seeds",
        "independent_additive_mean_std": independent_aggregate,
        "relative_false_penalty_window_reduction": float(
            1.0 - aggregate["false_penalty_windows"]["mean"]
            / independent_aggregate["false_penalty_windows"]["mean"]
        ),
        "relative_false_penalty_episode_reduction": float(
            1.0 - aggregate["episodes_with_false_penalty"]["mean"]
            / independent_aggregate["episodes_with_false_penalty"]["mean"]
        ),
        "coverage_in_93_97": bool(0.93 <= coverage_mean <= 0.97),
        "fewer_false_penalty_windows_than_independent": fewer_false_windows,
        "fewer_false_penalty_episodes_than_independent": fewer_false_episodes,
        "curve_plot": str(plot_path),
        "curve_seed": selected_seed,
        "curve_episodes": plotted_episodes,
        "production_ready": False,
        "note": "Development test is reported once and was not used to adjust q95.",
    }
    with (output / "bilateral_average_per_episode.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    write_json(output / "bilateral_average_evaluation.json", summary)
    return summary
