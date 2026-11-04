from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .stable_contact_experiment import _gates_for_rows, _load_evaluation_context


LOWER_COVERAGE = 0.05
UPPER_COVERAGE = 0.95
SIGMA_FLOOR_N = 1e-6
BILATERAL_SCALE_METHODS = ("mean", "rss", "max")


def bilateral_uncertainty_scale(
    sigma_left: np.ndarray,
    sigma_right: np.ndarray,
    method: str,
    *,
    floor_N: float = SIGMA_FLOOR_N,
) -> np.ndarray:
    """Combine side-specific predicted scales without claiming a joint stddev."""

    left = np.asarray(sigma_left, dtype=np.float64)
    right = np.asarray(sigma_right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"Left/right sigma shapes differ: {left.shape} versus {right.shape}"
        )
    if method == "mean":
        scale = 0.5 * (left + right)
    elif method == "rss":
        scale = 0.5 * np.sqrt(left**2 + right**2)
    elif method == "max":
        scale = np.maximum(left, right)
    else:
        raise ValueError(
            f"Unknown bilateral uncertainty scale {method!r}; "
            f"expected one of {BILATERAL_SCALE_METHODS}"
        )
    return np.maximum(scale, float(floor_N))


def _finite_sample_tail_quantile(
    values: np.ndarray, probability: float, *, upper: bool
) -> tuple[float, int]:
    """Return a split-conformal tail order statistic and its one-based rank."""

    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if not ordered.size:
        raise ValueError("Cannot calibrate an empty score array")
    unbounded_rank = (
        math.ceil((ordered.size + 1) * probability)
        if upper
        else math.floor((ordered.size + 1) * probability)
    )
    rank = min(ordered.size, max(1, int(unbounded_rank)))
    return float(ordered[rank - 1]), rank


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("batch_id", "")), int(row["episode_index"])


def _interval_metrics(
    target: np.ndarray,
    mu: np.ndarray,
    uncertainty_scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    residual = target - mu
    covered = (target >= lower) & (target <= upper)
    low_violation = target < lower
    high_violation = target > upper
    width = upper - lower
    return {
        "samples": int(target.size),
        "target_mean_N": float(np.mean(target)),
        "target_std_N": float(np.std(target)),
        "prediction_bias_mu_minus_target_N": float(np.mean(mu - target)),
        "mae_N": float(np.mean(np.abs(residual))),
        "rmse_N": float(np.sqrt(np.mean(residual**2))),
        "mean_uncertainty_scale_N": float(np.mean(uncertainty_scale)),
        "coverage": float(np.mean(covered)),
        "mean_interval_width_N": float(np.mean(width)),
        "interval_width_p05_N": float(np.quantile(width, 0.05)),
        "interval_width_p50_N": float(np.quantile(width, 0.50)),
        "interval_width_p95_N": float(np.quantile(width, 0.95)),
        "low_violation_windows": int(np.sum(low_violation)),
        "high_violation_windows": int(np.sum(high_violation)),
        "out_of_band_windows": int(np.sum(~covered)),
    }


def calibrate_bilateral_sigma_envelope(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calibrate an input-dependent bilateral force band without using test targets."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint, rows, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path)
    )
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else checkpoint_path.parent / "bilateral_sigma_calibration.json"
    )

    gates = {
        split: _gates_for_rows(grouped[split], history, history_mask, config)
        for split in ("calibration", "test")
    }
    masks = {
        split: gates[split][:, 0] & gates[split][:, 1]
        for split in ("calibration", "test")
    }
    if not np.any(masks["calibration"]):
        raise ValueError("No bilateral stable-contact calibration windows")
    if not np.any(masks["test"]):
        raise ValueError("No bilateral stable-contact test windows")

    values: dict[str, dict[str, np.ndarray]] = {}
    for split in ("calibration", "test"):
        left = predictions[split]["left"]
        right = predictions[split]["right"]
        values[split] = {
            "target": 0.5 * (left["target"] + right["target"]),
            "mu": 0.5 * (left["mu"] + right["mu"]),
            "uncertainty_scale": bilateral_uncertainty_scale(
                left["sigma"], right["sigma"], "mean"
            ),
        }

    cal_mask = masks["calibration"]
    cal = values["calibration"]
    cal_residual = cal["target"][cal_mask] - cal["mu"][cal_mask]
    standardized = cal_residual / cal["uncertainty_scale"][cal_mask]
    alpha05, alpha05_rank = _finite_sample_tail_quantile(
        standardized, LOWER_COVERAGE, upper=False
    )
    alpha95, alpha95_rank = _finite_sample_tail_quantile(
        standardized, UPPER_COVERAGE, upper=True
    )
    if not np.isfinite([alpha05, alpha95]).all() or alpha05 >= alpha95:
        raise RuntimeError(
            f"Invalid standardized-residual calibration: alpha05={alpha05}, "
            f"alpha95={alpha95}"
        )

    test_mask = masks["test"]
    test = values["test"]
    target = test["target"][test_mask]
    mu = test["mu"][test_mask]
    uncertainty_scale = test["uncertainty_scale"][test_mask]
    lower = mu + alpha05 * uncertainty_scale
    upper = mu + alpha95 * uncertainty_scale
    test_metrics = _interval_metrics(
        target, mu, uncertainty_scale, lower, upper
    )

    additive_q05, additive_q05_rank = _finite_sample_tail_quantile(
        cal_residual, LOWER_COVERAGE, upper=False
    )
    additive_q95, additive_q95_rank = _finite_sample_tail_quantile(
        cal_residual, UPPER_COVERAGE, upper=True
    )
    additive_metrics = _interval_metrics(
        target,
        mu,
        uncertainty_scale,
        mu + additive_q05,
        mu + additive_q95,
    )

    selected_positions = np.flatnonzero(test_mask)
    episode_positions: dict[tuple[str, int], list[int]] = {}
    for compact_index, position in enumerate(selected_positions):
        episode_positions.setdefault(
            _episode_key(grouped["test"][int(position)]), []
        ).append(compact_index)
    per_episode: list[dict[str, Any]] = []
    for (batch_id, episode), positions in sorted(episode_positions.items()):
        index = np.asarray(positions, dtype=np.int64)
        metrics = _interval_metrics(
            target[index], mu[index], uncertainty_scale[index], lower[index], upper[index]
        )
        per_episode.append({
            "batch_id": batch_id,
            "episode": episode,
            "stable_contact_windows": int(index.size),
            "coverage": metrics["coverage"],
            "mae_N": metrics["mae_N"],
            "rmse_N": metrics["rmse_N"],
            "mean_uncertainty_scale_N": metrics["mean_uncertainty_scale_N"],
            "mean_interval_width_N": metrics["mean_interval_width_N"],
            "low_violation_windows": metrics["low_violation_windows"],
            "high_violation_windows": metrics["high_violation_windows"],
        })

    train_raw_peaks: list[float] = []
    for row in rows:
        if row.get("split") != "train":
            continue
        side_peaks = [
            float(sensor["raw_max"])
            for sensor in row["tactile"]["per_sensor"].values()
            if sensor.get("raw_max") is not None
            and np.isfinite(float(sensor["raw_max"]))
        ]
        if side_peaks:
            train_raw_peaks.append(max(side_peaks))
    if not train_raw_peaks:
        raise ValueError("Train split has no finite raw tactile peak values")
    raw_peak_offset = float(
        config.get("stable_contact", {}).get("raw_peak_offset_N", 10.0)
    )
    if raw_peak_offset < 0.0:
        raise ValueError("stable_contact.raw_peak_offset_N must be non-negative")
    train_raw_max = float(np.max(train_raw_peaks))
    raw_peak_threshold = {
        "source": (
            "maximum zero-biased raw bilateral force magnitude over train split "
            f"windows plus {raw_peak_offset:g} N"
        ),
        "train_samples": len(train_raw_peaks),
        "train_raw_max_N": train_raw_max,
        "offset_N": raw_peak_offset,
        "threshold_N": train_raw_max + raw_peak_offset,
        "uses_validation_calibration_or_test": False,
    }

    task_override = config.get("data", {}).get("task_override")
    row_tasks = sorted({
        str(row["task"]).strip()
        for row in rows
        if row.get("task") is not None and str(row["task"]).strip()
    })
    if task_override is not None:
        task = str(task_override).strip()
    elif len(row_tasks) == 1:
        task = row_tasks[0]
    elif not row_tasks:
        task = None
    else:
        raise ValueError(
            "Calibration windows contain multiple task texts; set data.task_override "
            f"explicitly. Observed tasks: {row_tasks}"
        )
    report: dict[str, Any] = {
        "schema_version": 2,
        "method": "bilateral_stable_contact_average_uncertainty_scaled_conformal",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "task": task,
        "selection": "stable_prior_left AND stable_prior_right",
        "selection_uses_future_target": False,
        "target_definition": "y_avg = 0.5 * (y_left_top2_mean + y_right_top2_mean)",
        "center_definition": "mu_avg = 0.5 * (mu_left + mu_right)",
        "scale_method": "mean",
        "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
        "scale_is_standard_deviation": False,
        "standardized_residual_definition": (
            "s = (y_avg - mu_avg) / max(u_mean, scale_floor_N)"
        ),
        "lower_coverage": LOWER_COVERAGE,
        "lower_alpha_label": "alpha05",
        "lower_alpha": alpha05,
        "lower_alpha_source": (
            "calibration split finite-sample lower-tail order statistic of s"
        ),
        "lower_alpha_rank_one_based": alpha05_rank,
        "upper_coverage": UPPER_COVERAGE,
        "upper_alpha_label": "alpha95",
        "upper_alpha": alpha95,
        "upper_alpha_source": (
            "calibration split finite-sample upper-tail order statistic of s"
        ),
        "upper_alpha_rank_one_based": alpha95_rank,
        "lower_formula": "lower_avg = mu_avg + alpha05 * u_mean",
        "upper_formula": "upper_avg = mu_avg + alpha95 * u_mean",
        "scale_floor_N": SIGMA_FLOOR_N,
        "calibration_samples": int(np.sum(cal_mask)),
        "calibration_standardized_residual": {
            "mean": float(np.mean(standardized)),
            "std": float(np.std(standardized)),
            "q05": alpha05,
            "q50": float(np.quantile(standardized, 0.50)),
            "q95": alpha95,
        },
        "test_not_used_for_lower_alpha": True,
        "test_not_used_for_upper_alpha": True,
        "raw_peak_threshold": raw_peak_threshold,
        "test_report": {
            **test_metrics,
            "episodes": len(per_episode),
            "used_to_adjust_alpha": False,
        },
        "additive_ablation": {
            "method": "mu_avg plus calibration residual q05/q95",
            "q05_N": additive_q05,
            "q95_N": additive_q95,
            "q05_rank_one_based": additive_q05_rank,
            "q95_rank_one_based": additive_q95_rank,
            "test_report": additive_metrics,
        },
        "training_objective": checkpoint.get("loss"),
        "development_test_notice": (
            "Test metrics are reported without recalibrating on test targets. "
            "A new independent episode set is still required for blind evaluation."
        ),
    }
    write_json(output_path, report)
    csv_path = output_path.with_name(output_path.stem + "_per_episode.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_episode[0]))
        writer.writeheader()
        writer.writerows(per_episode)
    return report
