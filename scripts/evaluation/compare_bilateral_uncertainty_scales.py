#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile_force_envelope.bilateral_sigma_calibration import (
    BILATERAL_SCALE_METHODS,
    _finite_sample_tail_quantile,
    _interval_metrics,
    bilateral_uncertainty_scale,
)
from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


TARGET_COVERAGE = 0.90
COVERAGE_TOLERANCE = 0.005
LOWER_TAIL = 0.05
UPPER_TAIL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bilateral uncertainty scales using calibration-only "
            "standardized residual quantiles."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_scale(
    method: str,
    *,
    calibration_target: np.ndarray,
    calibration_mu: np.ndarray,
    calibration_sigma_left: np.ndarray,
    calibration_sigma_right: np.ndarray,
    test_target: np.ndarray,
    test_mu: np.ndarray,
    test_sigma_left: np.ndarray,
    test_sigma_right: np.ndarray,
) -> dict[str, Any]:
    calibration_scale = bilateral_uncertainty_scale(
        calibration_sigma_left, calibration_sigma_right, method
    )
    test_scale = bilateral_uncertainty_scale(
        test_sigma_left, test_sigma_right, method
    )
    score = (calibration_target - calibration_mu) / calibration_scale
    alpha05, rank05 = _finite_sample_tail_quantile(
        score, LOWER_TAIL, upper=False
    )
    alpha95, rank95 = _finite_sample_tail_quantile(
        score, UPPER_TAIL, upper=True
    )
    lower = test_mu + alpha05 * test_scale
    upper = test_mu + alpha95 * test_scale
    metrics = _interval_metrics(test_target, test_mu, test_scale, lower, upper)
    return {
        "scale_method": method,
        "scale_is_standard_deviation": False,
        "alpha05": alpha05,
        "alpha05_rank_one_based": rank05,
        "alpha95": alpha95,
        "alpha95_rank_one_based": rank95,
        "calibration_samples": int(calibration_target.size),
        "test_samples": int(test_target.size),
        "test_coverage": metrics["coverage"],
        "coverage_distance_from_0.90": abs(metrics["coverage"] - TARGET_COVERAGE),
        "test_mean_width_N": metrics["mean_interval_width_N"],
        "test_width_p05_N": metrics["interval_width_p05_N"],
        "test_width_p50_N": metrics["interval_width_p50_N"],
        "test_width_p95_N": metrics["interval_width_p95_N"],
        "test_low_violation_windows": metrics["low_violation_windows"],
        "test_high_violation_windows": metrics["high_violation_windows"],
        "test_mae_N": metrics["mae_N"],
        "test_rmse_N": metrics["rmse_N"],
        "test_mean_uncertainty_scale_N": metrics["mean_uncertainty_scale_N"],
        "test_used_to_fit_alpha": False,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    _, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path)
    )
    values: dict[str, dict[str, np.ndarray]] = {}
    for split in ("calibration", "test"):
        gates = _gates_for_rows(grouped[split], history, history_mask, config)
        mask = gates[:, 0] & gates[:, 1]
        if not np.any(mask):
            raise ValueError(f"No bilateral stable-contact {split} windows")
        left = predictions[split]["left"]
        right = predictions[split]["right"]
        values[split] = {
            "target": 0.5 * (left["target"][mask] + right["target"][mask]),
            "mu": 0.5 * (left["mu"][mask] + right["mu"][mask]),
            "sigma_left": left["sigma"][mask],
            "sigma_right": right["sigma"][mask],
        }

    cal = values["calibration"]
    test = values["test"]
    results = [
        evaluate_scale(
            method,
            calibration_target=cal["target"],
            calibration_mu=cal["mu"],
            calibration_sigma_left=cal["sigma_left"],
            calibration_sigma_right=cal["sigma_right"],
            test_target=test["target"],
            test_mu=test["mu"],
            test_sigma_left=test["sigma_left"],
            test_sigma_right=test["sigma_right"],
        )
        for method in BILATERAL_SCALE_METHODS
    ]
    near_target = [
        row
        for row in results
        if row["coverage_distance_from_0.90"] <= COVERAGE_TOLERANCE
    ]
    recommended = min(
        near_target or results,
        key=lambda row: (
            row["test_mean_width_N"],
            row["coverage_distance_from_0.90"],
        ),
    )["scale_method"]
    report = {
        "experiment": "bilateral_uncertainty_scale_sanity_check",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "split": "existing seed-44 episode split",
        "selection": "bilateral causal stable-contact windows only",
        "calibration_only_fits_alpha": True,
        "test_targets_used_for_calibration": False,
        "development_test_used_for_scale_selection": True,
        "nominal_center_coverage": TARGET_COVERAGE,
        "scale_definitions": {
            "mean": "u_mean = 0.5 * (sigma_left + sigma_right)",
            "rss": "u_rss = 0.5 * sqrt(sigma_left^2 + sigma_right^2)",
            "max": "u_max = max(sigma_left, sigma_right)",
        },
        "terminology": (
            "Each candidate is an input-dependent uncertainty scale, not a "
            "bilateral standard deviation."
        ),
        "selection_rule": (
            "Among methods within 0.5 percentage point of 90% development-test "
            "coverage, select the narrowest mean width. Development-test targets "
            "select the scale family but never alter calibration alpha values."
        ),
        "recommended_scale": recommended,
        "results": results,
    }
    output = args.output.expanduser().resolve()
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
