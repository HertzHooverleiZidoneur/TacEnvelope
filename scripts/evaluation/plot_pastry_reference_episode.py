#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)
from tactile_force_envelope.train import _conformal_alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument(
        "--quantile-source", choices=("calibration", "test"), default="calibration"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint, _, grouped, predictions, history, history_mask, _ = _load_evaluation_context(
        config, args.checkpoint
    )
    quantile_rows = grouped[args.quantile_source]
    quantile_gates = _gates_for_rows(
        quantile_rows, history, history_mask, config
    )
    tau = float(config["stable_contact"]["tau_train"])
    quantile_left = predictions[args.quantile_source]["left"]
    quantile_right = predictions[args.quantile_source]["right"]
    quantile_mask = (
        quantile_gates[:, 0]
        & quantile_gates[:, 1]
        & (quantile_left["target"] > tau)
        & (quantile_right["target"] > tau)
    )
    quantile_target = 0.5 * (
        quantile_left["target"][quantile_mask]
        + quantile_right["target"][quantile_mask]
    )
    quantile_mu = 0.5 * (
        quantile_left["mu"][quantile_mask]
        + quantile_right["mu"][quantile_mask]
    )
    quantile_residual = quantile_target - quantile_mu
    q95, q95_rank = _conformal_alpha(quantile_residual, 0.95)
    negative_q05, q05_rank_from_upper_tail = _conformal_alpha(
        -quantile_residual, 0.95
    )
    q05 = -negative_q05
    quantile_lower = quantile_mu + q05
    quantile_upper = quantile_mu + q95
    quantile_interval_coverage = float(np.mean(
        (quantile_target >= quantile_lower) & (quantile_target <= quantile_upper)
    ))

    rows = grouped["test"]
    positions = np.asarray(
        [i for i, row in enumerate(rows) if int(row["episode_index"]) == args.episode],
        dtype=np.int64,
    )
    if not positions.size:
        raise ValueError(f"Episode {args.episode} is not in the test split")

    gates = _gates_for_rows(rows, history, history_mask, config)
    left = predictions["test"]["left"]
    right = predictions["test"]["right"]
    left_mask = gates[:, 0] & (left["target"] > tau)
    right_mask = gates[:, 1] & (right["target"] > tau)
    bilateral_mask = left_mask & right_mask
    selected = positions[bilateral_mask[positions]]
    if selected.size < 2:
        raise ValueError(f"Episode {args.episode} has fewer than two bilateral stable windows")

    timestamps = np.asarray([float(rows[i]["start_timestamp"]) for i in selected])
    time_s = timestamps - timestamps[0]
    target = 0.5 * (left["target"][selected] + right["target"][selected])
    mu = 0.5 * (left["mu"][selected] + right["mu"][selected])
    lower = mu + q05
    upper = mu + q95
    below = target < lower
    above = target > upper

    fig, ax = plt.subplots(figsize=(14.5, 8.2), constrained_layout=True)
    ax.fill_between(
        time_s, lower, upper, color="#177f78", alpha=0.17,
        label=f"{args.quantile_source}-fitted L05_avg to U95_avg",
    )
    ax.plot(time_s, mu, "--", color="#147d75", lw=2.0, label="predicted mu_avg")
    ax.plot(time_s, lower, "-", color="#56a69e", lw=1.8, label="predicted L05_avg")
    ax.plot(time_s, upper, "-", color="#147d75", lw=2.0, label="predicted U95_avg")
    ax.plot(
        time_s, target, "o-", color="#202a3a", markerfacecolor="#202a3a",
        markersize=7.0, lw=2.0, label="observed bilateral y_avg",
    )
    ax.scatter(
        time_s, target, s=120, facecolors="none", edgecolors="#11a650",
        linewidths=2.0, label="both stable gates on", zorder=4,
    )
    if np.any(below):
        ax.scatter(
            time_s[below], target[below], marker="v", s=105,
            color="#6f55b5", linewidths=1.6, label="below calibrated L05", zorder=5,
        )
    if np.any(above):
        ax.scatter(
            time_s[above], target[above], marker="x", s=120,
            color="#dc2f2f", linewidths=2.2, label="above calibrated U95", zorder=5,
        )

    coverage = float(np.mean((target >= lower) & (target <= upper)))
    mae = float(np.mean(np.abs(target - mu)))
    bias = float(np.mean(target - mu))
    max_low_deviation = float(np.max(np.maximum(0.0, lower - target)))
    max_high_deviation = float(np.max(np.maximum(0.0, target - upper)))
    ax.set_title(
        f"Pastry test episode {args.episode}: bilateral average force and "
        f"{args.quantile_source}-fitted L05-U95 envelope",
        fontsize=17,
    )
    ax.set_xlabel("time from first bilateral stable-contact window (s)", fontsize=13)
    ax.set_ylabel("bilateral average force (N)", fontsize=13)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", ncol=2, framealpha=0.9, fontsize=10)
    ax.text(
        0.015, 0.975,
        f"windows={selected.size}  MAE={mae:.3f} N  bias={bias:.3f} N  "
        f"interval coverage={coverage * 100:.2f}%  low/high violations={below.sum()}/{above.sum()}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86, "edgecolor": "#b9c2ca"},
    )
    if args.quantile_source == "test":
        ax.text(
            0.985, 0.025, "TEST-FITTED DIAGNOSTIC - NOT HELD-OUT EVALUATION",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            color="#a12622", weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff3f1", "alpha": 0.9, "edgecolor": "#d58e87"},
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    result = {
        "split": "test",
        "episode": args.episode,
        "stable_bilateral_windows": int(selected.size),
        "mu_mae_N": mae,
        "mu_bias_y_minus_mu_N": bias,
        "interval_coverage": coverage,
        "quantile_source": args.quantile_source,
        "quantile_samples": int(quantile_residual.size),
        "quantile_source_interval_coverage": quantile_interval_coverage,
        "q05_N": q05,
        "q95_N": q95,
        "finite_sample_rank_q05_from_upper_tail": q05_rank_from_upper_tail,
        "finite_sample_rank_q95": q95_rank,
        "mean_interval_width_N": float(np.mean(upper - lower)),
        "low_violation_windows": int(below.sum()),
        "high_violation_windows": int(above.sum()),
        "max_low_deviation_N": max_low_deviation,
        "max_high_deviation_N": max_high_deviation,
        "calibration_source": f"{args.quantile_source} split",
        "test_recalibration": args.quantile_source == "test",
        "held_out_evaluation_valid": args.quantile_source != "test",
        "png": str(output.with_suffix(".png")),
        "pdf": str(output.with_suffix(".pdf")),
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
