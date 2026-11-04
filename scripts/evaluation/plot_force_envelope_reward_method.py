#!/usr/bin/env python3
"""Create the paper figure linking force-envelope violations to RL reward."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PREDICTIONS = Path(
    "/path/to/tactile_force_envelope/outputs/reports/"
    "pastry_pi05_expert_20260824/test_force_time/test_window_predictions.csv"
)
DEFAULT_WINDOWS = Path(
    "/path/to/tactile_force_envelope/outputs/training/"
    "franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/windows.jsonl"
)
DEFAULT_CALIBRATION = Path(
    "/path/to/tactile_force_envelope/outputs/training/"
    "franka_pastry_tactile_A298/pi05_jax_zrl_envelope_v1/"
    "sustained_contact_seed_44/stable_evaluation/bilateral_sigma_calibration.json"
)
DEFAULT_OUTPUT = Path(
    "/path/to/tactile_force_envelope/outputs/paper/"
    "method_force_envelope_reward_20260901"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--task-name", default="Pastry manipulation")
    parser.add_argument("--force-scale", type=float, default=2.0)
    parser.add_argument("--soft-weight", type=float, default=0.6)
    parser.add_argument("--peak-weight", type=float, default=0.4)
    parser.add_argument("--soft-episode-cap", type=float, default=0.6)
    parser.add_argument("--peak-episode-cap", type=float, default=0.4)
    parser.add_argument("--force-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_bilateral_predictions(path: Path, episode: int) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream"] != "bilateral" or int(row["episode"]) != episode:
                continue
            records.append(
                {
                    "window_index": int(row["window_index"]),
                    "timestamp": float(row["anchor_timestamp"]),
                    "target": float(row["target_N"]),
                    "mu": float(row["mu_N"]),
                    "sigma": float(row["sigma_N"]),
                }
            )
    records.sort(key=lambda row: float(row["timestamp"]))
    if len(records) < 4:
        raise RuntimeError(f"Episode {episode} has only {len(records)} bilateral test windows")
    return records


def read_raw_peaks(path: Path, window_ids: set[int]) -> dict[int, float]:
    peaks: dict[int, float] = {}
    with path.expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            window_index = int(row["window_index"])
            if window_index not in window_ids:
                continue
            per_sensor = row["tactile"]["per_sensor"]
            peaks[window_index] = max(
                float(sensor["raw_max"]) for sensor in per_sensor.values()
            )
            if len(peaks) == len(window_ids):
                break
    missing = sorted(window_ids - peaks.keys())
    if missing:
        raise RuntimeError(f"Raw force peaks missing for windows: {missing}")
    return peaks


def limited_increment(value: float, used: float, cap: float) -> float:
    return min(float(value), max(0.0, float(cap) - float(used)))


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = read_bilateral_predictions(args.predictions, args.episode)
    calibration = json.loads(args.calibration.expanduser().read_text(encoding="utf-8"))
    if calibration.get("method") != (
        "bilateral_stable_contact_average_uncertainty_scaled_conformal"
    ):
        raise RuntimeError("Paper figure requires uncertainty-scaled conformal calibration")
    alpha05 = float(calibration["lower_alpha"])
    alpha95 = float(calibration["upper_alpha"])
    peak_threshold = float(calibration["raw_peak_threshold"]["threshold_N"])
    raw_peaks = read_raw_peaks(
        args.windows, {int(row["window_index"]) for row in predictions}
    )

    t0 = float(predictions[0]["timestamp"])
    windows: list[dict[str, Any]] = []
    for row in predictions:
        target = float(row["target"])
        mu = float(row["mu"])
        uncertainty_scale = float(row["sigma"])
        lower = mu + alpha05 * uncertainty_scale
        upper = mu + alpha95 * uncertainty_scale
        low_deviation = max(0.0, lower - target)
        high_deviation = max(0.0, target - upper)
        band_deviation = low_deviation + high_deviation
        normalized = min(band_deviation / args.force_scale, 1.0)
        soft_cost = normalized**2
        raw_peak = raw_peaks[int(row["window_index"])]
        peak_cost = float(raw_peak > peak_threshold)
        windows.append(
            {
                **row,
                "time_s": float(row["timestamp"]) - t0,
                "lower": lower,
                "upper": upper,
                "low_deviation": low_deviation,
                "high_deviation": high_deviation,
                "soft_cost": soft_cost,
                "weighted_soft_penalty": args.soft_weight * soft_cost,
                "raw_peak": raw_peak,
                "peak_cost": peak_cost,
                "weighted_peak_penalty": args.peak_weight * peak_cost,
                "candidate_tactile_reward": -(
                    args.soft_weight * soft_cost + args.peak_weight * peak_cost
                ),
            }
        )

    transitions: list[dict[str, Any]] = []
    soft_used = 0.0
    peak_used = 0.0
    for transition_index, start in enumerate(range(0, len(windows) - 1, 2)):
        pair = windows[start : start + 2]
        soft_candidate = float(np.mean([row["weighted_soft_penalty"] for row in pair]))
        peak_candidate = float(np.mean([row["weighted_peak_penalty"] for row in pair]))
        soft_applied = limited_increment(soft_candidate, soft_used, args.soft_episode_cap)
        peak_applied = limited_increment(peak_candidate, peak_used, args.peak_episode_cap)
        soft_used += soft_applied
        peak_used += peak_applied
        transitions.append(
            {
                "transition": transition_index,
                "window_start": int(pair[0]["window_index"]),
                "window_end": int(pair[1]["window_index"]),
                "time_s": float(np.mean([row["time_s"] for row in pair])),
                "soft_candidate": soft_candidate,
                "peak_candidate": peak_candidate,
                "total_candidate": soft_candidate + peak_candidate,
                "soft_applied": soft_applied,
                "peak_applied": peak_applied,
                "reward_deduction": soft_applied + peak_applied,
                "tactile_reward": -(soft_applied + peak_applied),
                "cumulative_tactile_reward": -(soft_used + peak_used),
            }
        )

    metadata = {
        "task_name": args.task_name,
        "episode": args.episode,
        "split": "held-out test",
        "alpha05": alpha05,
        "alpha95": alpha95,
        "calibration_source": "standardized residual quantiles on calibration split only",
        "scale_definition": "u_mean = 0.5 * (sigma_left + sigma_right)",
        "scale_is_standard_deviation": False,
        "force_scale_N": args.force_scale,
        "soft_weight": args.soft_weight,
        "peak_weight": args.peak_weight,
        "soft_episode_cap": args.soft_episode_cap,
        "peak_episode_cap": args.peak_episode_cap,
        "raw_peak_threshold_N": peak_threshold,
        "h9_window_ms": 300,
        "e18_transition_ms": 600,
        "aggregation": "mean of two consecutive H9 penalties",
        "stable_contact_gate": "true for every displayed bilateral window",
        "peak_triggered": any(row["peak_cost"] > 0.0 for row in windows),
    }
    return windows, transitions, metadata


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_figure(
    windows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    configure_style()
    time_s = np.asarray([row["time_s"] for row in windows])
    target = np.asarray([row["target"] for row in windows])
    mu = np.asarray([row["mu"] for row in windows])
    lower = np.asarray([row["lower"] for row in windows])
    upper = np.asarray([row["upper"] for row in windows])
    low = np.asarray([row["low_deviation"] for row in windows])
    high = np.asarray([row["high_deviation"] for row in windows])
    soft = np.asarray([row["weighted_soft_penalty"] for row in windows])
    peak = np.asarray([row["weighted_peak_penalty"] for row in windows])

    fig = plt.figure(figsize=(7.25, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(2.65, 1.0, 1.25))
    ax_force = fig.add_subplot(grid[0])
    ax_violation = fig.add_subplot(grid[1], sharex=ax_force)
    ax_reward = fig.add_subplot(grid[2])

    band_color = "#b9ddd7"
    mu_color = "#147d75"
    observed_color = "#222a35"
    low_color = "#377eb8"
    high_color = "#d1495b"
    peak_color = "#e09131"

    ax_force.fill_between(
        time_s,
        lower,
        upper,
        color=band_color,
        alpha=0.62,
        linewidth=0,
        label=r"calibrated force band $[L_t,U_t]$",
    )
    ax_force.plot(time_s, lower, color="#5aa89f", lw=1.15)
    ax_force.plot(time_s, upper, color=mu_color, lw=1.3)
    ax_force.plot(time_s, mu, "--", color=mu_color, lw=1.55, label=r"predicted $\mu_{avg}$")
    ax_force.plot(
        time_s,
        target,
        "o-",
        color=observed_color,
        lw=1.65,
        ms=4.4,
        markerfacecolor="white",
        markeredgewidth=1.25,
        label=r"observed $F_{avg}$",
        zorder=4,
    )
    low_mask = low > 0.0
    high_mask = high > 0.0
    if np.any(low_mask):
        ax_force.scatter(
            time_s[low_mask], target[low_mask], marker="v", s=48,
            color=low_color, label=r"below $L_t$", zorder=5,
        )
    if np.any(high_mask):
        ax_force.scatter(
            time_s[high_mask], target[high_mask], marker="x", s=58,
            linewidths=1.8, color=high_color,
            label=r"above $U_t$", zorder=5,
        )
    gate_y = 0.32
    ax_force.plot(
        [time_s[0], time_s[-1]], [gate_y, gate_y], color="#2f9d58", lw=3.0,
        solid_capstyle="butt",
    )
    ax_force.text(
        time_s[0], gate_y, " causal gate ON", color="#247844", va="bottom", fontsize=7.5
    )
    force_top = max(float(np.max(target)), float(np.max(upper))) * 1.20
    ax_force.set_ylim(bottom=0.0, top=force_top)
    ax_force.set_ylabel("Bilateral average force (N)")
    ax_force.set_title("(a) Force-envelope prediction on a held-out test episode", loc="left", weight="bold")
    ax_force.grid(axis="y", alpha=0.2)
    ax_force.legend(loc="upper right", ncol=2, frameon=False)
    ax_force.tick_params(labelbottom=False)

    bar_width = max(0.12, float(np.median(np.diff(time_s))) * 0.55)
    ax_violation.bar(
        time_s, low, width=bar_width, color=low_color, alpha=0.85,
        label=r"low-force violation $[L_t-F_{avg}]_+$",
    )
    ax_violation.bar(
        time_s, high, width=bar_width, color=high_color, alpha=0.9,
        label=r"high-force violation $[F_{avg}-U_t]_+$",
    )
    ax_violation.set_ylabel("Violation (N)")
    ax_violation.set_title("(b) Distance outside the calibrated band", loc="left", weight="bold")
    ax_violation.grid(axis="y", alpha=0.2)
    ax_violation.legend(loc="upper right", ncol=2, frameon=False)
    ax_violation.tick_params(labelbottom=False)

    transition_time = np.asarray([row["time_s"] for row in transitions])
    candidate = np.asarray([row["total_candidate"] for row in transitions])
    applied = np.asarray([row["reward_deduction"] for row in transitions])
    transition_width = max(0.25, float(np.median(np.diff(transition_time))) * 0.55)
    ax_reward.bar(
        time_s,
        soft,
        width=bar_width,
        facecolor="none",
        edgecolor=high_color,
        linewidth=1.0,
        alpha=0.65,
        label=r"per-H9 candidate $0.6c^{soft}$",
    )
    if np.any(peak > 0.0):
        ax_reward.bar(
            time_s, peak, width=bar_width, color=peak_color, alpha=0.65,
            label=r"per-H9 peak term $0.4c^{peak}$",
        )
    ax_reward.bar(
        transition_time,
        candidate,
        width=transition_width,
        color="#b8bdc5",
        alpha=0.7,
        label="E18 candidate: mean of two H9 penalties",
    )
    ax_reward.bar(
        transition_time,
        applied,
        width=transition_width * 0.58,
        color="#743b8f",
        alpha=0.95,
        label=r"applied reward deduction $p_k$",
    )
    ax_reward.set_xlabel("Time from first stable-contact window (s)")
    ax_reward.set_ylabel("Reward deduction")
    ax_reward.set_ylim(0.0, max(0.65, float(max(candidate.max(initial=0), applied.max(initial=0))) * 1.22))
    ax_reward.set_title(
        "(c) Quadratic H9 cost, pairwise mean, and applied E18 reward",
        loc="left",
        weight="bold",
    )
    ax_reward.grid(axis="y", alpha=0.2)
    ax_reward.legend(loc="upper right", ncol=2, frameon=False)
    ax_reward.text(
        0.985,
        0.08,
        r"$r_k=r_k^{task}-p_k$",
        transform=ax_reward.transAxes,
        va="bottom",
        ha="right",
        fontsize=9.0,
        color="#4a4f56",
    )
    fig.suptitle(
        f"{metadata['task_name']}: force-envelope violations to dense Online RL reward",
        fontsize=13,
        weight="bold",
    )
    ax_force.text(
        0.995,
        0.062,
        (
            f"Calibration only: alpha05={metadata['alpha05']:.3f}, "
            f"alpha95={metadata['alpha95']:.3f} | "
            f"raw-peak threshold={metadata['raw_peak_threshold_N']:.2f} N "
            f"({'triggered' if metadata['peak_triggered'] else 'not triggered'})"
        ),
        transform=ax_force.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        color="#4a4f56",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
    )

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "force_envelope_violation_to_reward"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_method_two_panel(
    windows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    """Plot the calibrated band and final tactile reward as a compact paper pair."""
    configure_style()
    time_s = np.asarray([row["time_s"] for row in windows])
    target = np.asarray([row["target"] for row in windows])
    mu = np.asarray([row["mu"] for row in windows])
    lower = np.asarray([row["lower"] for row in windows])
    upper = np.asarray([row["upper"] for row in windows])
    low = np.asarray([row["low_deviation"] for row in windows])
    high = np.asarray([row["high_deviation"] for row in windows])
    band_color = "#c7e2de"
    boundary_color = "#4f9d96"
    mu_color = "#087f78"
    observed_color = "#202833"
    low_color = "#3478b7"
    high_color = "#cf3f4f"

    transition_time = np.asarray([row["time_s"] for row in transitions])
    tactile_reward = np.asarray([row["tactile_reward"] for row in transitions])

    fig, (ax_force, ax_reward) = plt.subplots(
        2,
        1,
        figsize=(8.9, 5.95),
        sharex=True,
        gridspec_kw={"height_ratios": (1.85, 1.0), "hspace": 0.14},
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.885)

    ax_force.fill_between(
        time_s,
        lower,
        upper,
        color=band_color,
        alpha=0.78,
        linewidth=0,
        label=r"band $[L_t,U_t]$",
    )
    ax_force.plot(time_s, lower, color=boundary_color, lw=1.35)
    ax_force.plot(time_s, upper, color=boundary_color, lw=1.35)
    ax_force.plot(
        time_s,
        mu,
        "--",
        color=mu_color,
        lw=2.15,
        label=r"mean $\mu_{avg}$",
    )
    ax_force.plot(
        time_s,
        target,
        "o-",
        color=observed_color,
        lw=2.0,
        ms=5.4,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label=r"force $F_{avg}$",
        zorder=4,
    )

    low_mask = low > 0.0
    high_mask = high > 0.0
    if np.any(low_mask):
        ax_force.scatter(
            time_s[low_mask],
            target[low_mask],
            marker="v",
            s=78,
            color=low_color,
            label=r"$F_{avg}<L_t$",
            zorder=5,
        )
    if np.any(high_mask):
        ax_force.scatter(
            time_s[high_mask],
            target[high_mask],
            marker="x",
            s=86,
            linewidths=2.2,
            color=high_color,
            label=r"$F_{avg}>U_t$",
            zorder=5,
        )

    force_min = min(float(np.min(target)), float(np.min(lower)))
    force_max = max(float(np.max(target)), float(np.max(upper)))
    force_pad = max(1.0, 0.08 * (force_max - force_min))
    ax_force.set_ylim(max(0.0, force_min - force_pad), force_max + force_pad)
    ax_force.set_ylabel("Bilateral average force (N)", fontsize=12)
    ax_force.set_title(
        "(a) Calibrated tactile envelope",
        loc="left",
        y=1.125,
        fontsize=12.5,
        weight="bold",
    )
    ax_force.grid(axis="y", color="#d8dde3", linewidth=0.8, alpha=0.8)
    ax_force.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        frameon=False,
        fontsize=8.2,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    ax_force.tick_params(labelsize=10)

    reward_color = "#743b8f"
    ax_reward.step(
        transition_time,
        tactile_reward,
        where="mid",
        color=reward_color,
        linewidth=2.2,
        label=r"tactile reward $r_k^{tac}$",
        zorder=3,
    )
    ax_reward.plot(
        transition_time,
        tactile_reward,
        "o",
        color=reward_color,
        markerfacecolor="white",
        markeredgewidth=1.5,
        markersize=5.1,
        zorder=4,
    )
    ax_reward.fill_between(
        transition_time,
        tactile_reward,
        0.0,
        step="mid",
        color=reward_color,
        alpha=0.12,
        linewidth=0,
    )
    ax_reward.axhline(0.0, color=observed_color, lw=1.0)
    ax_reward.set_xlabel("Time from first stable-contact window (s)", fontsize=11.5)
    ax_reward.set_ylabel("Tactile reward", fontsize=11.5)
    ax_reward.set_title(
        "(b) Tactile reward over time",
        loc="left",
        fontsize=12.5,
        weight="bold",
        pad=5,
    )
    ax_reward.grid(axis="y", color="#d8dde3", linewidth=0.8, alpha=0.8)
    ax_reward.legend(loc="center right", frameon=False, fontsize=8.7)
    ax_reward.tick_params(labelsize=9.5)

    reward_min = min(-0.08, float(tactile_reward.min(initial=0.0)))
    ax_reward.set_ylim(reward_min * 1.30, 0.045)
    value_offset = max(0.014, abs(reward_min) * 0.065)
    for time_value, reward_value in zip(
        transition_time, tactile_reward, strict=True
    ):
        if abs(float(reward_value)) < 0.005:
            continue
        ax_reward.text(
            time_value,
            reward_value - value_offset,
            f"{reward_value:.3f}",
            ha="center",
            va="top",
            fontsize=7.8,
            color=reward_color,
        )
    time_step = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.3
    ax_reward.set_xlim(time_s[0] - 0.35 * time_step, time_s[-1] + 0.35 * time_step)

    output.mkdir(parents=True, exist_ok=True)
    for name in ("force_envelope_and_tactile_reward", "force_envelope_and_violations"):
        stem = output / name
        fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_force_only(
    windows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    configure_style()
    time_s = np.asarray([row["time_s"] for row in windows])
    target = np.asarray([row["target"] for row in windows])
    mu = np.asarray([row["mu"] for row in windows])
    lower = np.asarray([row["lower"] for row in windows])
    upper = np.asarray([row["upper"] for row in windows])
    low = np.asarray([row["low_deviation"] for row in windows])
    high = np.asarray([row["high_deviation"] for row in windows])

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.15, top=0.73)

    band_color = "#b9ddd7"
    mu_color = "#147d75"
    observed_color = "#222a35"
    low_color = "#377eb8"
    high_color = "#d1495b"

    ax.fill_between(
        time_s,
        lower,
        upper,
        color=band_color,
        alpha=0.66,
        linewidth=0,
        label=r"calibrated force band $[L_t,U_t]$",
    )
    ax.plot(time_s, lower, color="#5aa89f", lw=1.8)
    ax.plot(time_s, upper, color=mu_color, lw=2.0)
    ax.plot(time_s, mu, "--", color=mu_color, lw=2.25, label=r"predicted $\mu_{avg}$")
    ax.plot(
        time_s,
        target,
        "o-",
        color=observed_color,
        lw=2.35,
        ms=6.2,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=r"observed $F_{avg}$",
        zorder=4,
    )
    low_mask = low > 0.0
    high_mask = high > 0.0
    if np.any(low_mask):
        ax.scatter(
            time_s[low_mask],
            target[low_mask],
            marker="v",
            s=82,
            color=low_color,
            label=r"below $L_t$",
            zorder=5,
        )
    if np.any(high_mask):
        ax.scatter(
            time_s[high_mask],
            target[high_mask],
            marker="x",
            s=90,
            linewidths=2.3,
            color=high_color,
            label=r"above $U_t$",
            zorder=5,
        )

    gate_y = 0.32
    ax.plot(
        [time_s[0], time_s[-1]],
        [gate_y, gate_y],
        color="#2f9d58",
        lw=4.0,
        solid_capstyle="butt",
    )
    ax.text(time_s[0], gate_y, " causal gate ON", color="#247844", va="bottom", fontsize=10)
    force_top = max(float(np.max(target)), float(np.max(upper))) * 1.08
    ax.set_ylim(bottom=0.0, top=force_top)
    ax.set_xlabel("Time from first stable-contact window (s)", fontsize=12)
    ax.set_ylabel("Bilateral average force (N)", fontsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(labelsize=10)

    fig.suptitle(
        f"(a) {metadata['task_name']}: calibrated tactile force envelope",
        fontsize=16,
        weight="bold",
        y=0.965,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=3.0,
        columnspacing=1.7,
    )

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "01_force_envelope"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_force_violation_over_time(
    windows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    """Plot only the observed distance below or above the calibrated force band."""
    configure_style()
    time_s = np.asarray([row["time_s"] for row in windows])
    low = np.asarray([row["low_deviation"] for row in windows])
    high = np.asarray([row["high_deviation"] for row in windows])
    bar_width = max(0.12, float(np.median(np.diff(time_s))) * 0.55)

    fig, ax = plt.subplots(figsize=(9.4, 4.25), constrained_layout=True)
    ax.bar(
        time_s,
        low,
        width=bar_width,
        color="#377eb8",
        alpha=0.88,
        label=r"low-force violation $[L_t-F_{avg}]_+$",
    )
    ax.bar(
        time_s,
        high,
        width=bar_width,
        color="#d1495b",
        alpha=0.92,
        label=r"high-force violation $[F_{avg}-U_t]_+$",
    )
    ax.axhline(0.0, color="#2b2f36", lw=1.0)
    ax.set_xlabel("Time from first stable-contact window (s)")
    ax.set_ylabel("Force-envelope violation (N)")
    ax.set_title(
        f"(b) {metadata['task_name']}: distance outside the calibrated force band",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    violation_max = max(float(low.max(initial=0.0)), float(high.max(initial=0.0)))
    ax.set_ylim(0.0, max(0.5, violation_max * 1.2))

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "02_force_violation"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_tactile_reward_over_time(
    windows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    """Plot the tactile reward actually attached to each Online RL transition."""
    configure_style()
    transition_time = np.asarray([row["time_s"] for row in transitions])
    tactile_reward = np.asarray([row["tactile_reward"] for row in transitions])

    fig, ax_reward = plt.subplots(figsize=(9.4, 4.25), constrained_layout=True)
    applied_color = "#743b8f"
    ax_reward.step(
        transition_time,
        tactile_reward,
        where="mid",
        color=applied_color,
        linewidth=2.4,
        label=r"final tactile reward $r_k^{tac}=-p_k$",
        zorder=3,
    )
    ax_reward.plot(
        transition_time,
        tactile_reward,
        "o",
        color=applied_color,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=6.2,
        zorder=4,
    )
    ax_reward.fill_between(
        transition_time,
        tactile_reward,
        0.0,
        step="mid",
        color=applied_color,
        alpha=0.12,
        linewidth=0,
    )
    ax_reward.axhline(0.0, color="#2b2f36", lw=1.0)
    ax_reward.set_xlabel("Time from first stable-contact window (s)")
    ax_reward.set_ylabel("Tactile reward")
    ax_reward.set_title(
        f"(d) {metadata['task_name']}: tactile reward over time",
        loc="left",
        weight="bold",
    )
    ax_reward.grid(axis="y", alpha=0.22)
    ax_reward.legend(loc="lower left", frameon=False)

    label_floor = min(-0.035, float(tactile_reward.min(initial=0.0)) * 0.12)
    for time_value, reward_value in zip(
        transition_time, tactile_reward, strict=True
    ):
        if abs(float(reward_value)) < 0.005:
            continue
        ax_reward.text(
            time_value,
            reward_value + label_floor,
            f"{reward_value:.3f}",
            ha="center",
            va="top",
            fontsize=7.7,
            color=applied_color,
        )
    reward_min = min(-0.08, float(tactile_reward.min(initial=0.0)))
    ax_reward.set_ylim(reward_min * 1.32, 0.055)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "04_tactile_reward_over_time"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_reward_deduction_over_time(
    transitions: list[dict[str, Any]],
    metadata: dict[str, Any],
    output: Path,
) -> None:
    """Plot the positive tactile deduction applied to each RL transition."""
    configure_style()
    transition_time = np.asarray([row["time_s"] for row in transitions])
    deduction = np.asarray([row["reward_deduction"] for row in transitions])
    transition_width = (
        max(0.25, float(np.median(np.diff(transition_time))) * 0.56)
        if len(transition_time) > 1
        else 0.35
    )

    fig, ax = plt.subplots(figsize=(9.4, 4.25), constrained_layout=True)
    color = "#743b8f"
    ax.bar(
        transition_time,
        deduction,
        width=transition_width,
        color=color,
        alpha=0.94,
        label=r"final reward deduction $p_k$",
    )
    ax.axhline(0.0, color="#2b2f36", lw=1.0)
    ax.set_xlabel("Time from first stable-contact window (s)")
    ax.set_ylabel("Reward deduction")
    ax.set_title(
        f"(c) {metadata['task_name']}: final tactile reward deduction",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0.0, max(0.08, float(deduction.max(initial=0.0)) * 1.32))
    for time_value, deduction_value in zip(
        transition_time, deduction, strict=True
    ):
        if float(deduction_value) < 0.005:
            continue
        ax.text(
            time_value,
            deduction_value + max(0.006, float(deduction.max()) * 0.035),
            f"{deduction_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color=color,
        )
    ax.text(
        0.985,
        0.94,
        f"episode total deduction = {float(deduction.sum()):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=color,
    )

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "03_reward_deduction"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_caption(output: Path, metadata: dict[str, Any]) -> None:
    caption = f"""# Suggested caption

**Force-envelope violations are converted into a dense tactile reward.**
Panel (a) shows the observed bilateral force, predicted mean, and calibrated
force band. At each 300 ms H9 subwindow, the Envelope Expert predicts the bilateral-average
force center $\\mu_{{avg}}$ and input-dependent uncertainty scale $u_{{mean}}$.
Calibration-only standardized-residual quantiles $\\alpha_{{05}}={metadata['alpha05']:.3f}$
and $\\alpha_{{95}}={metadata['alpha95']:.3f}$ define
$L_t=\\mu_{{avg}}+\\alpha_{{05}}u_{{mean}}$ and
$U_t=\\mu_{{avg}}+\\alpha_{{95}}u_{{mean}}$. Here $u_{{mean}}$ is an empirical
uncertainty scale, not a bilateral standard deviation. Deviations below or above the band are
normalized by {metadata['force_scale_N']:.1f} N, squared, and enabled only by the causal bilateral stable-contact
gate. Panel (b) shows the final tactile reward over time. Two consecutive H9
penalties are averaged, rather than summed, to form the
reward deduction for one 600 ms E18 Online RL transition. The independent raw-force
peak term is active without the contact gate; its {metadata['raw_peak_threshold_N']:.2f} N
threshold is {'triggered' if metadata['peak_triggered'] else 'not triggered'} in this held-out test example. Quantiles are estimated
only on the calibration split; the displayed trajectory is from the test split.

## 中文图注

**由触觉力包络超界得到稠密奖励。** 子图 (a) 展示观测双侧平均力、预测均值和
校准力包络。每个 300 ms 的 H9 子窗口中，Envelope
Expert 预测双侧平均力中心 $\\mu_{{avg}}$ 和输入相关 uncertainty scale $u_{{mean}}$；
该组合尺度不解释为双侧标准差。仅使用 calibration split 的标准化残差分位数构造动态区间
$[L_t,U_t]$。区间外偏差经 {metadata['force_scale_N']:.1f} N 归一化和平方映射，并且
只在因果双侧稳定接触 gate 开启时生效。子图 (b) 展示最终触觉奖励随时间的变化。
相邻两个 H9 penalty 取平均，形成一个
600 ms E18 Online RL transition 的 reward deduction，避免双频执行使奖励尺度翻倍。
原始力峰值项不受 gate 控制，本测试轨迹{'已' if metadata['peak_triggered'] else '未'}触发该项。
"""
    (output / "CAPTION.md").write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    windows, transitions, metadata = build_rows(args)
    output = args.output.expanduser().resolve()
    if args.force_only:
        plot_force_only(windows, metadata, output)
    else:
        plot_figure(windows, transitions, metadata, output)
    plot_method_two_panel(windows, transitions, metadata, output)
    plot_force_violation_over_time(windows, metadata, output)
    plot_reward_deduction_over_time(transitions, metadata, output)
    plot_tactile_reward_over_time(windows, transitions, metadata, output)
    write_csv(output / "h9_window_values.csv", windows)
    write_csv(output / "e18_transition_rewards.csv", transitions)
    (output / "summary.json").write_text(
        json.dumps(
            {
                **metadata,
                "h9_windows": len(windows),
                "e18_transitions": len(transitions),
                "low_violation_windows": sum(row["low_deviation"] > 0 for row in windows),
                "high_violation_windows": sum(row["high_deviation"] > 0 for row in windows),
                "peak_trigger_windows": sum(row["peak_cost"] > 0 for row in windows),
                "candidate_penalty_sum": sum(row["total_candidate"] for row in transitions),
                "applied_penalty_sum": sum(row["reward_deduction"] for row in transitions),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_caption(output, metadata)
    print(output)


if __name__ == "__main__":
    main()
