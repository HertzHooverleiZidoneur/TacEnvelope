from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.stats import spearmanr

from .bilateral_average import _episode_key, _evaluate_seed, _mean_std
from .common import write_json


NOTICE = (
    "q95 is estimated only on the calibration split. "
    "Test coverage is evaluated without recalibrating on test targets."
)
NOTICE_ZH = "q90/q95/alpha 只由校准集估计；测试集只用于计算最终指标，不反向调整上界。"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def _save_figure(figure: plt.Figure, stem: Path) -> dict[str, str]:
    paths = {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf"))}
    figure.savefig(paths["png"], dpi=180, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    return paths


def _episode_indices(rows: list[dict[str, Any]]) -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(_episode_key(row), []).append(index)
    return {
        key: np.asarray(sorted(indices, key=lambda i: float(rows[i]["start_timestamp"])))
        for key, indices in grouped.items()
    }


def _longest_false_run(
    rows: list[dict[str, Any]], gate: np.ndarray, penalty: np.ndarray,
) -> int:
    longest = 0
    expected_stride = None
    timestamps = np.asarray([float(row["start_timestamp"]) for row in rows])
    positive_deltas = np.diff(np.sort(np.unique(timestamps)))
    if positive_deltas.size:
        expected_stride = float(np.median(positive_deltas))
    for indices in _episode_indices(rows).values():
        run = 0
        previous_time = None
        for index in indices:
            timestamp = timestamps[index]
            discontinuous = (
                previous_time is not None
                and expected_stride is not None
                and timestamp - previous_time > 1.5 * expected_stride
            )
            if discontinuous or not gate[index] or not penalty[index]:
                run = 0
            else:
                run += 1
                longest = max(longest, run)
            previous_time = timestamp
    return longest


def _exceedance_stats(values: np.ndarray) -> dict[str, float | int]:
    positive = values[values > 0.0]
    result = {
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "positive_count": int(positive.size),
    }
    for name, quantile in (("mean", None), ("p90", 0.90), ("p95", 0.95), ("max", None)):
        key = f"positive_{name}"
        if not positive.size:
            result[key] = 0.0
        elif name == "mean":
            result[key] = float(np.mean(positive))
        elif name == "max":
            result[key] = float(np.max(positive))
        else:
            result[key] = float(np.quantile(positive, quantile))
    return result


def _false_episode_count(
    rows: list[dict[str, Any]], gate: np.ndarray, penalty: np.ndarray,
) -> int:
    return int(sum(
        np.any(gate[indices] & penalty[indices])
        for indices in _episode_indices(rows).values()
    ))


def _mu_error_stats(target: np.ndarray, mu: np.ndarray) -> dict[str, float]:
    residual = target - mu
    absolute = np.abs(residual)
    return {
        "bias_y_minus_mu": float(np.mean(residual)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
    }


def _sigma_diagnostics(
    target: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> dict[str, float | None]:
    error = np.abs(target - mu)
    correlation = spearmanr(sigma, error)
    rmse = float(np.sqrt(np.mean((target - mu) ** 2)))
    return {
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
        "empirical_rmse": rmse,
        "mean_sigma_over_rmse": float(np.mean(sigma) / max(rmse, 1e-8)),
        "spearman_sigma_vs_absolute_error": (
            None if not np.isfinite(correlation.statistic) else float(correlation.statistic)
        ),
        "standardized_residual_p90": float(np.quantile((target - mu) / (sigma + 1e-8), 0.90)),
        "standardized_residual_p95": float(np.quantile((target - mu) / (sigma + 1e-8), 0.95)),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _ensemble_evaluation(
    run_data: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    reference = run_data[0]["curves"]
    calibration_mask = np.asarray(reference["calibration_mask"], dtype=bool)
    test_gate = np.asarray(reference["test_mask"], dtype=bool)
    calibration_targets = []
    calibration_mus = []
    test_targets = []
    test_mus = []
    test_sigmas = []
    for data in run_data:
        curves = data["curves"]
        cal_target = 0.5 * (
            np.asarray(curves["calibration_left"]["target"])
            + np.asarray(curves["calibration_right"]["target"])
        )
        calibration_targets.append(cal_target)
        calibration_mus.append(0.5 * (
            np.asarray(curves["calibration_left"]["mu"])
            + np.asarray(curves["calibration_right"]["mu"])
        ))
        test_targets.append(np.asarray(curves["y_avg_all"]))
        test_mus.append(np.asarray(curves["mu_avg_all"]))
        test_sigmas.append(0.5 * np.sqrt(
            np.asarray(curves["test_left"]["sigma"])**2
            + np.asarray(curves["test_right"]["sigma"])**2
        ))
    for values in calibration_targets[1:]:
        if not np.allclose(values, calibration_targets[0]):
            raise ValueError("Calibration targets differ across seeds")
    for values in test_targets[1:]:
        if not np.allclose(values, test_targets[0]):
            raise ValueError("Test targets differ across seeds")
    calibration_target = calibration_targets[0]
    calibration_mu = np.mean(np.stack(calibration_mus), axis=0)
    calibration_residual = calibration_target[calibration_mask] - calibration_mu[calibration_mask]
    q90 = float(np.quantile(calibration_residual, 0.90))
    q95 = float(np.quantile(calibration_residual, 0.95))

    target_all = test_targets[0]
    mu_stack = np.stack(test_mus)
    sigma_stack = np.stack(test_sigmas)
    mu_all = np.mean(mu_stack, axis=0)
    within_sigma = np.mean(sigma_stack, axis=0)
    seed_mu_std = np.std(mu_stack, axis=0)
    total_sigma = np.sqrt(np.maximum(
        np.mean(sigma_stack**2 + mu_stack**2, axis=0) - mu_all**2, 0.0
    ))
    target = target_all[test_gate]
    mu = mu_all[test_gate]
    residual = target - mu
    absolute_error = np.abs(residual)
    observed_mean = float(np.mean(target))
    scale = max(abs(observed_mean), 1e-8)

    envelopes: dict[str, Any] = {}
    penalties: dict[str, np.ndarray] = {}
    for level, q in (("90", q90), ("95", q95)):
        upper_all = mu_all + q
        penalty_all = test_gate & (target_all > upper_all)
        penalties[level] = penalty_all
        exceedance = np.maximum(0.0, target - upper_all[test_gate])
        positive = exceedance[exceedance > 0.0]
        envelopes[level] = {
            "q": q,
            "q_percent_of_observed_mean": float(100.0 * q / scale),
            "upper_mean": float(np.mean(upper_all[test_gate])),
            "upper_min": float(np.min(upper_all[test_gate])),
            "upper_max": float(np.max(upper_all[test_gate])),
            "coverage": float(np.mean(target <= upper_all[test_gate])),
            "false_penalty_windows": int(np.sum(penalty_all)),
            "false_penalty_episodes": _false_episode_count(
                reference["rows"], test_gate, penalty_all
            ),
            "longest_false_run": _longest_false_run(
                reference["rows"], test_gate, penalty_all
            ),
            "exceedance_all": _distribution(exceedance),
            "exceedance_positive": (
                _distribution(positive) if positive.size else _distribution(np.zeros(1))
            ),
        }

    episode_rows: list[dict[str, Any]] = []
    for (batch_id, episode), indices in sorted(_episode_indices(reference["rows"]).items()):
        selected = indices[test_gate[indices]]
        if not selected.size:
            continue
        penalty = penalties["95"][selected]
        episode_rows.append({
            "seed": "ensemble",
            "batch_id": batch_id,
            "episode": episode,
            "bilateral_stable_windows": int(selected.size),
            "average_coverage": float(np.mean(~penalty)),
            "average_false_penalty_windows": int(np.sum(penalty)),
            "y_avg_mae": float(np.mean(np.abs(target_all[selected] - mu_all[selected]))),
        })

    summary = {
        "aggregation": "per-window mean prediction across seeds 42/43/44",
        "test_episodes": len(episode_rows),
        "test_windows_before_gate": len(reference["rows"]),
        "test_windows_after_bilateral_gate": int(test_gate.sum()),
        "observed_force": _distribution(target),
        "predicted_mu": _distribution(mu),
        "signed_error_y_minus_mu": _distribution(residual),
        "absolute_error": {
            **_distribution(absolute_error),
            "mean_percent_of_observed_mean": float(100.0 * np.mean(absolute_error) / scale),
            "p90_percent_of_observed_mean": float(100.0 * np.quantile(absolute_error, 0.90) / scale),
            "p95_percent_of_observed_mean": float(100.0 * np.quantile(absolute_error, 0.95) / scale),
            "max_percent_of_observed_mean": float(100.0 * np.max(absolute_error) / scale),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "rmse_percent_of_observed_mean": float(
                100.0 * np.sqrt(np.mean(residual**2)) / scale
            ),
        },
        "sigma": {
            "mean_within_model_sigma": float(np.mean(within_sigma[test_gate])),
            "mean_between_seed_mu_std": float(np.mean(seed_mu_std[test_gate])),
            "mean_total_ensemble_sigma": float(np.mean(total_sigma[test_gate])),
            "total_sigma_percent_of_observed_mean": float(
                100.0 * np.mean(total_sigma[test_gate]) / scale
            ),
        },
        "envelopes": envelopes,
    }
    curves = {
        "rows": reference["rows"],
        "calibration_residual": calibration_residual,
        "test_mask": test_gate,
        "penalty_all": penalties["95"],
        "y_avg_all": target_all,
        "mu_avg_all": mu_all,
        "upper_avg_all": mu_all + q95,
    }
    return summary, curves, episode_rows


def _plot_representative_episode(
    curves: dict[str, Any], per_episode: list[dict[str, Any]], output: Path, seed: int | str,
) -> tuple[dict[str, str], dict[str, Any]]:
    coverages = np.asarray([float(row["average_coverage"]) for row in per_episode])
    windows = np.asarray([int(row["bilateral_stable_windows"]) for row in per_episode])
    median_coverage = float(np.median(coverages))
    median_windows = float(np.median(windows))
    candidates = [
        row for row in per_episode
        if int(row["bilateral_stable_windows"]) >= median_windows
    ]
    selected = min(
        candidates,
        key=lambda row: (
            abs(float(row["average_coverage"]) - median_coverage),
            -int(row["bilateral_stable_windows"]),
        ),
    )
    key = (str(selected["batch_id"]), int(selected["episode"]))
    indices = _episode_indices(curves["rows"])[key]
    timestamps = np.asarray([
        float(curves["rows"][int(index)]["start_timestamp"]) for index in indices
    ])
    elapsed = timestamps - timestamps[0]
    gate = np.asarray(curves["test_mask"])[indices]
    penalty = np.asarray(curves["penalty_all"])[indices]
    observed = np.asarray(curves["y_avg_all"])[indices]
    predicted = np.asarray(curves["mu_avg_all"])[indices]
    upper = np.asarray(curves["upper_avg_all"])[indices]

    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(elapsed, observed, label="真实双侧平均力 y_avg", color="#2166ac", linewidth=2.0)
    axis.plot(elapsed, predicted, label="预测均值 mu_avg", color="#f28e2b", linewidth=1.8)
    axis.plot(elapsed, upper, label="95% 上界 mu_avg + q95", color="#2ca02c", linewidth=1.8)
    axis.fill_between(
        elapsed, upper, observed,
        where=penalty & (observed > upper), color="#d62728", alpha=0.35,
        label="真实力超过上界",
    )
    axis.scatter(
        elapsed[penalty], observed[penalty], color="#b2182b", marker="x", s=45,
        zorder=5,
    )
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("双侧平均力")
    axis.grid(alpha=0.25)
    gate_axis = axis.twinx()
    gate_axis.step(
        elapsed, gate.astype(float), where="post", color="#6a3d9a", alpha=0.6,
        linewidth=1.5, label="稳定接触 gate",
    )
    gate_axis.set_ylim(-0.05, 1.25)
    gate_axis.set_yticks([0, 1])
    gate_axis.set_ylabel("稳定接触 gate")
    handles, labels = axis.get_legend_handles_labels()
    gate_handles, gate_labels = gate_axis.get_legend_handles_labels()
    axis.legend(handles + gate_handles, labels + gate_labels, loc="best", ncol=2)
    axis.set_title(
        f"代表性测试 Episode：batch={key[0]}，episode={key[1]}，seed={seed}"
    )
    figure.text(0.01, 0.01, NOTICE_ZH, fontsize=8)
    paths = _save_figure(figure, output / "representative_episode_timeseries")
    selection = {
        "seed": seed,
        "batch_id": key[0],
        "episode": key[1],
        "explicit_success_field_available": False,
        "success_interpretation": "retained task demonstration; dataset has no explicit success field",
        "selection_rule": (
            "at least the median bilateral-stable window count, then coverage closest "
            "to the seed's episode median"
        ),
        "bilateral_stable_windows": int(selected["bilateral_stable_windows"]),
        "coverage": float(selected["average_coverage"]),
        "false_penalty_windows": int(selected["average_false_penalty_windows"]),
    }
    return paths, selection


def _plot_calibration_cdf(
    run_data: list[dict[str, Any]], output: Path,
) -> dict[str, str]:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    colors = ("#2166ac", "#f28e2b", "#2ca02c")
    notes = []
    for data, color in zip(run_data, colors, strict=False):
        residual = np.sort(np.asarray(data["curves"]["calibration_residual"]))
        cdf = np.arange(1, residual.size + 1) / residual.size
        q95 = float(data["report"]["q95"])
        seed = int(data["report"]["seed"])
        axis.plot(residual, cdf, color=color, linewidth=2, label=f"seed {seed}")
        axis.axvline(q95, color=color, linestyle="--", alpha=0.75)
        q90 = float(np.quantile(residual, 0.90))
        notes.append(f"seed {seed}: n={residual.size}, q90={q90:.3f}, q95={q95:.3f}")
        axis.axvline(q90, color=color, linestyle=":", alpha=0.65)
    axis.axhline(0.90, color="0.4", linestyle=":", linewidth=1)
    axis.axhline(0.95, color="black", linestyle=":", linewidth=1)
    axis.text(
        0.02, 0.96, "\n".join(notes), transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    axis.set_xlabel("校准残差：y_avg - mu_avg")
    axis.set_ylabel("经验累计分布 CDF")
    axis.set_title("双侧平均力校准残差 CDF（点线=q90，虚线=q95）")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.text(0.01, 0.01, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "calibration_residual_cdf")


def _plot_comparison(
    reports: list[dict[str, Any]], output: Path,
) -> dict[str, str]:
    definitions = (
        ("测试集 Coverage", "coverage", "joint_coverage", 100.0, "%"),
        ("误罚窗口数", "false_penalty_windows", "false_penalty_windows", 1.0, "数量"),
        ("含误罚的 Episode 数", "episodes_with_false_penalty", "episodes_with_false_penalty", 1.0, "数量"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.80, wspace=0.30)
    for axis, (title, avg_key, independent_key, scale, ylabel) in zip(
        axes, definitions, strict=True
    ):
        independent = np.asarray([
            report["independent_additive_comparison"][independent_key]
            for report in reports
        ]) * scale
        average = np.asarray([report[avg_key] for report in reports]) * scale
        means = [independent.mean(), average.mean()]
        stds = [independent.std(), average.std()]
        bars = axis.bar(
            [0, 1], means, yerr=stds, capsize=5,
            color=["#9e9e9e", "#4c78a8"], width=0.65,
        )
        axis.set_xticks([0, 1], ["左右独立\nEnvelope", "双侧平均\nEnvelope"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    figure.suptitle("稳定接触 Envelope 对比（三个 seed 的均值 +/- 标准差）")
    figure.text(0.01, 0.025, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "independent_vs_bilateral")


def _plot_episode_coverage(
    episode_rows: list[dict[str, Any]], seeds: list[int], output: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    values_by_seed: list[np.ndarray] = []
    statistics: dict[str, Any] = {}
    rng = np.random.default_rng(20260720)
    for position, seed in enumerate(seeds, start=1):
        values = np.asarray([
            float(row["average_coverage"]) for row in episode_rows
            if int(row["seed"]) == seed
        ])
        values_by_seed.append(values)
        below = values < 0.90
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        axis.scatter(
            np.full(values.size, position) + jitter, values * 100.0,
            c=np.where(below, "#d62728", "#4c78a8"), alpha=0.75, s=28,
        )
        statistics[str(seed)] = {
            "episodes": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "episodes_below_90_percent": int(below.sum()),
        }
        axis.text(
            position, 50,
            f"均值 {values.mean()*100:.1f}%\n中位数 {np.median(values)*100:.1f}%\n低于90%: {below.sum()}",
            ha="center", va="bottom", fontsize=9,
        )
    axis.boxplot(
        [values * 100.0 for values in values_by_seed], positions=range(1, len(seeds) + 1),
        widths=0.45, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    axis.axhline(90.0, color="#d62728", linestyle="--", linewidth=1.2, label="90%")
    axis.set_xticks(range(1, len(seeds) + 1), [f"seed {seed}" for seed in seeds])
    axis.set_ylabel("Episode Coverage（%）")
    axis.set_ylim(45, 102)
    axis.set_title("测试集 31 个 Episode 的 Coverage 分布")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower left")
    figure.text(0.01, 0.01, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "test_episode_coverage_distribution"), statistics


def _plot_90_95_envelopes(
    seed_rows: list[dict[str, Any]], output: Path,
) -> dict[str, str]:
    seeds = [int(row["seed"]) for row in seed_rows]
    positions = np.arange(len(seeds), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.80, wspace=0.30)
    q90 = np.asarray([row["additive_envelopes"]["90"]["q"] for row in seed_rows])
    q95 = np.asarray([row["additive_envelopes"]["95"]["q"] for row in seed_rows])
    coverage90 = np.asarray([
        row["additive_envelopes"]["90"]["test_coverage"] for row in seed_rows
    ]) * 100.0
    coverage95 = np.asarray([
        row["additive_envelopes"]["95"]["test_coverage"] for row in seed_rows
    ]) * 100.0
    false90 = np.asarray([
        row["additive_envelopes"]["90"]["false_penalty_windows"] for row in seed_rows
    ])
    false95 = np.asarray([
        row["additive_envelopes"]["95"]["false_penalty_windows"] for row in seed_rows
    ])
    for axis, first, second, title, ylabel in (
        (axes[0], q90, q95, "校准得到的加性上界宽度", "q 值（力单位）"),
        (axes[1], coverage90, coverage95, "测试集实际 Coverage", "Coverage（%）"),
        (axes[2], false90, false95, "测试集误罚窗口数", "窗口数"),
    ):
        bars90 = axis.bar(positions - width / 2, first, width, label="90% Envelope", color="#6baed6")
        bars95 = axis.bar(positions + width / 2, second, width, label="95% Envelope", color="#2171b5")
        axis.set_xticks(positions, [f"seed {seed}" for seed in seeds])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for bars in (bars90, bars95):
            for bar in bars:
                value = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.2f}" if ylabel != "窗口数" else f"{value:.0f}",
                    ha="center", va="bottom", fontsize=9,
                )
    axes[1].axhspan(90, 95, color="#2ca02c", alpha=0.08)
    axes[1].axhline(90, color="#6baed6", linestyle=":")
    axes[1].axhline(95, color="#2171b5", linestyle="--")
    axes[0].legend(loc="upper left")
    figure.suptitle("90% 与 95% 双侧平均力 Additive Envelope")
    figure.text(0.01, 0.025, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "envelope_90_95_details")


def _plot_mu_sigma_diagnostics(
    seed_rows: list[dict[str, Any]], output: Path,
) -> dict[str, str]:
    seeds = [int(row["seed"]) for row in seed_rows]
    positions = np.arange(len(seeds), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    figure.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.80, wspace=0.30)

    bias = np.asarray([row["mu_average_test_error"]["bias_y_minus_mu"] for row in seed_rows])
    mae = np.asarray([row["mu_average_test_error"]["mae"] for row in seed_rows])
    rmse = np.asarray([row["mu_average_test_error"]["rmse"] for row in seed_rows])
    width = 0.25
    axes[0].bar(positions - width, bias, width, label="Bias (y-mu)", color="#f28e2b")
    axes[0].bar(positions, mae, width, label="MAE", color="#4c78a8")
    axes[0].bar(positions + width, rmse, width, label="RMSE", color="#2ca02c")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("mu_avg 在测试集上的误差")
    axes[0].set_ylabel("力误差")
    axes[0].legend(fontsize=9)

    sigma_mean = np.asarray([row["sigma_average_diagnostic"]["mean_sigma"] for row in seed_rows])
    sigma_rmse = np.asarray([row["sigma_average_diagnostic"]["empirical_rmse"] for row in seed_rows])
    axes[1].bar(positions - 0.18, sigma_mean, 0.36, label="平均预测 sigma_avg", color="#9467bd")
    axes[1].bar(positions + 0.18, sigma_rmse, 0.36, label="实际 RMSE", color="#2ca02c")
    axes[1].set_title("sigma 尺度与实际误差")
    axes[1].set_ylabel("力尺度")
    axes[1].legend(fontsize=9)

    scaled90 = np.asarray([
        row["sigma_scaled_envelopes"]["90"]["test_coverage"] for row in seed_rows
    ]) * 100.0
    scaled95 = np.asarray([
        row["sigma_scaled_envelopes"]["95"]["test_coverage"] for row in seed_rows
    ]) * 100.0
    axes[2].bar(positions - 0.18, scaled90, 0.36, label="mu+alpha90*sigma", color="#6baed6")
    axes[2].bar(positions + 0.18, scaled95, 0.36, label="mu+alpha95*sigma", color="#2171b5")
    axes[2].axhspan(90, 95, color="#2ca02c", alpha=0.08)
    axes[2].set_title("Sigma-scaled Envelope 测试 Coverage")
    axes[2].set_ylabel("Coverage（%）")
    axes[2].legend(fontsize=9)

    for axis in axes:
        axis.set_xticks(positions, [f"seed {seed}" for seed in seeds])
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("mu 与 sigma 测试集诊断")
    figure.text(0.01, 0.025, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "mu_sigma_test_diagnostics")


def _plot_ensemble_calibration_cdf(
    ensemble: dict[str, Any], curves: dict[str, Any], output: Path,
) -> dict[str, str]:
    residual = np.sort(np.asarray(curves["calibration_residual"]))
    cdf = np.arange(1, residual.size + 1) / residual.size
    q90 = float(ensemble["envelopes"]["90"]["q"])
    q95 = float(ensemble["envelopes"]["95"]["q"])
    figure, axis = plt.subplots(figsize=(9, 5.8))
    axis.plot(residual, cdf, color="#2166ac", linewidth=2.2)
    axis.axhline(0.90, color="#6baed6", linestyle=":")
    axis.axhline(0.95, color="#2171b5", linestyle="--")
    axis.axvline(q90, color="#6baed6", linestyle=":", linewidth=2, label=f"q90={q90:.3f}")
    axis.axvline(q95, color="#2171b5", linestyle="--", linewidth=2, label=f"q95={q95:.3f}")
    axis.set_xlabel("校准残差 y_avg - ensemble_mu")
    axis.set_ylabel("经验累计分布 CDF")
    axis.set_title(f"集成模型校准残差（n={residual.size} 个双侧 gate 窗口）")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.text(0.01, 0.01, NOTICE_ZH, fontsize=8)
    return _save_figure(figure, output / "ensemble_calibration_cdf")


def _plot_ensemble_test_summary(
    ensemble: dict[str, Any], curves: dict[str, Any], output: Path,
) -> dict[str, str]:
    gate = np.asarray(curves["test_mask"], dtype=bool)
    target = np.asarray(curves["y_avg_all"])[gate]
    mu = np.asarray(curves["mu_avg_all"])[gate]
    absolute = np.abs(target - mu)
    upper95 = np.asarray(curves["upper_avg_all"])[gate]
    positive_exceedance = np.maximum(0.0, target - upper95)
    positive_exceedance = positive_exceedance[positive_exceedance > 0.0]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.91, hspace=0.32, wspace=0.24)

    axes[0, 0].hist(target, bins=35, color="#4c78a8", alpha=0.85)
    for key, color, style in (("mean", "#f28e2b", "-"), ("p95", "#2ca02c", "--"), ("max", "#d62728", ":")):
        value = ensemble["observed_force"][key]
        axes[0, 0].axvline(value, color=color, linestyle=style, linewidth=2, label=f"{key}={value:.2f}")
    axes[0, 0].set_title("测试集真实 y_avg 分布")
    axes[0, 0].set_xlabel("力值")
    axes[0, 0].set_ylabel("窗口数")
    axes[0, 0].legend()

    axes[0, 1].hist(absolute, bins=35, color="#f28e2b", alpha=0.85)
    for key, color, style in (("mean", "#4c78a8", "-"), ("p95", "#2ca02c", "--"), ("max", "#d62728", ":")):
        value = ensemble["absolute_error"][key]
        axes[0, 1].axvline(value, color=color, linestyle=style, linewidth=2, label=f"{key}={value:.2f}")
    axes[0, 1].set_title("集成 mu 的绝对误差分布")
    axes[0, 1].set_xlabel("|y_avg - mu_avg|")
    axes[0, 1].set_ylabel("窗口数")
    axes[0, 1].legend()

    levels = ("90", "95")
    coverage = [ensemble["envelopes"][level]["coverage"] * 100 for level in levels]
    false_windows = [ensemble["envelopes"][level]["false_penalty_windows"] for level in levels]
    bars = axes[1, 0].bar([0, 1], coverage, color=["#6baed6", "#2171b5"])
    axes[1, 0].axhline(90, color="#6baed6", linestyle=":")
    axes[1, 0].axhline(95, color="#2171b5", linestyle="--")
    axes[1, 0].set_xticks([0, 1], ["U90", "U95"])
    axes[1, 0].set_ylabel("测试 Coverage（%）")
    axes[1, 0].set_title("集成 Additive Envelope")
    for bar, value, count in zip(bars, coverage, false_windows, strict=True):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}%\n误罚{count}窗", ha="center", va="bottom")

    axes[1, 1].hist(positive_exceedance, bins=20, color="#d62728", alpha=0.8)
    positive_stats = ensemble["envelopes"]["95"]["exceedance_positive"]
    for key, color, style in (("mean", "#4c78a8", "-"), ("p95", "#2ca02c", "--"), ("max", "#7f0000", ":")):
        value = positive_stats[key]
        axes[1, 1].axvline(value, color=color, linestyle=style, linewidth=2, label=f"{key}={value:.2f}")
    axes[1, 1].set_title("只统计超过 U95 的超界量")
    axes[1, 1].set_xlabel("max(0, y_avg-U95)")
    axes[1, 1].set_ylabel("窗口数")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.20)
    figure.suptitle("单一测试集汇总：31 条 Episode，649 个双侧稳定接触窗口")
    figure.text(0.01, 0.01, "力单位来自 Xense SDK；确认 resultant 为 N 后可直接按牛顿解读。", fontsize=8)
    return _save_figure(figure, output / "ensemble_test_force_error_summary")


def _plot_ensemble_episode_coverage(
    episode_rows: list[dict[str, Any]], output: Path,
) -> dict[str, str]:
    rows = sorted(episode_rows, key=lambda row: (str(row["batch_id"]), int(row["episode"])))
    coverage = np.asarray([float(row["average_coverage"]) for row in rows]) * 100.0
    figure, axis = plt.subplots(figsize=(13, 5.8))
    positions = np.arange(len(rows))
    colors = np.where(coverage < 90.0, "#d62728", "#4c78a8")
    axis.bar(positions, coverage, color=colors)
    axis.axhline(90.0, color="#d62728", linestyle="--", label="90%")
    axis.axhline(float(np.mean(coverage)), color="#2ca02c", linestyle="-", label=f"均值 {np.mean(coverage):.2f}%")
    axis.axhline(float(np.median(coverage)), color="#f28e2b", linestyle=":", label=f"中位数 {np.median(coverage):.2f}%")
    axis.set_xticks(positions, [str(row["episode"]) for row in rows], rotation=70)
    axis.set_xlabel("测试 Episode")
    axis.set_ylabel("U95 Coverage（%）")
    axis.set_ylim(45, 102)
    axis.set_title(f"集成模型逐 Episode Coverage（低于90%：{int(np.sum(coverage < 90))}/{len(rows)}）")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return _save_figure(figure, output / "ensemble_episode_coverage")


def _report_image(row: dict[str, Any], field: str, size: tuple[int, int]) -> Image.Image:
    episode_dir = (
        Path(row["robot_dataset"])
        / "data"
        / f"episode_{int(row['source_episode_index']):06d}"
    )
    path = episode_dir / row["images"][field]["path"]
    try:
        with Image.open(path) as image:
            return ImageOps.fit(
                image.convert("RGB"), size, method=Image.Resampling.LANCZOS
            )
    except Exception:
        return Image.new("RGB", size, (225, 225, 225))


def _large_error_analysis(
    ensemble: dict[str, Any], curves: dict[str, Any], output: Path,
    camera_fields: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    for stale in output.glob("large_error_cases_page_*.png"):
        stale.unlink()
    for stale in output.glob("large_error_cases_page_*.pdf"):
        stale.unlink()
    gate = np.asarray(curves["test_mask"], dtype=bool)
    target_all = np.asarray(curves["y_avg_all"])
    mu_all = np.asarray(curves["mu_avg_all"])
    upper_all = np.asarray(curves["upper_avg_all"])
    indices = np.flatnonzero(gate)
    absolute = np.abs(target_all[indices] - mu_all[indices])
    observed_mean = float(ensemble["observed_force"]["mean"])
    thresholds = []
    for threshold in (1.0, 2.0, 3.0, 5.0, 10.0, 15.0):
        count = int(np.sum(absolute > threshold))
        signed = target_all[indices] - mu_all[indices]
        thresholds.append({
            "absolute_error_greater_than": threshold,
            "threshold_percent_of_observed_mean": float(
                100.0 * threshold / max(abs(observed_mean), 1e-8)
            ),
            "windows": count,
            "percent_of_gated_test_windows": float(100.0 * count / len(indices)),
            "underprediction_windows_y_minus_mu_gt_threshold": int(
                np.sum(signed > threshold)
            ),
            "overprediction_windows_mu_minus_y_gt_threshold": int(
                np.sum(signed < -threshold)
            ),
        })

    ranked_indices = indices[np.argsort(-absolute)]
    rows = curves["rows"]
    all_cases: list[dict[str, Any]] = []
    seen_episodes: set[tuple[str, int]] = set()
    representative_cases: list[dict[str, Any]] = []
    for index in ranked_indices:
        row = rows[int(index)]
        target = float(target_all[index])
        mu = float(mu_all[index])
        signed = target - mu
        case = {
            "rank_all_windows": len(all_cases) + 1,
            "window_index": int(row["window_index"]),
            "batch_id": str(row["batch_id"]),
            "episode": int(row["episode_index"]),
            "source_episode": int(row["source_episode_index"]),
            "anchor_timestamp": float(row["start_timestamp"]),
            "target_y_avg": target,
            "predicted_mu_avg": mu,
            "signed_error_y_minus_mu": signed,
            "absolute_error": abs(signed),
            "absolute_error_percent_of_observed_mean": float(
                100.0 * abs(signed) / max(abs(observed_mean), 1e-8)
            ),
            "upper95": float(upper_all[index]),
            "exceedance_y_minus_upper95": max(0.0, target - float(upper_all[index])),
            "camera_age_ms": {
                field: float(row["images"][field]["age_ms"])
                for field in camera_fields
            },
        }
        all_cases.append(case)
        episode_key = (case["batch_id"], case["episode"])
        if episode_key not in seen_episodes and len(representative_cases) < 12:
            seen_episodes.add(episode_key)
            representative_cases.append(case)

    csv_path = output / "large_error_windows.csv"
    csv_fields = [
        "rank_all_windows", "window_index", "batch_id", "episode", "source_episode",
        "anchor_timestamp", "target_y_avg", "predicted_mu_avg",
        "signed_error_y_minus_mu", "absolute_error",
        "absolute_error_percent_of_observed_mean", "upper95",
        "exceedance_y_minus_upper95",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: case[field] for field in csv_fields} for case in all_cases)

    rows_by_window = {int(row["window_index"]): row for row in rows}
    sheets: list[dict[str, str]] = []
    for page_start in range(0, len(representative_cases), 4):
        page = representative_cases[page_start:page_start + 4]
        figure, axes = plt.subplots(len(page), 4, figsize=(16, 3.8 * len(page)), squeeze=False)
        for row_position, case in enumerate(page):
            source = rows_by_window[int(case["window_index"])]
            for camera_index, field in enumerate(camera_fields):
                axis = axes[row_position, camera_index]
                axis.imshow(_report_image(source, field, (640, 480)))
                axis.set_title(
                    f"camera {camera_index + 1} | age={source['images'][field]['age_ms']:.1f} ms",
                    fontsize=9,
                )
                axis.axis("off")
            meta = axes[row_position, 3]
            meta.axis("off")
            meta.text(
                0.0, 1.0,
                "\n".join([
                    f"Distinct-episode case #{page_start + row_position + 1}",
                    f"All-window rank #{case['rank_all_windows']}",
                    f"batch={case['batch_id']}",
                    f"episode={case['episode']} window={case['window_index']}",
                    f"target y_avg={case['target_y_avg']:.3f}",
                    f"ensemble mu={case['predicted_mu_avg']:.3f}",
                    f"signed y-mu={case['signed_error_y_minus_mu']:.3f}",
                    f"absolute error={case['absolute_error']:.3f}",
                    f"error / test mean={case['absolute_error_percent_of_observed_mean']:.2f}%",
                    f"U95={case['upper95']:.3f}",
                    f"max(0,y-U95)={case['exceedance_y_minus_upper95']:.3f}",
                ]),
                va="top", family="monospace", fontsize=10,
            )
        figure.suptitle(
            "测试集大绝对误差窗口：每个 Episode 只展示最差窗口，三路锚点图像",
            fontsize=15,
        )
        figure.tight_layout()
        stem = output / f"large_error_cases_page_{page_start // 4 + 1:02d}"
        sheets.append(_save_figure(figure, stem))

    summary = {
        "gated_test_windows": int(len(indices)),
        "definition": "absolute_error = abs(y_avg - ensemble_mu)",
        "threshold_counts": thresholds,
        "absolute_error_distribution": _distribution(absolute),
        "top_distinct_episode_cases": representative_cases,
        "all_ranked_windows_csv": str(csv_path),
        "visual_review_limit": (
            "Images are anchor-frame observations only; visual co-occurrence does not prove "
            "the cause of prediction error."
        ),
    }
    return summary, representative_cases, sheets


def _write_markdown(
    output: Path, seed_rows: list[dict[str, Any]], summary: dict[str, Any],
) -> None:
    ensemble = summary["test_set_ensemble_summary"]
    large_error = summary["large_error_analysis"]
    lines = [
        "# 双侧平均力稳定接触 Envelope 测试摘要",
        "",
        NOTICE_ZH,
        "",
        "没有重新训练模型，测试标签没有参与 q90/q95 调整。",
        "",
        "## 单一测试集结果",
        "",
        f"- 测试 Episode：{ensemble['test_episodes']}",
        f"- 双侧稳定接触窗口：{ensemble['test_windows_after_bilateral_gate']}",
        f"- 真实力均值 / P95 / 最大值：{ensemble['observed_force']['mean']:.3f} / "
        f"{ensemble['observed_force']['p95']:.3f} / {ensemble['observed_force']['max']:.3f}",
        f"- MAE：{ensemble['absolute_error']['mean']:.3f} "
        f"({ensemble['absolute_error']['mean_percent_of_observed_mean']:.2f}%)",
        f"- RMSE：{ensemble['absolute_error']['rmse']:.3f} "
        f"({ensemble['absolute_error']['rmse_percent_of_observed_mean']:.2f}%)",
        f"- 最大绝对误差：{ensemble['absolute_error']['max']:.3f} "
        f"({ensemble['absolute_error']['max_percent_of_observed_mean']:.2f}%)",
        f"- U95 q / Coverage：{ensemble['envelopes']['95']['q']:.3f} / "
        f"{ensemble['envelopes']['95']['coverage']*100:.2f}%",
        "",
        "## 大误差数量",
        "",
        "| 绝对误差阈值 | 窗口数 | 占比 | 低估 | 高估 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in large_error["threshold_counts"]:
        lines.append(
            f"| > {row['absolute_error_greater_than']:.0f} | {row['windows']} | "
            f"{row['percent_of_gated_test_windows']:.2f}% | "
            f"{row['underprediction_windows_y_minus_mu_gt_threshold']} | "
            f"{row['overprediction_windows_mu_minus_y_gt_threshold']} |"
        )
    lines.extend([
        "",
        "最大绝对误差来自模型高估，不是安全侧的漏估。全部窗口排序见 "
        "`large_error_windows.csv`，三路图像见 `large_error_cases_page_*.png`。",
        "",
        "三个 seed 的重复实验明细保留在 `three_seed_summary.csv`，不作为主报告口径。",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_summary(config: dict[str, Any], workspace: Path) -> list[dict[str, Any]]:
    counts = {
        split: {"windows": 0, "episodes": set()}
        for split in ("train", "validation", "calibration", "test")
    }
    import json

    with (workspace / "windows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            counts[split]["windows"] += 1
            counts[split]["episodes"].add(
                (str(row.get("batch_id", "")), int(row["episode_index"]))
            )
    ratios = {
        "train": float(config["training"]["train_ratio"]),
        "validation": float(config["training"]["validation_ratio"]),
        "calibration": float(config["training"]["calibration_ratio"]),
        "test": float(config["training"]["test_ratio"]),
    }
    uses = {
        "train": "更新模型参数",
        "validation": "选择最佳 checkpoint 和 early stopping",
        "calibration": "只计算 q90/q95/alpha",
        "test": "只计算最终测试指标",
    }
    display_names = {"train": "训练集", "validation": "验证集", "calibration": "校准集", "test": "测试集"}
    return [{
        "split": split,
        "display_name": display_names[split],
        "configured_ratio": ratios[split],
        "episodes": len(counts[split]["episodes"]),
        "windows": int(counts[split]["windows"]),
        "use": uses[split],
        "updates_model_weights": split == "train",
    } for split in counts]


def _format_pm(metric: dict[str, float], *, percent: bool = False) -> str:
    if percent:
        return f"{metric['mean'] * 100.0:.2f}% +/- {metric['std'] * 100.0:.2f} pp"
    return f"{metric['mean']:.2f} +/- {metric['std']:.2f}"


def _write_html(
    output: Path, summary: dict[str, Any], episode_rows: list[dict[str, Any]],
) -> Path:
    aggregate = summary["aggregate_mean_std"]
    independent = summary["independent_envelope_mean_std"]
    ensemble = summary["test_set_ensemble_summary"]
    seed_table = "".join(
        "<tr>"
        f"<td>{row['seed']}</td><td>{row['calibration_samples']}</td>"
        f"<td>{row['test_samples']}</td><td>{row['observed_average_force_test']['mean']:.3f}</td>"
        f"<td>{row['predicted_mu_average_test']:.3f}</td>"
        f"<td>{row['mu_average_test_error']['mae']:.3f}</td>"
        f"<td>{row['mu_average_test_error']['mae_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['mu_average_test_error']['rmse']:.3f}</td>"
        f"<td>{row['mu_average_test_error']['rmse_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['q90']:.3f}</td><td>{row['additive_envelopes']['90']['q_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['additive_envelopes']['90']['mean_upper']:.3f}</td>"
        f"<td>{row['additive_envelopes']['90']['test_coverage']*100:.2f}%</td>"
        f"<td>{row['additive_envelopes']['90']['false_penalty_windows']}</td>"
        f"<td>{row['q95']:.3f}</td><td>{row['additive_envelopes']['95']['q_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['additive_envelopes']['95']['mean_upper']:.3f}</td>"
        f"<td>{row['test_coverage']*100:.2f}%</td>"
        f"<td>{row['false_penalty_windows']}</td><td>{row['false_penalty_episodes']}</td>"
        f"<td>{row['longest_consecutive_false_penalty']}</td>"
        f"<td>{row['exceedance']['positive_mean']:.3f}</td>"
        f"<td>{row['exceedance']['positive_mean_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['exceedance']['positive_p90']:.3f}</td>"
        f"<td>{row['exceedance']['positive_p95']:.3f}</td>"
        f"<td>{row['exceedance']['positive_max']:.3f}</td>"
        "</tr>"
        for row in summary["seed_summary"]
    )
    mu_sigma_table = "".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['mu_average_test_error']['bias_y_minus_mu']:.4f}</td>"
        f"<td>{row['mu_average_test_error']['bias_y_minus_mu_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['mu_average_test_error']['mae']:.4f}</td>"
        f"<td>{row['mu_average_test_error']['rmse']:.4f}</td>"
        f"<td>{row['mu_average_test_error']['absolute_error_p90']:.4f}</td>"
        f"<td>{row['mu_average_test_error']['absolute_error_p95']:.4f}</td>"
        f"<td>{row['sigma_average_diagnostic']['mean_sigma']:.4f}</td>"
        f"<td>{row['sigma_average_diagnostic']['mean_sigma_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['sigma_average_diagnostic']['empirical_rmse']:.4f}</td>"
        f"<td>{row['sigma_average_diagnostic']['mean_sigma_over_rmse']:.3f}</td>"
        f"<td>{row['sigma_average_diagnostic']['spearman_sigma_vs_absolute_error']:.3f}</td>"
        f"<td>{row['sigma_by_side_test']['left']['mean_sigma']:.4f}</td>"
        f"<td>{row['sigma_by_side_test']['left']['empirical_rmse']:.4f}</td>"
        f"<td>{row['sigma_by_side_test']['left']['spearman_sigma_vs_absolute_error']:.3f}</td>"
        f"<td>{row['sigma_by_side_test']['right']['mean_sigma']:.4f}</td>"
        f"<td>{row['sigma_by_side_test']['right']['empirical_rmse']:.4f}</td>"
        f"<td>{row['sigma_by_side_test']['right']['spearman_sigma_vs_absolute_error']:.3f}</td>"
        "</tr>"
        for row in summary["seed_summary"]
    )
    sigma_envelope_table = "".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['sigma_scaled_envelopes']['90']['alpha']:.4f}</td>"
        f"<td>{row['sigma_scaled_envelopes']['90']['mean_width']:.4f}</td>"
        f"<td>{row['sigma_scaled_envelopes']['90']['test_coverage']*100:.2f}%</td>"
        f"<td>{row['sigma_scaled_envelopes']['95']['alpha']:.4f}</td>"
        f"<td>{row['sigma_scaled_envelopes']['95']['mean_width']:.4f}</td>"
        f"<td>{row['sigma_scaled_envelopes']['95']['test_coverage']*100:.2f}%</td>"
        "</tr>"
        for row in summary["seed_summary"]
    )
    split_table = "".join(
        "<tr>"
        f"<td>{row['display_name']}</td><td>{row['configured_ratio']*100:.0f}%</td>"
        f"<td>{row['episodes']}</td><td>{row['windows']}</td>"
        f"<td>{html.escape(row['use'])}</td>"
        f"<td>{'是' if row['updates_model_weights'] else '否'}</td>"
        "</tr>"
        for row in summary["split_summary"]
    )
    large_error = summary["large_error_analysis"]
    error_threshold_table = "".join(
        "<tr>"
        f"<td>&gt; {row['absolute_error_greater_than']:.0f}</td>"
        f"<td>{row['threshold_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{row['windows']}</td>"
        f"<td>{row['percent_of_gated_test_windows']:.2f}%</td>"
        f"<td>{row['underprediction_windows_y_minus_mu_gt_threshold']}</td>"
        f"<td>{row['overprediction_windows_mu_minus_y_gt_threshold']}</td>"
        "</tr>"
        for row in large_error["threshold_counts"]
    )
    large_case_table = "".join(
        "<tr>"
        f"<td>{rank}</td><td>{case['rank_all_windows']}</td>"
        f"<td>{html.escape(case['batch_id'])}</td><td>{case['episode']}</td>"
        f"<td>{case['window_index']}</td><td>{case['target_y_avg']:.3f}</td>"
        f"<td>{case['predicted_mu_avg']:.3f}</td>"
        f"<td>{case['signed_error_y_minus_mu']:.3f}</td>"
        f"<td>{case['absolute_error']:.3f}</td>"
        f"<td>{case['absolute_error_percent_of_observed_mean']:.2f}%</td>"
        f"<td>{case['upper95']:.3f}</td>"
        f"<td>{case['exceedance_y_minus_upper95']:.3f}</td>"
        "</tr>"
        for rank, case in enumerate(large_error["top_distinct_episode_cases"], start=1)
    )
    large_error_sheets = "".join(
        f"<a href=\"{Path(paths['png']).name}\" target=\"_blank\">"
        f"<img class=\"report-figure\" src=\"{Path(paths['png']).name}\" "
        f"alt=\"大误差图片对照第 {index} 页\"></a>"
        f"<p class=\"downloads\"><a href=\"{Path(paths['png']).name}\">PNG</a> | "
        f"<a href=\"{Path(paths['pdf']).name}\">PDF</a></p>"
        for index, paths in enumerate(summary["large_error_case_sheets"], start=1)
    )
    episode_table = "".join(
        f"<tr class=\"{'low' if float(row['average_coverage']) < 0.90 else ''}\" "
        f"data-low=\"{str(float(row['average_coverage']) < 0.90).lower()}\">"
        f"<td>{html.escape(str(row['batch_id']))}</td>"
        f"<td>{row['episode']}</td><td>{row['bilateral_stable_windows']}</td>"
        f"<td>{float(row['average_coverage'])*100:.2f}%</td>"
        f"<td>{row['average_false_penalty_windows']}</td>"
        f"<td>{float(row['y_avg_mae']):.4f}</td></tr>"
        for row in episode_rows
    )
    episode_coverages = np.asarray([
        float(row["average_coverage"]) for row in episode_rows
    ])
    episode_notes = (
        f"<li>测试 Episode：{len(episode_rows)} 条</li>"
        f"<li>逐 Episode Coverage 均值：{episode_coverages.mean()*100:.2f}%</li>"
        f"<li>逐 Episode Coverage 中位数：{np.median(episode_coverages)*100:.2f}%</li>"
        f"<li>Coverage 低于 90%：{int(np.sum(episode_coverages < 0.90))} 条</li>"
    )
    figures = [
        ("代表性测试 Episode 时序", "representative_timeseries", (
            "使用三个 seed 的逐窗口平均预测。紫色阶梯是双侧稳定接触 gate；红叉和阴影表示真实平均力超过集成 U95。"
        )),
        ("测试集真实力与预测误差总览", "ensemble_test_force_error_summary", (
            "直接汇总同一个测试集的真实力分布、集成 mu 误差、U90/U95 Coverage 和 U95 超界量。"
        )),
        ("集成模型校准残差 CDF", "ensemble_calibration_cdf", (
            "q90 和 q95 只由校准集的集成预测残差得到；测试集没有参与阈值估计。"
        )),
        ("测试集逐 Episode Coverage", "ensemble_episode_coverage", (
            "每根柱是一条测试 Episode；红色表示 Coverage 低于 90%。"
        )),
    ]
    figure_html = "".join(
        f"<section id=\"{key}\"><h2>{title}</h2><p>{caption}</p>"
        f"<a href=\"{Path(summary['figures'][key]['png']).name}\" target=\"_blank\">"
        f"<img class=\"report-figure\" src=\"{Path(summary['figures'][key]['png']).name}\" alt=\"{title}\"></a>"
        f"<p class=\"downloads\"><a href=\"{Path(summary['figures'][key]['png']).name}\">PNG</a> | "
        f"<a href=\"{Path(summary['figures'][key]['pdf']).name}\">PDF</a></p></section>"
        for title, key, caption in figures
    )
    mean_mu_bias = float(ensemble["signed_error_y_minus_mu"]["mean"])
    mean_sigma_ratio = float(
        ensemble["sigma"]["mean_total_ensemble_sigma"]
        / max(ensemble["absolute_error"]["rmse"], 1e-8)
    )
    mean_sigma_correlation = float(np.mean([
        row["sigma_average_diagnostic"]["spearman_sigma_vs_absolute_error"]
        for row in summary["seed_summary"]
    ]))
    mean_cov90 = float(ensemble["envelopes"]["90"]["coverage"])
    mean_observed_force = float(ensemble["observed_force"]["mean"])
    mean_mu_mae = float(ensemble["absolute_error"]["mean"])
    mean_mu_rmse = float(ensemble["absolute_error"]["rmse"])
    mean_positive_exceedance = float(
        ensemble["envelopes"]["95"]["exceedance_positive"]["mean"]
    )
    mean_positive_exceedance_p95 = float(
        ensemble["envelopes"]["95"]["exceedance_positive"]["p95"]
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双侧平均力稳定接触 Envelope 测试报告</title>
<style>
:root {{ color-scheme: light; --ink:#18212b; --muted:#596574; --line:#ccd3da; --blue:#27689d; --red:#b4232f; --soft:#f4f6f8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:white; font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ border-bottom:1px solid var(--line); padding:28px max(24px,calc((100vw - 1320px)/2)); background:#f8fafb; }}
header h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
header p {{ margin:4px 0; color:var(--muted); }}
nav {{ position:sticky; top:0; z-index:2; background:white; border-bottom:1px solid var(--line); padding:9px max(24px,calc((100vw - 1320px)/2)); }}
nav a {{ color:var(--blue); margin-right:18px; text-decoration:none; font-weight:600; }}
main {{ max-width:1320px; margin:auto; padding:24px; }}
section {{ padding:22px 0 30px; border-bottom:1px solid var(--line); }}
h2 {{ margin:0 0 12px; font-size:21px; letter-spacing:0; }}
h3 {{ margin:22px 0 8px; font-size:17px; letter-spacing:0; }}
.notice {{ border-left:5px solid var(--blue); background:#edf5fb; padding:14px 16px; font-weight:650; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(150px,1fr)); border:1px solid var(--line); }}
.metric {{ padding:16px; border-right:1px solid var(--line); }}
.metric:last-child {{ border-right:0; }}
.metric b {{ display:block; font-size:21px; color:var(--blue); }}
.metric span {{ color:var(--muted); font-size:13px; }}
.formula {{ background:var(--soft); border:1px solid var(--line); padding:14px 16px; font-family:ui-monospace,SFMono-Regular,monospace; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right; }}
th {{ background:#eef1f4; font-weight:700; }}
th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
tr.low td {{ background:#fff0f1; color:#741923; }}
.report-figure {{ display:block; width:100%; max-height:900px; object-fit:contain; border:1px solid var(--line); background:white; }}
.downloads {{ text-align:right; margin:7px 0 0; }}
.downloads a {{ color:var(--blue); }}
.controls {{ display:flex; gap:14px; align-items:center; margin:10px 0; }}
select,input {{ padding:7px 9px; border:1px solid #9aa5b1; background:white; }}
.fine {{ color:var(--muted); font-size:13px; }}
@media (max-width:850px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .metric {{ border-bottom:1px solid var(--line); }} nav {{ overflow:auto; white-space:nowrap; }} }}
</style>
</head>
<body>
<header><h1>双侧平均力稳定接触 Envelope 测试报告</h1>
<p>使用 Variant B 的 3 个随机 seed；没有重新训练，没有修改模型或 loss。</p>
<p>3 个 seed 是 3 次模型重复实验，不是 3 条测试数据。每个模型都测试同一组 31 条测试 Episode。</p></header>
<nav><a href="#method">数据划分</a><a href="#summary">核心结果</a><a href="#large-errors">大误差图片</a><a href="#representative_timeseries">时序图</a><a href="#ensemble_test_force_error_summary">误差总览</a><a href="#ensemble_calibration_cdf">校准</a><a href="#ensemble_episode_coverage">逐 Episode 图</a><a href="#episodes">逐 Episode 表</a></nav>
<main>
<p class="notice">q90、q95 和 alpha 只在校准集上估计。<br>测试集只计算 Coverage 和误差，绝不使用测试标签重新调整上界。</p>
<section id="method"><h2>数据划分与指标定义</h2>
<div class="table-wrap"><table><thead><tr><th>数据集</th><th>设定比例</th><th>Episode 数</th><th>窗口数</th><th>用途</th><th>更新权重</th></tr></thead><tbody>{split_table}</tbody></table></div>
<p class="fine">按 Episode 划分，不按帧随机拆分。只有训练集更新模型参数；验证集、校准集、测试集都不参与梯度更新。</p>
<div class="formula">y_avg = 0.5 * (y_left + y_right)<br>mu_seed = 0.5 * (mu_left + mu_right)<br>ensemble_mu = mean(mu_seed42, mu_seed43, mu_seed44)<br>q95 = calibration quantile(y_avg - ensemble_mu, 0.95)<br>upper_avg = ensemble_mu + q95<br>test coverage = mean(y_avg &lt;= upper_avg)</div></section>
<section id="summary"><h2>单一测试集核心结果</h2>
<div class="metrics"><div class="metric"><b>{ensemble['observed_force']['mean']:.2f}</b><span>真实 y_avg 均值</span></div><div class="metric"><b>{ensemble['absolute_error']['mean']:.2f}</b><span>集成 mu 的 MAE</span></div><div class="metric"><b>{ensemble['absolute_error']['max']:.2f}</b><span>最大绝对误差</span></div><div class="metric"><b>{ensemble['envelopes']['95']['coverage']*100:.2f}%</b><span>U95 测试 Coverage</span></div><div class="metric"><b>{ensemble['envelopes']['95']['false_penalty_windows']}</b><span>U95 误罚窗口</span></div></div>
<p><strong>样本：</strong>测试集共 31 条 Episode、1227 个窗口；双侧稳定接触 gate 后统计 649 个窗口。三个 seed 的作用只是对同一窗口产生三次预测，主结果使用逐窗口预测均值。</p>
<p class="fine"><strong>单位：</strong>数值来自 Xense SDK ForceResultant。若厂家确认其单位为 N，下表可直接按牛顿读取；当前未做额外单位换算。</p>
<h3>测试集真实值与预测值分布</h3>
<div class="table-wrap"><table><thead><tr><th>数据</th><th>窗口数</th><th>均值</th><th>中位数</th><th>P90</th><th>P95</th><th>最大值</th></tr></thead><tbody><tr><td>真实 y_avg</td><td>{ensemble['test_windows_after_bilateral_gate']}</td><td>{ensemble['observed_force']['mean']:.3f}</td><td>{ensemble['observed_force']['median']:.3f}</td><td>{ensemble['observed_force']['p90']:.3f}</td><td>{ensemble['observed_force']['p95']:.3f}</td><td>{ensemble['observed_force']['max']:.3f}</td></tr><tr><td>预测 ensemble_mu</td><td>{ensemble['test_windows_after_bilateral_gate']}</td><td>{ensemble['predicted_mu']['mean']:.3f}</td><td>{ensemble['predicted_mu']['median']:.3f}</td><td>{ensemble['predicted_mu']['p90']:.3f}</td><td>{ensemble['predicted_mu']['p95']:.3f}</td><td>{ensemble['predicted_mu']['max']:.3f}</td></tr></tbody></table></div>
<h3>预测误差</h3>
<div class="table-wrap"><table><thead><tr><th>Bias(y-mu)</th><th>MAE</th><th>MAE/真实均值</th><th>RMSE</th><th>RMSE/真实均值</th><th>|误差|P90</th><th>|误差|P95</th><th>|误差|最大</th><th>最大误差/真实均值</th></tr></thead><tbody><tr><td>{ensemble['signed_error_y_minus_mu']['mean']:.3f}</td><td>{ensemble['absolute_error']['mean']:.3f}</td><td>{ensemble['absolute_error']['mean_percent_of_observed_mean']:.2f}%</td><td>{ensemble['absolute_error']['rmse']:.3f}</td><td>{ensemble['absolute_error']['rmse_percent_of_observed_mean']:.2f}%</td><td>{ensemble['absolute_error']['p90']:.3f}</td><td>{ensemble['absolute_error']['p95']:.3f}</td><td>{ensemble['absolute_error']['max']:.3f}</td><td>{ensemble['absolute_error']['max_percent_of_observed_mean']:.2f}%</td></tr></tbody></table></div>
<h3>U90 / U95</h3>
<div class="table-wrap"><table><thead><tr><th>上界</th><th>q</th><th>q/真实均值</th><th>U均值</th><th>U最小</th><th>U最大</th><th>测试Coverage</th><th>超界窗口</th><th>含超界Episode</th><th>最长连续超界</th></tr></thead><tbody><tr><td>U90</td><td>{ensemble['envelopes']['90']['q']:.3f}</td><td>{ensemble['envelopes']['90']['q_percent_of_observed_mean']:.2f}%</td><td>{ensemble['envelopes']['90']['upper_mean']:.3f}</td><td>{ensemble['envelopes']['90']['upper_min']:.3f}</td><td>{ensemble['envelopes']['90']['upper_max']:.3f}</td><td>{ensemble['envelopes']['90']['coverage']*100:.2f}%</td><td>{ensemble['envelopes']['90']['false_penalty_windows']}</td><td>{ensemble['envelopes']['90']['false_penalty_episodes']}</td><td>{ensemble['envelopes']['90']['longest_false_run']}</td></tr><tr><td>U95</td><td>{ensemble['envelopes']['95']['q']:.3f}</td><td>{ensemble['envelopes']['95']['q_percent_of_observed_mean']:.2f}%</td><td>{ensemble['envelopes']['95']['upper_mean']:.3f}</td><td>{ensemble['envelopes']['95']['upper_min']:.3f}</td><td>{ensemble['envelopes']['95']['upper_max']:.3f}</td><td>{ensemble['envelopes']['95']['coverage']*100:.2f}%</td><td>{ensemble['envelopes']['95']['false_penalty_windows']}</td><td>{ensemble['envelopes']['95']['false_penalty_episodes']}</td><td>{ensemble['envelopes']['95']['longest_false_run']}</td></tr></tbody></table></div>
<h3>真实值超过上界时，超过多少</h3>
<div class="table-wrap"><table><thead><tr><th>上界</th><th>超界窗口内均值</th><th>中位数</th><th>P90</th><th>P95</th><th>最大值</th></tr></thead><tbody><tr><td>U90</td><td>{ensemble['envelopes']['90']['exceedance_positive']['mean']:.3f}</td><td>{ensemble['envelopes']['90']['exceedance_positive']['median']:.3f}</td><td>{ensemble['envelopes']['90']['exceedance_positive']['p90']:.3f}</td><td>{ensemble['envelopes']['90']['exceedance_positive']['p95']:.3f}</td><td>{ensemble['envelopes']['90']['exceedance_positive']['max']:.3f}</td></tr><tr><td>U95</td><td>{ensemble['envelopes']['95']['exceedance_positive']['mean']:.3f}</td><td>{ensemble['envelopes']['95']['exceedance_positive']['median']:.3f}</td><td>{ensemble['envelopes']['95']['exceedance_positive']['p90']:.3f}</td><td>{ensemble['envelopes']['95']['exceedance_positive']['p95']:.3f}</td><td>{ensemble['envelopes']['95']['exceedance_positive']['max']:.3f}</td></tr></tbody></table></div>
<h3>集成 sigma</h3><div class="table-wrap"><table><thead><tr><th>模型内平均sigma</th><th>seed间mu标准差</th><th>总集成sigma</th><th>总sigma/真实均值</th><th>实际RMSE</th><th>总sigma/RMSE</th></tr></thead><tbody><tr><td>{ensemble['sigma']['mean_within_model_sigma']:.3f}</td><td>{ensemble['sigma']['mean_between_seed_mu_std']:.3f}</td><td>{ensemble['sigma']['mean_total_ensemble_sigma']:.3f}</td><td>{ensemble['sigma']['total_sigma_percent_of_observed_mean']:.2f}%</td><td>{ensemble['absolute_error']['rmse']:.3f}</td><td>{ensemble['sigma']['mean_total_ensemble_sigma']/ensemble['absolute_error']['rmse']:.3f}</td></tr></tbody></table></div>
<details><summary><strong>附录：三个 seed 的重复实验明细</strong></summary><p>下面只用于观察随机种子稳定性，不代表三组测试数据。</p><div class="table-wrap"><table><thead><tr><th>Seed</th><th>校准窗</th><th>测试窗</th><th>真实均力</th><th>预测mu均值</th><th>MAE</th><th>MAE%</th><th>RMSE</th><th>RMSE%</th><th>q90</th><th>q90%</th><th>平均U90</th><th>Cov90</th><th>误罚90</th><th>q95</th><th>q95%</th><th>平均U95</th><th>Cov95</th><th>误罚95</th><th>误罚Episode</th><th>最长连续</th><th>超界均值</th><th>超界%</th><th>超界P90</th><th>超界P95</th><th>超界最大</th></tr></thead><tbody>{seed_table}</tbody></table></div><h3>mu/sigma 明细</h3><div class="table-wrap"><table><thead><tr><th>Seed</th><th>Bias</th><th>Bias%</th><th>MAE</th><th>RMSE</th><th>|误差|P90</th><th>|误差|P95</th><th>sigma_avg</th><th>sigma%</th><th>实际RMSE</th><th>sigma/RMSE</th><th>相关</th><th>左sigma</th><th>左RMSE</th><th>左相关</th><th>右sigma</th><th>右RMSE</th><th>右相关</th></tr></thead><tbody>{mu_sigma_table}</tbody></table></div></details></section>
<section id="large-errors"><h2>大绝对误差有多少，以及对应图像</h2>
<p>这里的绝对误差定义为 <code>|y_avg - ensemble_mu|</code>。分母始终是 649 个双侧稳定接触测试窗口；阈值百分比以测试集真实力均值 {ensemble['observed_force']['mean']:.3f} 为基准。</p>
<div class="table-wrap"><table><thead><tr><th>绝对误差阈值</th><th>阈值/真实均值</th><th>窗口数</th><th>占649窗比例</th><th>低估窗口 y-mu&gt;阈值</th><th>高估窗口 mu-y&gt;阈值</th></tr></thead><tbody>{error_threshold_table}</tbody></table></div>
<p><strong>结论：</strong>最大误差 {ensemble['absolute_error']['max']:.3f} 是尾部极端窗口；下表和图片展示误差最大的 12 个不同 Episode，每个 Episode 只取最差窗口，避免连续相邻帧反复展示同一画面。</p>
<div class="table-wrap"><table><thead><tr><th>展示排名</th><th>全窗口排名</th><th>Batch</th><th>Episode</th><th>Window</th><th>真实y</th><th>预测mu</th><th>y-mu</th><th>|误差|</th><th>误差/均值</th><th>U95</th><th>超过U95</th></tr></thead><tbody>{large_case_table}</tbody></table></div>
<p class="fine">图像是动作锚点前最近的三路相机帧。它们可用于观察共现现象，但不能单凭画面断言误差的因果来源。</p>
{large_error_sheets}
<p><a href="large_error_windows.csv">下载全部 649 个测试窗口的误差排序 CSV</a></p></section>
<section><h2>结果解读</h2>
<ul><li>单一测试集上的 U90 Coverage 为 <strong>{mean_cov90*100:.2f}%</strong>；U95 Coverage 为 <strong>{ensemble['envelopes']['95']['coverage']*100:.2f}%</strong>。</li>
<li>双侧稳定接触测试窗口的真实 y_avg 均值约为 <strong>{mean_observed_force:.3f} 力单位</strong>。mu_avg 的平均 MAE 为 <strong>{mean_mu_mae:.3f}</strong>（约占真实均值的 {mean_mu_mae/mean_observed_force*100:.2f}%），RMSE 为 <strong>{mean_mu_rmse:.3f}</strong>（{mean_mu_rmse/mean_observed_force*100:.2f}%）。若 SDK 单位确认为 N，这些值分别就是 {mean_mu_mae:.3f} N 和 {mean_mu_rmse:.3f} N。</li>
<li>mu_avg 的平均 Bias(y-mu) 为 <strong>{mean_mu_bias:.4f}</strong>。负值表示预测均值 mu_avg 平均略高于真实 y_avg。</li>
<li>平均 sigma_avg / 实际 RMSE 为 <strong>{mean_sigma_ratio:.3f}</strong>，即 sigma 的数值尺度约为实际 RMSE 的 {mean_sigma_ratio*100:.1f}%，整体偏小。</li>
<li>sigma_avg 与绝对误差的平均 Spearman 相关系数为 <strong>{mean_sigma_correlation:.3f}</strong>，说明 sigma 能识别一部分高误差窗口，但不能完全覆盖尾部误差。</li>
<li>只统计真实力已经超过 95% 上界的窗口，平均超界量约为 <strong>{mean_positive_exceedance:.3f}</strong>，超界量 P95 约为 <strong>{mean_positive_exceedance_p95:.3f}</strong>。若 SDK 单位为 N，即分别约 {mean_positive_exceedance:.3f} N 和 {mean_positive_exceedance_p95:.3f} N。</li>
<li>以上结论只描述这 31 条测试 Episode；所有 q90/q95/alpha 均来自 30 条校准 Episode。</li></ul></section>
{figure_html}
<section id="episodes"><h2>测试集逐 Episode 结果</h2><ul>{episode_notes}</ul>
<div class="controls"><label><input id="lowFilter" type="checkbox"> 只看 Coverage 低于 90%</label></div>
<div class="table-wrap"><table id="episodeTable"><thead><tr><th>Batch</th><th>Episode</th><th>稳定窗口数</th><th>U95 Coverage</th><th>超界窗口</th><th>mu MAE</th></tr></thead><tbody>{episode_table}</tbody></table></div></section>
<section><h2>结果文件</h2><p><a href="summary.json">summary.json</a> | <a href="three_seed_summary.csv">three_seed_summary.csv</a> | <a href="REPORT.md">REPORT.md</a></p></section>
</main>
<script>
const low=document.getElementById('lowFilter');
function filterRows() {{ document.querySelectorAll('#episodeTable tbody tr').forEach(row => {{ row.hidden=low.checked&&row.dataset.low!=='true'; }}); }}
low.addEventListener('change',filterRows);
</script>
</body></html>"""
    path = output / "detailed_report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def generate_bilateral_report(config: dict[str, Any]) -> dict[str, Any]:
    stable = config["stable_contact"]
    seeds = [int(seed) for seed in stable.get("seeds", [42, 43, 44])]
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    root = workspace / str(stable.get("artifact_subdir", "stable_contact_ab"))
    output = root / "bilateral_average_report"
    output.mkdir(parents=True, exist_ok=True)

    run_data: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = root / f"B_seed_{seed}" / "stable_evaluation" / "stable_contact_calibrated_checkpoint.pt"
        report, per_episode, curves = _evaluate_seed(config, checkpoint, seed)
        gate = np.asarray(curves["test_mask"], dtype=bool)
        penalty = np.asarray(curves["penalty_all"], dtype=bool)
        calibration_mask = np.asarray(curves["calibration_mask"], dtype=bool)
        calibration_residual = np.asarray(curves["calibration_residual"])
        q90 = float(np.quantile(calibration_residual, 0.90))
        q95 = float(np.quantile(calibration_residual, 0.95))
        test_y = np.asarray(curves["y_avg_all"])
        test_mu = np.asarray(curves["mu_avg_all"])
        upper90 = test_mu + q90
        upper95 = test_mu + q95
        penalty90 = gate & (test_y > upper90)
        penalty95 = gate & (test_y > upper95)
        cal_left = curves["calibration_left"]
        cal_right = curves["calibration_right"]
        test_left = curves["test_left"]
        test_right = curves["test_right"]
        calibration_sigma_avg = 0.5 * np.sqrt(
            np.asarray(cal_left["sigma"])**2 + np.asarray(cal_right["sigma"])**2
        )
        test_sigma_avg = 0.5 * np.sqrt(
            np.asarray(test_left["sigma"])**2 + np.asarray(test_right["sigma"])**2
        )
        standardized_calibration = (
            calibration_residual / (calibration_sigma_avg[calibration_mask] + 1e-8)
        )
        alpha90 = float(np.quantile(standardized_calibration, 0.90))
        alpha95 = float(np.quantile(standardized_calibration, 0.95))
        scaled_upper90 = test_mu + alpha90 * test_sigma_avg
        scaled_upper95 = test_mu + alpha95 * test_sigma_avg
        exceedance = np.maximum(
            0.0,
            test_y[gate] - upper95[gate],
        )
        observed_mean = float(np.mean(test_y[gate]))
        observed_scale = max(abs(observed_mean), 1e-8)
        mu_error = _mu_error_stats(test_y[gate], test_mu[gate])
        for key in (
            "bias_y_minus_mu", "mae", "rmse", "absolute_error_p90", "absolute_error_p95",
        ):
            mu_error[f"{key}_percent_of_observed_mean"] = float(
                100.0 * mu_error[key] / observed_scale
            )
        sigma_by_side = {}
        for side, values in (("left", test_left), ("right", test_right)):
            sigma_by_side[side] = _sigma_diagnostics(
                np.asarray(values["target"])[gate],
                np.asarray(values["mu"])[gate],
                np.asarray(values["sigma"])[gate],
            )
        exceedance_summary = _exceedance_stats(exceedance)
        for key in (
            "mean", "p90", "p95", "max", "positive_mean", "positive_p90",
            "positive_p95", "positive_max",
        ):
            exceedance_summary[f"{key}_percent_of_observed_mean"] = float(
                100.0 * exceedance_summary[key] / observed_scale
            )
        seed_row = {
            "seed": seed,
            "calibration_samples": report["calibration_samples"],
            "test_samples": report["test_samples"],
            "mae": report["mae"],
            "q90": q90,
            "q95": q95,
            "test_coverage": float(np.mean(test_y[gate] <= upper95[gate])),
            "false_penalty_windows": int(penalty95.sum()),
            "false_penalty_episodes": _false_episode_count(curves["rows"], gate, penalty95),
            "longest_consecutive_false_penalty": _longest_false_run(
                curves["rows"], gate, penalty95
            ),
            "additive_envelopes": {
                "90": {
                    "q": q90,
                    "mean_upper": float(np.mean(upper90[gate])),
                    "q_percent_of_observed_mean": float(100.0 * q90 / observed_scale),
                    "test_coverage": float(np.mean(test_y[gate] <= upper90[gate])),
                    "false_penalty_windows": int(penalty90.sum()),
                    "false_penalty_episodes": _false_episode_count(
                        curves["rows"], gate, penalty90
                    ),
                },
                "95": {
                    "q": q95,
                    "mean_upper": float(np.mean(upper95[gate])),
                    "q_percent_of_observed_mean": float(100.0 * q95 / observed_scale),
                    "test_coverage": float(np.mean(test_y[gate] <= upper95[gate])),
                    "false_penalty_windows": int(penalty95.sum()),
                    "false_penalty_episodes": _false_episode_count(
                        curves["rows"], gate, penalty95
                    ),
                },
            },
            "observed_average_force_test": {
                "mean": observed_mean,
                "median": float(np.median(test_y[gate])),
                "p90": float(np.quantile(test_y[gate], 0.90)),
                "p95": float(np.quantile(test_y[gate], 0.95)),
                "max": float(np.max(test_y[gate])),
            },
            "predicted_mu_average_test": float(np.mean(test_mu[gate])),
            "mu_average_test_error": mu_error,
            "sigma_average_diagnostic": {
                **_sigma_diagnostics(test_y[gate], test_mu[gate], test_sigma_avg[gate]),
                "mean_sigma_percent_of_observed_mean": float(
                    100.0 * np.mean(test_sigma_avg[gate]) / observed_scale
                ),
                "definition": "0.5 * sqrt(sigma_left^2 + sigma_right^2), assuming independent side errors",
            },
            "sigma_by_side_test": sigma_by_side,
            "sigma_scaled_envelopes": {
                "90": {
                    "alpha": alpha90,
                    "mean_width": float(np.mean(alpha90 * test_sigma_avg[gate])),
                    "test_coverage": float(np.mean(test_y[gate] <= scaled_upper90[gate])),
                },
                "95": {
                    "alpha": alpha95,
                    "mean_width": float(np.mean(alpha95 * test_sigma_avg[gate])),
                    "test_coverage": float(np.mean(test_y[gate] <= scaled_upper95[gate])),
                },
            },
            "exceedance": exceedance_summary,
        }
        seed_rows.append(seed_row)
        episode_rows.extend(per_episode)
        run_data.append({"report": report, "per_episode": per_episode, "curves": curves})

    ensemble_summary, ensemble_curves, ensemble_episode_rows = _ensemble_evaluation(run_data)
    timeseries_paths, representative_selection = _plot_representative_episode(
        ensemble_curves, ensemble_episode_rows, output, "ensemble"
    )
    ensemble_cdf_paths = _plot_ensemble_calibration_cdf(
        ensemble_summary, ensemble_curves, output
    )
    ensemble_test_paths = _plot_ensemble_test_summary(
        ensemble_summary, ensemble_curves, output
    )
    ensemble_episode_paths = _plot_ensemble_episode_coverage(
        ensemble_episode_rows, output
    )
    large_error_summary, _, large_error_sheets = _large_error_analysis(
        ensemble_summary,
        ensemble_curves,
        output,
        list(config["vision"]["camera_order"]),
    )
    for stale_name in (
        "calibration_residual_cdf", "independent_vs_bilateral",
        "test_episode_coverage_distribution", "envelope_90_95_details",
        "mu_sigma_test_diagnostics",
    ):
        for suffix in (".png", ".pdf"):
            (output / f"{stale_name}{suffix}").unlink(missing_ok=True)

    aggregate_keys = (
        "mae", "q95", "test_coverage", "false_penalty_windows", "false_penalty_episodes",
        "longest_consecutive_false_penalty",
    )
    aggregate = {
        key: _mean_std([row[key] for row in seed_rows]) for key in aggregate_keys
    }
    exceedance_aggregate = {
        key: _mean_std([row["exceedance"][key] for row in seed_rows])
        for key in ("mean", "p90", "p95", "max")
    }
    independent = {
        "test_coverage": _mean_std([
            data["report"]["independent_additive_comparison"]["joint_coverage"]
            for data in run_data
        ]),
        "false_penalty_windows": _mean_std([
            data["report"]["independent_additive_comparison"]["false_penalty_windows"]
            for data in run_data
        ]),
        "false_penalty_episodes": _mean_std([
            data["report"]["independent_additive_comparison"]["episodes_with_false_penalty"]
            for data in run_data
        ]),
    }
    summary = {
        "title": "Bilateral average stable-contact test report",
        "notice": NOTICE,
        "retrained": False,
        "model_or_loss_modified": False,
        "test_used_to_adjust_q95": False,
        "force_unit": {
            "reported_unit": "Xense SDK resultant force unit",
            "newton_conversion_applied": False,
            "physical_newton_calibration_verified": False,
            "interpretation": "Values equal newtons only if the vendor confirms resultant is in N; verify with a known load before hard-safety use.",
        },
        "selection": "stable_prior_left AND stable_prior_right",
        "false_penalty_definition": "stable-gated observed y_avg exceeds mu_avg + calibration q95",
        "exceedance_definition": "max(0, y_avg - upper_avg), including zero values over all gated test windows",
        "std_definition": "population standard deviation across seeds",
        "seed_summary": seed_rows,
        "test_set_ensemble_summary": ensemble_summary,
        "large_error_analysis": large_error_summary,
        "large_error_case_sheets": large_error_sheets,
        "aggregate_mean_std": aggregate,
        "exceedance_mean_std": exceedance_aggregate,
        "independent_envelope_mean_std": independent,
        "split_summary": _split_summary(config, workspace),
        "representative_episode": representative_selection,
        "figures": {
            "representative_timeseries": timeseries_paths,
            "ensemble_calibration_cdf": ensemble_cdf_paths,
            "ensemble_test_force_error_summary": ensemble_test_paths,
            "ensemble_episode_coverage": ensemble_episode_paths,
        },
    }
    with (output / "three_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "seed", "calibration_samples", "test_samples", "mae", "q90", "q95",
            "test_coverage", "false_penalty_windows", "false_penalty_episodes",
            "longest_consecutive_false_penalty",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in seed_rows)
    _write_markdown(output, seed_rows, summary)
    html_path = _write_html(output, summary, ensemble_episode_rows)
    summary["html_report"] = str(html_path)
    write_json(output / "summary.json", summary)
    return summary
