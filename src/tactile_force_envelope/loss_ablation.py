from __future__ import annotations

import copy
import csv
import gc
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import write_json
from .stable_contact_experiment import _gates_for_rows, _load_evaluation_context
from .transformer_train import train_transformer_and_calibrate


VARIANTS = {
    "gaussian_nll": {
        "gaussian_nll_weight": 1.0,
        "smooth_l1_weight": 0.5,
        "label": "Gaussian NLL + 0.5 SmoothL1",
    },
    "smooth_l1_only": {
        "gaussian_nll_weight": 0.0,
        "smooth_l1_weight": 1.0,
        "label": "SmoothL1 only",
    },
}


def _episode_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("batch_id", "")), int(row["episode_index"])


def _central_interval_metrics(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    y = target[mask]
    center = mu[mask]
    low = lower[mask]
    high = upper[mask]
    residual = y - center
    low_violation = y < low
    high_violation = y > high
    violation = low_violation | high_violation
    selected_positions = np.flatnonzero(mask)
    episode_false: dict[tuple[str, int], int] = {}
    episode_coverage: dict[tuple[str, int], list[bool]] = {}
    for compact_index, position in enumerate(selected_positions):
        key = _episode_key(rows[int(position)])
        episode_false[key] = episode_false.get(key, 0) + int(violation[compact_index])
        episode_coverage.setdefault(key, []).append(not bool(violation[compact_index]))
    return {
        "samples": int(y.size),
        "episodes": len(episode_false),
        "target_mean_N": float(np.mean(y)),
        "target_std_N": float(np.std(y)),
        "bias_y_minus_mu_N": float(np.mean(residual)),
        "mae_N": float(np.mean(np.abs(residual))),
        "rmse_N": float(np.sqrt(np.mean(residual**2))),
        "coverage": float(np.mean(~violation)),
        "mean_interval_width_N": float(np.mean(high - low)),
        "low_violation_windows": int(np.sum(low_violation)),
        "high_violation_windows": int(np.sum(high_violation)),
        "false_penalty_windows": int(np.sum(violation)),
        "false_penalty_episodes": int(sum(value > 0 for value in episode_false.values())),
        "episode_coverage_mean": float(
            np.mean([np.mean(values) for values in episode_coverage.values()])
        ),
    }


def _evaluate_checkpoint(
    config: dict[str, Any], checkpoint_path: Path, variant: str, seed: int,
    label: str | None = None,
) -> dict[str, Any]:
    checkpoint, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path)
    )
    calibration_gate = _gates_for_rows(
        grouped["calibration"], history, history_mask, config
    )
    test_gate = _gates_for_rows(grouped["test"], history, history_mask, config)
    calibration_mask = calibration_gate[:, 0] & calibration_gate[:, 1]
    test_mask = test_gate[:, 0] & test_gate[:, 1]
    if not np.any(calibration_mask) or not np.any(test_mask):
        raise ValueError("Bilateral stable-contact calibration/test selection is empty")

    cal_left = predictions["calibration"]["left"]
    cal_right = predictions["calibration"]["right"]
    test_left = predictions["test"]["left"]
    test_right = predictions["test"]["right"]
    cal_y = 0.5 * (cal_left["target"] + cal_right["target"])
    cal_mu = 0.5 * (cal_left["mu"] + cal_right["mu"])
    test_y = 0.5 * (test_left["target"] + test_right["target"])
    test_mu = 0.5 * (test_left["mu"] + test_right["mu"])

    residual = cal_y[calibration_mask] - cal_mu[calibration_mask]
    q05 = float(np.quantile(residual, 0.05))
    q95 = float(np.quantile(residual, 0.95))
    additive_lower = test_mu + q05
    additive_upper = test_mu + q95
    report: dict[str, Any] = {
        "variant": variant,
        "label": label or VARIANTS[variant]["label"],
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_ran": len(
            json.loads(
                (checkpoint_path.parent / "transformer_train_history.json").read_text(
                    encoding="utf-8"
                )
            )
        ) if (checkpoint_path.parent / "transformer_train_history.json").is_file() else None,
        "loss": checkpoint["loss"],
        "selection": "stable_prior_left AND stable_prior_right",
        "selection_uses_future_target": False,
        "calibration_samples": int(np.sum(calibration_mask)),
        "test_samples": int(np.sum(test_mask)),
        "test_used_for_calibration": False,
        "additive_calibration": {
            "residual": "y_avg - mu_avg",
            "q05_N": q05,
            "q95_N": q95,
            "source": "calibration split bilateral stable-contact windows only",
        },
        "additive_test": _central_interval_metrics(
            grouped["test"], test_mask, test_y, test_mu,
            additive_lower, additive_upper,
        ),
    }

    if float(checkpoint["loss"].get("gaussian_nll_weight", 1.0)) > 0.0:
        cal_sigma = 0.5 * (cal_left["sigma"] + cal_right["sigma"])
        test_sigma = 0.5 * (test_left["sigma"] + test_right["sigma"])
        standardized = residual / (cal_sigma[calibration_mask] + 1e-8)
        alpha05 = float(np.quantile(standardized, 0.05))
        alpha95 = float(np.quantile(standardized, 0.95))
        scaled_lower = test_mu + alpha05 * test_sigma
        scaled_upper = test_mu + alpha95 * test_sigma
        sigma_metrics = _central_interval_metrics(
            grouped["test"], test_mask, test_y, test_mu,
            scaled_lower, scaled_upper,
        )
        sigma_metrics.update({
            "alpha05": alpha05,
            "alpha95": alpha95,
            "mean_sigma_avg_N": float(np.mean(test_sigma[test_mask])),
            "sigma_avg_definition": "0.5 * (sigma_left + sigma_right)",
        })
        side_nlls = []
        for side_values in (test_left, test_right):
            side_target = side_values["target"][test_mask]
            side_mu = side_values["mu"][test_mask]
            side_sigma = np.maximum(side_values["sigma"][test_mask], 1e-8)
            side_nlls.append(
                0.5 * (((side_target - side_mu) / side_sigma) ** 2 + np.log(side_sigma**2))
            )
        report["additive_test"]["gaussian_nll_physical"] = float(
            np.mean(np.stack(side_nlls, axis=1))
        )
        report["sigma_scaled_test"] = sigma_metrics
    else:
        report["sigma_scaled_test"] = None
    return report


def _aggregate(
    runs: list[dict[str, Any]], metric: str, source: str = "additive_test",
) -> dict[str, float]:
    values = np.asarray(
        [float(run[source][metric]) for run in runs], dtype=np.float64
    )
    return {"mean": float(values.mean()), "std": float(values.std())}


def _plot_summary(summary: dict[str, Any], output: Path) -> None:
    names = list(VARIANTS)
    labels = [VARIANTS[name]["label"] for name in names]
    metrics = [
        ("mae_N", "MAE (N)", None),
        ("rmse_N", "RMSE (N)", None),
        ("coverage", "Central 90% coverage", 0.90),
        ("mean_interval_width_N", "Mean interval width (N)", None),
        ("false_penalty_windows", "Out-of-band windows", None),
        ("false_penalty_episodes", "Episodes with violations", None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    colors = ["#2878B5", "#D95F02"]
    for axis, (key, title, target) in zip(axes.flat, metrics, strict=True):
        means = [summary["aggregate"][name][key]["mean"] for name in names]
        stds = [summary["aggregate"][name][key]["std"] for name in names]
        axis.bar(labels, means, yerr=stds, capsize=5, color=colors, width=0.65)
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=10)
    figure.suptitle("USB force-envelope loss ablation, three seeds")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"gaussian_loss_ablation_summary.{suffix}", dpi=200)
    plt.close(figure)


def _plot_sigma_summary(summary: dict[str, Any], output: Path) -> None:
    labels = ["Additive residual", "Sigma-scaled residual"]
    metrics = [
        ("coverage", "Central 90% coverage", 0.90),
        ("mean_interval_width_N", "Mean interval width (N)", None),
        ("false_penalty_windows", "Out-of-band windows", None),
        ("false_penalty_episodes", "Episodes with violations", None),
    ]
    additive = summary["aggregate"]["gaussian_nll"]
    scaled = summary["sigma_scaled_aggregate"]
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.5), constrained_layout=True)
    for axis, (key, title, target) in zip(axes, metrics, strict=True):
        means = [additive[key]["mean"], scaled[key]["mean"]]
        stds = [additive[key]["std"], scaled[key]["std"]]
        axis.bar(labels, means, yerr=stds, capsize=5, color=["#7F8C8D", "#2878B5"])
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle("Does predicted sigma improve calibrated USB force intervals?")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"gaussian_sigma_usefulness.{suffix}", dpi=200)
    plt.close(figure)


def _write_html(summary: dict[str, Any], output: Path) -> None:
    rows = []
    for run in summary["runs"]:
        metric = run["additive_test"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['variant'])}</td><td>{run['seed']}</td>"
            f"<td>{run['calibration_samples']}</td><td>{run['test_samples']}</td>"
            f"<td>{metric['mae_N']:.4f}</td><td>{metric['rmse_N']:.4f}</td>"
            f"<td>{metric['coverage'] * 100:.2f}%</td>"
            f"<td>{metric['mean_interval_width_N']:.4f}</td>"
            f"<td>{metric['false_penalty_windows']}</td>"
            f"<td>{metric['false_penalty_episodes']}</td></tr>"
        )
    sigma_rows = []
    for run in summary["runs"]:
        metric = run["sigma_scaled_test"]
        if metric is None:
            continue
        sigma_rows.append(
            "<tr>"
            f"<td>{run['seed']}</td><td>{metric['alpha05']:.4f}</td>"
            f"<td>{metric['alpha95']:.4f}</td><td>{metric['mean_sigma_avg_N']:.4f}</td>"
            f"<td>{metric['coverage'] * 100:.2f}%</td>"
            f"<td>{metric['mean_interval_width_N']:.4f}</td>"
            f"<td>{metric['false_penalty_windows']}</td>"
            f"<td>{metric['false_penalty_episodes']}</td></tr>"
        )
    comparison = summary["comparison"]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Gaussian Loss Ablation</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:32px auto;line-height:1.65;color:#20252b}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eef2f5;padding:2px 5px}}</style></head>
<body><h1>USB 触觉力包络 Gaussian Loss 消融</h1>
<p>保持数据、episode split、Pi0.5 z_rl、Transformer、200 ms 因果触觉历史、
300 ms 标签窗口和 q05/q95 residual calibration 不变。唯一训练差异是是否启用 Gaussian NLL。</p>
<p><strong>校准集仅用于估计 q05/q95，测试集没有参与训练、checkpoint 选择或区间校准。</strong></p>
<img src="gaussian_loss_ablation_summary.png" alt="ablation summary">
<h2>三种子结果</h2><table><thead><tr><th>Variant</th><th>Seed</th><th>Calibration</th>
<th>Test</th><th>MAE (N)</th><th>RMSE (N)</th><th>Coverage</th><th>Width (N)</th>
<th>越界窗口</th><th>越界 Episode</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Gaussian sigma 是否有用</h2>
<p>仅在 Gaussian 模型内比较。Additive 区间使用 <code>mu+q05, mu+q95</code>；
sigma-scaled 区间使用 calibration 集标准化残差得到的
<code>mu+alpha05*sigma, mu+alpha95*sigma</code>。两者都没有使用测试标签调参。</p>
<img src="gaussian_sigma_usefulness.png" alt="sigma usefulness">
<table><thead><tr><th>Seed</th><th>alpha05</th><th>alpha95</th><th>Mean sigma (N)</th>
<th>Coverage</th><th>Width (N)</th><th>越界窗口</th><th>越界 Episode</th></tr></thead>
<tbody>{''.join(sigma_rows)}</tbody></table>
<h2>结论</h2><p>{html.escape(comparison['conclusion'])}</p>
<p>Gaussian 相对 SmoothL1-only 的 MAE 改变量：<strong>{comparison['mae_relative_improvement_percent']:.2f}%</strong>；
RMSE 改变量：<strong>{comparison['rmse_relative_improvement_percent']:.2f}%</strong>；
校准区间宽度改变量：<strong>{comparison['width_relative_improvement_percent']:.2f}%</strong>。
正数表示 Gaussian 更优。</p>
<p>在 Gaussian 模型内，使用 sigma 缩放后，中心 90% coverage 为
<strong>{comparison['sigma_scaled_coverage_percent']:.2f}%</strong>，区间宽度相对纯 additive
calibration 缩窄 <strong>{comparison['sigma_scaled_width_reduction_percent']:.2f}%</strong>。
这说明 sigma 含有可利用的输入相关尺度信息，但不证明条件力分布严格服从高斯，也不表示
未经 calibration 的 Gaussian 分位数已经准确。</p></body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def run_gaussian_loss_ablation(
    config: dict[str, Any], *, seeds: list[int], force: bool = False,
) -> dict[str, Any]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    output = workspace / "gaussian_loss_ablation"
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for variant, settings in VARIANTS.items():
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config["training"].update({
                "seed": int(seed),
                "gaussian_nll_weight": settings["gaussian_nll_weight"],
                "smooth_l1_weight": settings["smooth_l1_weight"],
                "artifact_subdir": f"gaussian_loss_ablation/{variant}_seed_{seed}",
            })
            run_dir = workspace / run_config["training"]["artifact_subdir"]
            checkpoint_path = run_dir / "transformer_force_envelope_model.pt"
            if force or not checkpoint_path.is_file():
                print(f"Training {variant} seed={seed}", flush=True)
                train_transformer_and_calibrate(run_config)
            else:
                print(f"Reusing {checkpoint_path}", flush=True)
            runs.append(_evaluate_checkpoint(run_config, checkpoint_path, variant, seed))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate: dict[str, Any] = {}
    metric_names = (
        "mae_N", "rmse_N", "coverage", "mean_interval_width_N",
        "false_penalty_windows", "false_penalty_episodes",
    )
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        aggregate[variant] = {
            metric: _aggregate(selected, metric) for metric in metric_names
        }
    gaussian = aggregate["gaussian_nll"]
    deterministic = aggregate["smooth_l1_only"]
    gaussian_selected = [run for run in runs if run["variant"] == "gaussian_nll"]
    sigma_scaled_aggregate = {
        metric: _aggregate(gaussian_selected, metric, "sigma_scaled_test")
        for metric in metric_names
    }

    def improvement(metric: str) -> float:
        baseline = deterministic[metric]["mean"]
        return 100.0 * (baseline - gaussian[metric]["mean"]) / max(abs(baseline), 1e-8)

    gaussian_runs = {run["seed"]: run for run in runs if run["variant"] == "gaussian_nll"}
    deterministic_runs = {
        run["seed"]: run for run in runs if run["variant"] == "smooth_l1_only"
    }
    paired_mae_wins = sum(
        gaussian_runs[seed]["additive_test"]["mae_N"]
        < deterministic_runs[seed]["additive_test"]["mae_N"]
        for seed in seeds
    )
    paired_rmse_wins = sum(
        gaussian_runs[seed]["additive_test"]["rmse_N"]
        < deterministic_runs[seed]["additive_test"]["rmse_N"]
        for seed in seeds
    )
    supports_gaussian = (
        improvement("mae_N") > 0.0
        and improvement("rmse_N") > 0.0
        and paired_mae_wins >= 2
        and paired_rmse_wins >= 2
    )
    conclusion = (
        "在相同 additive residual calibration 下，Gaussian objective 对点预测具有一致收益；"
        "这支持将 Gaussian NLL 作为训练目标保留，但不等价于证明 sigma 已完成概率校准。"
        if supports_gaussian else
        "当前三种子结果不足以证明 Gaussian objective 优于更简单的 SmoothL1-only；"
        "Gaussian NLL 可以保留为概率建模选择，但不能据此声称它带来稳定性能收益。"
    )
    comparison = {
        "supports_gaussian_training_objective": supports_gaussian,
        "paired_mae_wins_out_of": [paired_mae_wins, len(seeds)],
        "paired_rmse_wins_out_of": [paired_rmse_wins, len(seeds)],
        "mae_relative_improvement_percent": improvement("mae_N"),
        "rmse_relative_improvement_percent": improvement("rmse_N"),
        "width_relative_improvement_percent": improvement("mean_interval_width_N"),
        "sigma_scaled_coverage_percent": 100.0
        * sigma_scaled_aggregate["coverage"]["mean"],
        "sigma_scaled_width_reduction_percent": 100.0
        * (
            gaussian["mean_interval_width_N"]["mean"]
            - sigma_scaled_aggregate["mean_interval_width_N"]["mean"]
        )
        / gaussian["mean_interval_width_N"]["mean"],
        "supports_sigma_as_heteroscedastic_scale": (
            abs(sigma_scaled_aggregate["coverage"]["mean"] - 0.90) <= 0.02
            and sigma_scaled_aggregate["mean_interval_width_N"]["mean"]
            < gaussian["mean_interval_width_N"]["mean"]
        ),
        "conclusion": conclusion,
    }
    summary = {
        "experiment": "gaussian_nll_vs_smooth_l1_only",
        "task": "USB insertion",
        "seeds": seeds,
        "controlled_variables": [
            "same windows and episode split",
            "same Pi0.5 z_rl condition cache",
            "same Transformer architecture and initialization seed",
            "same 200 ms causal tactile history",
            "same sustained-contact masks",
            "same calibration-only q05/q95 additive interval",
        ],
        "test_used_for_training_or_calibration": False,
        "runs": runs,
        "aggregate": aggregate,
        "sigma_scaled_aggregate": sigma_scaled_aggregate,
        "comparison": comparison,
    }
    write_json(output / "summary.json", summary)
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "variant", "seed", "calibration_samples", "test_samples", "best_epoch",
            "mae_N", "rmse_N", "coverage", "mean_interval_width_N",
            "false_penalty_windows", "false_penalty_episodes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            metric = run["additive_test"]
            writer.writerow({
                "variant": run["variant"],
                "seed": run["seed"],
                "calibration_samples": run["calibration_samples"],
                "test_samples": run["test_samples"],
                "best_epoch": run["best_epoch"],
                **{key: metric[key] for key in fieldnames[5:]},
            })
    _plot_summary(summary, output)
    _plot_sigma_summary(summary, output)
    _write_html(summary, output)
    return summary
