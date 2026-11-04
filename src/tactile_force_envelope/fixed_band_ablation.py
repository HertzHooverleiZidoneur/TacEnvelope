from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ablation_recalibration import _best_state_matches, _sha256
from .bilateral_sigma_calibration import (
    _finite_sample_tail_quantile,
    bilateral_uncertainty_scale,
)
from .common import write_json
from .stable_contact_experiment import _gates_for_rows, _load_evaluation_context
from .transformer_train import _side_value


ALPHA = 0.10
LOWER_PROBABILITY = 0.05
UPPER_PROBABILITY = 0.95


def interval_score(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    alpha: float = ALPHA,
) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    target = np.asarray(target, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (target.shape == lower.shape == upper.shape):
        raise ValueError("target/lower/upper shapes must match")
    if np.any(upper < lower):
        raise ValueError("upper bound is below lower bound")
    return (
        upper
        - lower
        + (2.0 / alpha) * np.maximum(lower - target, 0.0)
        + (2.0 / alpha) * np.maximum(target - upper, 0.0)
    )


def _metrics(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    width = upper - lower
    covered = (target >= lower) & (target <= upper)
    scores = interval_score(target, lower, upper)
    return {
        "samples": int(target.size),
        "coverage": float(np.mean(covered)),
        "mean_width_N": float(np.mean(width)),
        "width_p05_N": float(np.quantile(width, 0.05)),
        "width_p50_N": float(np.quantile(width, 0.50)),
        "width_p95_N": float(np.quantile(width, 0.95)),
        "mean_interval_score_alpha_0.1_N": float(np.mean(scores)),
        "interval_score_p50_N": float(np.quantile(scores, 0.50)),
        "interval_score_p95_N": float(np.quantile(scores, 0.95)),
        "low_violation_windows": int(np.sum(target < lower)),
        "high_violation_windows": int(np.sum(target > upper)),
        "out_of_band_windows": int(np.sum(~covered)),
    }


def _window_identity(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("batch_id", "")),
        int(row["episode_index"]),
        int(row["window_index"]),
    )


def _fingerprint(rows: list[dict[str, Any]], mask: np.ndarray) -> str:
    selected = [_window_identity(row) for row, keep in zip(rows, mask, strict=True) if keep]
    return hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _aggregate(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = np.asarray([float(run["test"][metric]) for run in runs])
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def _resolve_checkpoint_path(
    config: dict[str, Any], workspace: Path, seed: int
) -> Path:
    ablation_checkpoint = (
        workspace
        / "gaussian_loss_ablation"
        / f"gaussian_nll_seed_{seed}"
        / "transformer_force_envelope_model.pt"
    )
    if ablation_checkpoint.is_file():
        return ablation_checkpoint

    configured_seed = int(config["training"]["seed"])
    source_checkpoint = Path(
        config["stable_contact"]["source_checkpoint"]
    ).expanduser().resolve()
    if seed == configured_seed and source_checkpoint.is_file():
        return source_checkpoint

    raise FileNotFoundError(
        "Required validation-best checkpoint is missing; refusing to retrain. "
        f"Checked {ablation_checkpoint} and source_checkpoint={source_checkpoint} "
        f"for configured seed {configured_seed}."
    )


def _plot_episode(
    output: Path,
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_lower: float,
    fixed_upper: float,
    reference_seed: int,
) -> dict[str, Any]:
    selected_positions = np.flatnonzero(mask)
    by_episode: dict[tuple[str, int], list[int]] = defaultdict(list)
    for compact_index, position in enumerate(selected_positions):
        row = rows[int(position)]
        by_episode[(str(row.get("batch_id", "")), int(row["episode_index"]))].append(
            compact_index
        )
    episode_key, positions = sorted(
        by_episode.items(), key=lambda item: (-len(item[1]), item[0])
    )[0]
    index = np.asarray(positions, dtype=np.int64)
    timestamps = np.asarray(
        [float(rows[int(selected_positions[position])]["start_timestamp"]) for position in index]
    )
    time_s = timestamps - timestamps[0]

    figure, (axis, detail) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.25)},
        constrained_layout=True,
    )
    axis.axhspan(
        fixed_lower,
        fixed_upper,
        color="#D8DEE9",
        alpha=0.72,
        label="Global fixed band",
        zorder=0,
    )
    axis.fill_between(
        time_s,
        lower[index],
        upper[index],
        color="#72B7A6",
        alpha=0.36,
        label="Adaptive envelope",
        zorder=1,
    )
    axis.plot(
        time_s,
        mu[index],
        color="#14857C",
        linestyle="--",
        linewidth=2.0,
        label="Predicted mean",
        zorder=2,
    )
    axis.plot(
        time_s,
        target[index],
        color="#222B38",
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        label="Observed bilateral load",
        zorder=3,
    )
    axis.set_ylabel("Bilateral average force (N)", fontsize=12)
    axis.set_title("Full range: global fixed band", loc="left", fontsize=12)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False, fontsize=10, loc="upper right")

    detail.fill_between(
        time_s,
        lower[index],
        upper[index],
        color="#72B7A6",
        alpha=0.40,
        label="Adaptive envelope",
        zorder=1,
    )
    detail.plot(
        time_s,
        mu[index],
        color="#14857C",
        linestyle="--",
        linewidth=2.0,
        label="Predicted mean",
        zorder=2,
    )
    detail.plot(
        time_s,
        target[index],
        color="#222B38",
        marker="o",
        markersize=4.2,
        linewidth=1.8,
        label="Observed bilateral load",
        zorder=3,
    )
    detail.set_title("Detail (zoomed y-axis): input-adaptive envelope", loc="left", fontsize=12)
    detail.set_xlabel("Time from first bilateral stable-contact window (s)", fontsize=12)
    detail.set_ylabel("Bilateral average force (N)", fontsize=12)
    detail_values = np.concatenate((target[index], lower[index], upper[index]))
    detail_low = float(np.quantile(detail_values, 0.10))
    detail_high = float(np.quantile(detail_values, 0.95))
    detail_pad = max(1.0, 0.08 * (detail_high - detail_low))
    detail.set_ylim(detail_low - detail_pad, detail_high + detail_pad)
    detail.grid(axis="y", alpha=0.25)
    detail.legend(ncol=3, frameon=False, fontsize=9.5, loc="upper right")
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"global_fixed_vs_adaptive_episode_seed{reference_seed}.{suffix}",
            dpi=240,
        )
    plt.close(figure)
    return {
        "selection_rule": "test episode with the largest number of selected windows",
        "batch_id": episode_key[0],
        "episode_index": episode_key[1],
        "windows": int(index.size),
        "duration_s": float(time_s[-1] if time_s.size else 0.0),
    }


def _mean_std(metric: dict[str, float], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{metric['mean'] * scale:.3f} +/- {metric['std'] * scale:.3f}{suffix}"


def _write_markdown(output: Path, report: dict[str, Any]) -> None:
    fixed = report["global_fixed_band"]
    fixed_test = fixed["test"]
    adaptive = report["adaptive_envelope"]
    aggregate = adaptive["aggregate"]
    coverage_change = 100.0 * (aggregate["coverage"]["mean"] - fixed_test["coverage"])
    width_reduction = 100.0 * (
        fixed_test["mean_width_N"] - aggregate["mean_width_N"]["mean"]
    ) / fixed_test["mean_width_N"]
    score_reduction = 100.0 * (
        fixed_test["mean_interval_score_alpha_0.1_N"]
        - aggregate["mean_interval_score_alpha_0.1_N"]["mean"]
    ) / fixed_test["mean_interval_score_alpha_0.1_N"]
    seed_rows = []
    for run in adaptive["runs"]:
        metrics = run["test"]
        calibration = run["calibration"]
        seed_rows.append(
            f"| {run['seed']} | {calibration['alpha05']:.6f} | "
            f"{calibration['alpha95']:.6f} | {metrics['coverage'] * 100:.2f}% | "
            f"{metrics['mean_width_N']:.4f} | "
            f"{metrics['mean_interval_score_alpha_0.1_N']:.4f} | "
            f"{metrics['width_p05_N']:.4f}/{metrics['width_p50_N']:.4f}/"
            f"{metrics['width_p95_N']:.4f} |"
        )
    task_name = report["task"]
    calibration_samples = fixed["calibration_samples"]
    test_samples = fixed_test["samples"]
    seed_text = "/".join(str(run["seed"]) for run in adaptive["runs"])
    document = f"""# Global Fixed Band 与 Adaptive Envelope 消融

## 协议

- 任务：{task_name}；按 episode 固定划分；模型未重新训练。
- 两种方法均使用同一批 {calibration_samples} 个 calibration 和 {test_samples} 个 test 双侧稳定接触窗口。
- Global Fixed Band 只计算一次；Adaptive Envelope 使用 seed {seed_text}。
- Test target 未参与固定区间、alpha 或其他参数的拟合。
- 当前 test split 此前参与过方法诊断，因此属于 development evaluation。

执行命令：

```bash
{report['exact_command']}
```

## 固定区间

Train 双侧稳定接触窗口数为 {fixed['train_stable_contact_samples']}，其平均力为
`{fixed['train_center_N']:.4f} N`。在 calibration split 上计算
`r=y_avg-mu_train` 的有限样本 q05/q95：

- q05 = `{fixed['q05_residual_N']:.4f} N`
- q95 = `{fixed['q95_residual_N']:.4f} N`
- `L_fixed = {fixed['lower_fixed_N']:.4f} N`
- `U_fixed = {fixed['upper_fixed_N']:.4f} N`

该结果与直接对 calibration target 取有限样本 q05/q95 完全等价。

## 主结果

| Method | Coverage (target: 90%) | Mean Width (N) | Interval Score 0.1 (N) |
|---|---:|---:|---:|
| Global Fixed Band | {fixed_test['coverage'] * 100:.2f}% | {fixed_test['mean_width_N']:.3f} | {fixed_test['mean_interval_score_alpha_0.1_N']:.3f} |
| Adaptive Envelope | {_mean_std(aggregate['coverage'], percent=True)} | {_mean_std(aggregate['mean_width_N'])} | {_mean_std(aggregate['mean_interval_score_alpha_0.1_N'])} |

相对固定区间，自适应包络的 Coverage 更接近目标并提高 **{coverage_change:.2f} 个百分点**，
平均宽度减少 **{width_reduction:.2f}%**，Interval Score 降低 **{score_reduction:.2f}%**。

## Adaptive Per Seed

| Seed | alpha05 | alpha95 | Coverage | Mean Width | Interval Score | Width P05/P50/P95 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

## 解释边界

固定区间必须覆盖该任务中跨阶段、跨 episode 的整体负载分布，因此 calibration
得到 `[{fixed['lower_fixed_N']:.2f}, {fixed['upper_fixed_N']:.2f}] N` 的全局范围；
它在 test 上漏掉 {fixed_test['out_of_band_windows']}/{fixed_test['samples']} 个窗口。
自适应包络的中心和宽度随当前 VLA 表征、机器人状态、候选动作及触觉历史变化，
在接近 90% Coverage 时显著缩窄区间并降低 Interval Score。

这项离线消融支持“自适应包络更有效地描述成功示教中的条件接触负载”，
但不单独证明它会提高 Online RL 成功率；后者需要对应的在线奖励对照实验。
"""
    (output / "GLOBAL_FIXED_VS_ADAPTIVE_ABLATION_ZH.md").write_text(
        document, encoding="utf-8"
    )


def run_fixed_vs_adaptive_ablation(
    config: dict[str, Any],
    *,
    config_path: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    output = workspace / "stable_evaluation" / "fixed_vs_adaptive"
    output.mkdir(parents=True, exist_ok=True)
    adaptive_runs: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None

    for seed in seeds:
        checkpoint_path = _resolve_checkpoint_path(config, workspace, seed)
        checkpoint, rows, grouped, predictions, history, history_mask, _ = (
            _load_evaluation_context(config, checkpoint_path)
        )
        matches_best, best_path = _best_state_matches(checkpoint_path)
        if not matches_best:
            raise RuntimeError(f"Checkpoint is not the validation-best state: {checkpoint_path}")
        masks = {}
        for split in ("calibration", "test"):
            gates = _gates_for_rows(grouped[split], history, history_mask, config)
            masks[split] = gates[:, 0] & gates[:, 1]

        values = {}
        for split in ("calibration", "test"):
            left = predictions[split]["left"]
            right = predictions[split]["right"]
            values[split] = {
                "target": 0.5 * (left["target"] + right["target"]),
                "mu": 0.5 * (left["mu"] + right["mu"]),
                "scale": bilateral_uncertainty_scale(
                    left["sigma"], right["sigma"], "mean"
                ),
            }

        cal_mask = masks["calibration"]
        test_mask = masks["test"]
        cal_score = (
            values["calibration"]["target"][cal_mask]
            - values["calibration"]["mu"][cal_mask]
        ) / np.maximum(values["calibration"]["scale"][cal_mask], 1e-6)
        alpha05, rank05 = _finite_sample_tail_quantile(
            cal_score, LOWER_PROBABILITY, upper=False
        )
        alpha95, rank95 = _finite_sample_tail_quantile(
            cal_score, UPPER_PROBABILITY, upper=True
        )
        test_target = values["test"]["target"][test_mask]
        test_mu = values["test"]["mu"][test_mask]
        test_scale = values["test"]["scale"][test_mask]
        lower = test_mu + alpha05 * test_scale
        upper = test_mu + alpha95 * test_scale
        run = {
            "seed": int(seed),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "best_validation_state": best_path,
            "best_epoch": int(checkpoint["best_epoch"]),
            "calibration_samples": int(cal_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "calibration_window_fingerprint": _fingerprint(
                grouped["calibration"], cal_mask
            ),
            "test_window_fingerprint": _fingerprint(grouped["test"], test_mask),
            "calibration": {
                "score": "(y_avg - mu_avg) / max(u_mean, 1e-6)",
                "alpha05": alpha05,
                "alpha95": alpha95,
                "alpha05_rank_one_based": rank05,
                "alpha95_rank_one_based": rank95,
            },
            "test": _metrics(test_target, lower, upper),
        }
        adaptive_runs.append(run)
        if seed == 44 or reference is None:
            reference = {
                "rows": rows,
                "grouped": grouped,
                "history": history,
                "history_mask": history_mask,
                "masks": masks,
                "values": values,
                "test_target": test_target,
                "test_mu": test_mu,
                "test_lower": lower,
                "test_upper": upper,
                "seed": seed,
            }

    assert reference is not None
    fingerprints = {
        (run["calibration_window_fingerprint"], run["test_window_fingerprint"])
        for run in adaptive_runs
    }
    if len(fingerprints) != 1:
        raise RuntimeError("Adaptive seeds do not use identical selected windows")

    rows = reference["rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    train_gates = _gates_for_rows(
        train_rows, reference["history"], reference["history_mask"], config
    )
    train_mask = train_gates[:, 0] & train_gates[:, 1]
    train_target = np.asarray([
        0.5 * (
            _side_value(row, config, "left", "filtered_top2_mean")
            + _side_value(row, config, "right", "filtered_top2_mean")
        )
        for row in train_rows
    ])
    train_center = float(np.mean(train_target[train_mask]))

    calibration_target = reference["values"]["calibration"]["target"]
    calibration_mask = reference["masks"]["calibration"]
    fixed_residual = calibration_target[calibration_mask] - train_center
    q05, fixed_rank05 = _finite_sample_tail_quantile(
        fixed_residual, LOWER_PROBABILITY, upper=False
    )
    q95, fixed_rank95 = _finite_sample_tail_quantile(
        fixed_residual, UPPER_PROBABILITY, upper=True
    )
    fixed_lower = train_center + q05
    fixed_upper = train_center + q95
    test_target = reference["test_target"]
    fixed_lower_array = np.full_like(test_target, fixed_lower)
    fixed_upper_array = np.full_like(test_target, fixed_upper)
    fixed_metrics = _metrics(test_target, fixed_lower_array, fixed_upper_array)

    direct_q05, direct_rank05 = _finite_sample_tail_quantile(
        calibration_target[calibration_mask], LOWER_PROBABILITY, upper=False
    )
    direct_q95, direct_rank95 = _finite_sample_tail_quantile(
        calibration_target[calibration_mask], UPPER_PROBABILITY, upper=True
    )
    if not np.allclose([fixed_lower, fixed_upper], [direct_q05, direct_q95], atol=1e-10):
        raise RuntimeError("Residual and direct fixed-band definitions are not equivalent")

    aggregate_metrics = (
        "coverage",
        "mean_width_N",
        "width_p05_N",
        "width_p50_N",
        "width_p95_N",
        "mean_interval_score_alpha_0.1_N",
    )
    adaptive_aggregate = {
        metric: _aggregate(adaptive_runs, metric) for metric in aggregate_metrics
    }
    plot = _plot_episode(
        output,
        reference["grouped"]["test"],
        reference["masks"]["test"],
        reference["test_target"],
        reference["test_mu"],
        reference["test_lower"],
        reference["test_upper"],
        fixed_lower,
        fixed_upper,
        int(reference["seed"]),
    )

    repository = Path(__file__).resolve().parents[2]
    command = (
        f"cd {repository} && PYTHONPATH=src {sys.executable} "
        "scripts/evaluation/run_fixed_vs_adaptive_envelope_ablation.py "
        f"--config {config_path} --device {config['training']['device']} --seeds "
        + " ".join(str(seed) for seed in seeds)
    )
    report = {
        "schema_version": 1,
        "experiment": "global fixed band versus adaptive tactile envelope",
        "task": str(config["data"].get("task_override") or workspace.parent.name),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "exact_command": command,
        "models_retrained": False,
        "nominal_coverage": 0.90,
        "interval_score_alpha": ALPHA,
        "target_definition": "y_avg = 0.5 * (y_left_top2_mean + y_right_top2_mean)",
        "selection": "stable_prior_left AND stable_prior_right",
        "selection_uses_future_target": False,
        "calibration_source": "calibration split only",
        "test_used_for_calibration": False,
        "test_status": "development evaluation",
        "nonnegative_bound_clipping": False,
        "bound_clipping_reason": "The deployed adaptive envelope does not clip its lower bound.",
        "global_fixed_band": {
            "definition": (
                "train stable-contact mean plus calibration residual finite-sample "
                "q05/q95; exactly equivalent to calibration-target q05/q95"
            ),
            "train_stable_contact_samples": int(train_mask.sum()),
            "train_center_N": train_center,
            "calibration_samples": int(calibration_mask.sum()),
            "q05_residual_N": q05,
            "q95_residual_N": q95,
            "q05_rank_one_based": fixed_rank05,
            "q95_rank_one_based": fixed_rank95,
            "lower_fixed_N": fixed_lower,
            "upper_fixed_N": fixed_upper,
            "direct_calibration_q05_N": direct_q05,
            "direct_calibration_q95_N": direct_q95,
            "direct_ranks_one_based": [direct_rank05, direct_rank95],
            "residual_and_direct_definitions_equal": True,
            "test": fixed_metrics,
        },
        "adaptive_envelope": {
            "definition": "[mu_avg + alpha05*u_mean, mu_avg + alpha95*u_mean]",
            "uncertainty_scale": "u_mean = 0.5 * (sigma_left + sigma_right)",
            "uncertainty_scale_is_standard_deviation": False,
            "runs": adaptive_runs,
            "aggregate": adaptive_aggregate,
        },
        "consistency": {
            "identical_calibration_windows_across_seeds": True,
            "identical_test_windows_across_seeds": True,
            "calibration_window_fingerprint": next(iter(fingerprints))[0],
            "test_window_fingerprint": next(iter(fingerprints))[1],
            "same_targets_for_both_methods": True,
            "same_calibration_and_test_splits": True,
        },
        "representative_plot": plot,
        "interpretation": {
            "primary_question": (
                "At comparable coverage, does conditioning the band on the current "
                "interaction reduce width and interval score?"
            ),
            "does_not_establish_online_rl_gain": True,
            "strict_conformal_guarantee_claimed_for_fixed_band": False,
        },
    }
    write_json(output / "global_fixed_vs_adaptive_envelope_ablation.json", report)

    with (output / "adaptive_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "seed", "calibration_samples", "test_samples", "alpha05", "alpha95",
            "coverage", "mean_width_N", "width_p05_N", "width_p50_N", "width_p95_N",
            "mean_interval_score_alpha_0.1_N", "checkpoint", "checkpoint_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in adaptive_runs:
            writer.writerow({
                "seed": run["seed"],
                "calibration_samples": run["calibration_samples"],
                "test_samples": run["test_samples"],
                "alpha05": run["calibration"]["alpha05"],
                "alpha95": run["calibration"]["alpha95"],
                **{key: run["test"][key] for key in fields[5:11]},
                "checkpoint": run["checkpoint"],
                "checkpoint_sha256": run["checkpoint_sha256"],
            })
    _write_markdown(output, report)
    return report
