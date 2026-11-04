#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.fixed_band_ablation import run_fixed_vs_adaptive_ablation


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/paper/global_fixed_vs_adaptive_three_tasks"
TASKS = (
    (
        "USB insertion",
        ROOT / "configs/experiments/pi05_jax_zrl_stable_contact.yaml",
        (42, 43, 44),
    ),
    (
        "Ink squeezing",
        ROOT / "configs/experiments/pi05_jax_zrl_ink_beaker.yaml",
        (44,),
    ),
    (
        "Pastry transfer",
        ROOT / "configs/experiments/pi05_jax_zrl_pastry_A298.yaml",
        (44,),
    ),
)


def _seed_run(report: dict, seed: int = 44) -> dict:
    for run in report["adaptive_envelope"]["runs"]:
        if int(run["seed"]) == seed:
            return run
    raise RuntimeError(f"Adaptive result does not contain seed {seed}")


def _plot(rows: list[dict]) -> None:
    labels = [row["task"] for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    metrics = (
        ("coverage", "Coverage (%)", 100.0),
        ("mean_width_N", "Mean width (N)", 1.0),
        ("interval_score_N", "Interval score (N)", 1.0),
    )
    for axis, (key, ylabel, scale) in zip(axes, metrics, strict=True):
        fixed = np.asarray([row[f"fixed_{key}"] for row in rows]) * scale
        adaptive = np.asarray([row[f"adaptive_{key}"] for row in rows]) * scale
        axis.bar(x - width / 2, fixed, width, color="#7C8798", label="Global fixed")
        axis.bar(x + width / 2, adaptive, width, color="#16877E", label="Adaptive")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if key == "coverage":
            axis.axhline(90.0, color="#B23A48", linestyle="--", linewidth=1.3)
            axis.set_ylim(0.0, 100.0)
        for positions, values in ((x - width / 2, fixed), (x + width / 2, adaptive)):
            for position, value in zip(positions, values, strict=True):
                axis.text(
                    position,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0].legend(frameon=False, loc="lower right")
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"three_task_fixed_vs_adaptive.{suffix}", dpi=240)
    plt.close(figure)


def _markdown(rows: list[dict]) -> str:
    table = []
    effect_table = []
    for row in rows:
        table.append(
            f"| {row['task']} | {row['calibration_samples']} | {row['test_samples']} | "
            f"{row['fixed_lower_N']:.3f}--{row['fixed_upper_N']:.3f} | "
            f"{100.0 * row['fixed_coverage']:.2f}% | {row['fixed_mean_width_N']:.3f} | "
            f"{row['fixed_interval_score_N']:.3f} | "
            f"{100.0 * row['adaptive_coverage']:.2f}% | {row['adaptive_mean_width_N']:.3f} | "
            f"{row['adaptive_interval_score_N']:.3f} |"
        )
        effect_table.append(
            f"| {row['task']} | {row['coverage_delta_pp']:+.2f} pp | "
            f"{row['width_reduction_percent']:.2f}% | "
            f"{row['interval_score_reduction_percent']:.2f}% |"
        )
    return f"""# 三任务 Global Fixed Band vs. Adaptive Envelope 消融

## 协议

- 所有任务均按各自固定的 episode split 评估。
- 两种方法严格使用相同的双侧 stable-contact calibration/test 窗口。
- 区间参数仅由 train/calibration split 确定；test target 未参与校准。
- 没有重新训练 Envelope Expert。
- 为保证跨任务可比性，主表统一报告 seed 44。USB 的 seed 42/43/44 统计保留在其任务报告中。

## 结果

| Task | Cal. windows | Test windows | Fixed band (N) | Fixed cov. | Fixed width | Fixed IS | Adaptive cov. | Adaptive width | Adaptive IS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Coverage 的目标值为 90%。Interval Score 使用 alpha=0.1，越低越好。
固定区间只计算一次，不将其复制成三个伪随机种子。自适应区间采用
`mu_avg + alpha * u_mean`，其中 `u_mean=(sigma_left+sigma_right)/2` 是经验不确定性尺度，
不称为双侧平均力的标准差。

## 相对变化

| Task | Coverage change | Width reduction | Interval-score reduction |
|---|---:|---:|---:|
{chr(10).join(effect_table)}

## 结论边界

该消融比较的是成功示教测试窗口上的负载区间刻画能力。它可以支持自适应 envelope
相对全局固定阈值的离线优势，但不能单独证明 Online RL 成功率提升；后者需要真机在线对照。
当前 test episodes 已用于此前分析，最终论文结果仍建议用独立 blind episodes 复核。
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    task_reports: dict[str, dict] = {}
    rows: list[dict] = []
    for label, config_path, seeds in TASKS:
        config = load_config(config_path)
        config["training"]["device"] = "cpu"
        report = run_fixed_vs_adaptive_ablation(
            config, config_path=config_path, seeds=seeds
        )
        task_reports[label] = report
        fixed = report["global_fixed_band"]
        fixed_test = fixed["test"]
        adaptive = _seed_run(report)
        adaptive_test = adaptive["test"]
        row = {
                "task": label,
                "calibration_samples": fixed["calibration_samples"],
                "test_samples": fixed_test["samples"],
                "fixed_lower_N": fixed["lower_fixed_N"],
                "fixed_upper_N": fixed["upper_fixed_N"],
                "fixed_coverage": fixed_test["coverage"],
                "fixed_mean_width_N": fixed_test["mean_width_N"],
                "fixed_interval_score_N": fixed_test[
                    "mean_interval_score_alpha_0.1_N"
                ],
                "adaptive_seed": adaptive["seed"],
                "adaptive_coverage": adaptive_test["coverage"],
                "adaptive_mean_width_N": adaptive_test["mean_width_N"],
                "adaptive_interval_score_N": adaptive_test[
                    "mean_interval_score_alpha_0.1_N"
                ],
            }
        row["coverage_delta_pp"] = 100.0 * (
            row["adaptive_coverage"] - row["fixed_coverage"]
        )
        row["width_reduction_percent"] = 100.0 * (
            row["fixed_mean_width_N"] - row["adaptive_mean_width_N"]
        ) / row["fixed_mean_width_N"]
        row["interval_score_reduction_percent"] = 100.0 * (
            row["fixed_interval_score_N"] - row["adaptive_interval_score_N"]
        ) / row["fixed_interval_score_N"]
        rows.append(row)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_retrained": False,
        "test_used_for_calibration": False,
        "nominal_coverage": 0.90,
        "primary_comparison_seed": 44,
        "rows": rows,
        "task_report_paths": {
            label: str(
                Path(report["output_dir"])
                / "global_fixed_vs_adaptive_envelope_ablation.json"
            )
            for label, report in task_reports.items()
        },
    }
    write_json(OUTPUT / "three_task_fixed_vs_adaptive_summary.json", payload)
    with (OUTPUT / "three_task_fixed_vs_adaptive_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "THREE_TASK_FIXED_VS_ADAPTIVE_ZH.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    _plot(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
