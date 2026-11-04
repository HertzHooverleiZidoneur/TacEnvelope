#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.dual_rate_reward_ablation import analyze_paired_h9_windows
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/paper/dual_rate_validation_ablation"
TASKS = (
    {
        "task": "USB",
        "config": ROOT / "configs/experiments/pi05_jax_zrl_stable_contact.yaml",
        "workspace": "franka_tactile310",
        "force_scale": 5.0,
    },
    {
        "task": "Ink",
        "config": ROOT / "configs/experiments/pi05_jax_zrl_ink_beaker.yaml",
        "workspace": "franka_ink_beaker_tactile287",
        "force_scale": 2.0,
    },
    {
        "task": "Pastry",
        "config": ROOT / "configs/experiments/pi05_jax_zrl_pastry_A298.yaml",
        "workspace": "franka_pastry_tactile_A298",
        "force_scale": 2.0,
    },
)


def _artifact(task: dict, name: str) -> Path:
    return ROOT / (
        f"outputs/training/{task['workspace']}/pi05_jax_zrl_envelope_v1/"
        f"sustained_contact_seed_44/stable_evaluation/{name}"
    )


def _evaluate_task(task: dict) -> tuple[dict, list[dict], list[dict]]:
    config = copy.deepcopy(load_config(task["config"]))
    config["training"]["device"] = "cpu"
    config["training"]["num_workers"] = 0
    checkpoint_path = _artifact(task, "stable_contact_calibrated_checkpoint.pt")
    calibration_path = _artifact(task, "bilateral_sigma_calibration.json")
    _, _, grouped, predictions, history, history_mask, _ = _load_evaluation_context(
        config,
        checkpoint_path,
        split_names=("validation",),
    )
    rows = grouped["validation"]
    gates = _gates_for_rows(rows, history, history_mask, config)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("scale_method") != "mean":
        raise RuntimeError(f"{task['task']} does not use u_mean calibration")
    alpha05 = float(calibration["lower_alpha"])
    alpha95 = float(calibration["upper_alpha"])
    left = predictions["validation"]["left"]
    right = predictions["validation"]["right"]

    window_rows = []
    for position, row in enumerate(rows):
        observed = 0.5 * (
            float(left["target"][position]) + float(right["target"][position])
        )
        center = 0.5 * (
            float(left["mu"][position]) + float(right["mu"][position])
        )
        uncertainty_scale = 0.5 * (
            float(left["sigma"][position]) + float(right["sigma"][position])
        )
        window_rows.append(
            {
                "episode": int(row["episode_index"]),
                "window_index": int(row["window_index"]),
                "timestamp": float(row["start_timestamp"]),
                "observed": observed,
                "lower": center + alpha05 * uncertainty_scale,
                "upper": center + alpha95 * uncertainty_scale,
                "gate": bool(gates[position, 0] and gates[position, 1]),
                "success": 1,
            }
        )

    report, transitions, episodes = analyze_paired_h9_windows(
        window_rows,
        task=task["task"],
        source_file=str(checkpoint_path),
        force_scale=float(task["force_scale"]),
        source_protocol="validation split paired H9 counterfactual",
    )
    report.update(
        {
            "evaluation_split": "validation",
            "episode_split_seed": 44,
            "checkpoint_selected_on_validation": True,
            "calibration_file": str(calibration_path),
            "calibration_split_only": True,
            "evaluation_targets_used_for_calibration": False,
            "coverage_pairing": (
                "fresh and held bands use the identical window-2 subset "
                "where the fresh bilateral gate is active"
            ),
        }
    )
    return report, transitions, episodes


def _summary_rows(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        envelope = report["window_2_envelope"]
        held_coverage = float(envelope["held_band_coverage"])
        fresh_coverage = float(envelope["fresh_coverage"])
        held_score = float(envelope["held_band_interval_score_mean_N"])
        fresh_score = float(envelope["fresh_interval_score_mean_N"])
        material = report["reward"]["single_vs_dual_window_2_material"]
        rows.append(
            {
                "task": report["task"],
                "validation_episodes": report["episodes"],
                "paired_e18_transitions": report["accepted_transitions"],
                "window2_gate_samples": envelope["gate_active_samples"],
                "single_rate_coverage_percent": 100.0 * held_coverage,
                "dual_rate_coverage_percent": 100.0 * fresh_coverage,
                "coverage_gain_percentage_points": 100.0
                * (fresh_coverage - held_coverage),
                "single_rate_interval_score_N": held_score,
                "dual_rate_interval_score_N": fresh_score,
                "interval_score_reduction_percent": 100.0
                * (held_score - fresh_score)
                / held_score,
                "missed_triggers_relative_to_fresh": material[
                    "missed_penalty_windows"
                ],
                "extra_triggers_relative_to_fresh": material[
                    "false_penalty_windows"
                ],
            }
        )
    return rows


def _plot(rows: list[dict]) -> None:
    labels = [row["task"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    metrics = (
        (
            "single_rate_coverage_percent",
            "dual_rate_coverage_percent",
            "Window-2 coverage (%)",
        ),
        (
            "single_rate_interval_score_N",
            "dual_rate_interval_score_N",
            r"Interval score $IS_{0.1}$ (N)",
        ),
    )
    for axis, (single_key, dual_key, ylabel) in zip(axes, metrics, strict=True):
        single = np.asarray([row[single_key] for row in rows])
        dual = np.asarray([row[dual_key] for row in rows])
        axis.bar(x - width / 2, single, width, color="#7C8798", label="Single-rate")
        axis.bar(x + width / 2, dual, width, color="#16877E", label="Dual-rate")
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for patch in axis.patches:
            value = patch.get_height()
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"dual_rate_validation_ablation.{suffix}", dpi=240)
    plt.close(figure)


def _latex(rows: list[dict]) -> str:
    coverage_gains = [row["coverage_gain_percentage_points"] for row in rows]
    score_reductions = [row["interval_score_reduction_percent"] for row in rows]
    table_rows = []
    for row in rows:
        table_rows.extend(
            (
                f"{row['task']} & Single-rate & "
                f"{row['single_rate_coverage_percent']:.2f} & "
                f"{row['single_rate_interval_score_N']:.3f} \\\\",
                f"& Dual-rate & {row['dual_rate_coverage_percent']:.2f} & "
                f"\\textbf{{{row['dual_rate_interval_score_N']:.3f}}} \\\\",
                "\\hline",
            )
        )
    return rf"""We evaluate dual-rate inference through paired offline
comparisons with a single-rate counterfactual on the validation split
across all three tasks. Single-rate retains the first window's envelope
and contact gate in the second window, while dual-rate refreshes both.
Both settings use identical trajectories, expert models, calibration
coefficients, and reward parameters.

As shown in Table~\ref{{tab:dual_rate_ablation}}, dual-rate inference
increases second-window coverage by {min(coverage_gains):.2f}--{max(coverage_gains):.2f}
percentage points and reduces interval score by approximately
{min(score_reductions):.0f}--{max(score_reductions):.0f}\% relative to single-rate
evaluation. These results show that higher-rate tactile evaluation more
faithfully tracks contact-load changes within an action-execution cycle.

\begin{{table}}[!t]
\centering
\caption{{Paired offline comparison of single-rate and dual-rate tactile
envelope evaluation on the validation split.}}
\label{{tab:dual_rate_ablation}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}llcc@{{}}}}
\hline
Task & Setting & Cov. (\%) & $IS_{{0.1}}$ (N) $\downarrow$ \\
\hline
{chr(10).join(table_rows)}
\end{{tabular}}

\par\vspace{{3pt}}
\begin{{minipage}}{{\columnwidth}}
\footnotesize\raggedright
$IS_{{0.1}}$ denotes the mean interval score at a miscoverage level of
0.1, corresponding to nominal 90\% prediction intervals. Coverage for
both settings is computed on the same second-window subset selected by
the refreshed bilateral contact gate. The checkpoint was selected using
the validation split; these values are therefore development-set ablation
results rather than an unbiased final generalization estimate.
\end{{minipage}}
\end{{table}}
"""


def _markdown(rows: list[dict]) -> str:
    table = "\n".join(
        f"| {row['task']} | {row['validation_episodes']} | "
        f"{row['paired_e18_transitions']} | {row['window2_gate_samples']} | "
        f"{row['single_rate_coverage_percent']:.2f}% | "
        f"{row['dual_rate_coverage_percent']:.2f}% | "
        f"{row['coverage_gain_percentage_points']:+.2f} pp | "
        f"{row['single_rate_interval_score_N']:.3f} | "
        f"{row['dual_rate_interval_score_N']:.3f} | "
        f"{row['interval_score_reduction_percent']:.2f}% | "
        f"{row['missed_triggers_relative_to_fresh']} | "
        f"{row['extra_triggers_relative_to_fresh']} |"
        for row in rows
    )
    return f"""# 三任务 Dual-rate Validation 配对反事实消融

## 实验协议

- 三个任务全部只使用各自固定 seed-44 episode split 中的 validation 窗口。
- Dual-rate 在第二个 H9 窗口刷新 envelope 与 bilateral gate。
- Single-rate 在第二个 H9 窗口保持第一个窗口的 envelope 与 gate。
- 两种设置共享同一轨迹、实测双侧 Top2Mean 力、Expert、calibration 系数和奖励参数。
- calibration 系数仍只来自 calibration split，没有用 validation target 重校准。
- Coverage 对两种设置使用相同的 fresh-gate window-2 子集，以隔离 envelope 过时效应。
- 漏触发/额外触发以 fresh Dual-rate reward 为操作参考，不是独立人工标签。
- 每个 H9 必须覆盖完整 300 ms，且左右传感器各至少有4个有效样本；不连续的 H9 不配成 E18。

## 结果

| Task | Val. eps. | E18 pairs | Gate windows | Single cov. | Dual cov. | Gain | Single IS | Dual IS | IS reduction | Missed | Extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 结论边界

这里的力真值是 validation 窗口内传感器实测的双侧平均 Top2Mean，而不是
“validation 本身作为真值”。由于同一 validation split 曾用于 early stopping 和
checkpoint 选择，该表属于开发集消融，可能有选择偏差。它可以支持相同轨迹上的
reward fidelity 结论，但最终泛化指标仍应报告 test 或新的 blind episodes；
任务成功率提升需要真实在线 A/B。
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    transitions = []
    episodes = []
    for task in TASKS:
        report, task_transitions, task_episodes = _evaluate_task(task)
        reports.append(report)
        transitions.extend(task_transitions)
        episodes.extend(task_episodes)
    rows = _summary_rows(reports)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "validation",
        "episode_split_seed": 44,
        "models_retrained": False,
        "calibration_split_only": True,
        "all_tasks_are_paired_counterfactual": True,
        "single_rate_robot_reexecuted": False,
        "checkpoint_selected_on_evaluation_split": True,
        "valid_h9_definition": (
            "complete 300-ms window with at least four valid samples per side"
        ),
        "pair_definition": (
            "same episode, consecutive window index, anchor gap 0.25--0.38 s, "
            "non-overlapping H9 pairs"
        ),
        "rows": rows,
        "reports": reports,
    }
    write_json(OUTPUT / "summary.json", payload)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "paired_transitions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    with (OUTPUT / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    (OUTPUT / "DUAL_RATE_VALIDATION_ABLATION_ZH.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    (OUTPUT / "dual_rate_validation_table.tex").write_text(
        _latex(rows), encoding="utf-8"
    )
    _plot(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
