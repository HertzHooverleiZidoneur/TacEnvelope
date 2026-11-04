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
from tactile_force_envelope.dual_rate_reward_ablation import (
    analyze_dual_rate_log,
    analyze_paired_h9_windows,
)
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/paper/dual_rate_reward_ablation"
RUNS = (
    {
        "task": "Ink squeezing",
        # The archived reward values reproduce the historical 5 N sidecar scale.
        "force_scale": 5.0,
        "path": Path(
            "/path/to/franka_pi05_jax_online_rl_runs/"
            "pi05_ink_tactile_rtc18_20260910_101551/tactile_chunk_metrics.jsonl"
        ),
    },
    {
        "task": "Pastry transfer",
        "force_scale": 2.0,
        "path": Path(
            "/path/to/franka_pi05_jax_online_rl_runs/"
            "pi05_pastry_tactile_rtc18_20260827_110350/tactile_chunk_metrics.jsonl"
        ),
    },
)
USB = {
    "task": "USB insertion",
    "force_scale": 2.0,
    "config": ROOT / "configs/experiments/pi05_jax_zrl_stable_contact.yaml",
    "checkpoint": ROOT / (
        "outputs/training/franka_tactile310/pi05_jax_zrl_envelope_v1/"
        "sustained_contact_seed_44/stable_evaluation/"
        "stable_contact_calibrated_checkpoint.pt"
    ),
    "calibration": ROOT / (
        "outputs/training/franka_tactile310/pi05_jax_zrl_envelope_v1/"
        "sustained_contact_seed_44/stable_evaluation/"
        "bilateral_sigma_calibration.json"
    ),
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _analyze_usb_test() -> tuple[dict, list[dict], list[dict]]:
    config = copy.deepcopy(load_config(USB["config"]))
    config["training"]["num_workers"] = 0
    checkpoint, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, USB["checkpoint"])
    )
    del checkpoint
    rows = grouped["test"]
    gates = _gates_for_rows(rows, history, history_mask, config)
    calibration = json.loads(USB["calibration"].read_text(encoding="utf-8"))
    if calibration.get("scale_method") != "mean":
        raise RuntimeError("USB dual-rate ablation expects u_mean calibration")
    alpha05 = float(calibration["lower_alpha"])
    alpha95 = float(calibration["upper_alpha"])
    left = predictions["test"]["left"]
    right = predictions["test"]["right"]

    paired_rows = []
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
        paired_rows.append(
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
        paired_rows,
        task=USB["task"],
        source_file=str(USB["checkpoint"]),
        force_scale=float(USB["force_scale"]),
        source_protocol="held-out test split paired H9 counterfactual",
    )
    report["split"] = "test"
    report["episode_split_seed"] = 44
    report["calibration_file"] = str(USB["calibration"])
    report["calibration_uses_test"] = False
    report["test_targets_used_to_fit_envelope"] = False
    report["paired_window_selection"] = (
        "same episode, consecutive window_index, anchor gap in [0.25, 0.38] s; "
        "non-overlapping pairs"
    )
    return report, transitions, episodes


def _plot_summary(reports: list[dict]) -> None:
    labels = [report["task"] for report in reports]
    x = np.arange(len(reports))
    width = 0.35
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), constrained_layout=True)

    fresh_coverage = [
        100.0 * report["window_2_envelope"]["fresh_coverage"] for report in reports
    ]
    held_coverage = [
        100.0 * report["window_2_envelope"]["held_band_coverage"] for report in reports
    ]
    axes[0].bar(x - width / 2, held_coverage, width, color="#7C8798", label="600-ms held")
    axes[0].bar(x + width / 2, fresh_coverage, width, color="#16877E", label="300-ms fresh")
    axes[0].set_ylabel("Window-2 coverage (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(frameon=False, fontsize=9)

    missed = [
        report["reward"]["single_vs_dual_window_2_material"]["missed_penalty_windows"]
        for report in reports
    ]
    false = [
        report["reward"]["single_vs_dual_window_2_material"]["false_penalty_windows"]
        for report in reports
    ]
    axes[1].bar(x - width / 2, missed, width, color="#C84755", label="Missed")
    axes[1].bar(x + width / 2, false, width, color="#3478B8", label="False")
    axes[1].set_ylabel("Material penalty windows")
    axes[1].legend(frameon=False, fontsize=9)

    held_score = [
        report["window_2_envelope"]["held_band_interval_score_mean_N"]
        for report in reports
    ]
    fresh_score = [
        report["window_2_envelope"]["fresh_interval_score_mean_N"]
        for report in reports
    ]
    axes[2].bar(x - width / 2, held_score, width, color="#7C8798", label="600-ms held")
    axes[2].bar(x + width / 2, fresh_score, width, color="#16877E", label="300-ms fresh")
    axes[2].set_ylabel("Window-2 interval score (N)")

    for axis in axes:
        axis.set_xticks(x, labels, rotation=16, ha="right")
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
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"dual_rate_reward_ablation_summary.{suffix}", dpi=240)
    plt.close(figure)


def _plot_representative(task_slug: str, transitions: list[dict]) -> dict:
    by_episode: dict[int, list[dict]] = {}
    for row in transitions:
        by_episode.setdefault(row["episode"], []).append(row)
    episode, rows = max(
        by_episode.items(),
        key=lambda item: sum(abs(row["transition_soft_reward_error"]) for row in item[1]),
    )
    rows = sorted(rows, key=lambda row: row["chunk"])
    if all("window_2_timestamp_s" in row for row in rows):
        timestamps = np.asarray([row["window_2_timestamp_s"] for row in rows])
        time_s = timestamps - timestamps[0]
    else:
        time_s = 0.6 * np.arange(len(rows))
    observed = np.asarray([row["window_2_observed_N"] for row in rows])
    fresh_lower = np.asarray([row["window_2_fresh_lower_N"] for row in rows])
    fresh_upper = np.asarray([row["window_2_fresh_upper_N"] for row in rows])
    held_lower = np.asarray([row["window_2_held_lower_N"] for row in rows])
    held_upper = np.asarray([row["window_2_held_upper_N"] for row in rows])
    dual = np.asarray([row["dual_transition_soft_reward"] for row in rows])
    single = np.asarray([row["single_transition_soft_reward"] for row in rows])
    missed = np.asarray([row["material_missed_penalty"] for row in rows], dtype=bool)
    false = np.asarray([row["material_false_penalty"] for row in rows], dtype=bool)

    figure, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True, constrained_layout=True)
    axes[0].fill_between(
        time_s, fresh_lower, fresh_upper, color="#72B7A6", alpha=0.35, label="300-ms fresh band"
    )
    axes[0].plot(time_s, held_lower, color="#7C8798", linestyle="--", label="600-ms held lower")
    axes[0].plot(time_s, held_upper, color="#7C8798", linestyle="-.", label="600-ms held upper")
    axes[0].plot(time_s, observed, color="#222B38", marker="o", markersize=3.5, label="Window-2 force")
    axes[0].set_ylabel("Bilateral force (N)")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(time_s, dual, color="#16877E", marker="o", markersize=3.5, label="Dual-rate reward")
    axes[1].plot(time_s, single, color="#7C8798", linestyle="--", label="Single-rate reward")
    if missed.any():
        axes[1].scatter(time_s[missed], dual[missed], color="#C84755", marker="x", s=45, label="Missed penalty")
    if false.any():
        axes[1].scatter(time_s[false], single[false], color="#3478B8", marker="v", s=42, label="False penalty")
    axes[1].set_xlabel("Time within episode (s)")
    axes[1].set_ylabel("E18 soft reward deduction")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{task_slug}_representative_episode.{suffix}", dpi=240)
    plt.close(figure)
    return {"episode": episode, "transitions": len(rows)}


def _markdown(reports: list[dict], representatives: dict[str, dict]) -> str:
    rows = []
    details = []
    for report in reports:
        envelope = report["window_2_envelope"]
        material = report["reward"]["single_vs_dual_window_2_material"]
        reward = report["reward"]
        source = (
            "测试集配对反事实"
            if report["source_protocol"].startswith("held-out test")
            else "真机轨迹配对反事实"
        )
        rows.append(
            f"| {report['task']} | {source} | {report['episodes']} | {report['accepted_transitions']} | "
            f"{envelope['held_band_coverage'] * 100:.2f}% | "
            f"{envelope['fresh_coverage'] * 100:.2f}% | "
            f"{envelope['held_band_interval_score_mean_N']:.3f} | "
            f"{envelope['fresh_interval_score_mean_N']:.3f} | "
            f"{material['missed_penalty_windows']} ({material['miss_rate_among_dual_events'] * 100:.1f}%) | "
            f"{material['false_penalty_windows']} ({material['false_fraction_among_candidate_events'] * 100:.1f}%) | "
            f"{reward['transition_soft_reward_mae']:.5f} |"
        )
        capped = report["episode_caps"]
        details.append(
            f"### {report['task']}\n\n"
            f"- 有效 E18 配对：{report['accepted_transitions']}，episodes：{report['episodes']}。\n"
            f"- 第二 H9 的 fresh coverage 为 {envelope['fresh_coverage'] * 100:.2f}%，"
            f"保持旧包络时为 {envelope['held_band_coverage'] * 100:.2f}%，差值 "
            f"{(envelope['fresh_coverage'] - envelope['held_band_coverage']) * 100:+.2f} 个百分点。\n"
            f"- fresh/held interval score：{envelope['fresh_interval_score_mean_N']:.3f} / "
            f"{envelope['held_band_interval_score_mean_N']:.3f} N。\n"
            f"- Dual-rate 实质处罚事件 {material['dual_events']} 个；Single-rate 事件 "
            f"{material['candidate_events']} 个；漏处罚 {material['missed_penalty_windows']} 个 "
            f"({material['miss_rate_among_dual_events'] * 100:.1f}%)；误处罚 "
            f"{material['false_penalty_windows']} 个 "
            f"({material['false_fraction_among_candidate_events'] * 100:.1f}%)。\n"
            f"- cap 前 transition soft deduction 总和：Dual-rate "
            f"{reward['dual_transition_soft_reward_sum']:.3f}，Single-rate "
            f"{reward['single_transition_soft_reward_sum']:.3f}；cap 后 episode 总和："
            f"{capped['dual_capped_total']:.3f} / {capped['single_capped_total']:.3f}。"
        )
    return f"""# Dual-rate 触觉奖励配对反事实消融

## 定义

- Dual-rate：E18 动作周期内在 0 ms 和 300 ms 分别刷新 H9 包络。
- Single-rate 反事实：第二窗口继续使用第一窗口的包络和 stable-contact gate。
- 两种方法使用完全相同的真实轨迹、观测力、Expert、校准系数和奖励权重。
- 三个任务的 Single-rate 结果均为离线配对反事实，没有重新用 Single-rate 执行机器人。
- 两段 penalty 均取平均，未因触觉刷新频率翻倍而扩大扣分尺度。
- Material penalty 定义为单个 H9 加权 soft reward deduction >= 0.01。
- 主结果使用 episode cap 前的 soft reward；peak channel 两组完全相同，不计入漏罚/误罚。
- U 盘和糕点使用 `force_scale=2 N`；墨水存档由奖励复算确认使用历史 `force_scale=5 N`。
- U 盘沿用固定 seed-44 episode split，Expert 和区间系数均不使用 test targets 拟合。

## 主结果

| Task | 数据来源 | Episodes | E18 transitions | Held cov. | Fresh cov. | Held IS | Fresh IS | Missed penalty | False penalty | Transition reward MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 分任务结果

{chr(10).join(details)}

这里的漏处罚/误处罚以 300-ms fresh envelope 产生的在线 soft penalty 为参考。
`Held cov.` 与 `Fresh cov.` 只在第二窗口 fresh bilateral gate 开启的样本上计算，
用于隔离包络过时的影响；完整 Single-rate 反事实还会保持第一窗口 gate。
漏罚百分比以 Dual-rate material events 为分母；误罚百分比以 Single-rate
触发的 material events 为分母。零阈值的原始统计保存在 summary JSON 中。

## 解释边界

当前实现最终仍将两个 H9 penalty 平均成一个 E18 transition reward。因此该实验支持
“在相同轨迹上，提高触觉条件刷新频率可减少 stale-envelope reward error”，但不应
表述成 actor/critic 每 300 ms 接收一条独立 transition。三个任务全部是 paired
counterfactual：Dual-rate 一列使用每个 H9 的 fresh 预测，Single-rate 一列在相同的
第二 H9 观测上保持第一 H9 的 band 与 gate。没有任何任务实际执行 Single-rate 策略，
所以不能写成随机化在线 A/B，也不能据此直接声称 Dual-rate 提升了任务成功率。

任务之间的区别只在轨迹来源：墨水与糕点使用真实 Dual-rate Online RL rollout 日志；
U 盘使用独立 test episodes。两类来源不能合并成一个总体均值。

U 盘旧在线存档实际为 E14 且每周期只有一个触觉窗口，所以未伪造第二 H9 字段；本报告
改用 seed-44 test split 严格构造 H9+H9 的 E18 配对。U 盘 Coverage 是测试集统计；墨水、
糕点 Coverage 是部署轨迹统计，后者不是新的 conformal coverage 保证。

代表性 episode：{json.dumps(representatives, ensure_ascii=False)}。
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports = []
    all_transitions = []
    all_episodes = []
    representatives = {}
    usb_report, usb_transitions, usb_episodes = _analyze_usb_test()
    reports.append(usb_report)
    all_transitions.extend(usb_transitions)
    all_episodes.extend(usb_episodes)
    representatives[USB["task"]] = _plot_representative(
        "usb_insertion", usb_transitions
    )
    for run in RUNS:
        report, transitions, episodes = analyze_dual_rate_log(
            run["path"],
            task=run["task"],
            force_scale=run["force_scale"],
        )
        reports.append(report)
        all_transitions.extend(transitions)
        all_episodes.extend(episodes)
        slug = run["task"].lower().replace(" ", "_")
        representatives[run["task"]] = _plot_representative(slug, transitions)

    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "paired counterfactual: 600-ms held-envelope versus "
            "2x300-ms fresh-envelope reward"
        ),
        "models_retrained": False,
        "trajectories_reexecuted": False,
        "material_reward_threshold": 0.01,
        "peak_channel_in_comparison": False,
        "single_rate_robot_reexecuted": False,
        "all_tasks_are_paired_counterfactual": True,
        "reports": reports,
        "representative_episodes": representatives,
        "usb_status": (
            "included from seed-44 held-out test H9 pairs; archived USB online log "
            "was E14 and was not used"
        ),
    }
    write_json(OUTPUT / "dual_rate_reward_ablation_summary.json", payload)
    _write_csv(OUTPUT / "dual_rate_reward_transitions.csv", all_transitions)
    _write_csv(OUTPUT / "dual_rate_reward_per_episode.csv", all_episodes)
    (OUTPUT / "DUAL_RATE_REWARD_ABLATION_ZH.md").write_text(
        _markdown(reports, representatives), encoding="utf-8"
    )
    _plot_summary(reports)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
