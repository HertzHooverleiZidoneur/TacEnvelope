#!/usr/bin/env python3
"""Evaluate deployed dual-rate tactile inference with z_rl fixed per E18 cycle."""

from __future__ import annotations

import copy
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.dual_rate_reward_ablation import (
    analyze_paired_h9_windows,
    interval_score_scalar,
)
from tactile_force_envelope.model import TransformerEnvelopeExpert
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)
from tactile_force_envelope.transformer_train import TransformerWindowDataset


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/paper/dual_rate_fixed_zrl_test_ablation"
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


def _artifact(task: dict[str, Any], name: str) -> Path:
    return ROOT / (
        f"outputs/training/{task['workspace']}/pi05_jax_zrl_envelope_v1/"
        f"sustained_contact_seed_44/stable_evaluation/{name}"
    )


def _pair_positions(rows: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[int(row["episode_index"])].append(position)

    pairs: list[tuple[int, int]] = []
    rejected = 0
    for positions in grouped.values():
        ordered = sorted(positions, key=lambda p: float(rows[p]["start_timestamp"]))
        index = 0
        while index + 1 < len(ordered):
            first, second = ordered[index], ordered[index + 1]
            gap_s = float(rows[second]["start_timestamp"]) - float(
                rows[first]["start_timestamp"]
            )
            consecutive = int(rows[second]["window_index"]) == int(
                rows[first]["window_index"]
            ) + 1
            if consecutive and 0.25 <= gap_s <= 0.38:
                pairs.append((first, second))
                index += 2
            else:
                rejected += 1
                index += 1
    return pairs, rejected


def _build_model(checkpoint: dict[str, Any]) -> TransformerEnvelopeExpert:
    dimensions = checkpoint["dimensions"]
    model = TransformerEnvelopeExpert(
        condition_dim=int(dimensions["condition_dim"]),
        state_dim=int(dimensions["state_dim"]),
        action_dim=int(dimensions["action_dim"]),
        action_steps=int(dimensions["action_steps"]),
        config=checkpoint["model_config"],
        history_dim=int(dimensions.get("history_dim", 0)),
        history_steps=int(dimensions.get("history_steps", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _fixed_zrl_predictions(
    model: TransformerEnvelopeExpert,
    dataset: TransformerWindowDataset,
    condition_tokens: torch.Tensor,
    condition_mask: torch.Tensor,
    pairs: list[tuple[int, int]],
    normalization: dict[str, torch.Tensor],
    *,
    batch_size: int = 128,
) -> dict[tuple[int, int], dict[str, float]]:
    results: dict[tuple[int, int], dict[str, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            second_items = [dataset[second] for _, second in batch_pairs]
            first_window_indices = [
                int(dataset.rows[first]["window_index"]) for first, _ in batch_pairs
            ]
            kwargs: dict[str, torch.Tensor] = {
                "condition_tokens": torch.stack(
                    [condition_tokens[index] for index in first_window_indices]
                ).float(),
                "condition_mask": torch.stack(
                    [condition_mask[index] for index in first_window_indices]
                ).bool(),
                "state": torch.stack([item["state"] for item in second_items]),
                "actions": torch.stack([item["actions"] for item in second_items]),
                "action_mask": torch.stack(
                    [item["action_mask"] for item in second_items]
                ),
            }
            if "tactile_history" in second_items[0]:
                kwargs["tactile_history"] = torch.stack(
                    [item["tactile_history"] for item in second_items]
                )
                kwargs["tactile_history_mask"] = torch.stack(
                    [item["tactile_history_mask"] for item in second_items]
                )
            outputs = model(**kwargs)
            for row_index, pair in enumerate(batch_pairs):
                result: dict[str, float] = {}
                for side in ("left", "right"):
                    target_mean = float(normalization[f"target_mean_{side}"])
                    target_std = float(normalization[f"target_std_{side}"])
                    result[f"mu_{side}"] = (
                        float(outputs[f"mu_{side}"][row_index]) * target_std
                        + target_mean
                    )
                    result[f"sigma_{side}"] = (
                        float(torch.exp(0.5 * outputs[f"log_var_{side}"][row_index]))
                        * target_std
                    )
                results[pair] = result
    return results


def _band(
    mu_left: float,
    mu_right: float,
    sigma_left: float,
    sigma_right: float,
    alpha05: float,
    alpha95: float,
) -> tuple[float, float]:
    center = 0.5 * (mu_left + mu_right)
    scale = max(1e-6, 0.5 * (sigma_left + sigma_right))
    return center + alpha05 * scale, center + alpha95 * scale


def _evaluate_task(task: dict[str, Any]) -> tuple[dict[str, Any], list[dict], list[dict]]:
    config = copy.deepcopy(load_config(task["config"]))
    config["training"]["device"] = "cpu"
    config["training"]["num_workers"] = 0
    checkpoint_path = _artifact(task, "stable_contact_calibrated_checkpoint.pt")
    calibration_path = _artifact(task, "bilateral_sigma_calibration.json")
    checkpoint, _, grouped, predictions, history, history_mask, _ = (
        _load_evaluation_context(config, checkpoint_path, split_names=("test",))
    )
    rows = grouped["test"]
    gates = _gates_for_rows(rows, history, history_mask, config)
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    condition = torch.load(
        workspace / "condition_tokens.pt", map_location="cpu", weights_only=False
    )
    dataset = TransformerWindowDataset(
        rows,
        condition["condition_tokens"],
        condition["condition_mask"],
        checkpoint["normalization"],
        config,
        history,
        history_mask,
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("scale_method") != "mean":
        raise RuntimeError(f"{task['task']} does not use u_mean calibration")
    alpha05 = float(calibration["lower_alpha"])
    alpha95 = float(calibration["upper_alpha"])

    candidate_pairs, gap_rejections = _pair_positions(rows)
    pairs: list[tuple[int, int]] = []
    missing_first_zrl = 0
    candidate_gate2_samples = 0
    missing_first_zrl_gate2 = 0
    for first, second in candidate_pairs:
        gate2 = bool(gates[second, 0] and gates[second, 1])
        candidate_gate2_samples += int(gate2)
        first_window = int(rows[first]["window_index"])
        if not bool(condition["condition_mask"][first_window].any()):
            missing_first_zrl += 1
            missing_first_zrl_gate2 += int(gate2)
            continue
        pairs.append((first, second))
    fixed_predictions = _fixed_zrl_predictions(
        _build_model(checkpoint),
        dataset,
        condition["condition_tokens"],
        condition["condition_mask"],
        pairs,
        checkpoint["normalization"],
    )

    normal = predictions["test"]
    window_rows: list[dict[str, Any]] = []
    full_refresh_rows: list[dict[str, float]] = []
    for first, second in pairs:
        first_band = _band(
            float(normal["left"]["mu"][first]),
            float(normal["right"]["mu"][first]),
            float(normal["left"]["sigma"][first]),
            float(normal["right"]["sigma"][first]),
            alpha05,
            alpha95,
        )
        fixed = fixed_predictions[(first, second)]
        fixed_band = _band(
            fixed["mu_left"],
            fixed["mu_right"],
            fixed["sigma_left"],
            fixed["sigma_right"],
            alpha05,
            alpha95,
        )
        full_band = _band(
            float(normal["left"]["mu"][second]),
            float(normal["right"]["mu"][second]),
            float(normal["left"]["sigma"][second]),
            float(normal["right"]["sigma"][second]),
            alpha05,
            alpha95,
        )
        observed_first = 0.5 * (
            float(normal["left"]["target"][first])
            + float(normal["right"]["target"][first])
        )
        observed_second = 0.5 * (
            float(normal["left"]["target"][second])
            + float(normal["right"]["target"][second])
        )
        common = {
            "episode": int(rows[first]["episode_index"]),
            "success": 1,
        }
        window_rows.extend(
            [
                {
                    **common,
                    "window_index": int(rows[first]["window_index"]),
                    "timestamp": float(rows[first]["start_timestamp"]),
                    "observed": observed_first,
                    "lower": first_band[0],
                    "upper": first_band[1],
                    "gate": bool(gates[first, 0] and gates[first, 1]),
                },
                {
                    **common,
                    "window_index": int(rows[second]["window_index"]),
                    "timestamp": float(rows[second]["start_timestamp"]),
                    "observed": observed_second,
                    "lower": fixed_band[0],
                    "upper": fixed_band[1],
                    "gate": bool(gates[second, 0] and gates[second, 1]),
                },
            ]
        )
        second_window = int(rows[second]["window_index"])
        if (
            bool(gates[second, 0] and gates[second, 1])
            and bool(condition["condition_mask"][second_window].any())
        ):
            full_refresh_rows.append(
                {
                    "observed": observed_second,
                    "lower": full_band[0],
                    "upper": full_band[1],
                }
            )

    report, transitions, episodes = analyze_paired_h9_windows(
        window_rows,
        task=task["task"],
        source_file=str(checkpoint_path),
        force_scale=float(task["force_scale"]),
        source_protocol="test split paired H9, z_rl fixed within each E18 cycle",
    )
    full_coverage = float(
        np.mean(
            [
                row["lower"] <= row["observed"] <= row["upper"]
                for row in full_refresh_rows
            ]
        )
    )
    full_score = float(
        np.mean(
            [
                interval_score_scalar(row["observed"], row["lower"], row["upper"])
                for row in full_refresh_rows
            ]
        )
    )
    report.update(
        {
            "evaluation_split": "test",
            "episode_split_seed": 44,
            "z_rl_protocol": (
                "window 1 z_rl is reused for window 2; window-2 robot state, "
                "causal tactile history, action slice, and gate are refreshed"
            ),
            "calibration_file": str(calibration_path),
            "calibration_split_only": True,
            "test_targets_used_for_calibration": False,
            "candidate_pairs": len(candidate_pairs),
            "pairing_gap_rejections": gap_rejections,
            "missing_first_zrl_pairs": missing_first_zrl,
            "candidate_window2_gate_samples": candidate_gate2_samples,
            "missing_first_zrl_window2_gate_samples": missing_first_zrl_gate2,
            "total_test_episodes": len(
                {int(row["episode_index"]) for row in rows}
            ),
            "full_zrl_refresh_diagnostic": {
                "same_gate_active_samples": len(full_refresh_rows),
                "coverage": full_coverage,
                "interval_score_mean_N": full_score,
                "not_used_as_deployed_dual_rate_result": True,
            },
        }
    )
    return report, transitions, episodes


def _summary_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        envelope = report["window_2_envelope"]
        single_coverage = float(envelope["held_band_coverage"])
        dual_coverage = float(envelope["fresh_coverage"])
        single_score = float(envelope["held_band_interval_score_mean_N"])
        dual_score = float(envelope["fresh_interval_score_mean_N"])
        diagnostic = report["full_zrl_refresh_diagnostic"]
        rows.append(
            {
                "task": report["task"],
                "total_test_episodes": report["total_test_episodes"],
                "paired_test_episodes": report["episodes"],
                "paired_e18_transitions": report["accepted_transitions"],
                "window2_gate_samples": envelope["gate_active_samples"],
                "missing_first_zrl_pairs": report["missing_first_zrl_pairs"],
                "missing_first_zrl_window2_gate_samples": report[
                    "missing_first_zrl_window2_gate_samples"
                ],
                "single_rate_coverage_percent": 100.0 * single_coverage,
                "fixed_zrl_dual_rate_coverage_percent": 100.0 * dual_coverage,
                "coverage_gain_percentage_points": 100.0
                * (dual_coverage - single_coverage),
                "single_rate_interval_score_N": single_score,
                "fixed_zrl_dual_rate_interval_score_N": dual_score,
                "interval_score_reduction_percent": 100.0
                * (single_score - dual_score)
                / single_score,
                "full_zrl_refresh_coverage_percent_diagnostic": 100.0
                * float(diagnostic["coverage"]),
                "full_zrl_refresh_interval_score_N_diagnostic": float(
                    diagnostic["interval_score_mean_N"]
                ),
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]]) -> None:
    labels = [row["task"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
    metrics = (
        (
            "single_rate_coverage_percent",
            "fixed_zrl_dual_rate_coverage_percent",
            "Window-2 coverage (%)",
        ),
        (
            "single_rate_interval_score_N",
            "fixed_zrl_dual_rate_interval_score_N",
            r"Interval score $IS_{0.1}$ (N)",
        ),
    )
    for axis, (single_key, dual_key, ylabel) in zip(axes, metrics, strict=True):
        single = np.asarray([row[single_key] for row in rows])
        dual = np.asarray([row[dual_key] for row in rows])
        axis.bar(x - width / 2, single, width, color="#7C8798", label="Single-rate")
        axis.bar(x + width / 2, dual, width, color="#16877E", label=r"Dual-rate (fixed $z_{rl}$)")
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for patch in axis.patches:
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f"{patch.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"dual_rate_fixed_zrl_test_ablation.{suffix}", dpi=240)
    plt.close(figure)


def _latex(rows: list[dict[str, Any]]) -> str:
    table_rows: list[str] = []
    for row in rows:
        table_rows.extend(
            (
                f"{row['task']} & Single-rate & "
                f"{row['window2_gate_samples']} & "
                f"{row['single_rate_coverage_percent']:.2f} & "
                f"{row['single_rate_interval_score_N']:.3f} \\\\",
                f"& Dual-rate (fixed $z_{{\\mathrm{{rl}}}}$) & "
                f"{row['window2_gate_samples']} & "
                f"{row['fixed_zrl_dual_rate_coverage_percent']:.2f} & "
                f"\\textbf{{{row['fixed_zrl_dual_rate_interval_score_N']:.3f}}} \\\\",
                "\\hline",
            )
        )
    return """We compare deployed dual-rate tactile inference against a paired
single-rate counterfactual on the seed-44 test split. Within each E18
cycle, both methods reuse the same initial $z_{\\mathrm{rl}}$. Dual-rate
refreshes only the robot state, causal tactile history, next H9 action
slice, and stable-contact gate at the 300-ms midpoint, matching the
online implementation. Calibration coefficients remain fixed from the
calibration split; test targets are used only for evaluation.

\\begin{table}[!t]
\\centering
\\caption{Paired test-set comparison of single-rate and deployed dual-rate
tactile envelope evaluation with fixed $z_{\\mathrm{rl}}$.}
\\label{tab:dual_rate_fixed_zrl}
\\footnotesize
\\setlength{\\tabcolsep}{4pt}
\\renewcommand{\\arraystretch}{1.08}
\\begin{tabular}{@{}llccc@{}}
\\hline
Task & Setting & $N$ & Cov. (\\%) & $IS_{0.1}$ (N) $\\downarrow$ \\\\
\\hline
""" + "\n".join(table_rows) + """
\\end{tabular}

\\par\\vspace{3pt}
\\begin{minipage}{\\columnwidth}
\\footnotesize\\raggedright
Both settings use the same measured second-window bilateral Top2Mean
load. Calibration coefficients are estimated only on the calibration
split. Sparse feature caching excludes 3 Ink and 6 Pastry gate-active
pairs whose first-window $z_{\\mathrm{rl}}$ is unavailable.
\\end{minipage}
\\end{table}
"""


def _markdown(rows: list[dict[str, Any]]) -> str:
    table = "\n".join(
        "| {task} | {total_test_episodes} | {paired_test_episodes} | "
        "{paired_e18_transitions} | "
        "{window2_gate_samples} | {single_rate_coverage_percent:.2f}% | "
        "{fixed_zrl_dual_rate_coverage_percent:.2f}% | "
        "{coverage_gain_percentage_points:+.2f} pp | "
        "{single_rate_interval_score_N:.3f} | "
        "{fixed_zrl_dual_rate_interval_score_N:.3f} | "
        "{interval_score_reduction_percent:.2f}% | "
        "{full_zrl_refresh_coverage_percent_diagnostic:.2f}% |".format(**row)
        for row in rows
    )
    return f"""# 固定 z_rl 的 Dual-rate Test 消融

## 协议

- Episode split：seed 44 的 test episodes。
- Single-rate：第二个 H9 继续使用第一个 H9 的 envelope 和 gate。
- Dual-rate：一个 E18 周期内固定窗口 1 的 z_rl；在 300 ms 处只刷新机器人状态、
  过去 200 ms 因果触觉历史、后续 H9 动作和 gate。
- 两者使用相同第二窗口实测 bilateral Top2Mean 真值。
- alpha05/alpha95 只来自 calibration split，未使用 test target 重校准。
- Coverage 在相同的第二窗口 fresh bilateral-gate 子集上计算。

## 结果

| Task | Test eps. | Paired eps. | E18 pairs | Gate windows | Single cov. | Fixed-z Dual cov. | Gain | Single IS | Fixed-z Dual IS | IS reduction | Full-z refresh cov.（诊断） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 解释边界

主对比是 Single-rate 与 Fixed-z Dual-rate。最后一列完整刷新 z_rl 仅用于检查旧实验与
部署协议之间的差异，不作为在线 Dual-rate 结果。Ink/Pastry 的离线特征缓存只覆盖
sustained-contact 候选窗口，因此大量非接触配对没有窗口 1 z_rl。对主指标使用的
bilateral-gate 样本，Ink 排除 3 个、Pastry 排除 6 个；具体数量在 summary 中记录。
这些 test episodes 已参与此前开发诊断，仍建议最终使用新采集 blind episodes 做独立验证。
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for task in TASKS:
        report, task_transitions, task_episodes = _evaluate_task(task)
        reports.append(report)
        transitions.extend(task_transitions)
        episodes.extend(task_episodes)
    rows = _summary_rows(reports)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "test",
        "episode_split_seed": 44,
        "models_retrained": False,
        "calibration_split_only": True,
        "test_targets_used_for_calibration": False,
        "deployed_dual_rate_z_rl_fixed": True,
        "single_rate_robot_reexecuted": False,
        "all_tasks_are_paired_counterfactual": True,
        "rows": rows,
        "reports": reports,
    }
    write_json(OUTPUT / "summary.json", payload)
    for name, values in (
        ("summary.csv", rows),
        ("paired_transitions.csv", transitions),
        ("per_episode.csv", episodes),
    ):
        with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (OUTPUT / "DUAL_RATE_FIXED_ZRL_TEST_ABLATION_ZH.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    (OUTPUT / "dual_rate_fixed_zrl_test_table.tex").write_text(
        _latex(rows), encoding="utf-8"
    )
    _plot(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
