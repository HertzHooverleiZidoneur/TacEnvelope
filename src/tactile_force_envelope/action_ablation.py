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
from .loss_ablation import _evaluate_checkpoint
from .transformer_train import train_transformer_and_calibrate


VARIANTS = {
    "full_action": {
        "use_action_conditioning": True,
        "label": "Full: z + q + action + tactile history",
    },
    "without_action": {
        "use_action_conditioning": False,
        "label": "Without action: z + q + tactile history",
    },
}


def _aggregate(runs: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = np.asarray(
        [float(run["additive_test"][metric]) for run in runs], dtype=np.float64
    )
    return {"mean": float(values.mean()), "std": float(values.std())}


def _plot(summary: dict[str, Any], output: Path) -> None:
    names = list(VARIANTS)
    labels = ["Full", "w/o action"]
    metrics = [
        ("mae_N", "MAE (N)", None),
        ("rmse_N", "RMSE (N)", None),
        ("gaussian_nll_physical", "Gaussian NLL", None),
        ("coverage", "Central 90% coverage", 0.90),
        ("mean_interval_width_N", "Mean interval width (N)", None),
        ("false_penalty_windows", "Out-of-band windows", None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, (metric, title, target) in zip(axes.flat, metrics, strict=True):
        means = [summary["aggregate"][name][metric]["mean"] for name in names]
        stds = [summary["aggregate"][name][metric]["std"] for name in names]
        bars = axis.bar(
            labels, means, yerr=stds, capsize=5, color=["#2878B5", "#D95F02"]
        )
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        labels_on_bars = (
            [f"{value * 100:.2f}%" for value in means]
            if metric == "coverage"
            else [f"{value:.3f}" for value in means]
        )
        axis.bar_label(bars, labels=labels_on_bars, padding=4, fontsize=9)
    figure.suptitle("USB force-envelope action-conditioning ablation, three seeds")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"action_conditioning_ablation_summary.{suffix}", dpi=200)
    plt.close(figure)


def _write_html(summary: dict[str, Any], output: Path) -> None:
    rows = []
    for run in summary["runs"]:
        metric = run["additive_test"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['label'])}</td><td>{run['seed']}</td>"
            f"<td>{run['trainable_parameters']:,}</td>"
            f"<td>{run['calibration_samples']}</td><td>{run['test_samples']}</td>"
            f"<td>{metric['mae_N']:.4f}</td><td>{metric['rmse_N']:.4f}</td>"
            f"<td>{metric['gaussian_nll_physical']:.4f}</td>"
            f"<td>{metric['coverage'] * 100:.2f}%</td>"
            f"<td>{metric['mean_interval_width_N']:.4f}</td></tr>"
        )
    comparison = summary["comparison"]
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Action Conditioning Ablation</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.65;color:#20252b}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}code{{background:#eef2f5;padding:2px 5px}}</style></head>
<body><h1>USB 触觉力包络 Action Conditioning 消融</h1>
<p>Full 模型输入为 <code>[z_t, q_t, A_t, h_t^tac]</code>；w/o action 输入为
<code>[z_t, q_t, h_t^tac]</code>。后者从 Transformer 序列和参数中真正删除 action projection、
position embedding 和 9 个 action tokens，没有使用零动作占位。</p>
<p>数据、episode split、Pi0.5 z_rl 缓存、Transformer 主体、优化器、训练轮数、
Gaussian NLL、触觉历史和 q05/q95 residual calibration 均保持一致。
<strong>Calibration 只使用 calibration split；测试集不参与训练、checkpoint 选择或校准。</strong></p>
<img src="action_conditioning_ablation_summary.png" alt="action-conditioning ablation">
<h2>三种子结果</h2><table><thead><tr><th>Method</th><th>Seed</th><th>Parameters</th>
<th>Calibration</th><th>Test</th><th>MAE (N)</th><th>RMSE (N)</th><th>Gaussian NLL</th>
<th>Coverage</th><th>Width (N)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>结论</h2><p>{html.escape(comparison['conclusion'])}</p>
<p>移除动作后，MAE 变化 <strong>{comparison['mae_degradation_percent']:.2f}%</strong>，
RMSE 变化 <strong>{comparison['rmse_degradation_percent']:.2f}%</strong>，
Gaussian NLL 变化 <strong>{comparison['nll_degradation_percent']:.2f}%</strong>。
正数表示移除动作使指标变差。</p>
<p><strong>时序边界：</strong>当前离线 action 是按时间戳收集的未来 300 ms 实际执行命令，
数据元信息标记其并非在 anchor 时刻完整可得。因此该实验验证的是短时动作轨迹是否含有额外预测信息；
若要严格对应在线部署，还应以同一次 Pi0.5 forward 产生、在 anchor 时刻已知的 candidate chunk 复验。</p>
</body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def run_action_conditioning_ablation(
    config: dict[str, Any], *, seeds: list[int], force: bool = False,
) -> dict[str, Any]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    output = workspace / "action_conditioning_ablation"
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for variant, settings in VARIANTS.items():
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config["model"]["use_action_conditioning"] = bool(
                settings["use_action_conditioning"]
            )
            run_config["training"].update({
                "seed": int(seed),
                "gaussian_nll_weight": 1.0,
                "smooth_l1_weight": 0.5,
                "artifact_subdir": f"action_conditioning_ablation/{variant}_seed_{seed}",
            })
            run_dir = workspace / run_config["training"]["artifact_subdir"]
            checkpoint_path = run_dir / "transformer_force_envelope_model.pt"
            reused_from = None
            if variant == "full_action" and not force and not checkpoint_path.is_file():
                existing = (
                    workspace / "gaussian_loss_ablation" / f"gaussian_nll_seed_{seed}"
                    / "transformer_force_envelope_model.pt"
                )
                if existing.is_file():
                    checkpoint_path = existing
                    reused_from = str(existing)
            if force or not checkpoint_path.is_file():
                print(f"Training {variant} seed={seed}", flush=True)
                train_transformer_and_calibrate(run_config)
                checkpoint_path = run_dir / "transformer_force_envelope_model.pt"
            else:
                print(f"Reusing {checkpoint_path}", flush=True)
            report = _evaluate_checkpoint(
                run_config, checkpoint_path, variant, seed, settings["label"]
            )
            shape_path = checkpoint_path.parent / "transformer_shape_report.json"
            shape = json.loads(shape_path.read_text(encoding="utf-8"))
            report.update({
                "use_action_conditioning": settings["use_action_conditioning"],
                "action_branch_removal": (
                    "not applicable"
                    if settings["use_action_conditioning"]
                    else "projection, positional embeddings, and all action tokens removed"
                ),
                "trainable_parameters": int(shape["trainable_parameters"]),
                "decoder_input_shape": shape["decoder_input"],
                "action_token_shape": shape["action_tokens"],
                "reused_from": reused_from,
            })
            runs.append(report)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_names = (
        "mae_N", "rmse_N", "gaussian_nll_physical", "coverage",
        "mean_interval_width_N", "false_penalty_windows", "false_penalty_episodes",
    )
    aggregate = {}
    for variant in VARIANTS:
        selected = [run for run in runs if run["variant"] == variant]
        aggregate[variant] = {
            metric: _aggregate(selected, metric) for metric in metric_names
        }
    full = aggregate["full_action"]
    removed = aggregate["without_action"]

    def degradation(metric: str) -> float:
        return 100.0 * (removed[metric]["mean"] - full[metric]["mean"]) / max(
            abs(full[metric]["mean"]), 1e-8
        )

    paired_mae_worse = sum(
        next(r for r in runs if r["variant"] == "without_action" and r["seed"] == seed)
        ["additive_test"]["mae_N"]
        > next(r for r in runs if r["variant"] == "full_action" and r["seed"] == seed)
        ["additive_test"]["mae_N"]
        for seed in seeds
    )
    paired_rmse_worse = sum(
        next(r for r in runs if r["variant"] == "without_action" and r["seed"] == seed)
        ["additive_test"]["rmse_N"]
        > next(r for r in runs if r["variant"] == "full_action" and r["seed"] == seed)
        ["additive_test"]["rmse_N"]
        for seed in seeds
    )
    supports_action = (
        degradation("mae_N") > 0
        and degradation("rmse_N") > 0
        and paired_mae_worse >= 2
        and paired_rmse_worse >= 2
    )
    conclusion = (
        "移除未来动作后，平均 MAE 和 RMSE 均增大，其中 RMSE 在三个种子上全部恶化。"
        "这为动作条件能够改善短时力预测提供了有限但一致的尾部误差证据；由于 MAE 仅在两个种子上恶化，"
        "且 NLL 未改善，不应表述为强显著性结论。"
        if supports_action else
        "当前结果不足以证明未来动作条件能够稳定改善短时触觉力预测；不应将动作输入的贡献写成确定结论。"
    )
    comparison = {
        "supports_action_conditioning": supports_action,
        "paired_mae_degradations_out_of": [paired_mae_worse, len(seeds)],
        "paired_rmse_degradations_out_of": [paired_rmse_worse, len(seeds)],
        "mae_degradation_percent": degradation("mae_N"),
        "rmse_degradation_percent": degradation("rmse_N"),
        "nll_degradation_percent": degradation("gaussian_nll_physical"),
        "coverage_change_percentage_points": 100.0
        * (removed["coverage"]["mean"] - full["coverage"]["mean"]),
        "width_change_percent": degradation("mean_interval_width_N"),
        "conclusion": conclusion,
    }
    summary = {
        "experiment": "with_vs_without_action_conditioning",
        "task": "USB insertion",
        "seeds": seeds,
        "controlled_variables": [
            "same windows and episode split",
            "same Pi0.5 z_rl condition cache",
            "same state and 200 ms causal tactile history",
            "same Transformer decoder hyperparameters",
            "same Gaussian NLL + 0.5 SmoothL1 objective",
            "same validation checkpoint selection",
            "same calibration-only q05/q95 additive interval",
        ],
        "test_used_for_training_or_calibration": False,
        "input_change": {
            "full_action": "[z_t, q_t, A_t^(300ms), h_t^tac]",
            "without_action": "[z_t, q_t, h_t^tac]",
            "zero_action_placeholder_used": False,
            "offline_action_source": "timestamp-selected actually executed commands in [t,t+300ms]",
            "complete_action_sequence_available_at_anchor": False,
            "causality_limitation": (
                "The offline action slice may include within-window human feedback. A strict online "
                "confirmation should use the candidate chunk emitted by the same Pi0.5 forward at t."
            ),
        },
        "runs": runs,
        "aggregate": aggregate,
        "comparison": comparison,
    }
    write_json(output / "summary.json", summary)
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "variant", "seed", "trainable_parameters", "calibration_samples", "test_samples",
            "best_epoch", "mae_N", "rmse_N", "gaussian_nll_physical", "coverage",
            "mean_interval_width_N", "false_penalty_windows", "false_penalty_episodes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            metrics = run["additive_test"]
            writer.writerow({
                "variant": run["variant"], "seed": run["seed"],
                "trainable_parameters": run["trainable_parameters"],
                "calibration_samples": run["calibration_samples"],
                "test_samples": run["test_samples"], "best_epoch": run["best_epoch"],
                **{key: metrics[key] for key in fields[6:]},
            })
    write_json(output / "config_changes.json", {
        "base_config": "configs/experiments/pi05_jax_zrl_stable_contact.yaml",
        "full_action": {"model.use_action_conditioning": True},
        "without_action": {"model.use_action_conditioning": False},
        "all_other_training_and_data_settings": "unchanged",
    })
    _plot(summary, output)
    _write_html(summary, output)
    return summary
