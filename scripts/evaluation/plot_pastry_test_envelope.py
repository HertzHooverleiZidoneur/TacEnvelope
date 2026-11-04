#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from tactile_force_envelope.common import load_config, write_json
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


SIDES = ("left", "right")
COLORS = {
    "target": "#15191e",
    "mu": "#2878b5",
    "upper": "#d95f02",
    "exceed": "#c72e29",
}


def _metrics(target: np.ndarray, mu: np.ndarray, upper: np.ndarray) -> dict[str, float | int]:
    residual = target - mu
    upper_error = upper - target
    exceedance = np.maximum(0.0, target - upper)
    positive = exceedance[exceedance > 0.0]
    return {
        "samples": int(target.size),
        "target_mean_N": float(target.mean()),
        "target_p95_N": float(np.quantile(target, 0.95)),
        "target_max_N": float(target.max()),
        "mu_bias_N": float(residual.mean()),
        "mu_mae_N": float(np.abs(residual).mean()),
        "mu_rmse_N": float(np.sqrt(np.mean(residual**2))),
        "coverage": float(np.mean(target <= upper)),
        "mean_envelope_width_N": float(np.mean(upper - mu)),
        "mean_upper_headroom_N": float(upper_error.mean()),
        "upper_absolute_error_N": float(np.abs(upper_error).mean()),
        "false_penalty_windows": int(positive.size),
        "exceedance_mean_all_N": float(exceedance.mean()),
        "exceedance_mean_when_positive_N": float(positive.mean()) if positive.size else 0.0,
        "exceedance_p95_when_positive_N": (
            float(np.quantile(positive, 0.95)) if positive.size else 0.0
        ),
        "exceedance_max_N": float(exceedance.max()),
    }


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    return list(zip(changes[::2], changes[1::2], strict=True))


def _plot_series(
    ax: plt.Axes,
    time_s: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray,
    title: str,
) -> None:
    first = True
    for start, end in _segments(mask):
        segment = slice(start, end)
        ax.axvspan(time_s[start], time_s[end - 1], color="#e8f0f7", alpha=0.5, lw=0)
        ax.plot(
            time_s[segment], target[segment], "o-", ms=3.0, lw=1.25,
            color=COLORS["target"], label="Observed force" if first else None,
        )
        ax.plot(
            time_s[segment], mu[segment], "-", lw=1.25,
            color=COLORS["mu"], label="Predicted mu" if first else None,
        )
        ax.plot(
            time_s[segment], upper[segment], "-", lw=1.5,
            color=COLORS["upper"], label="Calibrated U95" if first else None,
        )
        exceeded = target[segment] > upper[segment]
        if np.any(exceeded):
            ax.scatter(
                time_s[segment][exceeded], target[segment][exceeded], marker="x", s=34,
                linewidths=1.4, color=COLORS["exceed"],
                label="Observed > U95" if first else None, zorder=5,
            )
        first = False
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylabel("Force (N)")
    ax.grid(True, alpha=0.22)


def _episode_figure(
    episode: int,
    time_s: np.ndarray,
    arrays: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11.0), sharex=True, constrained_layout=True)
    for axis, side, label in zip(axes[:2], SIDES, ("Left", "Right"), strict=True):
        values = arrays[side]
        _plot_series(
            axis, time_s, values["target"], values["mu"], values["upper"],
            masks[side], f"{label} stable-contact force envelope",
        )

    bilateral = arrays["bilateral"]
    _plot_series(
        axes[2], time_s, bilateral["target"], bilateral["mu"], bilateral["upper"],
        masks["bilateral"], "Bilateral average stable-contact force envelope",
    )

    mask = masks["bilateral"]
    residual = bilateral["target"] - bilateral["mu"]
    exceedance = np.maximum(0.0, bilateral["target"] - bilateral["upper"])
    axes[3].axhline(0.0, color="#777777", lw=0.8)
    first = True
    for start, end in _segments(mask):
        segment = slice(start, end)
        axes[3].plot(
            time_s[segment], residual[segment], "-", lw=1.2, color=COLORS["mu"],
            label="Center residual y - mu" if first else None,
        )
        axes[3].fill_between(
            time_s[segment], 0.0, exceedance[segment], color=COLORS["exceed"], alpha=0.45,
            label="Envelope exceedance max(0, y - U95)" if first else None,
        )
        first = False
    axes[3].set_title("Bilateral prediction error", loc="left", fontsize=10)
    axes[3].set_xlabel("Episode time (s)")
    axes[3].set_ylabel("Error (N)")
    axes[3].grid(True, alpha=0.22)

    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(loc="upper right", ncol=4, frameon=False, fontsize=8)
    if axes[3].get_legend_handles_labels()[0]:
        axes[3].legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"Pastry test episode {episode}: force envelope vs time", fontsize=14, y=1.0)
    return fig


def _summary_figure(records: list[dict[str, float | int]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for name, color in (("left", "#2878b5"), ("right", "#4a9b55"), ("bilateral", "#8c55a3")):
        rows = [row for row in records if row["stream"] == name]
        residual = np.asarray([float(row["target_N"]) - float(row["mu_N"]) for row in rows])
        upper_error = np.asarray([float(row["upper95_N"]) - float(row["target_N"]) for row in rows])
        axes[0, 0].hist(residual, bins=45, density=True, histtype="step", lw=1.5, color=color, label=name)
        axes[0, 1].hist(upper_error, bins=45, density=True, histtype="step", lw=1.5, color=color, label=name)
        sorted_abs = np.sort(np.abs(residual))
        axes[1, 0].plot(sorted_abs, np.arange(1, sorted_abs.size + 1) / sorted_abs.size, color=color, label=name)
        exceed = np.maximum(0.0, -upper_error)
        sorted_exceed = np.sort(exceed)
        axes[1, 1].plot(sorted_exceed, np.arange(1, sorted_exceed.size + 1) / sorted_exceed.size, color=color, label=name)
    axes[0, 0].axvline(0, color="#777", lw=0.8)
    axes[0, 0].set(title="Center residual y - mu", xlabel="Residual (N)", ylabel="Density")
    axes[0, 1].axvline(0, color=COLORS["exceed"], lw=1.0, ls="--")
    axes[0, 1].set(title="Upper error U95 - y", xlabel="Error (N)", ylabel="Density")
    axes[1, 0].set(title="Absolute center error CDF", xlabel="|y - mu| (N)", ylabel="CDF")
    axes[1, 1].set(title="Envelope exceedance CDF", xlabel="max(0, y - U95) (N)", ylabel="CDF")
    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Pastry Envelope Expert: held-out test error distributions", fontsize=14)
    return fig


def _write_html(output: Path, summary: dict[str, Any], episodes: list[int]) -> None:
    rows = []
    for stream in ("left", "right", "bilateral"):
        metric = summary["metrics"][stream]
        rows.append(
            "<tr>"
            f"<td>{stream}</td><td>{metric['samples']}</td>"
            f"<td>{metric['target_mean_N']:.3f} N</td>"
            f"<td>{metric['mu_mae_N']:.3f} N</td>"
            f"<td>{metric['mu_bias_N']:.3f} N</td>"
            f"<td>{metric['coverage'] * 100:.2f}%</td>"
            f"<td>{metric['mean_envelope_width_N']:.3f} N</td>"
            f"<td>{metric['upper_absolute_error_N']:.3f} N</td>"
            f"<td>{metric['exceedance_max_N']:.3f} N</td>"
            "</tr>"
        )
    episode_links = "\n".join(
        f'<li><a href="episodes/test_episode_{episode:03d}.png">test episode {episode}</a></li>'
        for episode in episodes
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>糕点触觉包络测试曲线</title>
<style>body{{font-family:Arial,sans-serif;max-width:1180px;margin:32px auto;color:#1b232b;line-height:1.55}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3d9;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #d8dde2}}code{{background:#eef2f5;padding:2px 5px}}</style></head><body>
<h1>糕点 Pi0.5 触觉 Envelope Expert：测试集力-时间曲线</h1>
<p>所有网络参数来自 train，最佳 checkpoint 由 validation 选择，<code>alpha95</code> 只由 calibration 估计。
以下 test 曲线没有重新校准。曲线只显示因果 stable-contact gate 开启且真实未来标签大于 1 N 的窗口。</p>
<table><thead><tr><th>序列</th><th>窗口</th><th>真实均值</th><th>mu MAE</th><th>mu bias (y-mu)</th><th>U95 coverage</th><th>平均包络宽度</th><th>|U95-y|</th><th>最大超界</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>测试集误差分布</h2><img src="test_error_summary.png" alt="test error summary">
<h2>逐 episode 曲线</h2><ul>{episode_links}</ul>
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint, _, grouped, predictions, history, history_mask, _ = _load_evaluation_context(
        config, args.checkpoint
    )
    rows = grouped["test"]
    gates = _gates_for_rows(rows, history, history_mask, config)
    tau = float(config["stable_contact"]["tau_train"])
    calibration = checkpoint["calibration"]
    output = args.output.expanduser().resolve()
    episode_dir = output / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, dict[str, np.ndarray]] = {}
    masks: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(SIDES):
        prediction = predictions["test"][side]
        alpha = float(calibration[f"alpha95_{side}"])
        arrays[side] = {
            "target": prediction["target"],
            "mu": prediction["mu"],
            "sigma": prediction["sigma"],
            "upper": prediction["mu"] + alpha * prediction["sigma"],
        }
        masks[side] = gates[:, side_index] & (prediction["target"] > tau)
    arrays["bilateral"] = {
        key: 0.5 * (arrays["left"][key] + arrays["right"][key])
        for key in ("target", "mu", "sigma", "upper")
    }
    masks["bilateral"] = masks["left"] & masks["right"]

    window_records: list[dict[str, float | int | str]] = []
    for stream in ("left", "right", "bilateral"):
        selected = np.flatnonzero(masks[stream])
        for position in selected:
            row = rows[int(position)]
            window_records.append({
                "stream": stream,
                "episode": int(row["episode_index"]),
                "window_index": int(row["window_index"]),
                "anchor_timestamp": float(row["start_timestamp"]),
                "target_N": float(arrays[stream]["target"][position]),
                "mu_N": float(arrays[stream]["mu"][position]),
                "sigma_N": float(arrays[stream]["sigma"][position]),
                "upper95_N": float(arrays[stream]["upper"][position]),
                "center_residual_N": float(arrays[stream]["target"][position] - arrays[stream]["mu"][position]),
                "upper_error_N": float(arrays[stream]["upper"][position] - arrays[stream]["target"][position]),
                "exceedance_N": float(max(0.0, arrays[stream]["target"][position] - arrays[stream]["upper"][position])),
            })

    episodes = sorted({int(row["episode_index"]) for row in rows})
    rendered: list[int] = []
    with PdfPages(output / "all_test_episode_curves.pdf") as pdf:
        for episode in episodes:
            positions = np.asarray(
                [i for i, row in enumerate(rows) if int(row["episode_index"]) == episode],
                dtype=np.int64,
            )
            if not positions.size or not np.any(masks["left"][positions] | masks["right"][positions]):
                continue
            anchors = np.asarray([float(rows[i]["start_timestamp"]) for i in positions])
            time_s = anchors - anchors.min()
            local_arrays = {
                stream: {key: value[positions] for key, value in stream_arrays.items()}
                for stream, stream_arrays in arrays.items()
            }
            local_masks = {stream: value[positions] for stream, value in masks.items()}
            fig = _episode_figure(episode, time_s, local_arrays, local_masks)
            fig.savefig(episode_dir / f"test_episode_{episode:03d}.png", dpi=170)
            fig.savefig(episode_dir / f"test_episode_{episode:03d}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            rendered.append(episode)

    metrics = {}
    for stream in ("left", "right", "bilateral"):
        mask = masks[stream]
        metrics[stream] = _metrics(
            arrays[stream]["target"][mask], arrays[stream]["mu"][mask], arrays[stream]["upper"][mask]
        )
    summary = {
        "split": "test",
        "test_episodes": len(episodes),
        "rendered_episodes": rendered,
        "calibration_source": "calibration split only",
        "test_recalibration": False,
        "selection": "causal stable-contact gate AND future test target > tau_train for evaluation only",
        "tau_train_N": tau,
        "metrics": metrics,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
    }
    write_json(output / "summary.json", summary)
    with (output / "test_window_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_records[0]))
        writer.writeheader()
        writer.writerows(window_records)
    fig = _summary_figure(window_records)
    fig.savefig(output / "test_error_summary.png", dpi=190)
    fig.savefig(output / "test_error_summary.pdf")
    plt.close(fig)
    _write_html(output, summary, rendered)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
