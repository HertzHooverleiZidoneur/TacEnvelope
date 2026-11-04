#!/usr/bin/env python3
"""Replay recent Online RL tactile windows against a calibration-only force band."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tactile_force_envelope.common import load_config
from tactile_force_envelope.stable_contact_experiment import (
    _gates_for_rows,
    _load_evaluation_context,
)


WINDOW_SECONDS = 0.300


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _finite(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    return float(value)


def _match_key(row: dict[str, Any], *, sidecar: bool) -> tuple[float, float, float]:
    observed_key = "target_avg_N" if sidecar else "observed_avg_N"
    return tuple(
        round(_finite(row.get(key)), 6)
        for key in ("mu_avg_N", observed_key, "raw_peak_N")
    )


def _attach_timestamps(
    rows: list[dict[str, Any]], sidecar_events: list[dict[str, Any]]
) -> None:
    event_by_key: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for event in sidecar_events:
        event_by_key[_match_key(event, sidecar=True)].append(event)

    matched_indices: list[int] = []
    event_position = {id(event): index for index, event in enumerate(sidecar_events)}
    for row in rows:
        candidates = event_by_key.get(_match_key(row, sidecar=False), [])
        if len(candidates) != 1:
            raise RuntimeError(
                "Cannot uniquely match Online RL row to tactile sidecar event: "
                f"episode={row.get('episode')} chunk={row.get('chunk')} "
                f"matches={len(candidates)}"
            )
        event = candidates[0]
        anchor_ns = int(event["request_start_ns"])
        row["anchor_timestamp_ns"] = anchor_ns
        row["anchor_timestamp_s"] = anchor_ns / 1e9
        row["anchor_iso_utc"] = datetime.fromtimestamp(
            anchor_ns / 1e9, tz=timezone.utc
        ).isoformat()
        matched_indices.append(event_position[id(event)])
    if any(current <= previous for previous, current in zip(matched_indices, matched_indices[1:])):
        raise RuntimeError("Matched sidecar events are not strictly chronological")


def _calibration_quantiles(
    config_path: Path, checkpoint_path: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    _, _, grouped, predictions, history, history_mask, _ = _load_evaluation_context(
        config, checkpoint_path
    )
    gates = _gates_for_rows(grouped["calibration"], history, history_mask, config)
    mask = gates[:, 0] & gates[:, 1]
    if not np.any(mask):
        raise RuntimeError("Calibration split has no bilateral stable-contact windows")
    left = predictions["calibration"]["left"]
    right = predictions["calibration"]["right"]
    target = 0.5 * (left["target"] + right["target"])
    mu = 0.5 * (left["mu"] + right["mu"])
    residual = target[mask] - mu[mask]
    return {
        "samples": int(mask.sum()),
        "q05_N": float(np.quantile(residual, 0.05)),
        "q95_N": float(np.quantile(residual, 0.95)),
        "residual_mean_N": float(np.mean(residual)),
        "residual_p50_N": float(np.quantile(residual, 0.50)),
        "source": "calibration split bilateral stable-contact windows only",
        "test_or_online_targets_used": False,
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {
            "count": 0,
            "mean_N": None,
            "p90_N": None,
            "p95_N": None,
            "max_N": None,
        }
    return {
        "count": int(values.size),
        "mean_N": float(values.mean()),
        "p90_N": float(np.quantile(values, 0.90)),
        "p95_N": float(np.quantile(values, 0.95)),
        "max_N": float(values.max()),
    }


def _longest_interval_seconds(rows: list[dict[str, Any]], violation_key: str) -> float:
    intervals = sorted(
        (
            float(row["anchor_timestamp_s"]),
            float(row["anchor_timestamp_s"]) + WINDOW_SECONDS,
        )
        for row in rows
        if bool(row[violation_key])
    )
    if not intervals:
        return 0.0
    longest = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end + 1e-6:
            end = max(end, next_end)
        else:
            longest = max(longest, end - start)
            start, end = next_start, next_end
    return float(max(longest, end - start))


def _prepare_rows(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> None:
    q05 = float(calibration["q05_N"])
    q95 = float(calibration["q95_N"])
    for row in rows:
        force = float(row["observed_avg_N"])
        mu = float(row["mu_avg_N"])
        lower = mu + q05
        upper = mu + q95
        active = bool(row["gate_bilateral"]) and bool(row["quality_valid"]) and force >= 0.0
        low_deviation = max(0.0, lower - force) if active else 0.0
        high_deviation = max(0.0, force - upper) if active else 0.0
        row.update(
            lower_N=lower,
            upper_N=upper,
            analysis_gate=active,
            low_violation=bool(low_deviation > 0.0),
            high_violation=bool(high_deviation > 0.0),
            low_deviation_N=low_deviation,
            high_deviation_N=high_deviation,
        )


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["analysis_gate"]]
    low = np.asarray([row["low_deviation_N"] for row in active if row["low_violation"]])
    high = np.asarray([row["high_deviation_N"] for row in active if row["high_violation"]])
    denominator = max(len(active), 1)
    episodes = sorted({int(row["episode"]) for row in rows})
    return {
        "episodes": len(episodes),
        "all_windows": len(rows),
        "stable_quality_windows": len(active),
        "low_violation_windows": int(sum(row["low_violation"] for row in active)),
        "high_violation_windows": int(sum(row["high_violation"] for row in active)),
        "low_violation_ratio": float(sum(row["low_violation"] for row in active) / denominator),
        "high_violation_ratio": float(sum(row["high_violation"] for row in active) / denominator),
        "low_deviation": _distribution(low),
        "high_deviation": _distribution(high),
    }


def _episode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row["analysis_gate"]]
    denominator = max(len(active), 1)
    low = np.asarray([row["low_deviation_N"] for row in active if row["low_violation"]])
    high = np.asarray([row["high_deviation_N"] for row in active if row["high_violation"]])
    first = rows[0]
    return {
        "episode": int(first["episode"]),
        "result": str(first["result"]),
        "success": int(first["success"]),
        "all_windows": len(rows),
        "stable_quality_windows": len(active),
        "low_violation_windows": int(sum(row["low_violation"] for row in active)),
        "high_violation_windows": int(sum(row["high_violation"] for row in active)),
        "low_violation_ratio": float(sum(row["low_violation"] for row in active) / denominator),
        "high_violation_ratio": float(sum(row["high_violation"] for row in active) / denominator),
        "low_deviation_mean_N": _distribution(low)["mean_N"] or 0.0,
        "low_deviation_p95_N": _distribution(low)["p95_N"] or 0.0,
        "low_deviation_max_N": _distribution(low)["max_N"] or 0.0,
        "high_deviation_mean_N": _distribution(high)["mean_N"] or 0.0,
        "high_deviation_p95_N": _distribution(high)["p95_N"] or 0.0,
        "high_deviation_max_N": _distribution(high)["max_N"] or 0.0,
        "longest_low_violation_s": _longest_interval_seconds(rows, "low_violation"),
        "longest_high_violation_s": _longest_interval_seconds(rows, "high_violation"),
    }


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "timestamp_ns",
        "episode",
        "chunk",
        "force_avg",
        "mu_avg",
        "lower",
        "upper",
        "low_violation",
        "high_violation",
        "low_deviation_N",
        "high_deviation_N",
        "stable_contact_gate",
        "quality_valid",
        "success_or_failure",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": row["anchor_iso_utc"],
                    "timestamp_ns": row["anchor_timestamp_ns"],
                    "episode": row["episode"],
                    "chunk": row["chunk"],
                    "force_avg": row["observed_avg_N"],
                    "mu_avg": row["mu_avg_N"],
                    "lower": row["lower_N"],
                    "upper": row["upper_N"],
                    "low_violation": int(row["low_violation"]),
                    "high_violation": int(row["high_violation"]),
                    "low_deviation_N": row["low_deviation_N"],
                    "high_deviation_N": row["high_deviation_N"],
                    "stable_contact_gate": int(row["gate_bilateral"]),
                    "quality_valid": int(row["quality_valid"]),
                    "success_or_failure": row["result"],
                }
            )


def _plot_episode(path: Path, rows: list[dict[str, Any]]) -> None:
    times = np.asarray([row["anchor_timestamp_s"] for row in rows])
    times -= times[0]
    force = np.asarray([row["observed_avg_N"] for row in rows])
    mu = np.asarray([row["mu_avg_N"] for row in rows])
    lower = np.asarray([row["lower_N"] for row in rows])
    upper = np.asarray([row["upper_N"] for row in rows])
    gate = np.asarray([bool(row["gate_bilateral"]) for row in rows])
    low = np.asarray([bool(row["low_violation"]) for row in rows])
    high = np.asarray([bool(row["high_violation"]) for row in rows])

    fig, (axis, gate_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
        constrained_layout=True,
    )
    axis.plot(times, force, "o-", linewidth=2, markersize=4, label="observed force_avg")
    axis.plot(times, mu, "o-", linewidth=1.5, markersize=3, label="predicted mu_avg")
    axis.plot(times, lower, "--", linewidth=1.5, label="lower = mu + q05")
    axis.plot(times, upper, "--", linewidth=1.5, label="upper = mu + q95")
    if np.any(low):
        axis.scatter(times[low], force[low], marker="v", s=60, color="#2667ff", label="below lower")
    if np.any(high):
        axis.scatter(times[high], force[high], marker="^", s=60, color="#d94841", label="above upper")
    axis.set_ylabel("Force (N)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    axis.set_title(
        f"Episode {rows[0]['episode']} | {rows[0]['result']} | "
        "calibration-only bilateral force band"
    )
    for time_value, active in zip(times, gate, strict=True):
        color = "#22c55e" if active else "#cbd5e1"
        gate_axis.broken_barh(
            [(float(time_value), WINDOW_SECONDS)],
            (0.0, 1.0),
            facecolors=color,
            alpha=0.55 if active else 0.35,
        )
    gate_axis.scatter(
        times,
        gate.astype(float),
        color=np.where(gate, "#15803d", "#64748b"),
        s=20,
        zorder=3,
    )
    gate_axis.set_yticks([0, 1], ["off", "on"])
    gate_axis.set_ylim(-0.1, 1.1)
    gate_axis.set_ylabel("gate")
    gate_axis.set_xlabel("Seconds from first anchor")
    gate_axis.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _number(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _write_html(
    output: Path,
    report: dict[str, Any],
    episode_summaries: list[dict[str, Any]],
) -> Path:
    calibration = report["calibration"]
    success = report["groups"]["success"]
    failure = report["groups"]["failure"]
    group_rows = []
    for name, label in (
        ("success", "成功 Episode"),
        ("failure", "失败 Episode"),
        ("all", "全部 Episode"),
    ):
        values = report["groups"][name]
        group_rows.append(
            "<tr>"
            f"<td>{label}</td><td>{values['episodes']}</td>"
            f"<td>{values['stable_quality_windows']}</td>"
            f"<td>{_percent(values['low_violation_ratio'])}</td>"
            f"<td>{_percent(values['high_violation_ratio'])}</td>"
            f"<td>{_number(values['low_deviation']['mean_N'])}</td>"
            f"<td>{_number(values['low_deviation']['p95_N'])}</td>"
            f"<td>{_number(values['low_deviation']['max_N'])}</td>"
            f"<td>{_number(values['high_deviation']['mean_N'])}</td>"
            f"<td>{_number(values['high_deviation']['p95_N'])}</td>"
            f"<td>{_number(values['high_deviation']['max_N'])}</td>"
            "</tr>"
        )
    episode_rows = []
    figures = {"success": [], "failure": []}
    for row in episode_summaries:
        result = str(row["result"])
        episode = int(row["episode"])
        csv_rel = f"episodes/{result}/episode_{episode:03d}.csv"
        image_rel = f"figures/{result}/episode_{episode:03d}.png"
        figures[result].append(
            f"<figure><img src='{image_rel}'><figcaption>"
            f"Episode {episode} | <a href='{csv_rel}'>CSV</a></figcaption></figure>"
        )
        episode_rows.append(
            "<tr>"
            f"<td>{episode}</td><td>{html.escape(result)}</td>"
            f"<td>{row['stable_quality_windows']}</td>"
            f"<td>{_percent(row['low_violation_ratio'])}</td>"
            f"<td>{_percent(row['high_violation_ratio'])}</td>"
            f"<td>{row['longest_low_violation_s']:.3f}</td>"
            f"<td>{row['longest_high_violation_s']:.3f}</td>"
            f"<td><a href='{csv_rel}'>数据</a></td>"
            "</tr>"
        )
    style = """
body{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:0;background:#f5f7fa;color:#17202a}
main{max-width:1500px;margin:auto;background:white;padding:28px 34px}
h1,h2{color:#15324a} .note{background:#eef6ff;border-left:4px solid #2878b5;padding:12px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd5df;padding:7px;text-align:right}
th{background:#eaf0f5}th:first-child,td:first-child{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px}
figure{margin:0;border:1px solid #d8e0e8;padding:8px}img{width:100%;height:auto}figcaption{text-align:center}
code{background:#eef1f4;padding:2px 5px}.small{color:#52606d;font-size:13px}
"""
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Online RL 双边力区间离线回放</title><style>{style}</style></head><body><main>
<h1>Online RL 双边力上下界离线回放</h1>
<p class="note">本报告不修改模型、不重新训练、不接入 Online RL。q05/q95 仅由 calibration split
中双侧 anchor stable-contact 窗口的残差计算；测试和在线目标均未参与校准。</p>
<h2>定义与口径</h2>
<p><code>force_avg=(left_top2_mean+right_top2_mean)/2</code>，
<code>mu_avg=(mu_left+mu_right)/2</code>，
<code>lower=mu_avg+q05</code>，<code>upper=mu_avg+q95</code>。</p>
<p>仅当 anchor 时刻的双侧 stable-contact gate 为真、且未来窗口质量有效时统计越界。
gate 在对应未来 300 ms 内保持固定，不读取未来力来开关。连续时长按真实
<code>[anchor, anchor+300ms]</code> 区间合并，不把相隔的 chunk 当作连续接触。</p>
<h2>Calibration</h2>
<table><tr><th>样本数</th><th>q05 (N)</th><th>q95 (N)</th><th>残差均值 (N)</th><th>来源</th></tr>
<tr><td>{calibration['samples']}</td><td>{calibration['q05_N']:.4f}</td>
<td>{calibration['q95_N']:.4f}</td><td>{calibration['residual_mean_N']:.4f}</td>
<td>{html.escape(calibration['source'])}</td></tr></table>
<h2>成功/失败对比</h2>
<table><thead><tr><th>分组</th><th>Episode</th><th>有效 gate 窗口</th>
<th>低于下界比例</th><th>高于上界比例</th><th>低偏差均值 N</th><th>低偏差 P95 N</th>
<th>低偏差最大 N</th><th>高偏差均值 N</th><th>高偏差 P95 N</th><th>高偏差最大 N</th>
</tr></thead><tbody>{''.join(group_rows)}</tbody></table>
<h2>从当前数据可以观察到什么</h2>
<ul>
<li>失败回合低于下界比例为 <strong>{_percent(failure['low_violation_ratio'])}</strong>，
成功回合为 <strong>{_percent(success['low_violation_ratio'])}</strong>。比例差异有限，但失败回合
低力偏差均值为 <strong>{_number(failure['low_deviation']['mean_N'])} N</strong>，
明显大于成功回合的 <strong>{_number(success['low_deviation']['mean_N'])} N</strong>。</li>
<li>失败回合高于上界比例为 <strong>{_percent(failure['high_violation_ratio'])}</strong>，
成功回合为 <strong>{_percent(success['high_violation_ratio'])}</strong>。请结合逐 Episode 图判断
该现象是否集中于少数失败轨迹，而不是直接解释为普遍因果。</li>
<li>当前 Online RL 每个 chunk 只评估一个 300 ms Envelope，锚点之间通常存在未覆盖间隔。
因此报告中的“最长连续时长”按真实 300 ms 区间合并；不能把相邻 chunk 的两个越界窗口
直接视为连续 600 ms。</li>
<li>这些结果支持继续验证双边区间，但仍属于最近在线回合的回放关联分析，不应直接用于
调整 q05/q95，也没有证明低力或高力是任务失败的原因。</li>
</ul>
<h2>逐 Episode 汇总</h2>
<table><thead><tr><th>Episode</th><th>结果</th><th>有效 gate 窗口</th>
<th>低于下界比例</th><th>高于上界比例</th><th>最长连续低偏差 s</th>
<th>最长连续高偏差 s</th><th>CSV</th></tr></thead><tbody>
{''.join(episode_rows)}</tbody></table>
<h2>失败 Episode 时序图</h2><div class="grid">{''.join(figures['failure'])}</div>
<h2>成功 Episode 时序图</h2><div class="grid">{''.join(figures['success'])}</div>
<p class="small">原始运行目录：{html.escape(report['online_run_dir'])}<br>
模型 checkpoint：{html.escape(report['checkpoint'])}<br>
Sidecar 日志 SHA256：{report['sidecar_sha256']}</p>
</main></body></html>"""
    path = output / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-run", type=Path, required=True)
    parser.add_argument("--sidecar-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    online_run = args.online_run.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else online_run / "bilateral_band_analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(online_run / "tactile_chunk_metrics.jsonl")
    sidecar_events = _read_jsonl(args.sidecar_log.expanduser().resolve())
    _attach_timestamps(rows, sidecar_events)
    calibration = _calibration_quantiles(
        args.config.expanduser().resolve(), args.checkpoint.expanduser().resolve()
    )
    _prepare_rows(rows, calibration)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode"])].append(row)
    episode_summaries = []
    for episode, episode_rows in sorted(grouped.items()):
        episode_rows.sort(key=lambda row: int(row["chunk"]))
        result = str(episode_rows[0]["result"])
        csv_dir = output / "episodes" / result
        figure_dir = output / "figures" / result
        csv_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        _write_episode_csv(csv_dir / f"episode_{episode:03d}.csv", episode_rows)
        _plot_episode(figure_dir / f"episode_{episode:03d}.png", episode_rows)
        episode_summaries.append(_episode_summary(episode_rows))

    success_rows = [row for row in rows if str(row["result"]) == "success"]
    failure_rows = [row for row in rows if str(row["result"]) == "failure"]
    report = {
        "schema_version": 1,
        "analysis": "bilateral_calibration_only_two_sided_band_online_replay",
        "online_run_dir": str(online_run),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "config": str(args.config.expanduser().resolve()),
        "sidecar_log": str(args.sidecar_log.expanduser().resolve()),
        "sidecar_sha256": hashlib.sha256(
            args.sidecar_log.expanduser().resolve().read_bytes()
        ).hexdigest(),
        "window_ms": int(WINDOW_SECONDS * 1000),
        "calibration": calibration,
        "gate": {
            "definition": "gate_left_at_anchor AND gate_right_at_anchor",
            "future_force_used": False,
            "held_for_window_ms": int(WINDOW_SECONDS * 1000),
        },
        "groups": {
            "success": _group_summary(success_rows),
            "failure": _group_summary(failure_rows),
            "all": _group_summary(rows),
        },
        "episode_summaries": episode_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary_csv(output / "per_episode_summary.csv", episode_summaries)
    report_path = _write_html(output, report, episode_summaries)
    print(json.dumps({"output": str(output), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
