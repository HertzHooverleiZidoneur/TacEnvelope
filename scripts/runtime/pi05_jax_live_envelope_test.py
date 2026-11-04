#!/usr/bin/env python3
"""Run Pi0.5 JAX and its matching force Envelope on a bounded Franka episode."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

_CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT))
    plt.rcParams["font.family"] = [
        font_manager.FontProperties(fname=str(_CJK_FONT)).get_name(),
        "DejaVu Sans",
    ]
plt.rcParams["axes.unicode_minus"] = False

from franka_pi05_tcp_client import (  # noqa: E402
    CAMERA_MAP,
    observation_to_openpi_raw,
    request,
    reset_remote,
    wait_for_queue_execution,
)
from rlt_openpi.policies.franka.jax_online_policy import JaxPi05OnlinePolicy  # noqa: E402
from tactile_force_envelope.data import median_filter  # noqa: E402
from tactile_force_envelope.pi05_online_service import (  # noqa: E402
    FEATURE_SOURCE,
    XenseRingClient,
    corrected_force,
    row_valid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/live_tests/pi05_jax_live_envelope_tests"
DEFAULT_CHECKPOINT = Path.home() / "Documents/tactile_model/24999"
DEFAULT_TASK = "Insert the red USB flash drive into the USB port on the black power strip"
BRIDGE_IMAGE_KEYS = tuple(CAMERA_MAP.values())
SIDES = ("left", "right")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sidecar_request(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(struct.pack("!I", len(body)) + body)
        header = sock.recv(4)
        if len(header) != 4:
            raise ConnectionError("Envelope sidecar returned a short header")
        size = struct.unpack("!I", header)[0]
        response = b""
        while len(response) < size:
            chunk = sock.recv(size - len(response))
            if not chunk:
                raise ConnectionError("Envelope sidecar closed the socket")
            response += chunk
    result = json.loads(response)
    if not result.get("ok", False):
        raise RuntimeError(str(result.get("error", "Envelope sidecar rejected request")))
    return result


def save_anchor_images(observation: dict[str, Any], output: Path, index: int) -> dict[str, str]:
    image_dir = output / "anchor_images"
    image_dir.mkdir(exist_ok=True)
    result: dict[str, str] = {}
    for camera_index, key in enumerate(BRIDGE_IMAGE_KEYS, start=1):
        path = image_dir / f"chunk_{index:04d}_camera_{camera_index}.jpg"
        path.write_bytes(base64.b64decode(observation["images"][key]["data"]))
        result[key] = str(path.relative_to(output))
    return result


def window_label(
    rows: list[dict[str, Any]],
    baseline: np.ndarray,
    start_ns: int,
    window_ms: int,
) -> dict[str, Any]:
    end_ns = start_ns + int(window_ms * 1e6)
    selected = [
        row
        for row in rows
        if start_ns <= int(row["pc_time_ns"]) < end_ns and row_valid(row)
    ]
    timestamps = np.asarray([int(row["pc_time_ns"]) for row in selected], dtype=np.int64)
    raw = np.asarray(
        [corrected_force(row, baseline) for row in selected],
        dtype=np.float32,
    )
    filtered = median_filter(raw, 3) if raw.size else raw
    ordered = np.sort(filtered)[::-1]
    return {
        "sample_count": len(selected),
        "quality_ok": len(selected) >= 4,
        "top2_mean_N": float(np.mean(ordered[:2])) if ordered.size else float("nan"),
        "raw_max_N": float(np.max(raw)) if raw.size else float("nan"),
        "max_gap_ms": (
            float(np.max(np.diff(timestamps)) / 1e6)
            if timestamps.size >= 2
            else None
        ),
        "timestamps_ns": timestamps.tolist(),
        "raw_force_N": raw.tolist(),
        "filtered_force_N": filtered.tolist(),
    }


def merge_rows(
    destination: dict[str, dict[tuple[int, int], dict[str, Any]]],
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    for side in SIDES:
        for row in rows.get(side, []):
            key = (int(row["pc_time_ns"]), int(row.get("source_frame", -1)))
            destination[side][key] = row


class TactileCollector:
    """Continuously copy Xense force rows without blocking the control loop."""

    def __init__(
        self,
        source: XenseRingClient,
        *,
        poll_interval_s: float = 0.5,
        overlap_ms: int = 100,
    ) -> None:
        self.source = source
        self.poll_interval_s = float(poll_interval_s)
        self.overlap_ns = int(overlap_ms * 1e6)
        self.rows: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
            side: {} for side in SIDES
        }
        self.errors: list[str] = []
        self._cursor_ns = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, start_ns: int) -> None:
        self._cursor_ns = int(start_ns)
        self._thread = threading.Thread(
            target=self._run,
            name="xense-force-collector",
            daemon=True,
        )
        self._thread.start()

    def _collect_until(self, end_ns: int) -> None:
        start_ns = max(0, self._cursor_ns - self.overlap_ns)
        rows = self.source.recent(start_ns, int(end_ns))
        with self._lock:
            merge_rows(self.rows, rows)
        self._cursor_ns = int(end_ns)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            try:
                self._collect_until(time.time_ns())
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def stop(self, end_ns: int) -> dict[str, list[dict[str, Any]]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_interval_s * 4.0))
        try:
            self._collect_until(int(end_ns))
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
        with self._lock:
            return {
                side: sorted(
                    self.rows[side].values(),
                    key=lambda row: int(row["pc_time_ns"]),
                )
                for side in SIDES
            }


def enrich_event(
    event: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    window_ms: int,
) -> dict[str, Any]:
    result = dict(event)
    for side in SIDES:
        label = window_label(rows[side], baselines[side], int(event["action_anchor_ns"]), window_ms)
        target = float(label["top2_mean_N"])
        mu = float(event[f"mu_{side}"])
        upper = float(event[f"upper95_{side}"])
        result.update(
            {
                f"target_{side}_N": target,
                f"raw_max_{side}_N": float(label["raw_max_N"]),
                f"sample_count_{side}": int(label["sample_count"]),
                f"quality_{side}": bool(label["quality_ok"]),
                f"max_gap_{side}_ms": label["max_gap_ms"],
                f"absolute_error_{side}_N": abs(target - mu),
                f"deviation_{side}_N": target - upper,
                f"upper95_{side}_N": upper,
                f"covered_{side}": bool(target <= upper) if np.isfinite(target) else False,
                f"label_{side}": label,
            }
        )
    result["target_avg_N"] = 0.5 * (
        float(result["target_left_N"]) + float(result["target_right_N"])
    )
    result["mu_avg_N"] = 0.5 * (
        float(result["mu_left"]) + float(result["mu_right"])
    )
    # This is the average of the two independently calibrated side envelopes.
    result["upper95_avg_N"] = 0.5 * (
        float(result["upper95_left"]) + float(result["upper95_right"])
    )
    result["sigma_avg_N"] = 0.5 * float(
        np.hypot(result["sigma_left"], result["sigma_right"])
    )
    result["stable_contact_avg"] = bool(
        result["stable_contact_left"] and result["stable_contact_right"]
    )
    result["quality_avg"] = bool(result["quality_left"] and result["quality_right"])
    result["absolute_error_avg_N"] = abs(result["target_avg_N"] - result["mu_avg_N"])
    result["deviation_avg_N"] = result["target_avg_N"] - result["upper95_avg_N"]
    result["covered_avg"] = bool(result["target_avg_N"] <= result["upper95_avg_N"])
    return result


def metric_summary(rows: list[dict[str, Any]], side: str, gated: bool) -> dict[str, Any]:
    gate_key = "stable_contact_avg" if side == "avg" else f"stable_contact_{side}"
    selected = [
        row
        for row in rows
        if row[f"quality_{side}"]
        and np.isfinite(float(row[f"target_{side}_N"]))
        and (not gated or row[gate_key])
    ]
    if not selected:
        return {
            "windows": 0,
            "mae_N": None,
            "coverage": None,
            "mean_target_N": None,
            "mean_upper_N": None,
            "exceedance_windows": 0,
        }
    return {
        "windows": len(selected),
        "mae_N": float(np.mean([row[f"absolute_error_{side}_N"] for row in selected])),
        "coverage": float(np.mean([row[f"covered_{side}"] for row in selected])),
        "mean_target_N": float(np.mean([row[f"target_{side}_N"] for row in selected])),
        "mean_upper_N": float(np.mean([row[f"upper95_{side}_N"] for row in selected])),
        "exceedance_windows": int(np.sum([row[f"deviation_{side}_N"] > 0 for row in selected])),
    }


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_report(
    rows: list[dict[str, Any]],
    tactile_rows: dict[str, list[dict[str, Any]]],
    baselines: dict[str, np.ndarray],
    output: Path,
) -> None:
    origin = int(rows[0]["action_anchor_ns"])
    times = np.asarray([(int(row["action_anchor_ns"]) - origin) / 1e9 for row in rows])
    colors = {"left": "#2563a6", "right": "#d9485f"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, SIDES, strict=True):
        target = np.asarray([row[f"target_{side}_N"] for row in rows])
        mu = np.asarray([row[f"mu_{side}"] for row in rows])
        upper = np.asarray([row[f"upper95_{side}"] for row in rows])
        gate = np.asarray([row[f"stable_contact_{side}"] for row in rows], dtype=bool)
        axis.fill_between(times, mu, upper, color=colors[side], alpha=0.16)
        axis.plot(times, mu, "--", color=colors[side], label="预测均值 μ")
        axis.plot(times, upper, color=colors[side], label="校准上界 U95")
        axis.plot(times, target, "o-", color="#202938", label="真实未来 300 ms top2")
        axis.scatter(
            times[gate],
            target[gate],
            s=65,
            facecolors="none",
            edgecolors="#16a34a",
            label="稳定接触 gate",
        )
        exceeded = target > upper
        axis.scatter(times[exceeded], target[exceeded], color="#dc2626", marker="x", s=75)
        axis.set_ylabel(f"{side} force (N)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=4, fontsize=9)
    axes[-1].set_xlabel("相对首个 action chunk 的时间 t (s)")
    fig.suptitle("Pi0.5 JAX：真实触觉力与左右 Force Envelope")
    save_figure(fig, output, "force_and_envelope")

    fig, axis = plt.subplots(figsize=(12, 6))
    target = np.asarray([row["target_avg_N"] for row in rows])
    mu = np.asarray([row["mu_avg_N"] for row in rows])
    upper = np.asarray([row["upper95_avg_N"] for row in rows])
    gate = np.asarray([row["stable_contact_avg"] for row in rows], dtype=bool)
    axis.fill_between(times, mu, upper, color="#0f766e", alpha=0.17)
    axis.plot(times, mu, "--", color="#0f766e", label="左右平均 μ")
    axis.plot(times, upper, color="#0f766e", label="左右独立 U95 的平均")
    axis.plot(times, target, "o-", color="#202938", label="真实左右平均力")
    axis.scatter(times[gate], target[gate], facecolors="none", edgecolors="#16a34a", s=75)
    exceeded = target > upper
    axis.scatter(times[exceeded], target[exceeded], color="#dc2626", marker="x", s=80)
    axis.set_xlabel("相对首个 action chunk 的时间 t (s)")
    axis.set_ylabel("bilateral average force (N)")
    axis.set_title("左右平均力与 Envelope（诊断值）")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, output, "bilateral_average_force_and_envelope")

    fig, axis = plt.subplots(figsize=(12, 5.5))
    for side in (*SIDES, "avg"):
        deviation = np.asarray([row[f"deviation_{side}_N"] for row in rows])
        axis.plot(times, deviation, "o-", label=f"{side}: actual - upper")
    axis.axhline(0.0, color="#111827")
    axis.set_xlabel("相对首个 action chunk 的时间 t (s)")
    axis.set_ylabel("真实力 - 预测上界 (N)")
    axis.set_title("Envelope 偏差；大于 0 表示真实力超出上界")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, output, "envelope_deviation")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for axis, side in zip(axes, SIDES, strict=True):
        samples = tactile_rows[side]
        sample_t = np.asarray([(int(row["pc_time_ns"]) - origin) / 1e9 for row in samples])
        sample_f = np.asarray(
            [corrected_force(row, baselines[side]) if row_valid(row) else np.nan for row in samples]
        )
        axis.plot(sample_t, sample_f, ".", color="#6b7280", markersize=2.5)
        for index, row in enumerate(rows):
            axis.hlines(
                row[f"upper95_{side}"],
                times[index],
                times[index] + 0.3,
                colors=colors[side],
                linewidth=3,
            )
        axis.set_ylabel(f"{side} force (N)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("相对首个 action chunk 的时间 t (s)")
    fig.suptitle("22 Hz 原始触觉与每个 300 ms 预测上界")
    save_figure(fig, output, "raw_force_timeline")


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "chunk_index",
        "time_t_s",
        "infer_s",
        "prediction_to_action_ms",
        "mu_left",
        "sigma_left",
        "upper95_left",
        "target_left_N",
        "deviation_left_N",
        "stable_contact_left",
        "mu_right",
        "sigma_right",
        "upper95_right",
        "target_right_N",
        "deviation_right_N",
        "stable_contact_right",
        "mu_avg_N",
        "sigma_avg_N",
        "upper95_avg_N",
        "target_avg_N",
        "deviation_avg_N",
        "stable_contact_avg",
    ]
    origin = int(rows[0]["action_anchor_ns"])
    with (output / "prediction_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compact = {field: row.get(field) for field in fields}
            compact["time_t_s"] = (int(row["action_anchor_ns"]) - origin) / 1e9
            writer.writerow(compact)


def render_html(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    origin = int(rows[0]["action_anchor_ns"])
    table_rows = []
    for row in rows:
        t = (int(row["action_anchor_ns"]) - origin) / 1e9
        table_rows.append(
            "<tr>"
            f"<td>{row['chunk_index']}</td><td>{t:.2f}</td>"
            f"<td>{row['target_left_N']:.3f}</td><td>{row['mu_left']:.3f}</td>"
            f"<td>{row['sigma_left']:.3f}</td><td>{row['upper95_left']:.3f}</td>"
            f"<td class={'bad' if row['deviation_left_N'] > 0 else 'good'}>{row['deviation_left_N']:.3f}</td>"
            f"<td>{row['target_right_N']:.3f}</td><td>{row['mu_right']:.3f}</td>"
            f"<td>{row['sigma_right']:.3f}</td><td>{row['upper95_right']:.3f}</td>"
            f"<td class={'bad' if row['deviation_right_N'] > 0 else 'good'}>{row['deviation_right_N']:.3f}</td>"
            f"<td>{row['target_avg_N']:.3f}</td><td>{row['upper95_avg_N']:.3f}</td>"
            "</tr>"
        )
    metric_rows = []
    for group in ("all_windows", "stable_gate_windows"):
        for side in (*SIDES, "avg"):
            metric = summary["metrics"][group][side]
            coverage = "-" if metric["coverage"] is None else f"{metric['coverage']*100:.1f}%"
            mae = "-" if metric["mae_N"] is None else f"{metric['mae_N']:.3f} N"
            metric_rows.append(
                f"<tr><td>{group}</td><td>{side}</td><td>{metric['windows']}</td>"
                f"<td>{coverage}</td><td>{mae}</td><td>{metric['exceedance_windows']}</td></tr>"
            )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Pi0.5 JAX + 触觉力包络实机报告</title><style>
body{{font-family:Arial,"Noto Sans CJK SC",sans-serif;margin:30px;color:#172033;max-width:1320px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #cbd5e1;padding:6px;text-align:right}}
th{{background:#eef2f7}}.bad{{color:#b91c1c;font-weight:700}}.good{{color:#166534}}
img{{max-width:100%;border:1px solid #d6dde8;margin:8px 0 24px}}
.note{{background:#edf6ff;border-left:4px solid #2563a6;padding:12px}}
</style></head><body><h1>Pi0.5 JAX + Force Envelope 实机测试</h1>
<div class="note">动作和 z_rl 来自同一次 Pi0.5 JAX forward。真实标签是 action 开始后 300 ms
内的 22 Hz 真实触觉，经零偏和三点中值滤波后的 top-2 均值。偏差 actual-upper 大于 0
表示真实力超出预测上界。</div>
<h2>汇总</h2><table><tr><th>范围</th><th>侧</th><th>窗口</th><th>Coverage</th><th>μ MAE</th><th>超界</th></tr>
{''.join(metric_rows)}</table>
<h2>左右真实力与 Envelope</h2><img src="force_and_envelope.png">
<h2>左右平均力</h2><img src="bilateral_average_force_and_envelope.png">
<h2>预测偏差</h2><img src="envelope_deviation.png">
<h2>原始触觉</h2><img src="raw_force_timeline.png">
<h2>逐窗口</h2><table><tr><th>Chunk</th><th>t</th><th>L actual</th><th>L μ</th><th>L σ</th><th>L U95</th><th>L 偏差</th>
<th>R actual</th><th>R μ</th><th>R σ</th><th>R U95</th><th>R 偏差</th><th>Avg actual</th><th>Avg upper</th></tr>
{''.join(table_rows)}</table>
<p>Pi0.5 checkpoint：{html.escape(summary['jax_checkpoint'])}<br>
Envelope checkpoint：{html.escape(summary['envelope_checkpoint'])}</p>
<p>左右平均上界是两个独立校准上界的算术平均，仅用于诊断；没有使用本次实机标签重新校准。</p>
</body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def self_test_report(output_root: Path) -> Path:
    output = output_root / f"self_test_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    origin_ns = time.time_ns()
    baselines = {side: np.zeros(3, dtype=np.float32) for side in SIDES}
    tactile_rows: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    for sample_index in range(90):
        timestamp_ns = origin_ns + int(sample_index / 22.0 * 1e9)
        for side_index, side in enumerate(SIDES):
            magnitude = 2.0 + side_index + 1.2 * np.sin(sample_index / 8.0)
            tactile_rows[side].append(
                {
                    "pc_time_ns": timestamp_ns,
                    "source_frame": sample_index,
                    "feature_names": [
                        "sample_valid",
                        "force_valid",
                        "force_fx",
                        "force_fy",
                        "force_fz",
                    ],
                    "values": [1.0, 1.0, float(magnitude), 0.0, 0.0],
                }
            )

    events: list[dict[str, Any]] = []
    for chunk_index in range(8):
        anchor_ns = origin_ns + int(chunk_index * 0.4 * 1e9)
        row: dict[str, Any] = {
            "chunk_index": chunk_index,
            "prediction_anchor_ns": anchor_ns - int(8e6),
            "action_anchor_ns": anchor_ns,
            "prediction_to_action_ms": 8.0,
            "infer_s": 0.11,
            "mu_left": 2.25 + 0.12 * chunk_index,
            "sigma_left": 0.65,
            "upper95_left": 3.45 + 0.12 * chunk_index,
            "stable_contact_left": chunk_index >= 2,
            "mu_right": 3.05 + 0.10 * chunk_index,
            "sigma_right": 0.72,
            "upper95_right": 4.35 + 0.10 * chunk_index,
            "stable_contact_right": chunk_index >= 2,
        }
        events.append(enrich_event(row, tactile_rows, baselines, 300))

    write_json(output / "prediction_windows.json", events)
    write_csv(events, output)
    plot_report(events, tactile_rows, baselines, output)
    summary = {
        "jax_checkpoint": str(DEFAULT_CHECKPOINT),
        "envelope_checkpoint": "self-test",
        "task": DEFAULT_TASK,
        "send_actions": False,
        "executed_chunks": len(events),
        "window_ms": 300,
        "history_ms": 200,
        "metrics": {
            group: {
                side: metric_summary(events, side, gated=group == "stable_gate_windows")
                for side in (*SIDES, "avg")
            }
            for group in ("all_windows", "stable_gate_windows")
        },
    }
    write_json(output / "summary.json", summary)
    render_html(events, summary, output)
    expected = (
        "report.html",
        "summary.json",
        "prediction_windows.csv",
        "force_and_envelope.png",
        "force_and_envelope.pdf",
        "bilateral_average_force_and_envelope.png",
        "envelope_deviation.png",
        "raw_force_timeline.png",
    )
    missing = [name for name in expected if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Report self-test missed outputs: {missing}")
    print(f"[pi05_envelope] report self-test passed: {output / 'report.html'}")
    return output


def run(args: argparse.Namespace) -> Path:
    output = args.output_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "command_config.json", vars(args))

    sidecar_metadata = sidecar_request(
        args.sidecar_host,
        args.sidecar_port,
        {"type": "metadata", "schema_version": 1},
    )
    expected_checkpoint = sidecar_metadata.get("expected_policy_checkpoint")
    if expected_checkpoint and Path(expected_checkpoint).resolve() != args.jax_checkpoint:
        raise RuntimeError(
            f"Envelope expects JAX checkpoint {expected_checkpoint}, got {args.jax_checkpoint}"
        )
    if sidecar_metadata.get("policy_feature_source") != FEATURE_SOURCE:
        raise RuntimeError(f"Unexpected sidecar feature source: {sidecar_metadata}")
    baseline_report = sidecar_metadata["baseline"]
    baselines = {
        side: np.asarray(baseline_report[side]["baseline_xyz"], dtype=np.float32)
        for side in SIDES
    }
    xense = XenseRingClient(args.xense_socket)
    policy = JaxPi05OnlinePolicy(args.jax_host, args.jax_port)
    if policy.z_dim != 512 or policy.action_dim != 8:
        raise RuntimeError(f"Unexpected JAX policy contract: {policy.metadata}")
    write_json(
        output / "runtime_contract.json",
        {
            "jax_metadata": policy.metadata,
            "sidecar_metadata": sidecar_metadata,
        },
    )
    sidecar_request(args.sidecar_host, args.sidecar_port, {"type": "reset_episode"})

    events: list[dict[str, Any]] = []
    pending: list[
        tuple[
            dict[str, Any],
            Future[dict[str, Any]],
            dict[str, Any],
        ]
    ] = []
    collector = TactileCollector(
        xense,
        poll_interval_s=args.tactile_poll_interval,
    )
    collector.start(time.time_ns() - int(args.history_ms * 1e6))
    prediction_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="envelope-predict",
    )
    sock = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
    sock.settimeout(args.socket_timeout)
    previous_chunk_size = 0
    completed_limit = False
    last_action_anchor_ns: int | None = None
    control_error: str | None = None
    try:
        status = request(sock, {"type": "status"})
        if not status.get("ok"):
            raise RuntimeError(f"NUC runtime is not ready: {status}")
        if args.send_actions and not status.get("execute"):
            raise RuntimeError("--send-actions requires NUC execute=True")
        if args.reset_before_run:
            reset_remote(sock, args)

        chunk_index = 0
        while args.max_chunks == 0 or chunk_index < args.max_chunks:
            if args.send_actions and args.queue_aware and previous_chunk_size > 0:
                min_execute = args.queue_min_execute or previous_chunk_size
                wait_for_queue_execution(
                    sock,
                    sent_chunk_size=previous_chunk_size,
                    min_execute=min_execute,
                    poll_interval=args.queue_poll_interval,
                    timeout=args.queue_wait_timeout,
                )

            observation = request(
                sock,
                {"type": "get_observation", "client_sent_at": time.time()},
            )
            if not observation.get("ok"):
                raise RuntimeError(f"Observation unavailable: {observation.get('missing')}")
            raw_observation = observation_to_openpi_raw(observation, args.task)

            infer_started = time.perf_counter()
            processed = policy.preprocess_obs(raw_observation)
            infer_s = time.perf_counter() - infer_started
            z_rl = processed["z_rl"][0].detach().cpu().numpy()
            model_state = processed["state"][0].detach().cpu().numpy()
            model_horizon = int(processed["ref_chunk"].shape[1])
            send_chunk_size = args.send_chunk_size or model_horizon
            send_chunk_size = min(send_chunk_size, model_horizon)
            action_chunk = (
                processed["ref_chunk"][0, :send_chunk_size, :8]
                .detach()
                .cpu()
                .numpy()
            )
            if len(action_chunk) < 9:
                raise RuntimeError(
                    f"Envelope requires at least 9 actions, got {len(action_chunk)}"
                )

            prediction_anchor_ns = time.time_ns()
            sent_ns = time.time_ns()
            if args.send_actions:
                acknowledgement = request(
                    sock,
                    {
                        "type": "enqueue_actions",
                        "actions": action_chunk.tolist(),
                        "replace": True,
                    },
                )
                if not acknowledgement.get("ok"):
                    raise RuntimeError(f"NUC rejected action chunk: {acknowledgement}")
            else:
                acknowledgement = {"ok": True, "execute": False}
            acknowledged_ns = time.time_ns()
            action_anchor_ns = (sent_ns + acknowledged_ns) // 2
            previous_chunk_size = len(action_chunk)
            last_action_anchor_ns = action_anchor_ns

            # Action delivery has priority. Envelope inference runs after enqueue
            # on one ordered worker and therefore cannot delay the control loop.
            prediction_future = prediction_executor.submit(
                sidecar_request,
                args.sidecar_host,
                args.sidecar_port,
                {
                    "type": "predict_chunk",
                    "schema_version": 1,
                    "policy_feature_source": FEATURE_SOURCE,
                    "policy_feature": z_rl.tolist(),
                    "robot_state": model_state.tolist(),
                    "proposed_action_chunk": action_chunk.tolist(),
                    "anchor_unix_ns": action_anchor_ns,
                },
            )
            event = {
                "chunk_index": chunk_index,
                "prediction_anchor_ns": prediction_anchor_ns,
                "action_anchor_ns": action_anchor_ns,
                "prediction_to_action_ms": (action_anchor_ns - prediction_anchor_ns) / 1e6,
                "infer_s": infer_s,
                "state": model_state,
                "z_rl_norm": float(np.linalg.norm(z_rl)),
                "action_chunk": action_chunk,
                "first_action": action_chunk[0],
                "last_action": action_chunk[-1],
                "image_age_s": {
                    key: float(value["age"])
                    for key, value in observation["images"].items()
                },
                "acknowledgement": acknowledgement,
            }
            pending.append((event, prediction_future, observation))
            print(
                f"[pi05_envelope] chunk={chunk_index + 1}"
                f"{f'/{args.max_chunks}' if args.max_chunks else ''} "
                f"send={args.send_actions} actions={len(action_chunk)}/{model_horizon} "
                f"infer={infer_s:.3f}s enqueue_before_envelope=True"
            )
            chunk_index += 1
        completed_limit = True
    except KeyboardInterrupt:
        print("\n[pi05_envelope] interrupted; stopping control and processing saved data")
    except Exception as exc:
        control_error = f"{type(exc).__name__}: {exc}"
        print(
            f"[pi05_envelope] control stopped with error: {control_error}; "
            "processing captured data"
        )
    finally:
        try:
            if (
                completed_limit
                and args.send_actions
                and args.queue_aware
                and previous_chunk_size > 0
            ):
                min_execute = args.queue_min_execute or previous_chunk_size
                wait_for_queue_execution(
                    sock,
                    sent_chunk_size=previous_chunk_size,
                    min_execute=min_execute,
                    poll_interval=args.queue_poll_interval,
                    timeout=args.queue_wait_timeout,
                )
        except Exception as exc:
            completion_error = f"{type(exc).__name__}: {exc}"
            control_error = control_error or completion_error
            print(
                f"[pi05_envelope] final queue wait failed: {completion_error}; "
                "holding robot and processing captured data"
            )
        finally:
            try:
                request(sock, {"type": "hold"})
            except Exception:
                pass
            sock.close()

    if not pending:
        prediction_executor.shutdown(wait=False, cancel_futures=True)
        collector.stop(time.time_ns())
        raise RuntimeError(f"No prediction windows were recorded; inspect {output}")

    # Ensure the final real 300 ms target window has arrived, then stop the
    # background force collector. All image writes and report work happen here.
    if last_action_anchor_ns is not None:
        remaining_s = (
            last_action_anchor_ns + int(args.window_ms * 1e6) - time.time_ns()
        ) / 1e9
        if remaining_s > 0:
            time.sleep(remaining_s)
    tactile_rows = collector.stop(time.time_ns() + int(0.05e9))
    prediction_executor.shutdown(wait=True)

    for event, prediction_future, observation in pending:
        prediction_response = prediction_future.result()
        event.update(prediction_response["prediction"])
        event["history_report"] = prediction_response["history_report"]
        event["image_paths"] = save_anchor_images(
            observation,
            output,
            int(event["chunk_index"]),
        )
        enriched = enrich_event(event, tactile_rows, baselines, args.window_ms)
        events.append(enriched)
        with (output / "prediction_windows.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, default=json_default) + "\n")

    for side in SIDES:
        with (output / f"tactile_{side}.jsonl").open("w", encoding="utf-8") as handle:
            for row in tactile_rows[side]:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    write_json(output / "prediction_windows.json", events)
    write_csv(events, output)
    plot_report(events, tactile_rows, baselines, output)
    summary = {
        "jax_checkpoint": str(args.jax_checkpoint),
        "envelope_checkpoint": str(sidecar_metadata["checkpoint"]),
        "task": args.task,
        "send_actions": args.send_actions,
        "executed_chunks": len(events),
        "window_ms": args.window_ms,
        "history_ms": args.history_ms,
        "control_path": {
            "action_enqueued_before_envelope": True,
            "images_written_after_control": True,
            "tactile_collected_in_background": True,
            "model_horizon": int(len(events[0]["action_chunk"])),
            "max_chunks": args.max_chunks,
            "collector_errors": collector.errors,
            "control_error": control_error,
        },
        "metrics": {
            group: {
                side: metric_summary(events, side, gated=group == "stable_gate_windows")
                for side in (*SIDES, "avg")
            }
            for group in ("all_windows", "stable_gate_windows")
        },
        "average_envelope_note": (
            "upper95_avg is the arithmetic mean of independently calibrated left/right "
            "upper95 values; live targets were not used for calibration"
        ),
    }
    write_json(output / "summary.json", summary)
    render_html(events, summary, output)
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"[pi05_envelope] report={output / 'report.html'}")
    if control_error:
        raise RuntimeError(
            f"Control stopped early ({control_error}); captured data was saved to {output}"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jax-host", default="127.0.0.1")
    parser.add_argument("--jax-port", type=int, default=8011)
    parser.add_argument("--jax-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sidecar-host", default="127.0.0.1")
    parser.add_argument("--sidecar-port", type=int, default=8021)
    parser.add_argument(
        "--xense-socket",
        type=Path,
        default=Path.home() / ".cache/franka_xense_capture/control.sock",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="0 runs until Ctrl-C; a positive value is only for bounded tests.",
    )
    parser.add_argument(
        "--send-chunk-size",
        type=int,
        default=0,
        help="0 uses the policy's native action horizon.",
    )
    parser.add_argument(
        "--queue-min-execute",
        type=int,
        default=0,
        help="0 executes the full sent chunk before replanning.",
    )
    parser.add_argument(
        "--queue-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--queue-poll-interval", type=float, default=0.03)
    parser.add_argument("--queue-wait-timeout", type=float, default=8.0)
    parser.add_argument("--tactile-poll-interval", type=float, default=0.5)
    parser.add_argument("--window-ms", type=int, default=300)
    parser.add_argument("--history-ms", type=int, default=200)
    parser.add_argument("--reset-before-run", action="store_true")
    parser.add_argument("--reset-seconds", type=float, default=8.0)
    parser.add_argument("--reset-timeout", type=float, default=90.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--self-test-report",
        action="store_true",
        help="Generate a synthetic report without connecting to JAX, ROS, or the robot.",
    )
    args = parser.parse_args()
    args.jax_checkpoint = args.jax_checkpoint.expanduser().resolve()
    args.xense_socket = args.xense_socket.expanduser()
    args.output_root = args.output_root.expanduser().resolve()
    if args.max_chunks < 0:
        parser.error("--max-chunks must be non-negative")
    if args.send_chunk_size < 0 or 0 < args.send_chunk_size < 9:
        parser.error("--send-chunk-size must be 0 or at least 9")
    if args.queue_min_execute < 0:
        parser.error("--queue-min-execute must be non-negative")
    if (
        args.send_chunk_size > 0
        and args.queue_min_execute > args.send_chunk_size
    ):
        parser.error("--queue-min-execute cannot exceed --send-chunk-size")
    if args.tactile_poll_interval <= 0:
        parser.error("--tactile-poll-interval must be positive")
    if args.window_ms != 300 or args.history_ms != 200:
        parser.error("This checkpoint requires window-ms=300 and history-ms=200")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.self_test_report:
        self_test_report(parsed_args.output_root)
    else:
        run(parsed_args)
