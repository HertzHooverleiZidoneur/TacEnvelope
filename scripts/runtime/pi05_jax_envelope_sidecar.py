#!/usr/bin/env python3
"""Serve Pi0.5 JAX z_rl force-envelope costs to Online RL."""

from __future__ import annotations

import argparse
import json
import signal
import socketserver
import struct
import threading
import time
from pathlib import Path
from typing import Any

import torch

from tactile_force_envelope.inference import TransformerForceEnvelopePredictor
from tactile_force_envelope.pi05_online_service import (
    Pi05EnvelopeCostService,
    XenseRingClient,
    calculate_baselines,
    checkpoint_sha256,
)


class EnvelopeRequestHandler(socketserver.BaseRequestHandler):
    @staticmethod
    def recv_exact(sock: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("client closed connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def handle(self) -> None:
        while True:
            try:
                header = self.recv_exact(self.request, 4)
            except ConnectionError:
                return
            try:
                size = struct.unpack("!I", header)[0]
                if size > 16 * 1024 * 1024:
                    raise ValueError(f"request too large: {size}")
                payload = json.loads(self.recv_exact(self.request, size))
                response = self.server.service.handle(payload)
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            body = json.dumps(response, separators=(",", ":")).encode()
            self.request.sendall(struct.pack("!I", len(body)) + body)


class EnvelopeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: Pi05EnvelopeCostService):
        self.service = service
        super().__init__(address, EnvelopeRequestHandler)


def calibrate_baseline(
    source: XenseRingClient,
    *,
    duration_s: float,
    max_residual_p95: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(
        f"[envelope_sidecar] collecting {duration_s:.1f}s no-contact baseline; "
        "keep both tactile sensors unloaded",
        flush=True,
    )
    started_ns = time.time_ns()
    time.sleep(duration_s)
    ended_ns = time.time_ns()
    rows = source.recent(started_ns, ended_ns)
    baselines, report = calculate_baselines(rows)
    noisy = {
        side: values["residual_p95_N"]
        for side, values in report.items()
        if values["residual_p95_N"] > max_residual_p95
    }
    if noisy:
        raise RuntimeError(
            "No-contact baseline is too noisy or loaded: "
            f"p95={noisy}, allowed<={max_residual_p95:.3f} N"
        )
    print(f"[envelope_sidecar] baseline={json.dumps(report)}", flush=True)
    return baselines, report


def parse_args() -> argparse.Namespace:
    default_checkpoint = (
        Path.home()
        / "Documents/tactile_force_envelope/outputs/training/franka_tactile310"
        / "pi05_jax_zrl_envelope_v1/sustained_contact_seed_44/stable_evaluation"
        / "stable_contact_calibrated_checkpoint.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--bilateral-calibration",
        type=Path,
        default=None,
        help=(
            "Calibration-only bilateral uncertainty-scaled conformal JSON. Defaults to "
            "<checkpoint-dir>/bilateral_sigma_calibration.json."
        ),
    )
    parser.add_argument(
        "--xense-socket",
        type=Path,
        default=Path.home() / ".cache/franka_xense_capture/control.sock",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--action-fps", type=float, default=30.0)
    parser.add_argument("--history-ms", type=int, default=200)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--max-baseline-residual-p95", type=float, default=1.0)
    parser.add_argument(
        "--tau-on",
        type=float,
        default=None,
        help="Stable-contact entry threshold. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--tau-off",
        type=float,
        default=None,
        help="Stable-contact exit threshold. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--force-scale", type=float, default=5.0)
    parser.add_argument("--cost-clip", type=float, default=1.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path.home() / ".cache/franka_tactile_envelope/chunks.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    calibration_path = (
        args.bilateral_calibration.expanduser().resolve()
        if args.bilateral_calibration is not None
        else args.checkpoint.parent / "bilateral_sigma_calibration.json"
    )
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    bilateral_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    expected_hash = str(bilateral_calibration.get("checkpoint_sha256", ""))
    actual_hash = checkpoint_sha256(args.checkpoint)
    if expected_hash != actual_hash:
        raise RuntimeError(
            "Bilateral calibration/checkpoint mismatch: "
            f"calibration={expected_hash or '(missing)'}, checkpoint={actual_hash}"
        )
    if int(bilateral_calibration.get("schema_version", -1)) != 2:
        raise RuntimeError(
            "Bilateral calibration must use schema_version=2 uncertainty-scaled calibration"
        )
    if bilateral_calibration.get("method") != (
        "bilateral_stable_contact_average_uncertainty_scaled_conformal"
    ):
        raise RuntimeError("Bilateral calibration is not uncertainty-scaled conformal")
    if bilateral_calibration.get("scale_method") != "mean":
        raise RuntimeError("Bilateral calibration must use the selected mean uncertainty scale")
    if bilateral_calibration.get("scale_definition") != (
        "u_mean = 0.5 * (sigma_left + sigma_right)"
    ):
        raise RuntimeError("Bilateral calibration has an unexpected uncertainty-scale formula")
    if bilateral_calibration.get("scale_is_standard_deviation") is not False:
        raise RuntimeError("Bilateral uncertainty scale must not be labeled a standard deviation")
    if not bilateral_calibration.get("test_not_used_for_upper_alpha", False):
        raise RuntimeError(
            "The upper bilateral alpha must be estimated without test-target tuning"
        )
    if not bilateral_calibration.get("test_not_used_for_lower_alpha", False):
        raise RuntimeError(
            "The lower bilateral alpha must be estimated without test-target tuning"
        )
    bilateral_lower_coverage = float(bilateral_calibration["lower_coverage"])
    bilateral_lower_alpha = float(bilateral_calibration["lower_alpha"])
    bilateral_upper_coverage = float(bilateral_calibration["upper_coverage"])
    bilateral_upper_alpha = float(bilateral_calibration["upper_alpha"])
    raw_peak_threshold = bilateral_calibration.get("raw_peak_threshold", {})
    if raw_peak_threshold.get("uses_validation_calibration_or_test", True):
        raise RuntimeError("Raw peak threshold must be derived from the train split only")
    raw_peak_threshold_N = float(raw_peak_threshold["threshold_N"])
    source = XenseRingClient(args.xense_socket)
    baselines, baseline_report = calibrate_baseline(
        source,
        duration_s=args.baseline_seconds,
        max_residual_p95=args.max_baseline_residual_p95,
    )
    print(
        f"[envelope_sidecar] loading checkpoint={args.checkpoint} device={args.device}",
        flush=True,
    )
    predictor = TransformerForceEnvelopePredictor(args.checkpoint, device=args.device)
    stable_contact = predictor.stable_contact_config
    tau_on = float(
        args.tau_on
        if args.tau_on is not None
        else stable_contact.get("tau_on", 5.0)
    )
    tau_off = float(
        args.tau_off
        if args.tau_off is not None
        else stable_contact.get("tau_off", 3.0)
    )
    if not 0.0 <= tau_off < tau_on:
        raise ValueError(
            f"Stable-contact thresholds require 0 <= tau_off < tau_on, got "
            f"tau_off={tau_off}, tau_on={tau_on}"
        )
    service = Pi05EnvelopeCostService(
        predictor,
        source,
        baselines,
        checkpoint_path=args.checkpoint,
        baseline_report=baseline_report,
        action_fps=args.action_fps,
        history_ms=args.history_ms,
        bilateral_lower_alpha=bilateral_lower_alpha,
        bilateral_lower_coverage=bilateral_lower_coverage,
        bilateral_upper_alpha=bilateral_upper_alpha,
        bilateral_upper_coverage=bilateral_upper_coverage,
        bilateral_calibration=bilateral_calibration,
        raw_peak_threshold_N=raw_peak_threshold_N,
        force_scale=args.force_scale,
        cost_clip=args.cost_clip,
        tau_on=tau_on,
        tau_off=tau_off,
        log_path=args.log,
    )
    server = EnvelopeServer((args.host, args.port), service)
    def shutdown(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(
        f"[envelope_sidecar] ready tcp://{args.host}:{args.port} "
        f"bilateral_L{int(round(100 * bilateral_lower_coverage)):02d}="
        f"mu{bilateral_lower_alpha:+.4f}*u_mean "
        f"bilateral_U{int(round(100 * bilateral_upper_coverage))}="
        f"mu{bilateral_upper_alpha:+.4f}*u_mean "
        f"stable_contact={tau_on:.4f}/{tau_off:.4f}N(on/off) "
        f"raw_peak_threshold={raw_peak_threshold_N:.4f}N "
        f"production_ready={predictor.production_ready}; use shadow first",
        flush=True,
    )
    server.serve_forever(poll_interval=0.2)
    server.server_close()


if __name__ == "__main__":
    main()
