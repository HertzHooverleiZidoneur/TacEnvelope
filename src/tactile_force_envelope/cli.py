from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit
from .bilateral_average import evaluate_bilateral_average
from .bilateral_sigma_calibration import calibrate_bilateral_sigma_envelope
from .bilateral_report import generate_bilateral_report
from .common import load_config
from .contact_recalibration import recalibrate_contact_envelope
from .history_analysis import analyze_history_checkpoint
from .stable_contact_experiment import (
    evaluate_stable_contact_checkpoint,
    run_stable_contact_ab,
    run_stable_gate_self_test,
)
from .train import train_and_calibrate
from .sweep import run_contact_weight_sweep
from .tactile_history import cache_tactile_history
from .vision import cache_visual_features
from .windows import build_windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision-conditioned tactile force envelope")
    parser.add_argument(
        "command", choices=(
            "audit", "build", "cache-features", "cache-vision", "train",
            "cache-tactile-history", "sweep-contact-weight", "recalibrate-contact",
            "analyze-history", "stable-evaluate", "stable-ab", "stable-gate-test",
            "bilateral-average", "calibrate-bilateral-sigma", "bilateral-report", "all",
        )
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    results = {}
    if args.command in {"audit", "all"}:
        results["audit"] = run_audit(config)
    if args.command in {"build", "all"}:
        results["build"] = build_windows(config)
    if args.command == "all" and bool(config.get("model", {}).get("use_tactile_history", False)):
        results["cache-tactile-history"] = cache_tactile_history(config)
    if args.command in {"cache-features", "cache-vision", "all"}:
        results["cache-features"] = cache_visual_features(config)
    if args.command == "cache-tactile-history":
        results["cache-tactile-history"] = cache_tactile_history(config)
    if args.command in {"train", "all"}:
        results["train"] = train_and_calibrate(config)
    if args.command == "sweep-contact-weight":
        results["sweep-contact-weight"] = run_contact_weight_sweep(config)
    if args.command == "recalibrate-contact":
        results["recalibrate-contact"] = recalibrate_contact_envelope(config)
    if args.command == "analyze-history":
        results["analyze-history"] = analyze_history_checkpoint(config)
    if args.command == "stable-evaluate":
        variant = (
            "B" if str(config.get("training", {}).get("objective")) == "sustained_contact"
            else "A"
        )
        results["stable-evaluate"] = evaluate_stable_contact_checkpoint(
            config,
            config["stable_contact"]["source_checkpoint"],
            output_dir=config["stable_contact"].get("evaluation_output_dir"),
            variant=variant,
        )
        results["bilateral-sigma-calibration"] = calibrate_bilateral_sigma_envelope(
            config,
            results["stable-evaluate"]["calibrated_checkpoint"],
        )
    if args.command == "stable-ab":
        results["stable-ab"] = run_stable_contact_ab(config)
    if args.command == "stable-gate-test":
        results["stable-gate-test"] = run_stable_gate_self_test(config)
    if args.command == "bilateral-average":
        results["bilateral-average"] = evaluate_bilateral_average(config)
    if args.command == "calibrate-bilateral-sigma":
        evaluation_dir = Path(
            config["stable_contact"]["evaluation_output_dir"]
        ).expanduser().resolve()
        results["calibrate-bilateral-sigma"] = calibrate_bilateral_sigma_envelope(
            config, evaluation_dir / "stable_contact_calibrated_checkpoint.pt"
        )
    if args.command == "bilateral-report":
        results["bilateral-report"] = generate_bilateral_report(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
