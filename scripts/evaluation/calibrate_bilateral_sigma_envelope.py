#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile_force_envelope.bilateral_sigma_calibration import (
    calibrate_bilateral_sigma_envelope,
)
from tactile_force_envelope.common import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a bilateral sigma-scaled force envelope."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = calibrate_bilateral_sigma_envelope(
        load_config(args.config),
        args.checkpoint,
        output_path=args.output,
    )
    print(json.dumps({
        "output": str(
            args.output.expanduser().resolve()
            if args.output is not None
            else args.checkpoint.expanduser().resolve().parent
            / "bilateral_sigma_calibration.json"
        ),
        "alpha05": report["lower_alpha"],
        "alpha95": report["upper_alpha"],
        "calibration_samples": report["calibration_samples"],
        "test": report["test_report"],
        "additive_ablation_test": report["additive_ablation"]["test_report"],
    }, indent=2))


if __name__ == "__main__":
    main()
