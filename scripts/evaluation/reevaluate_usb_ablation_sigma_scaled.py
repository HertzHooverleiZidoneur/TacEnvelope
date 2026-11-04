#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile_force_envelope.ablation_recalibration import reevaluate_usb_ablations
from tactile_force_envelope.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate existing USB envelope ablation checkpoints without retraining."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/pi05_jax_zrl_stable_contact.yaml",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config["training"]["device"] = args.device
    action, objective = reevaluate_usb_ablations(
        config, config_path=config_path, seeds=tuple(args.seeds)
    )
    print(json.dumps({
        "action_conditioning": action["aggregate"],
        "action_comparison": action["comparison"],
        "probabilistic_objective": objective["aggregate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

