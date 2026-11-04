#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tactile_force_envelope.action_ablation import run_action_conditioning_ablation
from tactile_force_envelope.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Pi0.5 force expert with and without future-action tokens."
    )
    parser.add_argument(
        "--config", default="configs/experiments/pi05_jax_zrl_stable_contact.yaml"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_action_conditioning_ablation(
        load_config(args.config), seeds=args.seeds, force=args.force
    )
    print(json.dumps(summary["comparison"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
