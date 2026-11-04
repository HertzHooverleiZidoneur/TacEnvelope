#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile_force_envelope.common import load_config
from tactile_force_envelope.fixed_band_ablation import run_fixed_vs_adaptive_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a global fixed force band with an adaptive envelope."
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
    report = run_fixed_vs_adaptive_ablation(
        config, config_path=config_path, seeds=tuple(args.seeds)
    )
    print(json.dumps({
        "global_fixed_band": report["global_fixed_band"],
        "adaptive_aggregate": report["adaptive_envelope"]["aggregate"],
        "representative_plot": report["representative_plot"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
