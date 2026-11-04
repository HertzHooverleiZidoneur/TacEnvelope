from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from tactile_force_envelope.common import load_config


MODULES = ("numpy", "pandas", "PIL", "scipy", "torch", "torchvision", "yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check tactile envelope training environment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"[OK] import {module}")
        except Exception as exc:
            failures.append(f"import {module}: {exc}")
            print(f"[FAIL] import {module}: {exc}")

    config = load_config(args.config)
    if config["vision"].get("encoder") == "smolvla_condition":
        try:
            importlib.import_module("lerobot.policies.smolvla.modeling_smolvla")
            print("[OK] import LeRobot SmolVLA")
        except Exception as exc:
            failures.append(f"LeRobot SmolVLA: {exc}")
            print(f"[FAIL] import LeRobot SmolVLA: {exc}")

    paths = {
        "collection manifest": config["data"].get("collection_manifest"),
        "robot dataset": config["data"]["robot_dataset"],
        "tactile root": config["data"]["tactile_root"],
        "SmolVLA checkpoint": config["vision"].get("checkpoint"),
    }
    for label, value in paths.items():
        path = Path(str(value)).expanduser()
        if path.exists():
            print(f"[OK] {label}: {path}")
        else:
            failures.append(f"{label} missing: {path}")
            print(f"[FAIL] {label}: {path}")

    output = Path(config["data"]["output_dir"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[OK] output directory writable: {output}")
    if failures:
        raise SystemExit(f"Environment check failed with {len(failures)} issue(s)")
    print("Environment check passed")


if __name__ == "__main__":
    main()

