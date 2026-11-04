#!/usr/bin/env python3
"""Compute a train-only fixed bilateral-force threshold for ablation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT = Path(
    "/path/to/tactile_force_envelope/outputs/training/"
    "franka_tactile310/pi05_jax_zrl_envelope_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--tau-on", type=float, default=5.0)
    parser.add_argument("--recent-samples", type=int, default=4)
    parser.add_argument("--required-samples", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/ablations/usb_fixed_force_q90.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_stable(
    history: np.ndarray,
    mask: np.ndarray,
    *,
    magnitude_index: int,
    tau_on: float,
    recent_samples: int,
    required_samples: int,
) -> bool:
    forces = history[:, magnitude_index][mask]
    forces = forces[-recent_samples:]
    return bool(
        forces.size >= recent_samples
        and np.count_nonzero(forces > tau_on) >= required_samples
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    if not 1 <= args.required_samples <= args.recent_samples:
        raise ValueError("required_samples must be in [1, recent_samples]")

    root = args.training_root.expanduser().resolve()
    windows_path = root / "windows.jsonl"
    history_path = root / "causal_tactile_history.pt"
    cache = torch.load(history_path, map_location="cpu", weights_only=False)
    history = np.asarray(cache["history"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=bool)
    feature_names = list(cache["report"]["features"])
    magnitude_index = feature_names.index("force_magnitude")

    values: list[float] = []
    train_windows = 0
    with windows_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            train_windows += 1
            window_index = int(row["window_index"])
            stable = [
                is_stable(
                    history[window_index, side_index],
                    mask[window_index, side_index],
                    magnitude_index=magnitude_index,
                    tau_on=args.tau_on,
                    recent_samples=args.recent_samples,
                    required_samples=args.required_samples,
                )
                for side_index in range(2)
            ]
            if not all(stable) or row.get("quality", {}).get("low_quality", False):
                continue
            sensors = row["tactile"]["per_sensor"]
            left = float(sensors["left_tip"]["filtered_top2_mean"])
            right = float(sensors["right_tip"]["filtered_top2_mean"])
            values.append(0.5 * (left + right))

    if not values:
        raise RuntimeError("No bilateral stable-contact train windows were found")
    forces = np.asarray(values, dtype=np.float64)
    threshold = float(np.quantile(forces, args.quantile))
    report = {
        "criterion": "fixed_upper",
        "task": "USB insertion",
        "threshold_N": threshold,
        "threshold_statistic": f"train bilateral stable-contact q{args.quantile:g}",
        "quantile": args.quantile,
        "source_split": "train",
        "test_or_calibration_used": False,
        "train_windows_total": train_windows,
        "train_bilateral_stable_windows": int(forces.size),
        "target": "0.5 * (left_filtered_top2_mean + right_filtered_top2_mean)",
        "gate": {
            "per_side": True,
            "tau_on_N": args.tau_on,
            "recent_samples": args.recent_samples,
            "required_samples_above_tau_on": args.required_samples,
            "bilateral_required": True,
        },
        "force_distribution_N": {
            "mean": float(np.mean(forces)),
            "std": float(np.std(forces)),
            "p50": float(np.quantile(forces, 0.50)),
            "p90": float(np.quantile(forces, 0.90)),
            "p95": float(np.quantile(forces, 0.95)),
            "max": float(np.max(forces)),
        },
        "windows_path": str(windows_path),
        "windows_sha256": sha256(windows_path),
        "history_path": str(history_path),
        "history_sha256": sha256(history_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
