from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


_UNRESOLVED_ENV = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")


def _expand_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        unresolved = _UNRESOLVED_ENV.findall(expanded)
        if unresolved:
            raise ValueError(
                "Configuration contains unset environment variables: "
                + ", ".join(sorted(set(unresolved)))
            )
        return expanded
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return _expand_config_value(config)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "median": None,
                "p95": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def rate_from_timestamps(timestamps: list[float]) -> dict[str, Any]:
    if len(timestamps) < 2:
        return {"hz_median": None, "hz_mean": None, "interval_ms": distribution([])}
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive = intervals[intervals > 0]
    return {
        "hz_median": float(1.0 / np.median(positive)) if positive.size else None,
        "hz_mean": float(1.0 / np.mean(positive)) if positive.size else None,
        "interval_ms": distribution((positive * 1000.0).tolist()),
        "duplicates": int(np.sum(intervals == 0)),
        "rollbacks": int(np.sum(intervals < 0)),
    }


def episode_split(
    episode_ids: list[int], train_ratio: float, calibration_ratio: float, seed: int
) -> dict[str, list[int]]:
    ids = sorted(set(episode_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n < 3:
        raise ValueError("At least three episodes are required for train/calibration/test")
    n_train = max(1, int(round(n * train_ratio)))
    n_cal = max(1, int(round(n * calibration_ratio)))
    if n_train + n_cal >= n:
        n_train = max(1, n - 2)
        n_cal = 1
    return {
        "train": sorted(ids[:n_train]),
        "calibration": sorted(ids[n_train:n_train + n_cal]),
        "test": sorted(ids[n_train + n_cal:]),
    }


def episode_split_four(
    episode_ids: list[int], train_ratio: float, validation_ratio: float,
    calibration_ratio: float, seed: int,
) -> dict[str, list[int]]:
    """Split whole episodes into four disjoint sets.

    The remainder is assigned to test so rounding cannot silently lose episodes.
    """
    ids = sorted(set(episode_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n < 4:
        raise ValueError("At least four episodes are required for train/validation/calibration/test")
    ratios = (train_ratio, validation_ratio, calibration_ratio)
    if any(value <= 0.0 for value in ratios) or sum(ratios) >= 1.0:
        raise ValueError(
            "train_ratio, validation_ratio, and calibration_ratio must be positive and sum to < 1"
        )
    n_train = max(1, int(round(n * train_ratio)))
    n_validation = max(1, int(round(n * validation_ratio)))
    n_calibration = max(1, int(round(n * calibration_ratio)))
    while n_train + n_validation + n_calibration >= n:
        largest = max(
            ((n_train, "train"), (n_validation, "validation"), (n_calibration, "calibration"))
        )[1]
        if largest == "train" and n_train > 1:
            n_train -= 1
        elif largest == "validation" and n_validation > 1:
            n_validation -= 1
        elif n_calibration > 1:
            n_calibration -= 1
        else:
            raise ValueError("Unable to create non-empty four-way episode split")
    validation_end = n_train + n_validation
    calibration_end = validation_end + n_calibration
    return {
        "train": sorted(ids[:n_train]),
        "validation": sorted(ids[n_train:validation_end]),
        "calibration": sorted(ids[validation_end:calibration_end]),
        "test": sorted(ids[calibration_end:]),
    }
