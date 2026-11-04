from __future__ import annotations

import copy
import concurrent.futures
import gc
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .transformer_train import train_transformer_and_calibrate


def _run_one(config: dict[str, Any], weight: float, seed: int) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config["training"]["seed"] = seed
    run_config["training"]["contact_sample_weight"] = weight
    run_config["training"]["artifact_subdir"] = f"contact_weight_sweep/weight_{weight:g}_seed_{seed}"
    print(f"Starting contact-weight sweep weight={weight:g} seed={seed}", flush=True)
    result = train_transformer_and_calibrate(run_config)
    return {"weight": weight, "seed": seed, **result}


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for side in ("left", "right"):
        keys = ("mae", "coverage", "mean_sigma", "mean_envelope_margin")
        values = {
            key: np.asarray([
                run["test_metrics"][side]["contact"][key] for run in runs
            ], dtype=np.float64)
            for key in keys
        }
        report[side] = {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in values.items()
        }
    report["best_epoch"] = {
        "mean": float(np.mean([run["best_epoch"] for run in runs])),
        "min": int(min(run["best_epoch"] for run in runs)),
        "max": int(max(run["best_epoch"] for run in runs)),
    }
    return report


def run_contact_weight_sweep(config: dict[str, Any]) -> dict[str, Any]:
    sweep = config.get("sweep", {})
    weights = [float(value) for value in sweep.get("contact_weights", [1.0, 1.5, 2.0])]
    seeds = [int(value) for value in sweep.get("seeds", [42, 43, 44])]
    parallel_seeds = int(sweep.get("parallel_seeds", 1))
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    all_runs: list[dict[str, Any]] = []
    by_weight: dict[str, list[dict[str, Any]]] = {}
    for weight in weights:
        weight_key = f"{weight:g}"
        by_weight[weight_key] = []
        if parallel_seeds > 1:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(parallel_seeds, len(seeds)), mp_context=context
            ) as executor:
                futures = [executor.submit(_run_one, config, weight, seed) for seed in seeds]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    by_weight[weight_key].append(record)
                    all_runs.append(record)
        else:
            for seed in seeds:
                record = _run_one(config, weight, seed)
                by_weight[weight_key].append(record)
                all_runs.append(record)
                gc.collect()
        by_weight[weight_key].sort(key=lambda item: item["seed"])
    summary = {
        "contact_definition": "either left or right raw max >= train contact threshold",
        "contact_threshold": float(config["label"].get("contact_threshold", 5.0)),
        "contact_weights": weights,
        "seeds": seeds,
        "parallel_seeds": parallel_seeds,
        "runs": all_runs,
        "aggregate_by_weight": {
            key: _aggregate(runs) for key, runs in by_weight.items()
        },
    }
    write_json(workspace / "contact_weight_sweep_summary.json", summary)
    return summary
