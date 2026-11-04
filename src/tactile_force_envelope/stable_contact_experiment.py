from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import read_jsonl, write_json
from .model import TransformerEnvelopeExpert
from .stable_contact import (
    SIDES,
    StableContactHysteresis,
    hard_force_limit_exceeded,
    normalized_soft_cost,
    stable_contact_from_samples,
    stable_contact_mask,
)
from .train import _conformal_alpha
from .transformer_train import (
    TransformerWindowDataset,
    _device,
    _predict,
    train_transformer_and_calibrate,
)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not values.size:
        return {f"p{p}": None for p in (50, 80, 90, 95, 99)}
    return {
        f"p{int(q * 100)}": float(np.quantile(values, q))
        for q in (0.50, 0.80, 0.90, 0.95, 0.99)
    }


def _noise_threshold_report(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    sensor_names = config["label"].get(
        "sensor_names", {"left": "left_tip", "right": "right_tip"}
    )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        key = (str(row.get("batch_id", "")), int(row["episode_index"]))
        unique.setdefault(key, row)
    report: dict[str, Any] = {
        "source": "verified contact-free first-second baseline diagnostics, train episodes only",
        "episodes": len(unique),
        "sides": {},
    }
    for side in SIDES:
        sensor = str(sensor_names[side])
        medians = np.asarray([
            float(row["quality"]["baseline_tactile"][sensor]["residual_median"])
            for row in unique.values()
        ])
        p95s = np.asarray([
            float(row["quality"]["baseline_tactile"][sensor]["residual_p95"])
            for row in unique.values()
        ])
        median = float(np.median(medians))
        mad = float(np.median(np.abs(medians - median)))
        robust_sigma = 1.4826 * mad
        report["sides"][side] = {
            "episode_residual_median": median,
            "episode_residual_median_mad": mad,
            "robust_sigma": robust_sigma,
            "median_plus_6_robust_sigma": median + 6.0 * robust_sigma,
            "baseline_residual_p95_quantiles": _quantiles(p95s),
            "configured_tau_on": float(config["stable_contact"]["tau_on"]),
        }
    return report


def _load_evaluation_context(
    config: dict[str, Any],
    checkpoint_path: Path,
    *,
    split_names: tuple[str, ...] = ("calibration", "test"),
) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]],
    dict[str, dict[str, np.ndarray]], torch.Tensor, torch.Tensor, torch.Tensor,
]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    rows_path = workspace / "windows.jsonl"
    rows = read_jsonl(rows_path)
    windows_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    condition = torch.load(workspace / "condition_tokens.pt", map_location="cpu", weights_only=False)
    history = torch.load(
        workspace / "causal_tactile_history.pt", map_location="cpu", weights_only=False
    )
    if condition.get("windows_sha256") != windows_hash:
        raise ValueError("condition_tokens.pt does not match windows.jsonl")
    if history.get("windows_sha256") != windows_hash:
        raise ValueError("causal_tactile_history.pt does not match windows.jsonl")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dimensions = checkpoint["dimensions"]
    model = TransformerEnvelopeExpert(
        condition_dim=int(dimensions["condition_dim"]),
        state_dim=int(dimensions["state_dim"]),
        action_dim=int(dimensions["action_dim"]),
        action_steps=int(dimensions["action_steps"]),
        config=checkpoint["model_config"],
        history_dim=int(dimensions.get("history_dim", 0)),
        history_steps=int(dimensions.get("history_steps", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = _device(config)
    model.to(device).eval()
    if not split_names:
        raise ValueError("split_names must not be empty")
    grouped = {
        name: [row for row in rows if row["split"] == name]
        for name in split_names
    }
    empty = [name for name, split_rows in grouped.items() if not split_rows]
    if empty:
        raise ValueError(f"No windows found for split(s): {empty}")
    datasets = {
        name: TransformerWindowDataset(
            split, condition["condition_tokens"], condition["condition_mask"],
            checkpoint["normalization"], config, history["history"], history["mask"],
        )
        for name, split in grouped.items()
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(config["training"].get("batch_size", 128)),
            shuffle=False,
            num_workers=int(config["training"].get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    predictions = {
        name: _predict(model, loader, device, checkpoint["normalization"])
        for name, loader in loaders.items()
    }
    return (
        checkpoint, rows, grouped, predictions,
        history["history"], history["mask"], history["timestamps"],
    )


def _gates_for_rows(
    rows: list[dict[str, Any]], history: torch.Tensor, history_mask: torch.Tensor,
    config: dict[str, Any],
) -> np.ndarray:
    indices = torch.tensor([int(row["window_index"]) for row in rows], dtype=torch.long)
    stable = config["stable_contact"]
    return stable_contact_mask(
        history[indices], history_mask[indices],
        magnitude_index=int(stable.get("magnitude_index", 3)),
        tau_on=float(stable.get("tau_on", 5.0)),
        recent_samples=int(stable.get("recent_samples", 4)),
        required_samples=int(stable.get("required_samples", 3)),
    )


def _method_metrics(
    target: np.ndarray, mu: np.ndarray, sigma: np.ndarray, upper: np.ndarray,
) -> dict[str, Any]:
    residual = target - mu
    return {
        "samples": int(target.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "signed_residual": _quantiles(residual),
        "coverage": float(np.mean(target <= upper)),
        "mean_sigma": float(np.mean(sigma)),
        "mean_envelope_width": float(np.mean(upper - mu)),
        "mean_upper_minus_target": float(np.mean(upper - target)),
        "false_penalty_windows": int(np.sum(target > upper)),
    }


def evaluate_stable_contact_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path, *,
    output_dir: str | Path | None = None, variant: str = "A",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint, rows, grouped, predictions, history, history_mask, timestamps = (
        _load_evaluation_context(config, checkpoint_path)
    )
    stable_cfg = config["stable_contact"]
    tau_train = float(stable_cfg.get("tau_train", 5.0))
    coverage_target = float(config["training"].get("coverage", 0.95))
    cal_gates = _gates_for_rows(grouped["calibration"], history, history_mask, config)
    test_gates = _gates_for_rows(grouped["test"], history, history_mask, config)
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint_path.parent / "stable_contact_evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)

    calibration: dict[str, Any] = {
        "mode": "stable_contact_sustained_only",
        "coverage": coverage_target,
        "window_ms": int(config["alignment"]["window_ms"]),
        "history_ms": int(config["causal_tactile_history"]["history_ms"]),
        "tau_on": float(stable_cfg.get("tau_on", 5.0)),
        "tau_off": float(stable_cfg.get("tau_off", 3.0)),
        "tau_train": tau_train,
        "entry_rule": "at least 3 of the last 4 valid pre-anchor samples > tau_on",
        "selection": "causal stable-prior gate AND future filtered_top2_mean > tau_train",
        "selection_future_label_is_offline_only": True,
    }
    report: dict[str, Any] = {
        "variant": variant,
        "source_checkpoint": str(checkpoint_path),
        "development_test_only": True,
        "noise_threshold_audit": _noise_threshold_report(rows, config),
        "sides": {},
    }
    per_episode_rows: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        cal = predictions["calibration"][side]
        cal_mask = cal_gates[:, side_index] & (cal["target"] > tau_train)
        if not np.any(cal_mask):
            raise ValueError(f"No sustained-contact calibration samples for {side}")
        scale_scores = (
            (cal["target"][cal_mask] - cal["mu"][cal_mask])
            / (cal["sigma"][cal_mask] + 1e-8)
        )
        additive_scores = cal["target"][cal_mask] - cal["mu"][cal_mask]
        alpha, alpha_rank = _conformal_alpha(scale_scores, coverage_target)
        q95, q95_rank = _conformal_alpha(additive_scores, coverage_target)
        calibration[f"alpha95_{side}"] = alpha
        calibration[f"q95_{side}"] = q95
        calibration[f"samples_{side}"] = int(cal_mask.sum())
        calibration[f"finite_sample_rank_scale_{side}"] = alpha_rank
        calibration[f"finite_sample_rank_additive_{side}"] = q95_rank

        test = predictions["test"][side]
        test_mask = test_gates[:, side_index] & (test["target"] > tau_train)
        if not np.any(test_mask):
            raise ValueError(f"No sustained-contact test samples for {side}")
        target = test["target"][test_mask]
        mu = test["mu"][test_mask]
        sigma = test["sigma"][test_mask]
        upper_scale = mu + alpha * sigma
        upper_add = mu + q95
        scale_metrics = _method_metrics(target, mu, sigma, upper_scale)
        add_metrics = _method_metrics(target, mu, sigma, upper_add)
        report["sides"][side] = {
            "calibration_samples": int(cal_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "calibration_standardized_residual": _quantiles(scale_scores),
            "calibration_additive_residual": _quantiles(additive_scores),
            "mu_plus_alpha_sigma": {"alpha95": alpha, **scale_metrics},
            "mu_plus_q95": {"q95": q95, **add_metrics},
        }

        selected_positions = np.flatnonzero(test_mask)
        episode_values: dict[tuple[str, int], list[int]] = {}
        for compact_index, position in enumerate(selected_positions):
            row = grouped["test"][int(position)]
            key = (str(row.get("batch_id", "")), int(row["episode_index"]))
            episode_values.setdefault(key, []).append(compact_index)
        episode_coverages = {"scale": [], "additive": []}
        for (batch_id, episode), positions in sorted(episode_values.items()):
            selected = np.asarray(positions, dtype=np.int64)
            for method, upper in (("scale", upper_scale), ("additive", upper_add)):
                episode_coverage = float(np.mean(target[selected] <= upper[selected]))
                episode_coverages[method].append(episode_coverage)
                per_episode_rows.append({
                    "variant": variant,
                    "side": side,
                    "method": method,
                    "batch_id": batch_id,
                    "episode": episode,
                    "assumed_successful_demonstration": True,
                    "sustained_contact_windows": int(selected.size),
                    "coverage": episode_coverage,
                    "false_penalty_windows": int(np.sum(target[selected] > upper[selected])),
                    "mae": float(np.mean(np.abs(target[selected] - mu[selected]))),
                    "mean_sigma": float(np.mean(sigma[selected])),
                    "mean_envelope_width": float(np.mean((upper - mu)[selected])),
                })
        report["sides"][side]["episode_level"] = {
            method: {
                "episodes": len(values),
                "mean_coverage": float(np.mean(values)),
                "min_coverage": float(np.min(values)),
                "p10_coverage": float(np.quantile(values, 0.10)),
            }
            for method, values in episode_coverages.items()
        }

    test_indices = torch.tensor(
        [int(row["window_index"]) for row in grouped["test"]], dtype=torch.long
    )
    selected_timestamps = timestamps[test_indices].numpy()
    selected_masks = history_mask[test_indices].numpy()
    anchors = np.asarray([float(row["start_timestamp"]) for row in grouped["test"]])
    valid_time = selected_timestamps[selected_masks]
    causal_deltas = selected_timestamps - anchors[:, None, None]
    causal_valid = causal_deltas[selected_masks]
    causal_audit = {
        "gate_inputs": "causal_tactile_history.pt only",
        "future_target_used_by_online_gate": False,
        "future_target_used_for_offline_selection_and_training_mask": True,
        "cached_causal_violations": int(
            checkpoint.get("tactile_history_metadata", {}).get("causal_violations", 0)
        ),
        "evaluated_history_samples": int(valid_time.size),
        "max_history_timestamp_minus_anchor_ms": float(np.max(causal_valid) * 1000.0),
        "all_history_samples_strictly_before_anchor": bool(np.all(causal_valid < 0.0)),
        "online_exit_gate_uses_future": False,
    }
    report["causality_audit"] = causal_audit
    report["production_ready"] = False
    report["production_blockers"] = [
        "The reused test split is a development test after multiple diagnostic passes.",
        "Blind shadow evaluation on newly collected independent episodes has not run.",
        "The live Online RL observation path does not yet supply 200 ms causal tactile history.",
        *([] if all(bool(row.get("action_sequence_available_at_anchor")) for row in rows) else [
            "Historical future action sequences include human feedback unavailable at the anchor."
        ]),
    ]

    with (output / "stable_contact_per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_episode_rows[0]))
        writer.writeheader()
        writer.writerows(per_episode_rows)
    write_json(output / "stable_contact_calibration.json", calibration)
    write_json(output / "stable_contact_gated_evaluation.json", report)
    write_json(output / "stable_contact_causality_audit.json", causal_audit)

    calibrated = copy.deepcopy(checkpoint)
    calibrated["calibration_before_stable_contact"] = checkpoint.get("calibration")
    calibrated["calibration"] = calibration
    calibrated["stable_contact"] = dict(stable_cfg)
    calibrated["production_ready"] = False
    calibrated["input_contract"] = {
        **dict(checkpoint.get("input_contract", {})),
        "online_gate": "per-side hysteresis from real valid tactile samples at or before t only",
        "soft_cost": "zero while gate is false; normalized and clipped positive exceedance",
        "hard_safety": "raw force limit remains active regardless of gate/model status",
    }
    calibrated_path = output / "stable_contact_calibrated_checkpoint.pt"
    torch.save(calibrated, calibrated_path)
    report["calibrated_checkpoint"] = str(calibrated_path)
    write_json(output / "stable_contact_gated_evaluation.json", report)
    return report


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side in SIDES:
        output[side] = {}
        for method in ("mu_plus_alpha_sigma", "mu_plus_q95"):
            output[side][method] = {}
            for key in (
                "mae", "rmse", "coverage", "mean_sigma", "mean_envelope_width",
                "false_penalty_windows",
            ):
                values = np.asarray([run["sides"][side][method][key] for run in runs])
                output[side][method][key] = {
                    "mean": float(values.mean()), "std": float(values.std())
                }
    return output


def run_stable_gate_self_test(config: dict[str, Any]) -> dict[str, Any]:
    stable = config["stable_contact"]
    tau_on = float(stable["tau_on"])
    tau_off = float(stable["tau_off"])
    checks: dict[str, bool] = {}
    checks["entry_three_of_four"] = stable_contact_from_samples(
        [6.0, 7.0, 1.0, 8.0], [True] * 4, tau_on=tau_on
    )
    checks["insufficient_history_stays_false"] = not stable_contact_from_samples(
        [6.0, 7.0, 8.0], [True] * 3, tau_on=tau_on
    )
    gate = StableContactHysteresis(tau_on=tau_on, tau_off=tau_off)
    states = [gate.update("left", value) for value in (6.0, 7.0, 1.0, 8.0, 2.0, 2.0)]
    checks["hysteresis_entry"] = states[3] is True
    checks["hysteresis_hold"] = states[4] is True
    checks["hysteresis_exit"] = states[5] is False
    checks["gate_false_zero_cost"] = normalized_soft_cost(
        100.0, 1.0, False, force_scale=5.0, cost_clip=2.0
    ) == 0.0
    checks["cost_is_clipped"] = normalized_soft_cost(
        100.0, 1.0, True, force_scale=5.0, cost_clip=2.0
    ) == 2.0
    checks["hard_limit_independent"] = hard_force_limit_exceeded(41.0, 40.0)
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "causality": {
            "gate_function_accepts_future_force": False,
            "gate_function_accepts_future_target": False,
            "entry_inputs": "past force magnitudes and valid mask only",
            "hysteresis_inputs": "streamed observed force magnitudes only",
        },
    }
    output = (
        Path(config["data"]["output_dir"]).expanduser().resolve()
        / str(stable.get("artifact_subdir", "stable_contact_ab"))
        / "stable_contact_gate_test_report.json"
    )
    write_json(output, report)
    if not report["passed"]:
        raise AssertionError(f"Stable-contact gate self-test failed: {checks}")
    return report


def run_stable_contact_ab(config: dict[str, Any]) -> dict[str, Any]:
    stable_cfg = config["stable_contact"]
    seeds = [int(value) for value in stable_cfg.get("seeds", [42, 43, 44])]
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    root = workspace / str(stable_cfg.get("artifact_subdir", "stable_contact_ab"))
    root.mkdir(parents=True, exist_ok=True)
    all_runs: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for variant in ("A", "B"):
        for seed in seeds:
            run_config = copy.deepcopy(config)
            run_config["training"]["seed"] = seed
            run_config["training"]["objective"] = (
                "all_windows" if variant == "A" else "sustained_contact"
            )
            run_config["training"]["contact_sample_weight"] = (
                float(stable_cfg.get("variant_a_contact_weight", 1.5))
                if variant == "A" else 1.0
            )
            relative = f"{stable_cfg.get('artifact_subdir', 'stable_contact_ab')}/{variant}_seed_{seed}"
            run_config["training"]["artifact_subdir"] = relative
            run_dir = workspace / relative
            evaluation_path = run_dir / "stable_evaluation/stable_contact_gated_evaluation.json"
            calibrated_path = run_dir / "stable_evaluation/stable_contact_calibrated_checkpoint.pt"
            if evaluation_path.is_file() and calibrated_path.is_file():
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            else:
                if variant == "A" and seed == int(stable_cfg.get("reuse_variant_a_seed", 44)):
                    checkpoint_path = Path(stable_cfg["source_checkpoint"]).expanduser().resolve()
                else:
                    training = train_transformer_and_calibrate(run_config)
                    checkpoint_path = Path(training["checkpoint"])
                evaluation = evaluate_stable_contact_checkpoint(
                    run_config, checkpoint_path,
                    output_dir=run_dir / "stable_evaluation", variant=variant,
                )
            record = {"variant": variant, "seed": seed, **evaluation}
            all_runs[variant].append(record)
            with (run_dir / "stable_evaluation/stable_contact_per_episode.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                for episode_row in csv.DictReader(handle):
                    episode_rows.append({"seed": seed, **episode_row})
            for side in SIDES:
                for method in ("mu_plus_alpha_sigma", "mu_plus_q95"):
                    metric = evaluation["sides"][side][method]
                    rows.append({
                        "variant": variant,
                        "seed": seed,
                        "side": side,
                        "method": method,
                        "samples": metric["samples"],
                        "mae": metric["mae"],
                        "rmse": metric["rmse"],
                        "coverage": metric["coverage"],
                        "mean_sigma": metric["mean_sigma"],
                        "mean_envelope_width": metric["mean_envelope_width"],
                        "false_penalty_windows": metric["false_penalty_windows"],
                    })
    summary = {
        "seeds": seeds,
        "same_episode_split": True,
        "variant_A": "200 ms history, current all-window weighted objective, online stable gate",
        "variant_B": "200 ms history, per-side sustained-contact-only masked objective",
        "aggregate": {variant: _aggregate_runs(runs) for variant, runs in all_runs.items()},
        "runs": all_runs,
        "selection_rule": (
            "Prefer 93%-97% sustained coverage, then narrower envelope and fewer successful-episode "
            "false penalties; development test is not a production sign-off."
        ),
        "production_ready": False,
        "production_blocker": "Independent blind shadow episodes are required.",
    }
    with (root / "stable_contact_ab_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "stable_contact_per_episode_all_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    run_stable_gate_self_test(config)
    write_json(root / "stable_contact_ab_summary.json", summary)

    b_candidates = []
    for run in all_runs["B"]:
        coverages = [
            run["sides"][side][method]["coverage"]
            for side in SIDES for method in ("mu_plus_alpha_sigma", "mu_plus_q95")
        ]
        widths = [
            run["sides"][side][method]["mean_envelope_width"]
            for side in SIDES for method in ("mu_plus_alpha_sigma", "mu_plus_q95")
        ]
        b_candidates.append((
            sum(abs(value - 0.95) for value in coverages),
            float(np.mean(widths)),
            run,
        ))
    _, _, selected = min(b_candidates, key=lambda item: (item[0], item[1]))
    source = Path(selected["calibrated_checkpoint"])
    candidate = root / "stable_contact_sustained_candidate.pt"
    shutil.copy2(source, candidate)
    summary["selected_sustained_checkpoint"] = str(candidate)
    summary["selected_seed"] = int(selected["seed"])
    write_json(root / "stable_contact_ab_summary.json", summary)
    return summary
