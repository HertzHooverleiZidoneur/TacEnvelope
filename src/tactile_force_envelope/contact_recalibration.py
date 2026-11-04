from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import read_jsonl, write_json
from .model import TransformerEnvelopeExpert
from .train import _conformal_alpha
from .transformer_train import (
    SIDES,
    TransformerWindowDataset,
    _device,
    _metrics,
    _predict,
)


def _contact_conditional_alpha(
    values: dict[str, np.ndarray], contact_threshold: float, coverage: float,
) -> tuple[float, int, np.ndarray]:
    contact = values["raw_max"] >= contact_threshold
    if not np.any(contact):
        raise ValueError("Calibration split has no contact samples for this sensor")
    scores = (
        (values["target"][contact] - values["mu"][contact])
        / (values["sigma"][contact] + 1e-8)
    )
    alpha, rank = _conformal_alpha(scores, coverage)
    return alpha, rank, contact


def recalibrate_contact_envelope(config: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    checkpoint_value = config["training"].get("contact_recalibration_checkpoint")
    if not checkpoint_value:
        raise ValueError("training.contact_recalibration_checkpoint must be configured")
    checkpoint_path = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Envelope checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "transformer_envelope":
        raise ValueError("Contact recalibration requires a transformer_envelope checkpoint")

    rows_path = workspace / "windows.jsonl"
    cache_path = workspace / "condition_tokens.pt"
    rows = read_jsonl(rows_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    windows_hash = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    if cache.get("windows_sha256") != windows_hash:
        raise ValueError("condition_tokens.pt does not match windows.jsonl; rerun cache-features")

    grouped = {
        name: [row for row in rows if row["split"] == name]
        for name in ("calibration", "test")
    }
    if any(not split for split in grouped.values()):
        raise ValueError("Calibration and test splits must both be non-empty")

    normalization = checkpoint["normalization"]
    condition_tokens = cache["condition_tokens"]
    condition_mask = cache["condition_mask"]
    dimensions = checkpoint["dimensions"]
    history_cache = None
    if int(dimensions.get("history_steps", 0)) > 0:
        history_path = workspace / "causal_tactile_history.pt"
        if not history_path.is_file():
            raise FileNotFoundError(f"Causal tactile history cache not found: {history_path}")
        history_cache = torch.load(history_path, map_location="cpu", weights_only=False)
        if history_cache.get("windows_sha256") != windows_hash:
            raise ValueError("causal_tactile_history.pt does not match windows.jsonl")
    datasets = {
        name: TransformerWindowDataset(
            split, condition_tokens, condition_mask, normalization, config,
            history_cache["history"] if history_cache else None,
            history_cache["mask"] if history_cache else None,
        )
        for name, split in grouped.items()
    }
    batch_size = int(config["training"].get("batch_size", 128))
    num_workers = int(config["training"].get("num_workers", 0))
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }

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

    calibration_values = _predict(model, loaders["calibration"], device, normalization)
    test_values = _predict(model, loaders["test"], device, normalization)
    coverage = float(config["training"].get("coverage", 0.95))
    contact_threshold = float(config["label"].get("contact_threshold", 5.0))
    old_calibration = checkpoint["calibration"]
    calibration = copy.deepcopy(old_calibration)
    calibration.update({
        "mode": "per_side_contact_conditional",
        "coverage": coverage,
        "contact_threshold": contact_threshold,
        "calibration_source": "per-side actual-contact samples only",
    })
    report: dict[str, Any] = {
        "source_checkpoint": str(checkpoint_path),
        "device": str(device),
        "coverage_target": coverage,
        "contact_threshold": contact_threshold,
        "sides": {},
    }

    contact_coverage_pass = True
    test_window_indices = torch.tensor(
        [int(row["window_index"]) for row in grouped["test"]], dtype=torch.long
    )
    for side in SIDES:
        cal_values = calibration_values[side]
        alpha, rank, cal_contact = _contact_conditional_alpha(
            cal_values, contact_threshold, coverage
        )
        old_alpha = float(old_calibration[f"alpha95_{side}"])
        calibration[f"alpha95_all_{side}"] = old_alpha
        calibration[f"alpha95_contact_{side}"] = alpha
        calibration[f"alpha95_{side}"] = alpha
        calibration[f"finite_sample_rank_contact_{side}"] = rank
        calibration[f"samples_contact_{side}"] = int(cal_contact.sum())

        cal_upper = cal_values["mu"] + alpha * cal_values["sigma"]
        test = test_values[side]
        old_upper = test["mu"] + old_alpha * test["sigma"]
        upper = test["mu"] + alpha * test["sigma"]
        old_metrics = _metrics(test, old_upper, contact_threshold)
        metrics = _metrics(test, upper, contact_threshold)
        test_contact = test["raw_max"] >= contact_threshold
        test_contact_scores = (
            (test["target"][test_contact] - test["mu"][test_contact])
            / (test["sigma"][test_contact] + 1e-8)
        )
        oracle_alpha, oracle_rank = _conformal_alpha(test_contact_scores, coverage)
        calibration_contact_scores = (
            (cal_values["target"][cal_contact] - cal_values["mu"][cal_contact])
            / (cal_values["sigma"][cal_contact] + 1e-8)
        )
        test_contact_coverage = float(metrics["contact"]["coverage"])
        contact_coverage_pass &= 0.93 <= test_contact_coverage <= 0.97
        report["sides"][side] = {
            "alpha95_all_samples": old_alpha,
            "alpha95_contact": alpha,
            "alpha_multiplier": alpha / old_alpha if old_alpha != 0 else None,
            "calibration_contact_samples": int(cal_contact.sum()),
            "finite_sample_rank": rank,
            "calibration_contact_coverage": float(
                np.mean(cal_values["target"][cal_contact] <= cal_upper[cal_contact])
            ),
            "standardized_residual_quantiles": {
                "calibration_contact": {
                    f"p{int(q * 100)}": float(np.quantile(calibration_contact_scores, q))
                    for q in (0.50, 0.90, 0.95, 0.99)
                },
                "test_contact": {
                    f"p{int(q * 100)}": float(np.quantile(test_contact_scores, q))
                    for q in (0.50, 0.90, 0.95, 0.99)
                },
            },
            "diagnostic_test_contact_oracle_alpha95": oracle_alpha,
            "diagnostic_test_contact_oracle_rank": oracle_rank,
            "diagnostic_oracle_alpha_multiplier_vs_calibration": oracle_alpha / alpha,
            "old_test_metrics": old_metrics,
            "contact_calibrated_test_metrics": metrics,
            "contact_margin_increase": float(
                metrics["contact"]["mean_envelope_margin"]
                - old_metrics["contact"]["mean_envelope_margin"]
            ),
            "no_contact_margin_increase": float(
                metrics["no_contact"]["mean_envelope_margin"]
                - old_metrics["no_contact"]["mean_envelope_margin"]
            ),
        }
        if history_cache is not None:
            side_index = SIDES.index(side)
            features = list(history_cache["report"]["features"])
            magnitude_index = features.index("force_magnitude")
            history = history_cache["history"][test_window_indices, side_index].numpy()
            history_mask = history_cache["mask"][test_window_indices, side_index].numpy()
            prior_contact = np.any(
                (history[..., magnitude_index] >= contact_threshold) & history_mask, axis=1
            )
            future_contact = test["raw_max"] >= contact_threshold
            causal_groups = {
                "ongoing_contact": future_contact & prior_contact,
                "new_contact_onset": future_contact & ~prior_contact,
                "future_no_contact": ~future_contact,
            }
            report["sides"][side]["causal_history_groups"] = {}
            for name, group_mask in causal_groups.items():
                if not np.any(group_mask):
                    continue
                report["sides"][side]["causal_history_groups"][name] = {
                    "samples": int(group_mask.sum()),
                    "coverage": float(np.mean(test["target"][group_mask] <= upper[group_mask])),
                    "mae": float(np.mean(np.abs(
                        test["target"][group_mask] - test["mu"][group_mask]
                    ))),
                    "bias_target_minus_mu": float(np.mean(
                        test["target"][group_mask] - test["mu"][group_mask]
                    )),
                    "mean_sigma": float(np.mean(test["sigma"][group_mask])),
                    "mean_envelope_margin": float(np.mean(
                        (upper - test["mu"])[group_mask]
                    )),
                }

    calibration["held_out_contact_coverage_in_93_97"] = bool(contact_coverage_pass)
    output_dir_value = config["training"].get("contact_recalibration_output_dir")
    output = (
        Path(str(output_dir_value)).expanduser().resolve()
        if output_dir_value
        else checkpoint_path.parent / "contact_recalibration"
    )
    output.mkdir(parents=True, exist_ok=True)
    calibrated_checkpoint = copy.deepcopy(checkpoint)
    calibrated_checkpoint["calibration"] = calibration
    calibrated_checkpoint["production_ready"] = False
    calibrated_checkpoint["contact_calibration_ready"] = bool(contact_coverage_pass)
    calibrated_path = output / "transformer_force_envelope_contact_calibrated.pt"
    torch.save(calibrated_checkpoint, calibrated_path)
    report["calibrated_checkpoint"] = str(calibrated_path)
    report["held_out_contact_coverage_in_93_97"] = bool(contact_coverage_pass)
    write_json(output / "transformer_contact_calibration.json", calibration)
    write_json(output / "transformer_contact_calibration_report.json", report)
    return report
