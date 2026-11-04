from __future__ import annotations

import json
import hashlib
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import read_jsonl, write_json
from .model import ForceEnvelopeModel, gaussian_nll


TARGET_FIELDS = {
    "raw_max": "y_raw_max",
    "filtered_top2_mean": "y_filtered_top2_mean",
    "max": "y_raw_max",
    "top2_mean": "y_filtered_top2_mean",
}


class WindowDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        visual: torch.Tensor,
        task_ids: dict[str, int],
        target_field: str,
        normalization: dict[str, torch.Tensor],
    ) -> None:
        self.rows = rows
        self.visual = visual
        self.task_ids = task_ids
        self.target_field = target_field
        self.norm = normalization

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        visual = self.visual[int(row["window_index"])].flatten().float()
        state = torch.tensor(row["state"], dtype=torch.float32)
        actions = torch.tensor(row["executed_actions"], dtype=torch.float32)
        target = torch.tensor(float(row["tactile"][self.target_field]), dtype=torch.float32)
        return {
            "visual": (visual - self.norm["visual_mean"]) / self.norm["visual_std"],
            "state": (state - self.norm["state_mean"]) / self.norm["state_std"],
            "actions": (actions - self.norm["action_mean"]) / self.norm["action_std"],
            "action_mask": torch.tensor(row["action_mask"], dtype=torch.float32),
            "task_id": torch.tensor(self.task_ids[str(row["task"])], dtype=torch.long),
            "target": (target - self.norm["target_mean"]) / self.norm["target_std"],
            "target_physical": target,
        }


def _normalization(
    rows: list[dict[str, Any]], visual: torch.Tensor, target_field: str
) -> dict[str, torch.Tensor]:
    indices = torch.tensor([int(row["window_index"]) for row in rows], dtype=torch.long)
    visual_values = visual[indices].flatten(1).float()
    states = torch.tensor([row["state"] for row in rows], dtype=torch.float32)
    actions = torch.tensor([row["executed_actions"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["tactile"][target_field] for row in rows], dtype=torch.float32)

    def stats(values: torch.Tensor, dimensions: int | tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=dimensions)
        std = values.std(dim=dimensions, unbiased=False).clamp_min(1e-6)
        return mean, std

    visual_mean, visual_std = stats(visual_values, 0)
    state_mean, state_std = stats(states, 0)
    action_mean, action_std = stats(actions, (0, 1))
    target_mean, target_std = stats(targets, 0)
    return {
        "visual_mean": visual_mean, "visual_std": visual_std,
        "state_mean": state_mean, "state_std": state_std,
        "action_mean": action_mean, "action_std": action_std,
        "target_mean": target_mean, "target_std": target_std,
    }


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _predict(
    model: ForceEnvelopeModel, loader: DataLoader, device: torch.device,
    target_mean: float, target_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            mu_norm, log_var_norm = model(
                batch["visual"].to(device), batch["state"].to(device),
                batch["actions"].to(device), batch["action_mask"].to(device),
                batch["task_id"].to(device),
            )
            mus.append((mu_norm.cpu().numpy() * target_std + target_mean))
            sigmas.append((torch.exp(0.5 * log_var_norm).cpu().numpy() * target_std))
            targets.append(batch["target_physical"].numpy())
    return np.concatenate(mus), np.concatenate(sigmas), np.concatenate(targets)


def _conformal_alpha(scores: np.ndarray, coverage: float) -> tuple[float, int]:
    if scores.size == 0:
        raise ValueError("Calibration split is empty")
    rank = min(scores.size, int(math.ceil((scores.size + 1) * coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def _group_coverage(
    rows: list[dict[str, Any]], targets: np.ndarray, upper: np.ndarray,
    contact_threshold: float, window_s: float, sampling_rate_bins: list[float],
) -> dict[str, dict[str, dict[str, float | int]]]:
    if len(rows) != targets.size or targets.size != upper.size:
        raise ValueError("Rows, targets, and upper bounds must have equal lengths")

    def summarize(indices: list[int]) -> dict[str, float | int]:
        selected = np.asarray(indices, dtype=np.int64)
        errors = targets[selected] - upper[selected]
        return {
            "samples": int(selected.size),
            "coverage": float(np.mean(errors <= 0.0)),
            "mean_exceedance": float(np.mean(np.maximum(errors, 0.0))),
            "max_exceedance": float(np.max(np.maximum(errors, 0.0))),
        }

    task_groups: dict[str, list[int]] = {}
    contact_groups: dict[str, list[int]] = {"contact": [], "no_contact": []}
    rate_groups: dict[str, list[int]] = {}
    edges = sorted(float(value) for value in sampling_rate_bins)
    for index, row in enumerate(rows):
        task_groups.setdefault(str(row["task"]), []).append(index)
        raw_max = float(row["tactile"].get("y_raw_max", targets[index]))
        contact_groups["contact" if raw_max >= contact_threshold else "no_contact"].append(index)
        counts = list(row["tactile"]["sensor_counts"].values())
        rate = min(counts) / window_s if counts else 0.0
        lower = float("-inf")
        label = ""
        for edge in edges:
            if rate < edge:
                label = f"{lower if lower != float('-inf') else 0:g}-{edge:g}Hz"
                break
            lower = edge
        if not label:
            label = f">={edges[-1]:g}Hz" if edges else "all"
        rate_groups.setdefault(label, []).append(index)

    return {
        "by_task": {key: summarize(value) for key, value in task_groups.items() if value},
        "by_contact_state": {
            key: summarize(value) for key, value in contact_groups.items() if value
        },
        "by_min_sensor_sampling_rate": {
            key: summarize(value) for key, value in rate_groups.items() if value
        },
    }


def train_mlp_and_calibrate(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    rows = read_jsonl(output / "windows.jsonl")
    cache = torch.load(output / "condition_features.pt", map_location="cpu", weights_only=False)
    windows_hash = hashlib.sha256((output / "windows.jsonl").read_bytes()).hexdigest()
    if cache.get("windows_sha256") != windows_hash:
        raise ValueError("condition_features.pt does not match windows.jsonl; rerun cache-vision")
    expected_encoder = str(config["vision"]["encoder"])
    if cache.get("encoder") != expected_encoder:
        raise ValueError(
            f"Cached feature encoder {cache.get('encoder')!r} != configured {expected_encoder!r}; "
            "rerun cache-vision"
        )
    for key in ("feature_contains_language", "feature_contains_robot_state"):
        expected = bool(config["vision"].get(key, False))
        if bool(cache.get(key, False)) != expected:
            raise ValueError(
                f"Cached {key}={cache.get(key)!r} != configured {expected!r}; rerun cache-vision"
            )
    visual = cache["features"]
    target_type = str(config["label"]["target_type"])
    if target_type not in TARGET_FIELDS:
        raise ValueError(f"target_type must be one of {sorted(TARGET_FIELDS)}")
    target_field = TARGET_FIELDS[target_type]
    grouped = {name: [row for row in rows if row["split"] == name]
               for name in ("train", "calibration", "test")}
    if any(not values for values in grouped.values()):
        raise ValueError(f"Every split must contain windows: { {k: len(v) for k,v in grouped.items()} }")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    normalization = _normalization(grouped["train"], visual, target_field)
    tasks = sorted({str(row["task"]) for row in rows})
    task_ids = {task: index for index, task in enumerate(tasks)}
    datasets = {
        name: WindowDataset(values, visual, task_ids, target_field, normalization)
        for name, values in grouped.items()
    }
    batch_size = int(config["training"]["batch_size"])
    loaders = {
        name: DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train",
            num_workers=int(config["training"].get("num_workers", 0)),
        )
        for name, dataset in datasets.items()
    }
    first = rows[0]
    model = ForceEnvelopeModel(
        visual_dim=int(np.prod(visual.shape[1:])),
        state_dim=len(first["state"]),
        action_dim=len(first["executed_actions"][0]),
        action_steps=len(first["executed_actions"]),
        num_tasks=len(task_ids),
        config=config["model"],
    )
    device = _device(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            mu, log_var = model(
                batch["visual"].to(device), batch["state"].to(device),
                batch["actions"].to(device), batch["action_mask"].to(device),
                batch["task_id"].to(device),
            )
            loss = gaussian_nll(mu, log_var, batch["target"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_gaussian_nll": float(np.mean(losses))})
    target_mean = float(normalization["target_mean"])
    target_std = float(normalization["target_std"])
    cal_mu, cal_sigma, cal_y = _predict(
        model, loaders["calibration"], device, target_mean, target_std
    )
    epsilon = 1e-8
    scores = (cal_y - cal_mu) / (cal_sigma + epsilon)
    coverage = float(config["training"]["coverage"])
    alpha95, rank = _conformal_alpha(scores, coverage)
    test_mu, test_sigma, test_y = _predict(
        model, loaders["test"], device, target_mean, target_std
    )
    upper = test_mu + alpha95 * test_sigma
    test_nll = 0.5 * np.mean(
        ((test_y - test_mu) ** 2) / np.maximum(test_sigma**2, 1e-12)
        + np.log(np.maximum(test_sigma**2, 1e-12))
    )
    calibration = {
        "coverage": coverage,
        "alpha95": alpha95,
        "target_type": target_type,
        "window_ms": int(config["alignment"]["window_ms"]),
        "calibration_samples": int(scores.size),
        "finite_sample_rank": rank,
        "score_definition": "(y - mu) / (sigma + 1e-8)",
        "quantile_method": "ceil((n + 1) * coverage)-th finite-sample order statistic",
        "normalization_source": "train episodes only",
    }
    contact_threshold = float(config["label"].get("contact_threshold", 5.0))
    sampling_rate_bins = list(config["training"].get("sampling_rate_bins_hz", [18.0, 21.0]))
    window_s = float(config["alignment"]["window_ms"]) / 1000.0
    metrics = {
        "test_samples": int(test_y.size),
        "test_coverage": float(np.mean(test_y <= upper)),
        "test_mae_mu": float(np.mean(np.abs(test_y - test_mu))),
        "test_rmse_mu": float(np.sqrt(np.mean((test_y - test_mu) ** 2))),
        "test_gaussian_nll": float(test_nll),
        "test_mean_sigma": float(np.mean(test_sigma)),
        "test_mean_upper_margin": float(np.mean(upper - test_mu)),
        "contact_threshold": contact_threshold,
        "group_coverage": _group_coverage(
            grouped["test"], test_y, upper, contact_threshold, window_s, sampling_rate_bins
        ),
    }
    uses_vla_features = cache["encoder"] == "smolvla_condition"
    causal_action_input = all(
        bool(row.get("action_sequence_available_at_anchor", False)) for row in rows
    )
    production_ready = uses_vla_features and causal_action_input
    calibration["calibration_empirical_coverage"] = float(
        np.mean(cal_y <= cal_mu + alpha95 * cal_sigma)
    )
    calibration["group_coverage"] = _group_coverage(
        grouped["calibration"], cal_y, cal_mu + alpha95 * cal_sigma,
        contact_threshold, window_s, sampling_rate_bins,
    )
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "model_config": config["model"],
        "dimensions": {
            "visual_dim": int(np.prod(visual.shape[1:])),
            "state_dim": len(first["state"]),
            "action_dim": len(first["executed_actions"][0]),
            "action_steps": len(first["executed_actions"]),
            "num_tasks": len(task_ids),
        },
        "normalization": normalization,
        "task_ids": task_ids,
        "target_type": target_type,
        "calibration": calibration,
        "image_fields": cache.get("image_fields", list(first["images"])),
        "vision_encoder": cache["encoder"],
        "feature_metadata": cache.get("report", {}),
        "control_window_ms": int(config["alignment"]["window_ms"]),
        "input_contract": {
            "action_sequence": (
                "commands actually queued for the complete fixed-horizon control window at t_k"
            ),
            "visual_features": (
                "frozen SmolVLA action-conditioning past-key-value features captured from the "
                "same forward pass that produces the candidate action chunk"
            ),
        },
        "production_ready": production_ready,
    }
    torch.save(checkpoint, output / "force_envelope_model.pt")
    write_json(output / "calibration.json", calibration)
    write_json(output / "test_metrics.json", metrics)
    write_json(output / "train_history.json", history)
    result = {
        "checkpoint": str(output / "force_envelope_model.pt"),
        "device": str(device),
        "split_sizes": {key: len(value) for key, value in grouped.items()},
        "calibration": calibration,
        "test_metrics": metrics,
        "last_train_nll": history[-1]["train_gaussian_nll"],
        "production_ready": production_ready,
        "production_blockers": [
            blocker for blocker, active in (
                ("Features are not SmolVLA action-conditioning prefix features.",
                 not uses_vla_features),
                ("Demonstration future actions include post-contact human corrections unavailable at t_k.",
                 not causal_action_input),
            ) if active
        ],
    }
    write_json(output / "training_result.json", result)
    return result


def train_and_calibrate(config: dict[str, Any]) -> dict[str, Any]:
    model_type = str(config["model"].get("model_type", "mlp_baseline"))
    if model_type == "transformer_envelope":
        from .transformer_train import train_transformer_and_calibrate

        return train_transformer_and_calibrate(config)
    if model_type == "mlp_baseline":
        return train_mlp_and_calibrate(config)
    raise ValueError(f"Unsupported model_type {model_type!r}")
