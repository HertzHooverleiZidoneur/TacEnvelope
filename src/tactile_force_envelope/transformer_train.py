from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .common import read_jsonl, write_json
from .model import TransformerEnvelopeExpert, gaussian_nll
from .stable_contact import stable_contact_mask
from .train import _conformal_alpha


SIDES = ("left", "right")


def _sensor_name(config: dict[str, Any], side: str) -> str:
    names = config["label"].get("sensor_names", {"left": "left_tip", "right": "right_tip"})
    return str(names[side])


def _side_value(row: dict[str, Any], config: dict[str, Any], side: str, field: str) -> float:
    sensor = _sensor_name(config, side)
    try:
        value = row["tactile"]["per_sensor"][sensor][field]
    except KeyError as error:
        raise KeyError(f"Missing {side} sensor {sensor!r} field {field!r}") from error
    if value is None:
        raise ValueError(f"Invalid {side} target in window {row['window_index']}")
    return float(value)


def _normalization(
    rows: list[dict[str, Any]], config: dict[str, Any],
    tactile_history: torch.Tensor | None = None,
    tactile_history_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    states = torch.tensor([row["state"] for row in rows], dtype=torch.float32)
    actions = torch.tensor([row["executed_actions"] for row in rows], dtype=torch.float32)
    targets = {
        side: torch.tensor([
            _side_value(row, config, side, "filtered_top2_mean") for row in rows
        ], dtype=torch.float32)
        for side in SIDES
    }

    def stats(values: torch.Tensor, dims: int | tuple[int, ...]):
        return values.mean(dim=dims), values.std(dim=dims, unbiased=False).clamp_min(1e-6)

    state_mean, state_std = stats(states, 0)
    action_mean, action_std = stats(actions, (0, 1))
    result = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    for side in SIDES:
        result[f"target_mean_{side}"], result[f"target_std_{side}"] = stats(targets[side], 0)
    if tactile_history is not None and tactile_history_mask is not None:
        indices = torch.tensor([int(row["window_index"]) for row in rows], dtype=torch.long)
        selected = tactile_history[indices].float()
        selected_mask = tactile_history_mask[indices].bool()
        history_means = []
        history_stds = []
        for side_index in range(2):
            values = selected[:, side_index][selected_mask[:, side_index]]
            if not values.numel():
                raise ValueError(f"No valid train tactile history for side index {side_index}")
            mean, std = stats(values, 0)
            history_means.append(mean)
            history_stds.append(std)
        result["history_mean"] = torch.stack(history_means)
        result["history_std"] = torch.stack(history_stds)
    return result


class TransformerWindowDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, Any]], condition_tokens: torch.Tensor,
        condition_mask: torch.Tensor, normalization: dict[str, torch.Tensor],
        config: dict[str, Any], tactile_history: torch.Tensor | None = None,
        tactile_history_mask: torch.Tensor | None = None,
    ) -> None:
        self.rows = rows
        self.condition_tokens = condition_tokens
        self.condition_mask = condition_mask
        self.norm = normalization
        self.config = config
        self.tactile_history = tactile_history
        self.tactile_history_mask = tactile_history_mask
        self.contact_threshold = float(config["label"].get("contact_threshold", 5.0))
        self.stable_prior = None
        if tactile_history is not None and tactile_history_mask is not None:
            stable_cfg = self.config.get("stable_contact", {})
            self.stable_prior = torch.from_numpy(stable_contact_mask(
                tactile_history,
                tactile_history_mask,
                magnitude_index=int(stable_cfg.get("magnitude_index", 3)),
                tau_on=float(stable_cfg.get("tau_on", 5.0)),
                recent_samples=int(stable_cfg.get("recent_samples", 4)),
                required_samples=int(stable_cfg.get("required_samples", 3)),
            ))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        window_index = int(row["window_index"])
        state = torch.tensor(row["state"], dtype=torch.float32)
        actions = torch.tensor(row["executed_actions"], dtype=torch.float32)
        item = {
            # Keep cached tokens in their stored dtype through collation and PCIe transfer.
            # Autocast promotes operations as needed on the GPU.
            "condition_tokens": self.condition_tokens[window_index],
            "condition_mask": self.condition_mask[window_index],
            "state": (state - self.norm["state_mean"]) / self.norm["state_std"],
            "actions": (actions - self.norm["action_mean"]) / self.norm["action_std"],
            "action_mask": torch.tensor(row["action_mask"], dtype=torch.bool),
            "window_index": torch.tensor(window_index, dtype=torch.long),
        }
        for side in SIDES:
            target = torch.tensor(
                _side_value(row, self.config, side, "filtered_top2_mean"), dtype=torch.float32
            )
            raw_max = torch.tensor(
                _side_value(row, self.config, side, "raw_max"), dtype=torch.float32
            )
            item[f"target_{side}"] = (
                target - self.norm[f"target_mean_{side}"]
            ) / self.norm[f"target_std_{side}"]
            item[f"target_physical_{side}"] = target
            item[f"raw_max_{side}"] = raw_max
            item[f"contact_{side}"] = (raw_max >= self.contact_threshold).float()
        item["window_contact"] = torch.maximum(item["contact_left"], item["contact_right"])
        if self.tactile_history is not None and self.tactile_history_mask is not None:
            history_mask = self.tactile_history_mask[window_index]
            raw_history = self.tactile_history[window_index].float()
            history = (
                raw_history - self.norm["history_mean"][:, None]
            ) / self.norm["history_std"][:, None]
            item["tactile_history"] = torch.where(
                history_mask[..., None], history, torch.zeros_like(history)
            )
            item["tactile_history_mask"] = history_mask
            stable_cfg = self.config.get("stable_contact", {})
            tau_train = float(stable_cfg.get("tau_train", 5.0))
            for side_index, side in enumerate(SIDES):
                stable = bool(self.stable_prior[window_index, side_index])
                item[f"stable_prior_{side}"] = torch.tensor(stable, dtype=torch.bool)
                item[f"sustained_mask_{side}"] = torch.tensor(
                    stable and float(item[f"target_physical_{side}"]) > tau_train,
                    dtype=torch.bool,
                )
        return item


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _forward(model, batch, device):
    return model(
        # Transfer the compact FP16 cache first, then convert on the GPU. This
        # works both inside autocast and in the FP32 calibration/test path.
        batch["condition_tokens"].to(device, non_blocking=True).float(),
        batch["condition_mask"].to(device, non_blocking=True),
        batch["state"].to(device, non_blocking=True),
        batch["actions"].to(device, non_blocking=True),
        batch["action_mask"].to(device, non_blocking=True),
        batch["tactile_history"].to(device, non_blocking=True)
        if "tactile_history" in batch else None,
        batch["tactile_history_mask"].to(device, non_blocking=True)
        if "tactile_history_mask" in batch else None,
    )


def _loss(
    outputs, batch, device, contact_sample_weight: float, smooth_l1_weight: float,
    gaussian_nll_weight: float = 1.0,
):
    weights = torch.where(
        batch["window_contact"].to(device).bool(),
        torch.tensor(contact_sample_weight, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )
    nlls = []
    smooth_losses = []
    for side in SIDES:
        mu = outputs[f"mu_{side}"].float()
        log_var = outputs[f"log_var_{side}"].float()
        target = batch[f"target_{side}"].to(device).float()
        nlls.append(0.5 * ((target - mu).square() * torch.exp(-log_var) + log_var))
        smooth_losses.append(F.smooth_l1_loss(mu, target, reduction="none"))
    per_window_nll = torch.stack(nlls, dim=1).mean(dim=1)
    per_window_smooth = torch.stack(smooth_losses, dim=1).mean(dim=1)
    denominator = weights.sum().clamp_min(1.0)
    weighted_nll = (per_window_nll * weights).sum() / denominator
    weighted_smooth = (per_window_smooth * weights).sum() / denominator
    total = gaussian_nll_weight * weighted_nll + smooth_l1_weight * weighted_smooth
    return total, weighted_nll, weighted_smooth


def _sustained_contact_loss(
    outputs, batch, device, smooth_l1_weight: float,
    gaussian_nll_weight: float = 1.0,
):
    """Per-side masked objective for already-established sustained contact."""
    side_nlls = []
    side_smooth = []
    for side in SIDES:
        mu = outputs[f"mu_{side}"].float()
        log_var = outputs[f"log_var_{side}"].float()
        target = batch[f"target_{side}"].to(device).float()
        mask = batch[f"sustained_mask_{side}"].to(device).float()
        denominator = mask.sum().clamp_min(1.0)
        nll_values = 0.5 * ((target - mu).square() * torch.exp(-log_var) + log_var)
        smooth_values = F.smooth_l1_loss(mu, target, reduction="none")
        side_nlls.append((nll_values * mask).sum() / denominator)
        side_smooth.append((smooth_values * mask).sum() / denominator)
    nll = 0.5 * (side_nlls[0] + side_nlls[1])
    smooth = 0.5 * (side_smooth[0] + side_smooth[1])
    return gaussian_nll_weight * nll + smooth_l1_weight * smooth, nll, smooth


def _predict(model, loader, device, normalization):
    result = {
        side: {"mu": [], "sigma": [], "target": [], "raw_max": [], "contact_probability": []}
        for side in SIDES
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = _forward(model, batch, device)
            for side in SIDES:
                target_mean = float(normalization[f"target_mean_{side}"])
                target_std = float(normalization[f"target_std_{side}"])
                result[side]["mu"].append(
                    outputs[f"mu_{side}"].float().cpu().numpy() * target_std + target_mean
                )
                result[side]["sigma"].append(
                    torch.exp(0.5 * outputs[f"log_var_{side}"].float()).cpu().numpy() * target_std
                )
                result[side]["target"].append(batch[f"target_physical_{side}"].numpy())
                result[side]["raw_max"].append(batch[f"raw_max_{side}"].numpy())
                result[side]["contact_probability"].append(
                    torch.sigmoid(outputs[f"contact_logit_{side}"].float()).cpu().numpy()
                )
    return {
        side: {key: np.concatenate(value) for key, value in values.items()}
        for side, values in result.items()
    }


def _metrics(values: dict[str, np.ndarray], upper: np.ndarray, contact_threshold: float):
    target = values["target"]
    mu = values["mu"]
    sigma = values["sigma"]
    nll = 0.5 * np.mean(
        ((target - mu) ** 2) / np.maximum(sigma**2, 1e-12)
        + np.log(np.maximum(sigma**2, 1e-12))
    )
    result: dict[str, Any] = {
        "samples": int(target.size),
        "mae": float(np.mean(np.abs(target - mu))),
        "rmse": float(np.sqrt(np.mean((target - mu) ** 2))),
        "gaussian_nll": float(nll),
        "coverage": float(np.mean(target <= upper)),
        "mean_envelope_margin": float(np.mean(upper - mu)),
        "mean_target_headroom": float(np.mean(upper - target)),
        "max_exceedance": float(np.max(np.maximum(target - upper, 0.0))),
    }
    contact = values["raw_max"] >= contact_threshold
    for name, mask in (("contact", contact), ("no_contact", ~contact)):
        if np.any(mask):
            result[name] = {
                "samples": int(mask.sum()),
                "coverage": float(np.mean(target[mask] <= upper[mask])),
                "mae": float(np.mean(np.abs(target[mask] - mu[mask]))),
                "mean_exceedance": float(np.mean(np.maximum(target[mask] - upper[mask], 0.0))),
                "mean_sigma": float(np.mean(sigma[mask])),
                "mean_envelope_margin": float(np.mean((upper - mu)[mask])),
            }
    return result


def _fixed_threshold_baseline(
    train_rows: list[dict[str, Any]], test_values: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any], coverage: float,
) -> dict[str, Any]:
    train_by_side = {
        side: np.asarray([
            _side_value(row, config, side, "filtered_top2_mean") for row in train_rows
        ]) for side in SIDES
    }
    global_targets = np.maximum(train_by_side["left"], train_by_side["right"])
    global_rank = min(global_targets.size, int(math.ceil((global_targets.size + 1) * coverage)))
    global_threshold = float(np.partition(global_targets, global_rank - 1)[global_rank - 1])
    global_report: dict[str, Any] = {"threshold": global_threshold, "by_side": {}}
    for side in SIDES:
        targets = test_values[side]["target"]
        global_report["by_side"][side] = {
            "coverage": float(np.mean(targets <= global_threshold)),
            "mean_threshold_minus_target": float(np.mean(global_threshold - targets)),
            "max_exceedance": float(np.max(np.maximum(targets - global_threshold, 0.0))),
        }
    per_side_report = {}
    for side in SIDES:
        train_targets = train_by_side[side]
        rank = min(train_targets.size, int(math.ceil((train_targets.size + 1) * coverage)))
        threshold = float(np.partition(train_targets, rank - 1)[rank - 1])
        targets = test_values[side]["target"]
        per_side_report[side] = {
            "threshold": threshold,
            "coverage": float(np.mean(targets <= threshold)),
            "mean_threshold_minus_target": float(np.mean(threshold - targets)),
            "max_exceedance": float(np.max(np.maximum(targets - threshold, 0.0))),
        }
    return {
        "single_global_train_p95": global_report,
        "per_side_train_p95": per_side_report,
    }


def train_transformer_and_calibrate(config: dict[str, Any]) -> dict[str, Any]:
    workspace_output = Path(config["data"]["output_dir"]).expanduser().resolve()
    artifact_subdir = str(config["training"].get("artifact_subdir", "")).strip()
    output = workspace_output / artifact_subdir if artifact_subdir else workspace_output
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(workspace_output / "windows.jsonl")
    cache_path = workspace_output / "condition_tokens.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    windows_hash = hashlib.sha256((workspace_output / "windows.jsonl").read_bytes()).hexdigest()
    if cache.get("windows_sha256") != windows_hash:
        raise ValueError("condition_tokens.pt does not match windows.jsonl; rerun cache-features")
    supported_condition_encoders = {
        "smolvla_condition_tokens",
        "pi05_jax_z_rl",
    }
    if cache.get("encoder") not in supported_condition_encoders:
        raise ValueError(
            "Transformer Envelope requires a token condition cache; "
            f"got encoder={cache.get('encoder')!r}"
        )
    condition_tokens = cache["condition_tokens"]
    condition_mask = cache["condition_mask"]
    use_tactile_history = bool(config["model"].get("use_tactile_history", False))
    history_cache = None
    if use_tactile_history:
        history_cache_path = workspace_output / "causal_tactile_history.pt"
        if not history_cache_path.is_file():
            raise FileNotFoundError(
                f"Causal tactile history cache not found: {history_cache_path}; "
                "run cache-tactile-history first"
            )
        history_cache = torch.load(history_cache_path, map_location="cpu", weights_only=False)
        if history_cache.get("windows_sha256") != windows_hash:
            raise ValueError("causal_tactile_history.pt does not match windows.jsonl")
    objective = str(config["training"].get("objective", "all_windows"))
    if objective not in {"all_windows", "sustained_contact"}:
        raise ValueError(f"Unknown transformer training objective: {objective}")
    if objective == "sustained_contact" and not use_tactile_history:
        raise ValueError("sustained_contact objective requires causal tactile history")
    selected_window_mask = cache.get("selected_window_mask")
    if objective == "sustained_contact" and selected_window_mask is not None:
        rows = [
            row for row in rows
            if bool(selected_window_mask[int(row["window_index"])])
        ]
    grouped = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "validation", "calibration", "test")
    }
    if any(not split for split in grouped.values()):
        raise ValueError(f"Every four-way split must be non-empty: { {k: len(v) for k,v in grouped.items()} }")
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    normalization = _normalization(
        grouped["train"], config,
        history_cache["history"] if history_cache else None,
        history_cache["mask"] if history_cache else None,
    )
    datasets = {
        name: TransformerWindowDataset(
            split, condition_tokens, condition_mask, normalization, config,
            history_cache["history"] if history_cache else None,
            history_cache["mask"] if history_cache else None,
        ) for name, split in grouped.items()
    }
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        ) for name, dataset in datasets.items()
    }
    first = rows[0]
    model = TransformerEnvelopeExpert(
        condition_dim=int(condition_tokens.shape[-1]),
        state_dim=len(first["state"]),
        action_dim=len(first["executed_actions"][0]),
        action_steps=len(first["executed_actions"]),
        config=config["model"],
        history_dim=int(history_cache["history"].shape[-1]) if history_cache else 0,
        history_steps=int(history_cache["history"].shape[-2]) if history_cache else 0,
    )
    device = _device(config)
    model.to(device)
    shape_batch = next(iter(loaders["train"]))
    shape_report = model.token_shapes(
        shape_batch["condition_tokens"].to(device).float(),
        shape_batch["condition_mask"].to(device),
        shape_batch["state"].to(device),
        shape_batch["actions"].to(device),
        shape_batch["action_mask"].to(device),
        shape_batch["tactile_history"].to(device) if use_tactile_history else None,
        shape_batch["tactile_history_mask"].to(device) if use_tactile_history else None,
    )
    shape_report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Transformer Envelope one-batch shape gate:")
    print(json.dumps(shape_report, indent=2))
    write_json(output / "transformer_shape_report.json", shape_report)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    contact_sample_weight = float(config["training"].get("contact_sample_weight", 1.0))
    smooth_l1_weight = float(config["training"].get("smooth_l1_weight", 0.5))
    gaussian_nll_weight = float(config["training"].get("gaussian_nll_weight", 1.0))
    if gaussian_nll_weight < 0.0 or smooth_l1_weight < 0.0:
        raise ValueError("Loss weights must be non-negative")
    if gaussian_nll_weight == 0.0 and smooth_l1_weight == 0.0:
        raise ValueError("At least one regression loss weight must be positive")
    patience = int(config["training"].get("early_stopping_patience", 6))
    minimum_epochs = int(config["training"].get("minimum_epochs", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 1e-4))
    use_amp = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_values = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = _forward(model, batch, device)
                if objective == "sustained_contact":
                    loss, nll, smooth = _sustained_contact_loss(
                        outputs, batch, device, smooth_l1_weight,
                        gaussian_nll_weight,
                    )
                else:
                    loss, nll, smooth = _loss(
                        outputs, batch, device, contact_sample_weight,
                        smooth_l1_weight, gaussian_nll_weight,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
            optimizer.step()
            train_values.append((float(loss.detach()), float(nll.detach()), float(smooth.detach())))
        model.eval()
        validation_values = []
        with torch.inference_mode():
            for batch in loaders["validation"]:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = _forward(model, batch, device)
                    if objective == "sustained_contact":
                        loss, nll, smooth = _sustained_contact_loss(
                            outputs, batch, device, smooth_l1_weight,
                            gaussian_nll_weight,
                        )
                    else:
                        loss, nll, smooth = _loss(
                            outputs, batch, device, contact_sample_weight,
                            smooth_l1_weight, gaussian_nll_weight,
                        )
                validation_values.append((float(loss), float(nll), float(smooth)))
        train_mean = np.mean(train_values, axis=0)
        validation_mean = np.mean(validation_values, axis=0)
        record = {
            "epoch": epoch,
            "train_total": float(train_mean[0]),
            "train_gaussian_nll": float(train_mean[1]),
            "train_weighted_smooth_l1": float(train_mean[2]),
            "validation_total": float(validation_mean[0]),
            "validation_gaussian_nll": float(validation_mean[1]),
            "validation_weighted_smooth_l1": float(validation_mean[2]),
            "gaussian_nll_weight": gaussian_nll_weight,
            "smooth_l1_weight": smooth_l1_weight,
            "contact_sample_weight": contact_sample_weight,
            "objective": objective,
        }
        history.append(record)
        print(json.dumps(record))
        if float(validation_mean[0]) < best_validation - min_delta:
            best_validation = float(validation_mean[0])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            temporary_best = output / "transformer_best_training_state.pt.tmp"
            torch.save(
                {
                    "epoch": best_epoch,
                    "validation_total": best_validation,
                    "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
                },
                temporary_best,
            )
            temporary_best.replace(output / "transformer_best_training_state.pt")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if epoch >= minimum_epochs and stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    model.load_state_dict(best_state)

    calibration_values = _predict(model, loaders["calibration"], device, normalization)
    test_values = _predict(model, loaders["test"], device, normalization)
    coverage = float(config["training"].get("coverage", 0.95))
    calibration: dict[str, Any] = {
        "coverage": coverage,
        "target_type": "per_side_filtered_top2_mean",
        "window_ms": int(config["alignment"]["window_ms"]),
        "normalization_source": "train episodes only",
    }
    metrics: dict[str, Any] = {}
    contact_threshold = float(config["label"].get("contact_threshold", 5.0))
    for side in SIDES:
        values = calibration_values[side]
        scores = (values["target"] - values["mu"]) / (values["sigma"] + 1e-8)
        alpha, rank = _conformal_alpha(scores, coverage)
        calibration[f"alpha95_{side}"] = alpha
        calibration[f"finite_sample_rank_{side}"] = rank
        calibration[f"samples_{side}"] = int(scores.size)
        test = test_values[side]
        upper = test["mu"] + alpha * test["sigma"]
        metrics[side] = _metrics(test, upper, contact_threshold)
        metrics[side]["alpha95"] = alpha
    fixed = _fixed_threshold_baseline(grouped["train"], test_values, config, coverage)
    old_metrics_path = Path(str(config["training"].get("old_mlp_metrics", ""))).expanduser()
    old_mlp = None
    if str(old_metrics_path) and old_metrics_path.is_file():
        old_mlp = json.loads(old_metrics_path.read_text(encoding="utf-8"))
    comparison = {
        "fixed_threshold_baselines": fixed,
        "old_mlp_global_max_baseline": old_mlp,
        "note": (
            "Old MLP predicts max(left,right) on the former 70/15/15 split, so its metrics are "
            "reported as a historical non-like-for-like baseline."
        ),
    }
    use_action_conditioning = bool(config["model"].get("use_action_conditioning", True))
    causal_action_input = (
        not use_action_conditioning
        or all(bool(row.get("action_sequence_available_at_anchor", False)) for row in rows)
    )
    coverage_pass = all(float(metrics[side]["coverage"]) >= coverage for side in SIDES)
    production_ready = causal_action_input and coverage_pass
    checkpoint = {
        "model_type": "transformer_envelope",
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_config": config["model"],
        "dimensions": {
            "condition_dim": int(condition_tokens.shape[-1]),
            "condition_steps": int(condition_tokens.shape[1]),
            "state_dim": len(first["state"]),
            "action_dim": len(first["executed_actions"][0]),
            "action_steps": len(first["executed_actions"]),
            "history_dim": int(history_cache["history"].shape[-1]) if history_cache else 0,
            "history_steps": int(history_cache["history"].shape[-2]) if history_cache else 0,
        },
        "normalization": normalization,
        "calibration": calibration,
        "control_window_ms": int(config["alignment"]["window_ms"]),
        "feature_metadata": cache["report"],
        "tactile_history_metadata": history_cache["report"] if history_cache else None,
        "stable_contact": dict(config.get("stable_contact", {})),
        "best_epoch": best_epoch,
        "loss": {
            "type": (
                (
                    "per_side_masked_smooth_l1_only"
                    if objective == "sustained_contact"
                    else "weighted_smooth_l1_only"
                )
                if gaussian_nll_weight == 0.0
                else (
                    "per_side_masked_gaussian_nll_plus_smooth_l1"
                    if objective == "sustained_contact"
                    else "weighted_gaussian_nll_plus_smooth_l1"
                )
            ),
            "objective": objective,
            "contact_definition": (
                "per-side causal stable prior and future filtered_top2_mean > tau_train"
                if objective == "sustained_contact"
                else "max(left_raw_max,right_raw_max) >= contact_threshold"
            ),
            "contact_threshold": contact_threshold,
            "contact_sample_weight": 1.0 if objective == "sustained_contact" else contact_sample_weight,
            "smooth_l1_weight": smooth_l1_weight,
            "gaussian_nll_weight": gaussian_nll_weight,
            "sigma_trained": gaussian_nll_weight > 0.0,
        },
        "production_ready": production_ready,
        "input_contract": {
            "condition_tokens": (
                "same Pi0.5 JAX forward z_rl as the candidate action chunk"
                if cache.get("encoder") == "pi05_jax_z_rl"
                else "same SmolVLA forward as the candidate action chunk"
            ),
            "actions": (
                "complete candidate chunk prefix actually queued at t_k for 300 ms"
                if use_action_conditioning
                else "disabled for action-conditioning ablation; no action tokens are created"
            ),
            "action_conditioning_enabled": use_action_conditioning,
            "tactile_history": (
                "real valid samples strictly before t_k" if history_cache else "not used"
            ),
        },
    }
    checkpoint_path = output / "transformer_force_envelope_model.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(output / "transformer_calibration.json", calibration)
    write_json(output / "transformer_test_metrics.json", metrics)
    write_json(output / "transformer_comparison.json", comparison)
    write_json(output / "transformer_train_history.json", history)
    result = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "split_sizes": {key: len(value) for key, value in grouped.items()},
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "contact_sample_weight": contact_sample_weight,
        "smooth_l1_weight": smooth_l1_weight,
        "gaussian_nll_weight": gaussian_nll_weight,
        "shape_report": shape_report,
        "calibration": calibration,
        "test_metrics": metrics,
        "comparison": comparison,
        "production_ready": production_ready,
        "production_blockers": [
            blocker for blocker, active in (
                (
                    "Historical future actions include human feedback unavailable at anchor time.",
                    not causal_action_input,
                ),
                (
                    "At least one side does not reach the requested 95% coverage on held-out test episodes.",
                    not coverage_pass,
                ),
            ) if active
        ],
    }
    write_json(output / "transformer_training_result.json", result)
    return result
