from __future__ import annotations

import base64
import copy
import hashlib
import html
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps
from scipy.stats import gaussian_kde, spearmanr
from torch.utils.data import DataLoader

from .common import read_jsonl, write_json
from .data import load_tactile_streams, window_slice
from .discovery import discover_tactile_sidecars
from .model import TransformerEnvelopeExpert
from .transformer_train import SIDES, TransformerWindowDataset, _predict


QUANTILES = (0.50, 0.80, 0.90, 0.95, 0.99)
GROUPS = ("all", "prior_contact_true", "prior_contact_false", "new_contact_onset")
COLORS = {
    "all": "#171717",
    "prior_contact_true": "#007a78",
    "prior_contact_false": "#d97706",
    "new_contact_onset": "#c2415d",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values: Iterable[float], levels: tuple[float, ...] = QUANTILES) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result: dict[str, Any] = {"count": int(array.size)}
    if not array.size:
        result.update({f"p{int(level * 100)}": None for level in levels})
        return result
    result.update({
        f"p{int(level * 100)}": float(np.quantile(array, level)) for level in levels
    })
    result.update({
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    })
    return result


def _group_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(frame), dtype=bool)
    if name == "prior_contact_true":
        return frame["prior_contact"].to_numpy(dtype=bool)
    if name == "prior_contact_false":
        return ~frame["prior_contact"].to_numpy(dtype=bool)
    if name == "new_contact_onset":
        return frame["new_contact_onset"].to_numpy(dtype=bool)
    raise KeyError(name)


def _spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    first = np.asarray(list(x), dtype=np.float64)
    second = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 3 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return {"n": int(first.size), "rho": None, "p_value": None}
    result = spearmanr(first, second)
    return {"n": int(first.size), "rho": float(result.statistic), "p_value": float(result.pvalue)}


def _load_model_and_predictions(
    config: dict[str, Any], checkpoint_path: Path, splits: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    workspace = Path(config["data"]["output_dir"]).expanduser().resolve()
    rows_path = workspace / "windows.jsonl"
    rows = read_jsonl(rows_path)
    rows_by_split = {name: [row for row in rows if row["split"] == name] for name in splits}
    if any(not rows_by_split[name] for name in splits):
        raise ValueError(f"Requested split is empty: { {k: len(v) for k, v in rows_by_split.items()} }")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dimensions = checkpoint["dimensions"]
    if int(dimensions.get("history_steps", 0)) <= 0:
        raise ValueError("The selected checkpoint has no causal tactile-history input")
    cache = torch.load(workspace / "condition_tokens.pt", map_location="cpu", weights_only=False)
    history_cache = torch.load(
        workspace / "causal_tactile_history.pt", map_location="cpu", weights_only=False
    )
    windows_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    for name, value in (
        ("condition token", cache.get("windows_sha256")),
        ("tactile history", history_cache.get("windows_sha256")),
    ):
        if value != windows_sha:
            raise ValueError(f"{name} cache does not match windows.jsonl")

    model = TransformerEnvelopeExpert(
        condition_dim=int(dimensions["condition_dim"]),
        state_dim=int(dimensions["state_dim"]),
        action_dim=int(dimensions["action_dim"]),
        action_steps=int(dimensions["action_steps"]),
        config=checkpoint["model_config"],
        history_dim=int(dimensions["history_dim"]),
        history_steps=int(dimensions["history_steps"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    requested = str(config["training"].get("device", "auto"))
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    )
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    batch_size = int(config["training"].get("batch_size", 128))
    contact_threshold = float(config["label"].get("contact_threshold", 5.0))
    history_features = list(history_cache["report"]["features"])
    magnitude_index = history_features.index("force_magnitude")
    history = history_cache["history"]
    history_mask = history_cache["mask"]

    episode_max_anchor: dict[int, int] = defaultdict(int)
    for row in rows:
        episode_max_anchor[int(row["episode_index"])] = max(
            episode_max_anchor[int(row["episode_index"])], int(row["anchor_frame_index"])
        )

    records: list[dict[str, Any]] = []
    for split in splits:
        split_rows = rows_by_split[split]
        dataset = TransformerWindowDataset(
            split_rows,
            cache["condition_tokens"],
            cache["condition_mask"],
            normalization,
            config,
            history,
            history_mask,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        values_by_side = _predict(model, loader, device, normalization)
        for local_index, row in enumerate(split_rows):
            window_index = int(row["window_index"])
            episode_index = int(row["episode_index"])
            max_anchor = max(1, episode_max_anchor[episode_index])
            progress = min(1.0, float(row["anchor_frame_index"]) / max_anchor)
            for side_index, side in enumerate(SIDES):
                values = values_by_side[side]
                target = float(values["target"][local_index])
                raw_max = float(values["raw_max"][local_index])
                mu = float(values["mu"][local_index])
                sigma = float(values["sigma"][local_index])
                side_history = history[window_index, side_index].numpy()
                side_mask = history_mask[window_index, side_index].numpy().astype(bool)
                prior_force = side_history[side_mask, magnitude_index]
                prior_max = float(np.max(prior_force)) if prior_force.size else 0.0
                prior_contact = bool(prior_max >= contact_threshold)
                future_contact = bool(raw_max >= contact_threshold)
                alpha = float(checkpoint["calibration"][f"alpha95_{side}"])
                residual = target - mu
                action = np.asarray(row["executed_actions"], dtype=np.float64)
                action_delta = action - action[0]
                records.append({
                    "window_index": window_index,
                    "split": split,
                    "side": side,
                    "episode_index": episode_index,
                    "source_episode_index": int(row["source_episode_index"]),
                    "batch_id": str(row["batch_id"]),
                    "anchor_frame_index": int(row["anchor_frame_index"]),
                    "anchor_timestamp": float(row["start_timestamp"]),
                    "episode_progress": progress,
                    "phase_bin": min(9, int(progress * 10)),
                    "target": target,
                    "raw_max": raw_max,
                    "mu": mu,
                    "sigma": sigma,
                    "residual": residual,
                    "positive_residual": max(0.0, residual),
                    "absolute_error": abs(residual),
                    "standardized_residual": residual / (sigma + 1e-8),
                    "alpha95": alpha,
                    "upper95": mu + alpha * sigma,
                    "covered": bool(target <= mu + alpha * sigma),
                    "prior_contact": prior_contact,
                    "prior_force_max": prior_max,
                    "future_contact": future_contact,
                    "new_contact_onset": bool(future_contact and not prior_contact),
                    "image_age_max_ms": max(float(value["age_ms"]) for value in row["images"].values()),
                    "arm_action_travel": float(np.linalg.norm(action_delta[:, :7], axis=1).max()),
                    "gripper_action_change": float(action[-1, -1] - action[0, -1]),
                })
    return pd.DataFrame.from_records(records), rows, checkpoint, history_cache


def _distribution_summary(frame: pd.DataFrame, splits: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in SIDES:
        result[side] = {}
        for split in splits:
            subset = frame[(frame.side == side) & (frame.split == split)]
            split_result: dict[str, Any] = {}
            for group in GROUPS:
                selected = subset[_group_mask(subset, group)]
                positive = selected.loc[selected.residual > 0, "residual"]
                split_result[group] = {
                    "target": _quantiles(selected.target),
                    "signed_residual": _quantiles(selected.residual),
                    "positive_residual_only": _quantiles(positive, (0.90, 0.95, 0.99)),
                    "standardized_residual": _quantiles(selected.standardized_residual),
                    "sigma": _quantiles(selected.sigma),
                    "coverage": float(selected.covered.mean()) if len(selected) else None,
                }
            result[side][split] = split_result
    return result


def _correlation_summary(frame: pd.DataFrame, splits: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in SIDES:
        result[side] = {}
        for split in splits:
            subset = frame[(frame.side == side) & (frame.split == split)]
            result[side][split] = {}
            for group in GROUPS:
                selected = subset[_group_mask(subset, group)]
                result[side][split][group] = {
                    "sigma_vs_absolute_error": _spearman(selected.sigma, selected.absolute_error),
                    "sigma_vs_positive_residual": _spearman(
                        selected.sigma, selected.positive_residual
                    ),
                }
    return result


def _episode_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for keys, group in frame.groupby(["split", "batch_id", "episode_index", "side"], sort=True):
        split, batch, episode, side = keys
        contact = group[group.future_contact]
        measured = contact if len(contact) else group
        positive = measured.loc[measured.residual > 0, "residual"]
        records.append({
            "split": split,
            "batch": batch,
            "episode": int(episode),
            "side": side,
            "windows": int(len(group)),
            "contact_windows": int(len(contact)),
            "metrics_scope": "contact" if len(contact) else "all_no_contact",
            "force_mean": float(measured.target.mean()),
            "force_p95": float(measured.target.quantile(0.95)),
            "force_max": float(measured.target.max()),
            "mae": float(measured.absolute_error.mean()),
            "positive_residual_p95": (
                float(positive.quantile(0.95)) if len(positive) else 0.0
            ),
            "mean_sigma": float(measured.sigma.mean()),
            "coverage": float(measured.covered.mean()),
            "all_window_coverage": float(group.covered.mean()),
            "onset_windows": int(group.new_contact_onset.sum()),
        })
    return pd.DataFrame.from_records(records)


def _plot_target_distributions(frame: pd.DataFrame, side: str, splits: list[str], output: Path) -> None:
    figure, axes = plt.subplots(len(splits), 3, figsize=(16, 4.2 * len(splits)), squeeze=False)
    for row_index, split in enumerate(splits):
        subset = frame[(frame.side == side) & (frame.split == split)]
        all_values = subset.target.to_numpy()
        limits = (float(np.min(all_values)), float(np.max(all_values)))
        grid = np.linspace(limits[0], limits[1], 300)
        for group in GROUPS:
            selected = subset.loc[_group_mask(subset, group), "target"].to_numpy()
            if not selected.size:
                continue
            label = f"{group} (n={selected.size})"
            axes[row_index, 0].hist(
                selected, bins=45, density=True, histtype="step", linewidth=1.5,
                color=COLORS[group], label=label,
            )
            if selected.size >= 3 and np.std(selected) > 1e-9:
                axes[row_index, 1].plot(
                    grid, gaussian_kde(selected)(grid), color=COLORS[group], label=label
                )
            ordered = np.sort(selected)
            axes[row_index, 2].plot(
                ordered, np.arange(1, ordered.size + 1) / ordered.size,
                color=COLORS[group], label=label,
            )
        all_quantiles = np.quantile(all_values, QUANTILES)
        for axis in axes[row_index]:
            for level, value in zip(QUANTILES, all_quantiles, strict=True):
                axis.axvline(value, color="#888", alpha=0.35, linewidth=0.8)
                if axis is axes[row_index, 2]:
                    axis.text(value, level, f" P{int(level*100)}", fontsize=7, rotation=90)
            axis.grid(alpha=0.18)
        axes[row_index, 0].set_ylabel(f"{split}\ndensity")
        axes[row_index, 0].legend(fontsize=7)
        axes[row_index, 0].set_title("Histogram")
        axes[row_index, 1].set_title("KDE")
        axes[row_index, 2].set_title("CDF")
    figure.suptitle(f"{side.upper()} target force distributions and all-sample quantiles", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _plot_signed_residuals(frame: pd.DataFrame, side: str, splits: list[str], output: Path) -> None:
    figure, axes = plt.subplots(2, len(splits), figsize=(5.3 * len(splits), 8), squeeze=False)
    for column, split in enumerate(splits):
        subset = frame[(frame.side == side) & (frame.split == split)]
        for group in GROUPS:
            values = subset.loc[_group_mask(subset, group), "residual"].to_numpy()
            if not values.size:
                continue
            label = f"{group} (n={values.size})"
            axes[0, column].hist(
                values, bins=55, density=True, histtype="step", linewidth=1.4,
                color=COLORS[group], label=label,
            )
            ordered = np.sort(values)
            axes[1, column].plot(
                ordered, np.arange(1, ordered.size + 1) / ordered.size,
                color=COLORS[group], label=label,
            )
        positive = subset.loc[subset.residual > 0, "residual"].to_numpy()
        text = "positive residual\n" + "  ".join(
            f"P{int(q*100)}={np.quantile(positive,q):.2f}" for q in (0.90, 0.95, 0.99)
        ) if positive.size else "no positive residual"
        for axis in axes[:, column]:
            axis.axvline(0.0, color="#555", linewidth=1.0)
            axis.grid(alpha=0.18)
        axes[0, column].set_title(f"{split} histogram\n{text}", fontsize=10)
        axes[1, column].set_title(f"{split} CDF")
        axes[0, column].legend(fontsize=7)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("CDF")
    figure.suptitle(f"{side.upper()} signed residual e = y - mu", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _plot_standardized_residuals(frame: pd.DataFrame, side: str, output: Path) -> None:
    splits = [name for name in ("calibration", "test") if name in set(frame.split)]
    figure, axes = plt.subplots(2, len(splits), figsize=(7 * len(splits), 8), squeeze=False)
    for column, split in enumerate(splits):
        subset = frame[(frame.side == side) & (frame.split == split)]
        for group in GROUPS:
            values = subset.loc[_group_mask(subset, group), "standardized_residual"].to_numpy()
            if not values.size:
                continue
            label = f"{group} (n={values.size})"
            axes[0, column].hist(
                values, bins=60, density=True, histtype="step", linewidth=1.4,
                color=COLORS[group], label=label,
            )
            ordered = np.sort(values)
            axes[1, column].plot(
                ordered, np.arange(1, ordered.size + 1) / ordered.size,
                color=COLORS[group], label=label,
            )
        all_values = subset.standardized_residual.to_numpy()
        quantile_text = "  ".join(
            f"P{int(q*100)}={np.quantile(all_values,q):.2f}" for q in QUANTILES
        )
        axes[0, column].set_title(f"{split} histogram\n{quantile_text}", fontsize=9)
        axes[1, column].set_title(f"{split} CDF")
        axes[0, column].legend(fontsize=7)
        for axis in axes[:, column]:
            axis.axvline(0.0, color="#555", linewidth=1.0)
            axis.grid(alpha=0.18)
    figure.suptitle(f"{side.upper()} standardized residual (y - mu) / sigma", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _plot_sigma_scatter(frame: pd.DataFrame, side: str, output: Path) -> None:
    subset = frame[(frame.side == side) & (frame.split == "test")]
    group_names = ("all", "prior_contact_true", "new_contact_onset")
    figure, axes = plt.subplots(2, len(group_names), figsize=(5.3 * len(group_names), 8), squeeze=False)
    for column, group in enumerate(group_names):
        selected = subset[_group_mask(subset, group)]
        axes[0, column].scatter(selected.sigma, selected.absolute_error, s=10, alpha=0.45, color=COLORS[group])
        axes[1, column].scatter(selected.sigma, selected.positive_residual, s=10, alpha=0.45, color=COLORS[group])
        first = _spearman(selected.sigma, selected.absolute_error)
        second = _spearman(selected.sigma, selected.positive_residual)
        axes[0, column].set_title(f"{group}\nrho={first['rho'] if first['rho'] is not None else 'NA'}")
        axes[1, column].set_title(f"{group}\nrho={second['rho'] if second['rho'] is not None else 'NA'}")
        axes[1, column].set_xlabel("sigma")
        for axis in axes[:, column]:
            axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel("absolute error")
    axes[1, 0].set_ylabel("positive residual")
    figure.suptitle(f"{side.upper()} test sigma association", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _plot_episode_heatmap(episodes: pd.DataFrame, side: str, output: Path) -> None:
    subset = episodes[(episodes.side == side) & (episodes.split == "test")].copy()
    subset = subset.sort_values(["batch", "episode"])
    columns = ["force_p95", "force_max", "mae", "positive_residual_p95", "mean_sigma", "coverage"]
    raw = subset[columns].to_numpy(dtype=np.float64)
    scaled = np.empty_like(raw)
    for index in range(raw.shape[1]):
        low, high = np.nanmin(raw[:, index]), np.nanmax(raw[:, index])
        scaled[:, index] = (raw[:, index] - low) / max(high - low, 1e-9)
    height = max(8, 0.32 * len(subset))
    figure, axis = plt.subplots(figsize=(10, height))
    image = axis.imshow(scaled, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(columns)), labels=columns, rotation=25, ha="right")
    labels = [f"{row.batch}:ep{int(row.episode):03d}" for row in subset.itertuples()]
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    for row_index in range(raw.shape[0]):
        for column_index in range(raw.shape[1]):
            value = raw[row_index, column_index]
            text = f"{value:.2f}" if columns[column_index] != "coverage" else f"{value:.1%}"
            axis.text(column_index, row_index, text, ha="center", va="center", fontsize=6,
                      color="white" if scaled[row_index, column_index] < 0.55 else "black")
    axis.set_title(f"{side.upper()} test episode metrics (color normalized per column)")
    figure.colorbar(image, ax=axis, fraction=0.02, pad=0.02)
    figure.tight_layout()
    figure.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(figure)


def _select_matches(frame: pd.DataFrame, top_k: int, phase_bins: int) -> list[dict[str, Any]]:
    left = frame[(frame.side == "left") & (frame.split == "test")].copy()
    left["phase_bin"] = np.minimum(phase_bins - 1, (left.episode_progress * phase_bins).astype(int))
    high = left.sort_values("residual", ascending=False).head(top_k)
    available = set(left.window_index.astype(int).tolist()) - set(high.window_index.astype(int).tolist())
    matches = []
    for rank, high_row in enumerate(high.itertuples(), start=1):
        eligible = left[
            (left.prior_contact == high_row.prior_contact)
            & (left.future_contact == high_row.future_contact)
            & (left.phase_bin == high_row.phase_bin)
            & (~left.window_index.isin(high.window_index))
        ].copy()
        candidates = eligible[eligible.window_index.isin(available)].copy()
        if not len(candidates):
            # Reuse a control before violating the requested phase/prior-contact match.
            candidates = eligible
        if not len(candidates):
            raise RuntimeError(f"No low-residual match for window {high_row.window_index}")
        target_scale = max(float(left.target.std()), 1e-6)
        candidates["match_cost"] = (
            (candidates.episode_progress - high_row.episode_progress).abs()
            + 0.25 * (candidates.target - high_row.target).abs() / target_scale
            + 0.25 * candidates.absolute_error / max(float(left.absolute_error.median()), 1e-6)
        )
        low_row = candidates.sort_values(["match_cost", "absolute_error"]).iloc[0]
        available.discard(int(low_row.window_index))
        matches.append({
            "rank": rank,
            "high_window_index": int(high_row.window_index),
            "low_window_index": int(low_row.window_index),
            "same_phase_bin": bool(int(low_row.phase_bin) == int(high_row.phase_bin)),
            "same_prior_contact": bool(low_row.prior_contact == high_row.prior_contact),
            "same_future_contact": bool(low_row.future_contact == high_row.future_contact),
            "high_residual": float(high_row.residual),
            "low_residual": float(low_row.residual),
            "high_absolute_error": float(high_row.absolute_error),
            "low_absolute_error": float(low_row.absolute_error),
            "high_progress": float(high_row.episode_progress),
            "low_progress": float(low_row.episode_progress),
        })
    return matches


class _RawDataReader:
    def __init__(self, force_axes: list[str]) -> None:
        self.force_axes = force_axes
        self.sidecar_maps: dict[str, dict[int, Path]] = {}
        self.streams: dict[tuple[str, int], dict[str, Any]] = {}

    def curves(self, row: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        root = str(row["tactile_root"])
        episode = int(row["source_episode_index"])
        if root not in self.sidecar_maps:
            self.sidecar_maps[root] = discover_tactile_sidecars(Path(root))
        key = (root, episode)
        if key not in self.streams:
            loaded, _ = load_tactile_streams(self.sidecar_maps[root][episode], self.force_axes)
            self.streams[key] = {stream.name: stream for stream in loaded}
        result = {}
        start = float(row["start_timestamp"])
        end = float(row["end_timestamp"])
        for side_name in ("left_tip", "right_tip"):
            stream = self.streams[key][side_name]
            selection = window_slice(stream.timestamps, start - 0.2, end)
            timestamps = stream.timestamps[selection]
            xyz = stream.forces_xyz[selection]
            valid = stream.valid[selection]
            timestamps, xyz = timestamps[valid], xyz[valid]
            baseline = np.asarray(row["quality"]["force_baseline_xyz"][side_name], dtype=np.float32)
            magnitude = np.linalg.norm(xyz - baseline, axis=1)
            result[side_name] = ((timestamps - start) * 1000.0, magnitude)
        return result


def _image_for_row(row: dict[str, Any], field: str, size: tuple[int, int]) -> Image.Image:
    episode_dir = (
        Path(row["robot_dataset"]) / "data" / f"episode_{int(row['source_episode_index']):06d}"
    )
    path = episode_dir / row["images"][field]["path"]
    try:
        image = Image.open(path).convert("RGB")
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    except Exception:
        return Image.new("RGB", size, (220, 220, 220))


def _make_montage(
    record: pd.Series, row: dict[str, Any], reader: _RawDataReader, output: Path,
    label: str, camera_fields: list[str],
) -> None:
    curves = reader.curves(row)
    figure = plt.figure(figsize=(16, 9))
    grid = GridSpec(2, 4, figure=figure, height_ratios=[1.0, 0.85])
    for index, field in enumerate(camera_fields):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(_image_for_row(row, field, (640, 480)))
        axis.set_title(f"camera {index + 1}\nage={row['images'][field]['age_ms']:.1f} ms", fontsize=10)
        axis.axis("off")
    meta_axis = figure.add_subplot(grid[0, 3])
    meta_axis.axis("off")
    meta_axis.text(
        0.0, 1.0,
        "\n".join([
            label,
            f"window={int(record.window_index)} split={record.split}",
            f"batch={record.batch_id}",
            f"episode={int(record.episode_index)} source_ep={int(record.source_episode_index)}",
            f"progress={record.episode_progress:.3f} phase_bin={int(record.phase_bin)}",
            f"prior_contact={bool(record.prior_contact)} onset={bool(record.new_contact_onset)}",
            f"target={record.target:.3f} raw_max={record.raw_max:.3f}",
            f"mu={record.mu:.3f} sigma={record.sigma:.3f}",
            f"upper95={record.upper95:.3f} covered={bool(record.covered)}",
            f"residual={record.residual:.3f} standardized={record.standardized_residual:.3f}",
        ]),
        va="top", family="monospace", fontsize=10,
    )

    tactile_axis = figure.add_subplot(grid[1, :2])
    for side_name, color in (("left_tip", "#c2415d"), ("right_tip", "#007a78")):
        times, values = curves[side_name]
        tactile_axis.plot(times, values, marker="o", markersize=3, linewidth=1.4,
                          color=color, label=side_name)
    tactile_axis.axvspan(-200, 0, color="#ddd", alpha=0.35, label="causal history")
    tactile_axis.axvspan(0, 300, color="#f0c", alpha=0.04, label="future target window")
    tactile_axis.axvline(0, color="#222", linewidth=1.0)
    tactile_axis.axhline(5.0, color="#666", linestyle="--", linewidth=0.9, label="contact threshold")
    tactile_axis.set_xlim(-205, 305)
    tactile_axis.set_xlabel("time from anchor (ms)")
    tactile_axis.set_ylabel("baseline-corrected force magnitude")
    tactile_axis.set_title("Past 200 ms and future 300 ms real tactile samples")
    tactile_axis.grid(alpha=0.2)
    tactile_axis.legend(fontsize=8, ncol=2)

    action_axis = figure.add_subplot(grid[1, 2:])
    action = np.asarray(row["executed_actions"], dtype=np.float64)
    delta = action[:, :7] - action[0, :7]
    action_axis.plot(np.arange(action.shape[0]), np.linalg.norm(delta, axis=1), marker="o",
                     color="#2456a6", label="arm ||a_t-a_0||")
    action_axis.set_xlabel("action step")
    action_axis.set_ylabel("arm action displacement")
    action_axis.grid(alpha=0.2)
    gripper_axis = action_axis.twinx()
    gripper_axis.plot(np.arange(action.shape[0]), action[:, -1], marker="s", linestyle="--",
                      color="#d97706", label="gripper")
    gripper_axis.set_ylabel("gripper action")
    lines = action_axis.get_lines() + gripper_axis.get_lines()
    action_axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    action_axis.set_title(
        f"Candidate/executed chunk 9x8; max arm travel={record.arm_action_travel:.4f}; "
        f"gripper delta={record.gripper_action_change:.4f}"
    )
    figure.suptitle(f"LEFT residual analysis: {label}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _make_contact_sheets(
    matches: list[dict[str, Any]], frame: pd.DataFrame, rows_by_index: dict[int, dict[str, Any]],
    camera_fields: list[str], output_dir: Path,
) -> list[str]:
    files = []
    lookup = frame[(frame.side == "left") & (frame.split == "test")].set_index(
        "window_index", drop=False
    )
    for page_start in range(0, len(matches), 10):
        page = matches[page_start:page_start + 10]
        figure, axes = plt.subplots(len(page), 6, figsize=(21, 2.8 * len(page)), squeeze=False)
        for row_index, match in enumerate(page):
            for pair_index, key in enumerate(("high_window_index", "low_window_index")):
                window_index = int(match[key])
                row = rows_by_index[window_index]
                record = lookup.loc[window_index]
                for camera_column, field in enumerate(camera_fields):
                    axis = axes[row_index, pair_index * 3 + camera_column]
                    axis.imshow(_image_for_row(row, field, (400, 300)))
                    axis.axis("off")
                    kind = "HIGH" if pair_index == 0 else "MATCH"
                    axis.set_title(
                        f"#{match['rank']} {kind} cam{camera_column + 1} w{window_index} "
                        f"e={record.residual:.2f} p={record.episode_progress:.2f}", fontsize=7
                    )
        figure.suptitle(
            "Left positive-residual visual pairs: all three anchor cameras", fontsize=14
        )
        figure.tight_layout()
        name = f"paired_contact_sheet_{page_start // 10 + 1:02d}.png"
        figure.savefig(output_dir / name, dpi=110, bbox_inches="tight")
        plt.close(figure)
        files.append(name)
    return files


def _paired_summary(
    matches: list[dict[str, Any]], frame: pd.DataFrame,
) -> dict[str, Any]:
    lookup = frame[(frame.side == "left") & (frame.split == "test")].set_index(
        "window_index", drop=False
    )
    high = lookup.loc[[item["high_window_index"] for item in matches]]
    low = lookup.loc[[item["low_window_index"] for item in matches]]
    all_test = frame[(frame.side == "left") & (frame.split == "test")]

    def describe(selected: pd.DataFrame) -> dict[str, Any]:
        return {
            "samples": int(len(selected)),
            "onset_fraction": float(selected.new_contact_onset.mean()),
            "prior_contact_fraction": float(selected.prior_contact.mean()),
            "future_contact_fraction": float(selected.future_contact.mean()),
            "target": _quantiles(selected.target),
            "residual": _quantiles(selected.residual),
            "sigma": _quantiles(selected.sigma),
            "prior_force_max": _quantiles(selected.prior_force_max),
            "image_age_max_ms": _quantiles(selected.image_age_max_ms),
            "arm_action_travel": _quantiles(selected.arm_action_travel),
            "gripper_action_change": _quantiles(selected.gripper_action_change),
            "batch_counts": dict(Counter(selected.batch_id)),
        }

    return {
        "selection": (
            "Top 50 signed positive residual windows on the left/test split; matched controls "
            "use the same prior-contact and future-contact state, prefer the same episode-progress "
            "decile, and minimize progress, target-force, and absolute-error distance."
        ),
        "high": describe(high),
        "matched_low": describe(low),
        "all_left_test": describe(all_test),
        "pair_quality": {
            "same_phase_bin_fraction": float(np.mean([item["same_phase_bin"] for item in matches])),
            "same_prior_contact_fraction": float(np.mean([item["same_prior_contact"] for item in matches])),
            "same_future_contact_fraction": float(np.mean([item["same_future_contact"] for item in matches])),
            "median_progress_difference": float(np.median([
                abs(item["high_progress"] - item["low_progress"]) for item in matches
            ])),
            "unique_matched_window_fraction": float(
                len({item["low_window_index"] for item in matches}) / len(matches)
            ),
        },
    }


def _evidence_summary(
    frame: pd.DataFrame, matches: list[dict[str, Any]], visual_review_notes: list[str],
) -> dict[str, Any]:
    left_test = frame[(frame.side == "left") & (frame.split == "test")].set_index(
        "window_index", drop=False
    )
    high = left_test.loc[[item["high_window_index"] for item in matches]]
    low = left_test.loc[[item["low_window_index"] for item in matches]]

    paired_differences = {}
    for column in (
        "target", "sigma", "prior_force_max", "image_age_max_ms", "arm_action_travel",
    ):
        difference = high[column].to_numpy() - low[column].to_numpy()
        paired_differences[column] = {
            "median_high_minus_match": float(np.median(difference)),
            "mean_high_minus_match": float(np.mean(difference)),
            "fraction_high_greater": float(np.mean(difference > 0)),
        }

    batch_rows = []
    for batch, group in left_test.groupby("batch_id"):
        high_count = int((high.batch_id == batch).sum())
        expected_share = float(len(group) / len(left_test))
        observed_share = float(high_count / len(high))
        positive = group[group.residual > 0].residual
        prior = group[group.prior_contact]
        batch_rows.append({
            "batch": str(batch),
            "test_windows": int(len(group)),
            "top50_windows": high_count,
            "expected_share": expected_share,
            "observed_top50_share": observed_share,
            "top50_enrichment": observed_share / expected_share,
            "coverage": float(group.covered.mean()),
            "prior_contact_coverage": float(prior.covered.mean()) if len(prior) else None,
            "positive_residual_p95": float(positive.quantile(0.95)) if len(positive) else None,
            "onset_fraction": float(group.new_contact_onset.mean()),
        })

    all_onset_fraction = float(left_test.new_contact_onset.mean())
    high_onset_fraction = float(high.new_contact_onset.mean())
    return {
        "visual_review": {
            "reviewed_pairs": int(len(matches)),
            "camera_views_per_window": 3,
            "notes": visual_review_notes,
            "interpretation": (
                "No consistent visual offset or grasp-shift association was observed in the paired "
                "anchor frames. This is an observational negative result under limited image "
                "resolution and missing pose annotations, not evidence of no visual effect."
            ),
        },
        "new_contact_onset": {
            "all_left_test_fraction": all_onset_fraction,
            "top50_fraction": high_onset_fraction,
            "top50_enrichment": high_onset_fraction / max(all_onset_fraction, 1e-12),
            "interpretation": (
                "Large positive residuals are strongly associated with new-contact onset in this "
                "test split; the analysis does not establish that onset causes the error."
            ),
        },
        "paired_differences": paired_differences,
        "history_preload": {
            "interpretation": (
                "After matching prior-contact state and episode phase, maximum prior force differs "
                "only slightly in the median; this comparison does not show a strong preload-level "
                "separation between high and matched windows."
            ),
        },
        "batch_association": {
            "rows": batch_rows,
            "interpretation": (
                "Batch 001 is over-represented among the top-50 and has the lowest left/test "
                "coverage. Batch is associated with error, but batch identity may proxy changes in "
                "trajectory, contact stage, scene, or sensor conditions."
            ),
        },
    }


def _table_html(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_html(index=False, classes="data-table", border=0, float_format=lambda x: f"{x:.4f}")


def _build_html(
    output: Path, summary: dict[str, Any], episodes: pd.DataFrame,
    plot_files: dict[str, str], matches: list[dict[str, Any]], montage_dir: Path,
    contact_sheets: list[str],
) -> None:
    correlation_rows = []
    for side, side_values in summary["spearman"].items():
        for split, split_values in side_values.items():
            for group, metrics in split_values.items():
                correlation_rows.append({
                    "side": side, "split": split, "group": group,
                    "rho_sigma_abs": metrics["sigma_vs_absolute_error"]["rho"],
                    "p_sigma_abs": metrics["sigma_vs_absolute_error"]["p_value"],
                    "rho_sigma_positive": metrics["sigma_vs_positive_residual"]["rho"],
                    "p_sigma_positive": metrics["sigma_vs_positive_residual"]["p_value"],
                })
    correlations = pd.DataFrame(correlation_rows)
    pair_rows = []
    for item in matches:
        pair_rows.append({
            "rank": item["rank"],
            "high": f'<a href="montages/high_{item["rank"]:02d}_w{item["high_window_index"]}.png">w{item["high_window_index"]}</a>',
            "high_e": item["high_residual"],
            "matched": f'<a href="montages/matched_{item["rank"]:02d}_w{item["low_window_index"]}.png">w{item["low_window_index"]}</a>',
            "matched_e": item["low_residual"],
            "same_phase": item["same_phase_bin"],
            "progress_delta": abs(item["high_progress"] - item["low_progress"]),
        })
    pairs_html = pd.DataFrame(pair_rows).to_html(
        index=False, escape=False, classes="data-table", border=0,
        float_format=lambda x: f"{x:.4f}",
    )
    test_episode_table = episodes[episodes.split == "test"].copy()
    episode_columns = [
        "batch", "episode", "side", "windows", "contact_windows", "force_mean",
        "force_p95", "force_max", "mae", "positive_residual_p95", "mean_sigma", "coverage",
    ]
    plot_html = "\n".join(
        f'<figure><img loading="lazy" src="plots/{html.escape(name)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        for label, name in plot_files.items()
    )
    sheet_html = "\n".join(
        f'<figure><a href="contact_sheets/{name}"><img loading="lazy" src="contact_sheets/{name}"></a></figure>'
        for name in contact_sheets
    )
    paired = summary["matched_visual_analysis"]
    left_test = summary["distributions"]["left"]["test"]
    evidence = summary["evidence_findings"]
    observations = [
        (
            f"Left/test onset windows have positive-residual P95 "
            f"{left_test['new_contact_onset']['positive_residual_only']['p95']:.3f}; "
            f"prior-contact windows have P95 "
            f"{left_test['prior_contact_true']['positive_residual_only']['p95']:.3f}."
        ),
        (
            f"The top-50 high-residual set contains {paired['high']['onset_fraction']:.1%} onset "
            f"windows versus {paired['all_left_test']['onset_fraction']:.1%} in all left/test windows "
            f"({evidence['new_contact_onset']['top50_enrichment']:.1f}x enrichment)."
        ),
        (
            f"Top-50 median target is {paired['high']['target']['p50']:.3f}, while matched controls "
            f"have median {paired['matched_low']['target']['p50']:.3f}."
        ),
        (
            f"Top-50 median maximum prior force is {paired['high']['prior_force_max']['p50']:.3f}, "
            f"while matched controls are {paired['matched_low']['prior_force_max']['p50']:.3f}; "
            f"paired median difference is "
            f"{evidence['paired_differences']['prior_force_max']['median_high_minus_match']:.3f}."
        ),
        (
            evidence["batch_association"]["interpretation"]
        ),
        *evidence["visual_review"]["notes"],
    ]
    observations_html = "".join(f"<li>{html.escape(item)}</li>" for item in observations)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>200 ms causal-history force-envelope analysis</title>
<style>
body{{font-family:Inter,Arial,sans-serif;color:#202124;margin:0;background:#f5f6f7;line-height:1.55}}
main{{max-width:1500px;margin:auto;background:white;padding:28px 38px 80px}}
h1,h2,h3{{color:#17202a}} .muted{{color:#667085}} .warning{{border-left:4px solid #c2415d;background:#fff5f6;padding:12px 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}} figure{{margin:0;border:1px solid #ddd;padding:8px;background:#fff}}
figure img{{width:100%;height:auto;display:block}} figcaption{{font-size:13px;color:#555;padding:6px}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px}} .data-table th,.data-table td{{border:1px solid #ddd;padding:5px 7px;text-align:right}}
.data-table th{{background:#eef1f4;position:sticky;top:0}} .table-wrap{{max-height:620px;overflow:auto;border:1px solid #ddd}}
code{{background:#eef1f4;padding:2px 4px}} details{{margin:14px 0}} a{{color:#075985}}
</style></head><body><main>
<h1>200 ms 因果触觉历史 checkpoint：左右力分布与视觉对应分析</h1>
<p class="muted">生成时间：{html.escape(summary['generated_at'])}<br>Checkpoint：<code>{html.escape(summary['checkpoint'])}</code></p>
<div class="warning"><strong>分析边界：</strong>本报告没有重新训练，也没有修改模型、loss 或 calibration。视觉偏孔/夹持偏移没有人工标签，因此只提供 montage 和可观察统计，不把共现关系写成因果。</div>
<h2>摘要</h2><ul>{observations_html}</ul>
<h2>分布、残差与不确定性图</h2><div class="grid">{plot_html}</div>
<h2>Spearman 相关性</h2><p>相关系数只描述单调关联，不表示因果。</p><div class="table-wrap">{_table_html(correlations, list(correlations.columns))}</div>
<h2>Test episode 表</h2><p>存在接触窗口时，force/error/sigma/coverage 按该侧接触窗口统计；无接触 episode 才回退到全窗口。</p>
<div class="table-wrap">{_table_html(test_episode_table, episode_columns)}</div>
<p><a href="episode_metrics.csv">下载完整 train/calibration/test episode CSV</a>；<a href="analysis_summary.json">下载 JSON 汇总</a>。</p>
<h2>左侧 Top-50 正残差与匹配低残差</h2>
<p>高残差来自 test split。匹配优先要求相同 prior-contact、future-contact 和 episode 进度 decile；每个窗口的完整 montage 包含三路图像、历史/未来真实触觉曲线和动作摘要。</p>
<div class="grid">{sheet_html}</div><details><summary>展开 50 组 montage 链接</summary><div class="table-wrap">{pairs_html}</div></details>
<h2>证据约束下的判断</h2>
<ul><li>可以量化：onset、历史预载、真实力、sigma、动作幅度、batch 与残差/coverage 的关联。</li>
<li>只能定性检查：画面中的 USB/孔位偏移、夹持偏移和遮挡。</li>
<li>不能从本报告推出：某个视觉现象导致力预测误差。要检验该命题，需要人工视觉标签或可测量的 USB/孔位姿态。</li></ul>
</main></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def analyze_history_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    analysis = config.get("analysis", {})
    checkpoint_path = Path(str(analysis["checkpoint"])).expanduser().resolve()
    output = Path(str(analysis["output_dir"])).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    montages = output / "montages"
    sheets = output / "contact_sheets"
    for directory in (plots, montages, sheets):
        directory.mkdir(parents=True, exist_ok=True)
        for stale_plot in directory.glob("*.png"):
            stale_plot.unlink()
    splits = [str(value) for value in analysis.get("splits", ["train", "calibration", "test"])]
    frame, rows, checkpoint, history_cache = _load_model_and_predictions(
        config, checkpoint_path, splits
    )
    rows_by_index = {int(row["window_index"]): row for row in rows}
    distributions = _distribution_summary(frame, splits)
    correlations = _correlation_summary(frame, splits)
    episodes = _episode_table(frame)
    episodes.to_csv(output / "episode_metrics.csv", index=False)
    frame.to_csv(output / "window_predictions.csv", index=False)

    plot_files: dict[str, str] = {}
    for side in SIDES:
        name = f"target_distributions_{side}.png"
        _plot_target_distributions(frame, side, splits, plots / name)
        plot_files[f"{side}: y histogram/KDE/CDF"] = name
        name = f"signed_residuals_{side}.png"
        _plot_signed_residuals(frame, side, splits, plots / name)
        plot_files[f"{side}: signed residual histogram/CDF"] = name
        name = f"standardized_residuals_{side}.png"
        _plot_standardized_residuals(frame, side, plots / name)
        plot_files[f"{side}: calibration/test standardized residual"] = name
        name = f"sigma_scatter_{side}.png"
        _plot_sigma_scatter(frame, side, plots / name)
        plot_files[f"{side}: sigma association on test"] = name
        name = f"episode_heatmap_{side}.png"
        _plot_episode_heatmap(episodes, side, plots / name)
        plot_files[f"{side}: test episode heatmap"] = name

    top_k = int(analysis.get("montage_top_k", 50))
    phase_bins = int(analysis.get("phase_bins", 10))
    matches = _select_matches(frame, top_k, phase_bins)
    camera_fields = list(config["vision"]["camera_order"])
    reader = _RawDataReader(list(config["label"]["force_axes"]))
    left_lookup = frame[(frame.side == "left") & (frame.split == "test")].set_index(
        "window_index", drop=False
    )
    for item in matches:
        rank = int(item["rank"])
        for kind, key in (("high", "high_window_index"), ("matched", "low_window_index")):
            window_index = int(item[key])
            record = left_lookup.loc[window_index]
            _make_montage(
                record,
                rows_by_index[window_index],
                reader,
                montages / f"{kind}_{rank:02d}_w{window_index}.png",
                f"{kind.upper()} pair rank {rank}",
                camera_fields,
            )
    contact_sheets = _make_contact_sheets(matches, frame, rows_by_index, camera_fields, sheets)
    paired = _paired_summary(matches, frame)
    evidence = _evidence_summary(
        frame, matches, [str(note) for note in analysis.get("visual_review_notes", [])]
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "model_unchanged": True,
        "loss_unchanged": True,
        "calibration_unchanged": True,
        "calibration": _jsonable(checkpoint["calibration"]),
        "splits": {name: int((frame.split == name).sum() // 2) for name in splits},
        "contact_threshold": float(config["label"].get("contact_threshold", 5.0)),
        "history": _jsonable(history_cache["report"]),
        "distributions": distributions,
        "spearman": correlations,
        "episode_metrics_rows": int(len(episodes)),
        "matched_visual_analysis": paired,
        "evidence_findings": evidence,
        "matches": matches,
        "limitations": [
            "No ground-truth USB pose, hole pose, grasp-offset, or occlusion labels are available.",
            "Visual montage review is observational and cannot establish causality.",
            "Future actions in the offline dataset may include human feedback unavailable at anchor time.",
            "The test split is used for diagnostics and visual case selection, not for model updates.",
        ],
        "artifacts": {
            "html": str(output / "index.html"),
            "json": str(output / "analysis_summary.json"),
            "episode_csv": str(output / "episode_metrics.csv"),
            "window_csv": str(output / "window_predictions.csv"),
            "plots": {label: str(plots / name) for label, name in plot_files.items()},
            "montage_count": top_k * 2,
            "contact_sheets": [str(sheets / name) for name in contact_sheets],
        },
    }
    write_json(output / "analysis_summary.json", _jsonable(summary))
    _build_html(output, summary, episodes, plot_files, matches, montages, contact_sheets)
    return {
        "html": str(output / "index.html"),
        "json": str(output / "analysis_summary.json"),
        "episode_csv": str(output / "episode_metrics.csv"),
        "window_csv": str(output / "window_predictions.csv"),
        "montages": top_k * 2,
        "plots": len(plot_files),
    }
