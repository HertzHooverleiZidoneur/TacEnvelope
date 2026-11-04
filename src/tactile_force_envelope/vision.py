from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import pil_to_tensor

from .common import read_jsonl, write_json
from .smolvla_condition import (
    FEATURE_TAP,
    MASK_TAP,
    TOKEN_FEATURE_TAP,
    FrozenSmolVLAConditionEncoder,
    checkpoint_fingerprint,
    read_preprocessor_camera_map,
)


class _ImageDataset(Dataset):
    def __init__(self, records: list[tuple[int, str, Path]], transform: Any):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[int, str, torch.Tensor]:
        window_index, field, path = self.records[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return window_index, field, tensor


def _encoder(config: dict[str, Any]) -> tuple[nn.Module, Any, int]:
    name = str(config["vision"]["encoder"])
    if name != "resnet18":
        raise ValueError(f"Only the validated frozen encoder is currently supported: {name}")
    weights_name = str(config["vision"].get("weights", "default")).lower()
    weights = ResNet18_Weights.DEFAULT if weights_name == "default" else None
    model = resnet18(weights=weights)
    feature_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    model.requires_grad_(False)
    model.eval()
    transform = weights.transforms() if weights else ResNet18_Weights.DEFAULT.transforms()
    return model, transform, feature_dim


def _source_image_path(row: dict[str, Any], dataset: Path, field: str) -> Path:
    row_dataset = Path(row.get("robot_dataset") or dataset).expanduser().resolve()
    source_episode = int(row.get("source_episode_index", row["episode_index"]))
    return row_dataset / "data" / f"episode_{source_episode:06d}" / str(row["images"][field]["path"])


def _cache_resnet_features(config: dict[str, Any]) -> dict[str, Any]:
    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    windows = read_jsonl(output / "windows.jsonl")
    if not windows:
        raise ValueError("windows.jsonl is empty")
    image_fields = list(windows[0]["images"])
    records: list[tuple[int, str, Path]] = []
    missing: list[str] = []
    for row in windows:
        for field in image_fields:
            path = _source_image_path(row, dataset, field)
            if path.exists():
                records.append((int(row["window_index"]), field, path))
            else:
                missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} source images are unavailable. The training mirror must include images.\n{preview}"
        )
    device_cfg = str(config["training"].get("device", "auto"))
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available()
                          else "cpu" if device_cfg == "auto" else device_cfg)
    model, transform, feature_dim = _encoder(config)
    model.to(device)
    loader = DataLoader(
        _ImageDataset(records, transform),
        batch_size=int(config["vision"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["vision"].get(
            "num_workers", config["training"].get("num_workers", 0)
        )),
        pin_memory=device.type == "cuda",
    )
    field_index = {field: index for index, field in enumerate(image_fields)}
    features = torch.zeros((len(windows), len(image_fields), feature_dim), dtype=torch.float32)
    with torch.inference_mode():
        for window_indices, fields, images in loader:
            encoded = model(images.to(device, non_blocking=True)).cpu()
            for batch_index, (window_index, field) in enumerate(zip(window_indices.tolist(), fields)):
                features[window_index, field_index[str(field)]] = encoded[batch_index]
    cache = {
        "features": features,
        "image_fields": image_fields,
        "encoder": str(config["vision"]["encoder"]),
        "feature_dim_per_camera": feature_dim,
        "windows_sha256": hashlib.sha256((output / "windows.jsonl").read_bytes()).hexdigest(),
    }
    torch.save(cache, output / "condition_features.pt")
    report = {
        "windows": len(windows), "images": len(records), "image_fields": image_fields,
        "feature_shape": list(features.shape), "device": str(device),
        "cache": str(output / "condition_features.pt"),
        "feature_contains_language": False,
        "feature_contains_robot_state": False,
        "ablation_only": True,
    }
    write_json(output / "visual_features.json", report)
    return report


class _SmolVLAWindowDataset(Dataset):
    def __init__(self, windows: list[dict[str, Any]], dataset: Path, camera_order: list[str]):
        self.windows = windows
        self.dataset = dataset
        self.camera_order = camera_order

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.windows[index]
        result: dict[str, Any] = {
            "window_index": int(row["window_index"]),
            "state": torch.tensor(row["state"], dtype=torch.float32),
            "task": str(row["task"]),
        }
        for field in self.camera_order:
            path = _source_image_path(row, self.dataset, field)
            with Image.open(path) as image:
                result[field] = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        return result


def _resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint /= "pretrained_model"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Invalid SmolVLA checkpoint: {checkpoint}")
    return checkpoint


def _smolvla_cache_signature(
    windows_hash: str, checkpoint_hash: str, config: dict[str, Any], camera_order: list[str],
    cache_kind: str = "pooled",
) -> str:
    extractor_source = Path(__file__).with_name("smolvla_condition.py")
    payload = {
        "cache_schema_version": 3 if cache_kind == "tokens" else 2,
        "windows_sha256": windows_hash,
        "checkpoint_sha256": checkpoint_hash,
        "feature_tap": TOKEN_FEATURE_TAP if cache_kind == "tokens" else FEATURE_TAP,
        "mask_tap": MASK_TAP,
        "pooling": (
            "none" if cache_kind == "tokens"
            else "per_layer_key_value_masked_token_mean_then_head_mean_then_concat"
        ),
        "cache_kind": cache_kind,
        "camera_order": camera_order,
        "feature_contains_language": True,
        "feature_contains_robot_state": True,
        "vision_config": config["vision"],
        "extractor_code_sha256": hashlib.sha256(extractor_source.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


PI05_JAX_CAMERA_MAP = {
    "observation.images.camera_camera_1_color_image_raw": "exterior_image_1_left",
    "observation.images.camera_camera_2_color_image_raw": "exterior_image_2_left",
    "observation.images.d515_color_image_raw": "wrist_image_left",
}


def _pi05_checkpoint_fingerprint(checkpoint: Path) -> str:
    """Hash the small files that identify an Orbax checkpoint and its assets."""
    candidates = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params" / "_METADATA",
        checkpoint / "params" / "_sharding",
    ]
    candidates.extend(sorted((checkpoint / "assets").rglob("norm_stats.json")))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"No Pi0.5 checkpoint metadata found under {checkpoint}"
        )
    digest = hashlib.sha256()
    for path in existing:
        digest.update(str(path.relative_to(checkpoint)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pi05_jax_cache_signature(
    windows_hash: str,
    checkpoint_hash: str,
    config: dict[str, Any],
    camera_order: list[str],
    metadata: dict[str, Any],
) -> str:
    payload = {
        "cache_schema_version": 1,
        "windows_sha256": windows_hash,
        "checkpoint_sha256": checkpoint_hash,
        "encoder": "pi05_jax_z_rl",
        "feature_tap": "joint JAX RLTPolicy response.z_rl",
        "pooling": "none; flattened RL token exposed by the policy server",
        "camera_order": camera_order,
        "feature_contains_language": True,
        "feature_contains_robot_state": False,
        "server_contract": {
            key: metadata.get(key)
            for key in (
                "has_rl_token",
                "z_dim",
                "proprio_dim",
                "chunk_len",
                "action_dim",
            )
        },
        "vision_config": config["vision"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_rgb_array(row: dict[str, Any], dataset: Path, field: str) -> np.ndarray:
    path = _source_image_path(row, dataset, field)
    if not path.is_file():
        raise FileNotFoundError(f"Pi0.5 source image is unavailable: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


class _LocalV3VideoReader:
    """Sequentially decode selected source frames from a local LeRobot v3 dataset."""

    def __init__(
        self,
        root: Path,
        windows: list[dict[str, Any]],
        camera_order: list[str],
        source_episode_count: int,
        removed_source_episode_ids: list[int],
    ) -> None:
        import av
        import pyarrow.parquet as pq

        self._av = av
        self.root = root.expanduser().resolve()
        self.camera_order = camera_order
        removed = {int(value) for value in removed_source_episode_ids}
        source_ids = [
            episode_id for episode_id in range(source_episode_count)
            if episode_id not in removed
        ]
        info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        total_episodes = int(info["total_episodes"])
        if len(source_ids) != total_episodes:
            raise ValueError(
                "Local v3 source episode mapping does not match dataset: "
                f"{len(source_ids)} mapped != {total_episodes} v3 episodes"
            )
        self._source_to_v3 = {
            source_episode: v3_episode
            for v3_episode, source_episode in enumerate(source_ids)
        }
        requested: set[tuple[int, int]] = set()
        for row in windows:
            source_episode = int(row.get("source_episode_index", row["episode_index"]))
            if source_episode not in self._source_to_v3:
                raise ValueError(f"Source episode {source_episode} is absent from local v3 mapping")
            for field in camera_order:
                match = re.search(r"frame_(\d+)\.[^.]+$", str(row["images"][field]["path"]))
                if match is None:
                    raise ValueError(
                        f"Cannot recover source frame index from {row['images'][field]['path']}"
                    )
                requested.add((self._source_to_v3[source_episode], int(match.group(1))))

        self._locations: dict[tuple[int, int], tuple[int, int]] = {}
        self._data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        for file_index, path in enumerate(self._data_files):
            table = pq.read_table(path, columns=["episode_index", "frame_index"])
            episodes = table["episode_index"].to_numpy()
            frames = table["frame_index"].to_numpy()
            for row_index, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
                key = (int(episode), int(frame))
                if key in requested:
                    self._locations[key] = (file_index, row_index)
        missing = requested - set(self._locations)
        if missing:
            raise ValueError(f"Local v3 videos are missing {len(missing)} requested frame locations")

        self._streams: dict[str, dict[str, Any]] = {
            field: {"file_index": None, "container": None, "iterator": None, "frame_index": -1}
            for field in camera_order
        }

    def _video_path(self, field: str, file_index: int) -> Path:
        parquet_path = self._data_files[file_index]
        relative = parquet_path.relative_to(self.root / "data").with_suffix(".mp4")
        return self.root / "videos" / field / relative

    def read(self, row: dict[str, Any], field: str) -> np.ndarray:
        source_episode = int(row.get("source_episode_index", row["episode_index"]))
        match = re.search(r"frame_(\d+)\.[^.]+$", str(row["images"][field]["path"]))
        if match is None:
            raise ValueError(f"Cannot parse image frame path: {row['images'][field]['path']}")
        key = (self._source_to_v3[source_episode], int(match.group(1)))
        file_index, target_index = self._locations[key]
        stream = self._streams[field]
        if stream["file_index"] != file_index:
            if stream["container"] is not None:
                stream["container"].close()
            path = self._video_path(field, file_index)
            if not path.is_file():
                raise FileNotFoundError(f"Local LeRobot v3 video is unavailable: {path}")
            container = self._av.open(str(path))
            video = container.streams.video[0]
            stream.update({
                "file_index": file_index,
                "container": container,
                "iterator": iter(container.decode(video)),
                "frame_index": -1,
            })
        if target_index < int(stream["frame_index"]):
            raise RuntimeError("Local v3 frame requests must be monotonically ordered")
        frame = None
        while int(stream["frame_index"]) < target_index:
            try:
                frame = next(stream["iterator"])
            except StopIteration as error:
                raise RuntimeError(
                    f"Video ended before frame {target_index}: {self._video_path(field, file_index)}"
                ) from error
            stream["frame_index"] = int(stream["frame_index"]) + 1
        if frame is None:
            raise RuntimeError("Duplicate local v3 frame request is not supported")
        return frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        for stream in self._streams.values():
            if stream["container"] is not None:
                stream["container"].close()


def _cache_pi05_jax_z_rl(config: dict[str, Any]) -> dict[str, Any]:
    """Cache the exact z_rl exposed by the resident Pi0.5 JAX policy server."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    windows_path = output / "windows.jsonl"
    windows = read_jsonl(windows_path)
    if not windows:
        raise ValueError("windows.jsonl is empty")

    vision = config["vision"]
    checkpoint = Path(str(vision["checkpoint"])).expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Invalid Pi0.5 JAX checkpoint: {checkpoint}")
    checkpoint_hash = _pi05_checkpoint_fingerprint(checkpoint)
    windows_hash = hashlib.sha256(windows_path.read_bytes()).hexdigest()

    camera_order = list(vision.get("camera_order") or [])
    if camera_order != list(windows[0]["images"]):
        raise ValueError(
            "Configured Pi0.5 camera_order does not match window camera order: "
            f"{camera_order} != {list(windows[0]['images'])}"
        )
    missing_mappings = [field for field in camera_order if field not in PI05_JAX_CAMERA_MAP]
    if missing_mappings:
        raise ValueError(f"No Pi0.5 input mapping for cameras: {missing_mappings}")

    host = str(vision.get("server_host", "127.0.0.1"))
    port = int(vision.get("server_port", 8011))
    client = WebsocketClientPolicy(host, port)
    metadata = dict(client.get_server_metadata())
    expected = {
        "has_rl_token": True,
        "z_dim": int(vision.get("feature_dim", 512)),
        "proprio_dim": 8,
        "action_dim": 8,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"Pi0.5 JAX server metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )

    signature = _pi05_jax_cache_signature(
        windows_hash, checkpoint_hash, config, camera_order, metadata
    )
    cache_path = output / "condition_tokens.pt"
    if cache_path.is_file():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        if existing.get("cache_signature") == signature:
            report = dict(existing["report"])
            report["reused"] = True
            return report

    z_dim = int(metadata["z_dim"])
    selected_windows = windows
    selected_window_mask = torch.ones(len(windows), dtype=torch.bool)
    if bool(vision.get("cache_sustained_contact_only", False)):
        from .stable_contact import stable_contact_mask

        history_path = output / "causal_tactile_history.pt"
        if not history_path.is_file():
            raise FileNotFoundError(
                "cache_sustained_contact_only requires causal_tactile_history.pt; "
                "run cache-tactile-history first"
            )
        history_cache = torch.load(history_path, map_location="cpu", weights_only=False)
        if history_cache.get("windows_sha256") != windows_hash:
            raise ValueError("causal_tactile_history.pt does not match windows.jsonl")
        stable_cfg = config.get("stable_contact", {})
        stable = stable_contact_mask(
            history_cache["history"],
            history_cache["mask"],
            magnitude_index=int(stable_cfg.get("magnitude_index", 3)),
            tau_on=float(stable_cfg.get("tau_on", 5.0)),
            recent_samples=int(stable_cfg.get("recent_samples", 4)),
            required_samples=int(stable_cfg.get("required_samples", 3)),
        )
        tau_train = float(stable_cfg.get("tau_train", 5.0))
        sensor_names = config["label"].get(
            "sensor_names", {"left": "left_tip", "right": "right_tip"}
        )
        selected_window_mask.zero_()
        for row in windows:
            index = int(row["window_index"])
            selected_window_mask[index] = any(
                bool(stable[index, side_index])
                and float(row["tactile"]["per_sensor"][str(sensor_names[side])]["filtered_top2_mean"])
                > tau_train
                for side_index, side in enumerate(("left", "right"))
            )
        selected_windows = [
            row for row in windows if bool(selected_window_mask[int(row["window_index"])])
        ]
        if not selected_windows:
            raise ValueError("No sustained-contact windows are available for feature caching")

    condition_tokens = torch.zeros((len(windows), 1, z_dim), dtype=torch.float32)
    condition_masks = torch.zeros((len(windows), 1), dtype=torch.bool)
    records: list[dict[str, Any] | None] = [None] * len(windows)
    batch_size = int(vision.get("batch_size", 8))
    if not 1 <= batch_size <= 16:
        raise ValueError("Pi0.5 JAX cache batch_size must be in [1, 16]")

    local_v3_root = vision.get("local_v3_dataset")
    local_reader = None
    if local_v3_root:
        local_reader = _LocalV3VideoReader(
            Path(str(local_v3_root)),
            selected_windows,
            camera_order,
            int(vision["source_episode_count"]),
            list(vision.get("removed_source_episode_ids", [])),
        )

    try:
        for start in range(0, len(selected_windows), batch_size):
            batch_rows = selected_windows[start : start + batch_size]
            requests = []
            for row in batch_rows:
                state = np.asarray(row["state"], dtype=np.float32).reshape(-1)
                if state.shape != (8,) or not np.isfinite(state).all():
                    raise ValueError(
                        f"window {row['window_index']} has invalid 8D state {state.shape}"
                    )
                request: dict[str, Any] = {
                    "state": state,
                    "prompt": str(row["task"]),
                }
                for field in camera_order:
                    request[PI05_JAX_CAMERA_MAP[field]] = (
                        local_reader.read(row, field)
                        if local_reader is not None
                        else _load_rgb_array(row, dataset, field)
                    )
                requests.append(request)

            response = client.infer({"batch": requests})
            results = response.get("batch_results")
            if not isinstance(results, list) or len(results) != len(batch_rows):
                raise RuntimeError(
                    f"Pi0.5 JAX batch response has {len(results) if isinstance(results, list) else 'invalid'} "
                    f"results for {len(batch_rows)} requests"
                )
            for row, result in zip(batch_rows, results, strict=True):
                window_index = int(row["window_index"])
                z_rl = np.asarray(result.get("z_rl"), dtype=np.float32).reshape(-1)
                if z_rl.shape != (z_dim,) or not np.isfinite(z_rl).all():
                    raise RuntimeError(
                        f"window {window_index} returned invalid z_rl {z_rl.shape}"
                    )
                condition_tokens[window_index, 0] = torch.from_numpy(z_rl.copy())
                condition_masks[window_index, 0] = True
                records[window_index] = {
                    "episode_id": int(row["episode_index"]),
                    "window_id": window_index,
                    "anchor_timestamp": float(row["start_timestamp"]),
                    "image_timestamps": {
                        field: float(row["images"][field]["stamp"])
                        for field in camera_order
                    },
                    "feature_shape": [1, z_dim],
                    "condition_valid_tokens": 1,
                    "task_text": str(row["task"]),
                }
            print(
                "Pi0.5 z_rl cache: "
                f"{min(start + len(batch_rows), len(selected_windows))}/{len(selected_windows)} "
                f"selected ({len(windows)} total)",
                flush=True,
            )
    finally:
        if local_reader is not None:
            local_reader.close()

    report = {
        "windows": len(windows),
        "cached_windows": len(selected_windows),
        "selection": (
            "per-side causal stable prior and future filtered_top2_mean > tau_train"
            if len(selected_windows) != len(windows) else "all windows"
        ),
        "condition_token_shape": list(condition_tokens.shape),
        "condition_mask_shape": list(condition_masks.shape),
        "cache_dtype": str(condition_tokens.dtype),
        "cache": str(cache_path),
        "encoder": "pi05_jax_z_rl",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "feature_tap": "joint JAX RLTPolicy response.z_rl",
        "pooling": "none; z_rl is one 512D condition token",
        "server": {"host": host, "port": port, "metadata": metadata},
        "camera_order_raw": camera_order,
        "camera_order_after_preprocessor": [
            PI05_JAX_CAMERA_MAP[field] for field in camera_order
        ],
        "image_preprocessing": {
            "source": "resident Pi0.5 JAX policy server input transforms",
            "client_input": "uint8 RGB",
            "missing_camera_rule": "all three configured cameras are required",
            "frame_source": str(Path(str(local_v3_root)).expanduser().resolve()) if local_v3_root else str(dataset),
        },
        "language": {
            "included": True,
            "task_format": "same prompt string sent to online JAX policy",
        },
        "robot_state": {
            "included": False,
            "format": "8D state is sent to policy, but z_rl encodes the image/language prefix; "
            "the Envelope Expert receives robot state separately",
        },
        "cache_signature": signature,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    cache = {
        "condition_tokens": condition_tokens,
        "condition_mask": condition_masks,
        "selected_window_mask": selected_window_mask,
        "records": records,
        "report": report,
        "encoder": "pi05_jax_z_rl",
        "condition_dim": z_dim,
        "feature_contains_language": True,
        "feature_contains_robot_state": False,
        "windows_sha256": windows_hash,
        "checkpoint_sha256": checkpoint_hash,
        "cache_signature": signature,
    }
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    write_json(output / "condition_tokens.json", report)
    return report


def _load_smolvla_cache_context(config: dict[str, Any]):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    windows_path = output / "windows.jsonl"
    windows = read_jsonl(windows_path)
    if not windows:
        raise ValueError("windows.jsonl is empty")
    checkpoint = _resolve_checkpoint(str(config["vision"]["checkpoint"]))
    checkpoint_hash = checkpoint_fingerprint(checkpoint)
    windows_hash = hashlib.sha256(windows_path.read_bytes()).hexdigest()
    camera_map = read_preprocessor_camera_map(checkpoint)
    configured_order = list(config["vision"].get("camera_order") or [])
    if not configured_order:
        raise ValueError("SmolVLA cache requires an explicit raw camera_order")
    if configured_order != list(windows[0]["images"]):
        raise ValueError(
            "Configured SmolVLA camera_order does not match the window camera order: "
            f"{configured_order} != {list(windows[0]['images'])}"
        )
    renamed_order = [camera_map.get(field) for field in configured_order]
    if any(value is None for value in renamed_order):
        raise ValueError(f"Checkpoint preprocessor does not map all cameras: {camera_map}")
    device = torch.device(str(config["training"].get("device", "cuda")))
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
    policy.requires_grad_(False)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("SmolVLA was not completely frozen")
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return (
        dataset, output, windows_path, windows, checkpoint, checkpoint_hash, windows_hash,
        configured_order, renamed_order, device, policy, preprocessor,
    )


def _cache_smolvla_tokens(config: dict[str, Any]) -> dict[str, Any]:
    (
        dataset, output, _windows_path, windows, checkpoint, checkpoint_hash, windows_hash,
        configured_order, renamed_order, device, policy, preprocessor,
    ) = _load_smolvla_cache_context(config)
    signature = _smolvla_cache_signature(
        windows_hash, checkpoint_hash, config, configured_order, cache_kind="tokens"
    )
    cache_path = output / "condition_tokens.pt"
    if cache_path.is_file():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        if existing.get("cache_signature") == signature:
            report = dict(existing["report"])
            report["reused"] = True
            return report

    encoder = FrozenSmolVLAConditionEncoder(policy).to(device)
    loader = DataLoader(
        _SmolVLAWindowDataset(windows, dataset, configured_order),
        batch_size=int(config["vision"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["vision"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    condition_tokens: torch.Tensor | None = None
    condition_masks: torch.Tensor | None = None
    records: list[dict[str, Any] | None] = [None] * len(windows)
    for batch in loader:
        indices = batch.pop("window_index").tolist()
        raw_batch = {
            "observation.state": batch.pop("state"),
            "task": batch.pop("task"),
            **batch,
        }
        processed = preprocessor(raw_batch)
        amp = (
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and bool(getattr(policy.config, "use_amp", False))
            else nullcontext()
        )
        with amp:
            batch_tokens, batch_mask = encoder.tokens(processed)
        batch_tokens = batch_tokens.to(dtype=torch.float16, device="cpu")
        batch_mask = batch_mask.to(device="cpu")
        if condition_tokens is None:
            condition_tokens = torch.empty(
                (len(windows), *batch_tokens.shape[1:]), dtype=torch.float16
            )
            condition_masks = torch.empty(
                (len(windows), batch_mask.shape[1]), dtype=torch.bool
            )
        assert condition_masks is not None
        condition_tokens[indices] = batch_tokens
        condition_masks[indices] = batch_mask
        for local_index, window_index in enumerate(indices):
            row = windows[window_index]
            records[window_index] = {
                "episode_id": int(row["episode_index"]),
                "window_id": int(row["window_index"]),
                "anchor_timestamp": float(row["start_timestamp"]),
                "image_timestamps": {
                    field: float(row["images"][field]["stamp"]) for field in configured_order
                },
                "feature_shape": list(batch_tokens[local_index].shape),
                "condition_valid_tokens": int(batch_mask[local_index].sum()),
                "task_text": str(row["task"]),
            }
    if condition_tokens is None or condition_masks is None:
        raise RuntimeError("SmolVLA token cache produced no batches")
    report = {
        "windows": len(windows),
        "condition_token_shape": list(condition_tokens.shape),
        "condition_mask_shape": list(condition_masks.shape),
        "cache_dtype": str(condition_tokens.dtype),
        "device": str(device),
        "cache": str(cache_path),
        "encoder": "smolvla_condition_tokens",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "feature_tap": TOKEN_FEATURE_TAP,
        "mask_tap": MASK_TAP,
        "pooling": "none",
        "camera_order_raw": configured_order,
        "camera_order_after_preprocessor": renamed_order,
        "image_preprocessing": {
            "source": str(checkpoint / "policy_preprocessor.json"),
            "policy_resize_with_padding": list(policy.config.resize_imgs_with_padding),
            "policy_pixel_transform": "[0,1] -> resize_with_pad -> [-1,1]",
            "missing_camera_rule": (
                f"checkpoint empty_cameras={policy.config.empty_cameras}; no missing cameras accepted"
            ),
        },
        "language": {"included": True, "task_format": "checkpoint-native preprocessor"},
        "robot_state": {"included": True, "format": "checkpoint-normalized 8D observation.state"},
        "cache_signature": signature,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    cache = {
        "condition_tokens": condition_tokens,
        "condition_mask": condition_masks,
        "records": records,
        "report": report,
        "encoder": "smolvla_condition_tokens",
        "condition_dim": int(condition_tokens.shape[-1]),
        "feature_contains_language": True,
        "feature_contains_robot_state": True,
        "windows_sha256": windows_hash,
        "checkpoint_sha256": checkpoint_hash,
        "cache_signature": signature,
    }
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    write_json(output / "condition_tokens.json", report)
    return report


def _cache_smolvla_features(config: dict[str, Any]) -> dict[str, Any]:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dataset = Path(config["data"]["robot_dataset"]).expanduser().resolve()
    output = Path(config["data"]["output_dir"]).expanduser().resolve()
    windows_path = output / "windows.jsonl"
    windows = read_jsonl(windows_path)
    if not windows:
        raise ValueError("windows.jsonl is empty")
    checkpoint = _resolve_checkpoint(str(config["vision"]["checkpoint"]))
    checkpoint_hash = checkpoint_fingerprint(checkpoint)
    windows_hash = hashlib.sha256(windows_path.read_bytes()).hexdigest()
    camera_map = read_preprocessor_camera_map(checkpoint)
    configured_order = list(config["vision"].get("camera_order") or [])
    if not configured_order:
        raise ValueError("SmolVLA cache requires an explicit raw camera_order")
    if configured_order != list(windows[0]["images"]):
        raise ValueError(
            "Configured SmolVLA camera_order does not match the window camera order: "
            f"{configured_order} != {list(windows[0]['images'])}"
        )
    renamed_order = [camera_map.get(field) for field in configured_order]
    if any(value is None for value in renamed_order):
        raise ValueError(f"Checkpoint preprocessor does not map all cameras: {camera_map}")
    signature = _smolvla_cache_signature(windows_hash, checkpoint_hash, config, configured_order)
    cache_path = output / "condition_features.pt"
    if cache_path.is_file():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        if existing.get("cache_signature") == signature:
            report = dict(existing["report"])
            report["reused"] = True
            return report

    device = torch.device(str(config["training"].get("device", "cuda")))
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("SmolVLA was not completely frozen")
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    encoder = FrozenSmolVLAConditionEncoder(policy).to(device)
    loader = DataLoader(
        _SmolVLAWindowDataset(windows, dataset, configured_order),
        batch_size=int(config["vision"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["vision"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    features = torch.empty((len(windows), encoder.embedding_dim), dtype=torch.float32)
    records: list[dict[str, Any] | None] = [None] * len(windows)
    raw_shapes: set[tuple[int, ...]] = set()
    for batch in loader:
        indices = batch.pop("window_index").tolist()
        raw_batch = {
            "observation.state": batch.pop("state"),
            "task": batch.pop("task"),
            **batch,
        }
        processed = preprocessor(raw_batch)
        amp = (
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and bool(getattr(policy.config, "use_amp", False))
            else nullcontext()
        )
        with amp:
            pooled, mask, past_key_values = encoder(processed)
        pooled = pooled.float().cpu()
        first_layer = past_key_values[min(past_key_values)]
        raw_shapes.add(tuple(first_layer["key_states"].shape[1:]))
        for local_index, window_index in enumerate(indices):
            features[window_index] = pooled[local_index]
            row = windows[window_index]
            records[window_index] = {
                "episode_id": int(row["episode_index"]),
                "window_id": int(row["window_index"]),
                "anchor_timestamp": float(row["start_timestamp"]),
                "image_timestamps": {
                    field: float(row["images"][field]["stamp"]) for field in configured_order
                },
                "feature_shape": [encoder.embedding_dim],
                "task_text": str(row["task"]),
            }
    report = {
        "windows": len(windows),
        "feature_shape": list(features.shape),
        "raw_condition_token_shapes": [list(value) for value in sorted(raw_shapes)],
        "device": str(device),
        "cache": str(cache_path),
        "encoder": "smolvla_condition",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "feature_tap": FEATURE_TAP,
        "mask_tap": MASK_TAP,
        "pooling": "per_layer_key_value_masked_token_mean_then_head_mean_then_concat",
        "camera_order_raw": configured_order,
        "camera_order_after_preprocessor": renamed_order,
        "image_preprocessing": {
            "source": str(checkpoint / "policy_preprocessor.json"),
            "policy_resize_with_padding": list(policy.config.resize_imgs_with_padding),
            "policy_pixel_transform": "[0,1] -> resize_with_pad -> [-1,1]",
            "missing_camera_rule": f"checkpoint empty_cameras={policy.config.empty_cameras}; no missing cameras accepted",
        },
        "language": {
            "included": True,
            "task_format": "checkpoint processor adds trailing newline and tokenizes to max_length=48",
        },
        "robot_state": {
            "included": True,
            "format": "8D observation.state normalized by checkpoint policy preprocessor",
        },
        "cache_signature": signature,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    cache = {
        "features": features,
        "smolvla_feature": features,
        "records": records,
        "report": report,
        "encoder": "smolvla_condition",
        "feature_dim": encoder.embedding_dim,
        "feature_contains_language": True,
        "feature_contains_robot_state": True,
        "windows_sha256": windows_hash,
        "checkpoint_sha256": checkpoint_hash,
        "cache_signature": signature,
    }
    temporary = cache_path.with_suffix(".pt.tmp")
    torch.save(cache, temporary)
    temporary.replace(cache_path)
    write_json(output / "condition_features.json", report)
    return report


def cache_visual_features(config: dict[str, Any]) -> dict[str, Any]:
    encoder = str(config["vision"]["encoder"])
    if encoder == "pi05_jax_z_rl":
        if str(config["model"].get("model_type")) != "transformer_envelope":
            raise ValueError("pi05_jax_z_rl is only supported by transformer_envelope")
        return _cache_pi05_jax_z_rl(config)
    if encoder == "smolvla_condition":
        if str(config["model"].get("model_type", "mlp_baseline")) == "transformer_envelope":
            return _cache_smolvla_tokens(config)
        return _cache_smolvla_features(config)
    if encoder == "resnet18":
        return _cache_resnet_features(config)
    raise ValueError(f"Unsupported feature encoder {encoder!r}; no automatic fallback is allowed")
