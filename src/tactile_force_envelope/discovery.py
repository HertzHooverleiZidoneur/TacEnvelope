from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import iter_jsonl


@dataclass
class DatasetSchema:
    timestamp_field: str
    state_field: str
    action_field: str
    image_fields: list[str]
    image_stamp_fields: dict[str, str | None]
    state_dim: int
    action_dim: int
    action_chunk_start_field: str | None
    task_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_paths(dataset: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted((dataset / "data").glob("episode_*/frames.jsonl")):
        match = re.search(r"episode_(\d+)", str(path.parent.name))
        if match:
            result[int(match.group(1))] = path
    return result


def _vector_field(row: dict[str, Any], preferred: str | None, candidates: list[str]) -> str:
    if preferred:
        if not isinstance(row.get(preferred), list):
            raise ValueError(f"Configured vector field is absent or not a list: {preferred}")
        return preferred
    for key in candidates:
        if isinstance(row.get(key), list) and row[key]:
            return key
    raise ValueError(f"None of the vector fields exist: {candidates}")


def discover_schema(dataset: Path, config: dict[str, Any]) -> DatasetSchema:
    paths = episode_paths(dataset)
    if not paths:
        raise FileNotFoundError(f"No data/episode_*/frames.jsonl under {dataset}")
    row = next(iter_jsonl(next(iter(paths.values()))))
    data_cfg = config["data"]
    timestamp = data_cfg.get("timestamp_field") or "timestamp"
    if timestamp not in row:
        raise ValueError(f"Timestamp field not found: {timestamp}")
    state = _vector_field(
        row, data_cfg.get("state_field"), ["observation.state", "robot.state", "robot.q"]
    )
    action = _vector_field(row, data_cfg.get("action_field"), ["action", "executed_action"])
    configured_images = data_cfg.get("image_fields")
    if configured_images:
        images = list(configured_images)
        missing = [key for key in images if key not in row]
        if missing:
            raise ValueError(f"Configured image fields are absent: {missing}")
    else:
        images = sorted(
            key for key in row
            if key.startswith("observation.images.") and not key.endswith(".stamp")
        )
    if not images:
        raise ValueError("No raw image fields were discovered")
    stamp_fields = {key: f"{key}.stamp" if f"{key}.stamp" in row else None for key in images}
    chunk_candidates = [
        "action_chunk_start", "action_chunk_id", "execution_chunk_id", "chunk_id"
    ]
    chunk_field = next((key for key in chunk_candidates if key in row), None)
    task_field = str(data_cfg.get("task_field") or "task")
    episodes_meta = dataset / "meta" / "episodes.jsonl"
    task_source = task_field if task_field in row else (
        "meta/episodes.jsonl" if episodes_meta.exists() else "missing"
    )
    return DatasetSchema(
        timestamp_field=timestamp,
        state_field=state,
        action_field=action,
        image_fields=images,
        image_stamp_fields=stamp_fields,
        state_dim=len(row[state]),
        action_dim=len(row[action]),
        action_chunk_start_field=chunk_field,
        task_source=task_source,
    )


def episode_tasks(dataset: Path, task_field: str) -> dict[int, str]:
    result: dict[int, str] = {}
    path = dataset / "meta" / "episodes.jsonl"
    if path.exists():
        for row in iter_jsonl(path):
            if "episode_index" in row and task_field in row:
                result[int(row["episode_index"])] = str(row[task_field])
            elif "episode_index" in row and "task" in row:
                result[int(row["episode_index"])] = str(row["task"])
    return result


def discover_tactile_sidecars(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for manifest_path in sorted(root.glob("episode_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "episode_index" in manifest and not bool(manifest.get("active", False)):
            result[int(manifest["episode_index"])] = manifest_path.parent
    return result

