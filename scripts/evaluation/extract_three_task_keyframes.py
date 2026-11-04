#!/usr/bin/env python3
"""Extract synchronized wrist/side keyframes for the three paper tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


WRIST_FIELD = "observation.images.camera_camera_1_color_image_raw"
SIDE_FIELD = "observation.images.d515_color_image_raw"
CAMERAS = (("wrist", "Wrist view", WRIST_FIELD), ("side", "Side view", SIDE_FIELD))


@dataclass(frozen=True)
class Stage:
    slug: str
    label: str
    time_s: float


@dataclass(frozen=True)
class Task:
    slug: str
    title: str
    dataset_root: Path
    episode: int
    stages: tuple[Stage, ...]


TASKS = (
    Task(
        slug="usb_insertion",
        title="USB insertion",
        dataset_root=Path(
            "/path/to/VLA_RL_dataset/"
            "franka_usb_tactile_310_20260716/vla_dataset"
        ),
        episode=15,
        stages=(
            Stage("approach", "Approach", 2.70),
            Stage("grasp", "Grasp", 5.43),
            Stage("align", "Align", 10.90),
            Stage("insert", "Insert", 12.27),
        ),
    ),
    Task(
        slug="ink_squeezing",
        title="Ink squeezing",
        dataset_root=Path(
            "/path/to/VLA_RL_dataset/"
            "franka_ink_beaker_tactile_287_20260807"
        ),
        episode=180,
        stages=(
            Stage("approach", "Approach", 5.13),
            Stage("grasp", "Grasp dropper", 10.30),
            Stage("squeeze", "Squeeze", 20.63),
            Stage("release", "Release", 28.30),
        ),
    ),
    Task(
        slug="pastry_transfer",
        title="Pastry transfer",
        dataset_root=Path(
            "/path/to/VLA_RL_dataset/"
            "franka_pastry_tactile_A298_20260820"
        ),
        episode=294,
        stages=(
            Stage("approach", "Approach", 7.17),
            Stage("grasp", "Grasp", 9.57),
            Stage("transfer", "Transfer", 11.97),
            Stage("place", "Place", 13.10),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/path/to/tactile_force_envelope/outputs/paper/"
            "task_keyframes_20260908"
        ),
    )
    return parser.parse_args()


def find_episode_row(dataset_root: Path, episode: int) -> dict:
    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata parquet under {dataset_root}")

    for path in episode_files:
        table = pq.read_table(path)
        episode_indices = table.column("episode_index").to_numpy()
        matches = np.flatnonzero(episode_indices == episode)
        if matches.size:
            return table.slice(int(matches[0]), 1).to_pylist()[0]
    raise KeyError(f"Episode {episode} not found in {dataset_root}")


def video_path(dataset_root: Path, row: dict, field: str) -> Path:
    prefix = f"videos/{field}"
    chunk_index = int(row[f"{prefix}/chunk_index"])
    file_index = int(row[f"{prefix}/file_index"])
    path = dataset_root / prefix / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_nearest_frame(path: Path, target_s: float) -> tuple[np.ndarray, float]:
    """Decode the frame nearest an absolute timestamp in a LeRobot video file."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        seek_s = max(0.0, target_s - 1.0)
        container.seek(
            int(seek_s / time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )

        best_frame = None
        best_time = None
        best_distance = float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * frame.time_base)
            distance = abs(frame_time - target_s)
            if distance < best_distance:
                best_frame = frame.to_ndarray(format="rgb24")
                best_time = frame_time
                best_distance = distance
            if best_frame is not None and frame_time > target_s + 0.12:
                break

    if best_frame is None or best_time is None:
        raise RuntimeError(f"Could not decode a frame near {target_s:.3f}s from {path}")
    if best_distance > 0.10:
        raise RuntimeError(
            f"Nearest frame in {path} is {best_distance:.3f}s from target {target_s:.3f}s"
        )
    return best_frame, best_time


def save_task_figure(task: Task, frames: dict, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(14.2, 5.65))
    for row_index, (camera_slug, camera_label, _) in enumerate(CAMERAS):
        for column_index, stage in enumerate(task.stages):
            ax = axes[row_index, column_index]
            ax.imshow(frames[(camera_slug, stage.slug)])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#808080")
            if row_index == 0:
                ax.set_title(
                    f"({column_index + 1}) {stage.label}",
                    fontsize=13,
                    fontweight="bold",
                    pad=8,
                )
            if column_index == 0:
                ax.set_ylabel(camera_label, fontsize=13, fontweight="bold", labelpad=9)

    fig.suptitle(task.title, fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.025, top=0.89, wspace=0.025, hspace=0.075)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"{task.slug}_wrist_side_flow.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def save_overview_figure(frames_by_task: dict, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    fig, axes = plt.subplots(6, 4, figsize=(14.2, 16.3))

    for task_index, task in enumerate(TASKS):
        task_frames = frames_by_task[task.slug]
        for camera_index, (camera_slug, camera_label, _) in enumerate(CAMERAS):
            row_index = task_index * 2 + camera_index
            for column_index, stage in enumerate(task.stages):
                ax = axes[row_index, column_index]
                ax.imshow(task_frames[(camera_slug, stage.slug)])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.7)
                    spine.set_color("#808080")
                if camera_index == 0:
                    ax.set_title(stage.label, fontsize=12, fontweight="bold", pad=7)
                if column_index == 0:
                    ax.set_ylabel(
                        f"{task.title}\n{camera_label}",
                        fontsize=11.5,
                        fontweight="bold",
                        labelpad=10,
                    )

    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.015, top=0.985, wspace=0.025, hspace=0.16)
    for extension in ("png", "pdf"):
        fig.savefig(
            figure_dir / f"three_tasks_wrist_side_keyframes.{extension}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_mapping": {
            "wrist": WRIST_FIELD,
            "side": SIDE_FIELD,
        },
        "tasks": [],
    }
    frames_by_task = {}

    for task in TASKS:
        row = find_episode_row(task.dataset_root, task.episode)
        task_dir = raw_dir / task.slug
        task_dir.mkdir(parents=True, exist_ok=True)
        task_frames = {}
        task_record = {
            "task": task.slug,
            "title": task.title,
            "dataset_root": str(task.dataset_root),
            "episode": task.episode,
            "dataset_task_text": [str(value) for value in row["tasks"]],
            "stages": [],
        }

        for stage_index, stage in enumerate(task.stages, start=1):
            stage_record = {
                "index": stage_index,
                "slug": stage.slug,
                "label": stage.label,
                "episode_time_s": stage.time_s,
                "cameras": {},
            }
            for camera_slug, camera_label, field in CAMERAS:
                source_path = video_path(task.dataset_root, row, field)
                episode_start_s = float(row[f"videos/{field}/from_timestamp"])
                requested_video_time_s = episode_start_s + stage.time_s
                frame, decoded_video_time_s = extract_nearest_frame(
                    source_path, requested_video_time_s
                )
                filename = f"{stage_index:02d}_{stage.slug}_{camera_slug}.png"
                destination = task_dir / filename
                Image.fromarray(frame).save(destination, compress_level=3)
                task_frames[(camera_slug, stage.slug)] = frame
                stage_record["cameras"][camera_slug] = {
                    "label": camera_label,
                    "field": field,
                    "source_video": str(source_path),
                    "requested_video_time_s": requested_video_time_s,
                    "decoded_video_time_s": decoded_video_time_s,
                    "decoded_episode_time_s": decoded_video_time_s - episode_start_s,
                    "time_error_ms": 1000.0 * (decoded_video_time_s - requested_video_time_s),
                    "output": str(destination.relative_to(output_dir)),
                    "shape": list(frame.shape),
                }
            task_record["stages"].append(stage_record)

        frames_by_task[task.slug] = task_frames
        manifest["tasks"].append(task_record)
        save_task_figure(task, task_frames, output_dir)

    save_overview_figure(frames_by_task, output_dir)
    manifest_path = output_dir / "keyframes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote paper keyframes to {output_dir}")


if __name__ == "__main__":
    main()
