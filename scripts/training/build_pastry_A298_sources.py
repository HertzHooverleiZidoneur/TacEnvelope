#!/usr/bin/env python3
"""Build the timestamp-preserving source index for the pastry A298 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


TASK = "Pick up the pastry and place it into the blue box."


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_link(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        raise RuntimeError(f"Existing symlink has a different target: {path}")
    if path.exists():
        raise RuntimeError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home()
        / "Documents/VLA_RL_dataset/franka_pastry_tactile_A298_20260820",
    )
    parser.add_argument(
        "--tactile-root",
        type=Path,
        default=Path.home() / "Documents/VLA_RL_dataset/franka_remote_tactile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "Documents/VLA_RL_dataset/franka_pastry_tactile_A298_sources",
    )
    parser.add_argument("--remote", default="user@robot-host")
    parser.add_argument(
        "--remote-datasets-root",
        default="/home/user/Documents/gello_software/ros2/datasets",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    tactile_root = args.tactile_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    provenance = dataset / "provenance/episodes.csv"
    rows = list(csv.DictReader(provenance.open(encoding="utf-8")))
    if len(rows) != 298:
        raise RuntimeError(f"Expected 298 selected episodes, found {len(rows)}")
    if [int(row["new_episode"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("Selected episode indices are not contiguous from zero")

    relative_frames: list[str] = []
    for row in rows:
        source = Path(row["source_path"])
        try:
            relative = source.relative_to(args.remote_datasets_root)
        except ValueError as error:
            raise RuntimeError(f"Source is outside remote dataset root: {source}") from error
        relative_frames.append(str(relative / "frames.jsonl"))

    raw_frames = output / "raw_frames"
    raw_frames.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--files-from=-",
            "-e",
            "ssh",
            f"{args.remote}:{args.remote_datasets_root.rstrip('/')}/",
            f"{raw_frames}/",
        ],
        input="\n".join(relative_frames) + "\n",
        text=True,
        check=True,
    )

    source_data = output / "source_frames/data"
    remapped_tactile = output / "tactile_raw"
    source_data.mkdir(parents=True, exist_ok=True)
    remapped_tactile.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, str | int]] = []

    for row in rows:
        new_episode = int(row["new_episode"])
        source_episode = int(row["source_episode"])
        batch = row["batch"]
        raw_frame_dir = raw_frames / batch / "data" / f"episode_{source_episode:06d}"
        frame_file = raw_frame_dir / "frames.jsonl"
        if not frame_file.is_file():
            raise FileNotFoundError(frame_file)
        ensure_link(source_data / f"episode_{new_episode:06d}", raw_frame_dir)

        sidecars = sorted((tactile_root / batch).glob(f"episode_{source_episode:06d}_*"))
        if len(sidecars) != 1:
            raise RuntimeError(
                f"Expected one tactile sidecar for {batch} episode {source_episode}, "
                f"found {len(sidecars)}"
            )
        original = sidecars[0]
        manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
        session_id = str(manifest["session_id"])
        remapped = remapped_tactile / f"episode_{new_episode:06d}_{session_id}"
        remapped.mkdir(parents=True, exist_ok=True)
        for sensor in manifest.get("sensors", []):
            sensor_name = str(sensor["name"])
            ensure_link(remapped / sensor_name, original / sensor_name)
        manifest.update(
            {
                "episode_index": new_episode,
                "dataset_name": "franka_pastry_tactile_A298_sources",
                "task": TASK,
                "source_batch": batch,
                "source_episode_index": source_episode,
                "source_sidecar": str(original),
            }
        )
        atomic_json(remapped / "manifest.json", manifest)
        mapping_rows.append(
            {
                "episode_index": new_episode,
                "source_batch": batch,
                "source_episode_index": source_episode,
                "frames_jsonl": str(frame_file),
                "tactile_sidecar": str(original),
            }
        )

    mapping_path = output / "episodes.csv"
    temporary = mapping_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    os.replace(temporary, mapping_path)
    atomic_json(
        output / "manifest.json",
        {
            "format": "franka_pastry_tactile_A298_envelope_sources_v1",
            "task": TASK,
            "episodes": len(mapping_rows),
            "robot_source": str(output / "source_frames"),
            "tactile_source": str(remapped_tactile),
            "local_v3_dataset": str(dataset),
            "selection": "A_bilateral_direct only",
        },
    )
    print(f"Built {len(mapping_rows)} timestamp-preserving episode mappings at {output}")


if __name__ == "__main__":
    main()
