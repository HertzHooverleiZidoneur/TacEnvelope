# Franka USB tactile collection (310 episodes)

This is a self-contained collection assembled from four recording sessions. `vla_dataset/` is a
single merged LeRobot v3 dataset with globally indexed episodes 0--309.

The top-level data directories are:

- `vla_dataset`: merged LeRobot v3 data, including all 30 Hz camera videos, state, action, tactile
  image videos, and merged tactile columns.
- `source_frames`: frame metadata plus the original JPGs referenced by accepted 300 ms force
  windows. These permit rebuilding the frozen visual-feature cache.
- `tactile_raw`: complete original workstation Xense sidecars at approximately 22 Hz. These are
  the authoritative force-label source.
- `audits`: compact per-batch audit JSON and summary files.

`force_envelope/windows.jsonl` is the combined 14,296-window index. Every row records both a
global `episode_index` and its original `source_episode_index`, `batch_id`, `robot_dataset`, and
`tactile_root`. Splits in `force_envelope/episode_splits.json` are recomputed globally by episode.
The cached `visual_features.pt` already covers all 14,296 windows and three environment cameras.

The merged nearest-sample tactile columns in existing v3 files contain a small number of future
or stale values from the old merge implementation. Use the original sidecars for force labels;
the force-envelope windows already do this. New captures use the corrected invalid-sample logic.

Full audit conclusions are copied to `DATA_AUDIT.md`.

To cache visual features and train the combined force model:

```bash
cd ~/Documents/gello_software
CONFIG=configs/experiments/recent_310_collection.yaml \
  ./run.sh cache-vision
CONFIG=configs/experiments/recent_310_collection.yaml \
  ./run.sh train
```

The combined windows have already been built. Do not run `audit` or `build` with this collection
configuration; the compact source audit results are retained under `audits/`.
