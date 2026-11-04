#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLT_REPO="${RLT_REPO:-/path/to/rlt_openpi_franka}"
PYTHON_BIN="${PYTHON_BIN:-/path/to/RL_Token_update/.venv/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/experiments/pi05_jax_zrl_stable_contact.yaml}"
JAX_LAUNCHER="${JAX_LAUNCHER:-/path/to/lerobot/examples/franka/run_franka_pi05_jax_online_rl.sh}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/tactile_model/24999}"
export DATASET_ROOT="${DATASET_ROOT:-/path/to/VLA_RL_dataset/franka_usb_tactile_310_20260716/vla_dataset}"

export PYTHONPATH="$ROOT/src:$RLT_REPO/third_party/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-all}" in
  start-server)
    exec "$JAX_LAUNCHER" --server-only
    ;;
  audit|build|cache-features|cache-tactile-history|train|stable-evaluate)
    exec "$PYTHON_BIN" -m tactile_force_envelope.cli "$1" --config "$CONFIG"
    ;;
  all)
    "$JAX_LAUNCHER" --server-only
    "$PYTHON_BIN" -m tactile_force_envelope.cli audit --config "$CONFIG"
    "$PYTHON_BIN" -m tactile_force_envelope.cli build --config "$CONFIG"
    "$PYTHON_BIN" -m tactile_force_envelope.cli cache-features --config "$CONFIG"
    "$PYTHON_BIN" -m tactile_force_envelope.cli cache-tactile-history --config "$CONFIG"
    "$PYTHON_BIN" -m tactile_force_envelope.cli train --config "$CONFIG"
    exec "$PYTHON_BIN" -m tactile_force_envelope.cli stable-evaluate --config "$CONFIG"
    ;;
  *)
    echo "usage: $0 [start-server|audit|build|cache-features|cache-tactile-history|train|stable-evaluate|all]" >&2
    exit 2
    ;;
esac
