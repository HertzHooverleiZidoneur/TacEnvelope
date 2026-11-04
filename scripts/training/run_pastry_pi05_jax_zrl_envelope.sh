#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-$ROOT/configs/experiments/pi05_jax_zrl_pastry_A298.yaml}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/tactile_model/pastry_model/24999}"
export DATASET_ROOT="${DATASET_ROOT:-/path/to/VLA_RL_dataset/franka_pastry_tactile_A298_20260820}"
export TASK="${TASK:-Pick up the pastry and place it into the blue box.}"
export JAX_POLICY_PORT="${JAX_POLICY_PORT:-8041}"

case "${1:-all}" in
  build-sources)
    exec "$ROOT/scripts/training/build_pastry_A298_sources.py"
    ;;
  start-server)
    exec /path/to/lerobot/examples/franka/run_franka_pi05_jax_server.sh
    ;;
  all)
    for phase in audit build cache-tactile-history cache-features train stable-evaluate; do
      "$ROOT/scripts/training/run_pi05_jax_zrl_envelope.sh" "$phase"
    done
    ;;
  audit|build|cache-features|cache-tactile-history|train|stable-evaluate)
    exec "$ROOT/scripts/training/run_pi05_jax_zrl_envelope.sh" "$1"
    ;;
  *)
    echo "usage: $0 [build-sources|start-server|audit|build|cache-tactile-history|cache-features|train|stable-evaluate|all]" >&2
    exit 2
    ;;
esac
