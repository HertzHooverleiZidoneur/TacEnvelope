#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-$ROOT/configs/experiments/pi05_jax_zrl_ink_beaker.yaml}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/tactile_model/ink_model/24999}"
export DATASET_ROOT="${DATASET_ROOT:-/path/to/VLA_RL_dataset/franka_ink_beaker_tactile_287_20260807}"
export TASK="${TASK:-Squeeze the ink into the beaker.}"
export JAX_POLICY_PORT="${JAX_POLICY_PORT:-8000}"

case "${1:-all}" in
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
    echo "usage: $0 [start-server|audit|build|cache-features|cache-tactile-history|train|stable-evaluate|all]" >&2
    exit 2
    ;;
esac
