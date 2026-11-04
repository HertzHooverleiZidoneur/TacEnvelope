#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-/path/to/lerobot}"
CLIENT_DIR="${CLIENT_DIR:-${LEROBOT_DIR}/examples/franka}"
PYTHON_BIN="${PYTHON_BIN:-${LEROBOT_DIR}/.venv/bin/python}"

export PYTHONPATH="${LEROBOT_DIR}/src:${CLIENT_DIR}:${ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-franka-envelope}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "${ROOT}/scripts/runtime/live_franka_envelope_test.py" "$@"
