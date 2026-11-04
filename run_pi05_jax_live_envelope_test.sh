#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-/path/to/lerobot}"
RLT_REPO="${RLT_REPO:-/path/to/rlt_openpi_franka}"
OPENPI_REPO="${OPENPI_REPO:-${RLT_REPO}/third_party/openpi}"
CLIENT_DIR="${LEROBOT_DIR}/examples/franka"
PYTHON_BIN="${PYTHON_BIN:-/path/to/RL_Token_update/.venv/bin/python}"
JAX_CHECKPOINT="${JAX_CHECKPOINT:-/path/to/tactile_model/24999}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/VLA_RL_dataset/franka_usb_tactile_310_20260716/vla_dataset}"
JAX_HOST="${JAX_POLICY_HOST:-127.0.0.1}"
JAX_PORT="${JAX_POLICY_PORT:-8011}"
STATE_DIR="${HOME}/.cache/franka_pi05_jax_tactile_envelope"
PID_FILE="${STATE_DIR}/server.pid"
CHECKPOINT_FILE="${STATE_DIR}/checkpoint.txt"
LOG_FILE="${STATE_DIR}/server.log"

mkdir -p "${STATE_DIR}"

server_ready() {
  timeout 1 bash -c "</dev/tcp/${JAX_HOST}/${JAX_PORT}" >/dev/null 2>&1
}

server_matches_checkpoint() {
  local pid cmdline
  while read -r pid; do
    [ -r "/proc/${pid}/cmdline" ] || continue
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    if [[ "${cmdline}" == *"serve_rlt_policy.py"* ]] \
      && [[ "${cmdline}" == *"--port ${JAX_PORT}"* ]] \
      && [[ "${cmdline}" == *"--checkpoint-dir ${JAX_CHECKPOINT}"* ]]; then
      echo "${pid}" >"${PID_FILE}"
      echo "${JAX_CHECKPOINT}" >"${CHECKPOINT_FILE}"
      return 0
    fi
  done < <(pgrep -f "serve_rlt_policy.py" || true)
  return 1
}

mode="${1:-}"
case "${1:-}" in
  --stop-jax-server)
    if [ -s "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill -INT "$(cat "${PID_FILE}")"
    fi
    rm -f "${PID_FILE}" "${CHECKPOINT_FILE}"
    echo "[pi05_envelope] dedicated JAX server stopped"
    exit 0
    ;;
  --jax-server-status)
    if server_ready; then
      echo "[pi05_envelope] JAX server ready ws://${JAX_HOST}:${JAX_PORT}"
      cat "${CHECKPOINT_FILE}" 2>/dev/null || true
      exit 0
    fi
    echo "[pi05_envelope] JAX server is not running"
    exit 1
    ;;
esac

export PYTHONPATH="${RLT_REPO}/src:${OPENPI_REPO}/src:${OPENPI_REPO}/packages/openpi-client/src:${LEROBOT_DIR}/src:${CLIENT_DIR}:${ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-pi05-envelope}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

if [ "${mode}" = "--self-test-report" ]; then
  exec "${PYTHON_BIN}" "${ROOT}/scripts/runtime/pi05_jax_live_envelope_test.py" "$@"
fi

if server_ready; then
  if { [ ! -f "${CHECKPOINT_FILE}" ] || [ "$(cat "${CHECKPOINT_FILE}")" != "${JAX_CHECKPOINT}" ]; } \
    && ! server_matches_checkpoint; then
    echo "[pi05_envelope] port ${JAX_PORT} is occupied by an unverified JAX checkpoint" >&2
    echo "[pi05_envelope] use another JAX_POLICY_PORT or stop the process on this port" >&2
    exit 2
  fi
  echo "[pi05_envelope] reusing JAX checkpoint ${JAX_CHECKPOINT}"
else
  echo "[pi05_envelope] starting dedicated JAX server"
  echo "[pi05_envelope] checkpoint=${JAX_CHECKPOINT}"
  echo "[pi05_envelope] log=${LOG_FILE}"
  nohup setsid env \
    CHECKPOINT_DIR="${JAX_CHECKPOINT}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    JAX_POLICY_PORT="${JAX_PORT}" \
    JAX_POLICY_BIND_HOST="${JAX_HOST}" \
    "${CLIENT_DIR}/run_franka_pi05_jax_server.sh" \
    >"${LOG_FILE}" 2>&1 &
  server_pid=$!
  echo "${server_pid}" >"${PID_FILE}"
  echo "${JAX_CHECKPOINT}" >"${CHECKPOINT_FILE}"

  deadline=$((SECONDS + ${JAX_POLICY_START_TIMEOUT:-240}))
  until server_ready; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[pi05_envelope] JAX server exited during startup" >&2
      tail -100 "${LOG_FILE}" >&2
      exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "[pi05_envelope] timeout waiting for JAX server" >&2
      tail -100 "${LOG_FILE}" >&2
      exit 1
    fi
    sleep 1
  done
fi

if [ "${mode}" = "--start-jax-server" ]; then
  echo "[pi05_envelope] JAX server ready ws://${JAX_HOST}:${JAX_PORT}"
  echo "[pi05_envelope] checkpoint=${JAX_CHECKPOINT}"
  exit 0
fi

"${ROOT}/run_pi05_jax_envelope_sidecar.sh" start

exec "${PYTHON_BIN}" "${ROOT}/scripts/runtime/pi05_jax_live_envelope_test.py" \
  --jax-host "${JAX_HOST}" \
  --jax-port "${JAX_PORT}" \
  --jax-checkpoint "${JAX_CHECKPOINT}" \
  "$@"
