#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XENSE_PYTHON="${XENSE_PYTHON:-/home/user/.venvs/xense/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/path/to/RL_Token_update/.venv/bin/python}"
XENSE_SCRIPT="${ROOT}/integrations/xense/pc_xense_capture.py"
SIDECAR_SCRIPT="${ROOT}/scripts/runtime/pi05_jax_envelope_sidecar.py"
XENSE_LEFT_SERIAL="${XENSE_LEFT_SERIAL:-OG000836}"
XENSE_RIGHT_SERIAL="${XENSE_RIGHT_SERIAL:-OG000837}"
XENSE_SERIALS="${XENSE_LEFT_SERIAL},${XENSE_RIGHT_SERIAL}"
XENSE_NAMES="left_tip,right_tip"
CHECKPOINT="${TACTILE_ENVELOPE_CHECKPOINT:-${ROOT}/checkpoints/production/pi05_jax_zrl_stable_contact.pt}"
BILATERAL_CALIBRATION="${TACTILE_BILATERAL_CALIBRATION:-${ROOT}/checkpoints/production/pi05_jax_zrl_bilateral_sigma_calibration.json}"
HOST="${TACTILE_SIDECAR_HOST:-127.0.0.1}"
PORT="${TACTILE_SIDECAR_PORT:-8022}"
STATE_DIR="${TACTILE_SIDECAR_STATE_DIR:-${HOME}/.cache/franka_tactile_envelope_dualrate}"
PID_FILE="${STATE_DIR}/sidecar.pid"
LOG_FILE="${STATE_DIR}/sidecar.log"
CHUNK_LOG="${STATE_DIR}/chunks.jsonl"
COMMAND="${1:-start}"

mkdir -p "${STATE_DIR}"

is_running() {
  [ -s "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

fresh_samples_ok() {
  local end_ns start_ns response
  end_ns="$("${XENSE_PYTHON}" -c 'import time; print(time.time_ns())')"
  start_ns=$((end_ns - 1000000000))
  response="$("${XENSE_PYTHON}" "${XENSE_SCRIPT}" recent \
    --request-json "{\"start_ns\":${start_ns},\"end_ns\":${end_ns}}" 2>/dev/null)" || return 1
  printf '%s' "${response}" | "${XENSE_PYTHON}" -c '
import json
import sys

payload = json.load(sys.stdin)
sensors = payload.get("sensors", {})
for name in ("left_tip", "right_tip"):
    valid = 0
    for row in sensors.get(name, []):
        feature_names = row.get("feature_names", [])
        values = row.get("values", [])
        sample = dict(zip(feature_names, values))
        if sample.get("sample_valid", 0.0) > 0.5 and sample.get("force_valid", 0.0) > 0.5:
            valid += 1
    if valid < 4:
        raise SystemExit(1)
'
}

probe() {
  "${MODEL_PYTHON}" - "${HOST}" "${PORT}" <<'PY'
import json
import os
import socket
import struct
import sys

host, port = sys.argv[1], int(sys.argv[2])
payload = json.dumps({"type": "metadata", "schema_version": 2}).encode()
with socket.create_connection((host, port), timeout=2.0) as sock:
    sock.sendall(struct.pack("!I", len(payload)) + payload)
    header = sock.recv(4)
    if len(header) != 4:
        raise RuntimeError("short metadata header")
    size = struct.unpack("!I", header)[0]
    response = b""
    while len(response) < size:
        response += sock.recv(size - len(response))
metadata = json.loads(response)
reward = metadata.get("reward_envelope") or {}
stable = metadata.get("stable_contact") or {}
supported = metadata.get("supported_reward_criteria", ["adaptive_envelope"])
required_criterion = os.environ.get(
    "TACTILE_REWARD_CRITERION", "adaptive_envelope"
)
print(json.dumps({
    "ok": metadata.get("ok"),
    "feature": metadata.get("policy_feature_source"),
    "checkpoint": metadata.get("checkpoint"),
    "reward_type": reward.get("type"),
    "lower_coverage": reward.get("lower_coverage"),
    "lower_alpha_label": reward.get("lower_alpha_label"),
    "lower_alpha": reward.get("lower_alpha"),
    "upper_coverage": reward.get("upper_coverage"),
    "upper_alpha_label": reward.get("upper_alpha_label"),
    "upper_alpha": reward.get("upper_alpha"),
    "cost_power": reward.get("cost_power"),
    "cost_clip": reward.get("cost_clip"),
    "tau_on_N": stable.get("tau_on"),
    "tau_off_N": stable.get("tau_off"),
    "raw_peak_threshold_N": (metadata.get("raw_peak_penalty") or {}).get("threshold_N"),
    "production_ready": metadata.get("production_ready"),
    "supported_reward_criteria": supported,
    "required_reward_criterion": required_criterion,
}))
if (
    not metadata.get("ok")
    or metadata.get("policy_feature_source") != "pi05_jax_z_rl"
    or reward.get("type") != "bilateral_average_uncertainty_scaled_band"
    or reward.get("calibration_method") != "standardized_residual_quantiles"
    or reward.get("uncertainty_scale_method") != "mean"
    or reward.get("scale_definition") != "u_mean = 0.5 * (sigma_left + sigma_right)"
    or reward.get("scale_is_standard_deviation") is not False
    or reward.get("lower_coverage") != 0.05
    or reward.get("lower_alpha_label") != "alpha05"
    or reward.get("lower_alpha") is None
    or reward.get("upper_coverage") != 0.95
    or reward.get("upper_alpha_label") != "alpha95"
    or reward.get("upper_alpha") is None
    or reward.get("cost_power") != 2
    or reward.get("cost_clip") != 1.0
    or (metadata.get("raw_peak_penalty") or {}).get("threshold_N") is None
    or required_criterion not in supported
):
    raise SystemExit(1)
PY
}

case "${COMMAND}" in
  status)
    probe
    exit 0
    ;;
  stop)
    if is_running; then
      kill "$(cat "${PID_FILE}")"
      for _ in $(seq 1 50); do
        is_running || break
        sleep 0.1
      done
    fi
    rm -f "${PID_FILE}"
    echo "[envelope_sidecar] stopped"
    exit 0
    ;;
  restart)
    "${BASH_SOURCE[0]}" stop
    ;;
  start)
    ;;
  *)
    echo "usage: $0 [start|status|stop|restart]" >&2
    exit 2
    ;;
esac

SIDECAR_COMPATIBLE=0
if probe >/dev/null 2>&1; then
  SIDECAR_COMPATIBLE=1
fi

# An older daemon may not support recent samples, and a stale daemon can remain
# alive after USB disconnect while returning empty sample arrays.
MAPPING_OK=0
XENSE_RESTARTED=0
if STATUS_JSON="$("${XENSE_PYTHON}" "${XENSE_SCRIPT}" status 2>/dev/null)"; then
  if printf '%s' "${STATUS_JSON}" | "${XENSE_PYTHON}" -c \
    'import json,sys; x=json.load(sys.stdin); m={s["name"]:s["serial"] for s in x["sensors"]}; raise SystemExit(0 if m=={"left_tip":sys.argv[1],"right_tip":sys.argv[2]} else 1)' \
    "${XENSE_LEFT_SERIAL}" "${XENSE_RIGHT_SERIAL}"; then
    MAPPING_OK=1
  fi
fi
if [ "${MAPPING_OK}" != "1" ] || ! fresh_samples_ok; then
  echo "[envelope_sidecar] restarting stale or mismatched Xense daemon"
  "${XENSE_PYTHON}" "${XENSE_SCRIPT}" shutdown >/dev/null 2>&1 || true
  XENSE_RESTARTED=1
fi
"${XENSE_PYTHON}" "${XENSE_SCRIPT}" ensure \
  --serials "${XENSE_SERIALS}" \
  --names "${XENSE_NAMES}"

SAMPLES_READY=0
for _ in $(seq 1 100); do
  if fresh_samples_ok; then
    SAMPLES_READY=1
    break
  fi
  sleep 0.1
done
if [ "${SAMPLES_READY}" != "1" ]; then
  echo "[envelope_sidecar] no fresh valid samples from both tactile sensors after 10s" >&2
  "${XENSE_PYTHON}" "${XENSE_SCRIPT}" status >&2 || true
  exit 1
fi

# A responsive model service is reusable only while the tactile source is also
# fresh. If USB recovery restarted Xense, recalibrate the sidecar baseline too.
if [ "${SIDECAR_COMPATIBLE}" = "1" ] && [ "${XENSE_RESTARTED}" = "0" ]; then
  echo "[envelope_sidecar] reusing tcp://${HOST}:${PORT}; tactile samples are fresh"
  probe
  exit 0
fi

if is_running; then
  echo "[envelope_sidecar] replacing stale/incompatible managed sidecar"
  kill "$(cat "${PID_FILE}")"
  for _ in $(seq 1 50); do
    is_running || break
    sleep 0.1
  done
  rm -f "${PID_FILE}"
fi

echo "[envelope_sidecar] keep both tactile sensors unloaded during baseline"
TACTILE_GATE_ARGS=()
if [ -n "${TACTILE_TAU_ON:-}" ]; then
  TACTILE_GATE_ARGS+=(--tau-on "${TACTILE_TAU_ON}")
fi
if [ -n "${TACTILE_TAU_OFF:-}" ]; then
  TACTILE_GATE_ARGS+=(--tau-off "${TACTILE_TAU_OFF}")
fi
PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" \
nohup setsid "${MODEL_PYTHON}" "${SIDECAR_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --bilateral-calibration "${BILATERAL_CALIBRATION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --action-fps "${POLICY_EXECUTE_FPS:-30}" \
  --baseline-seconds "${TACTILE_BASELINE_SECONDS:-1.0}" \
  --max-baseline-residual-p95 "${TACTILE_MAX_BASELINE_RESIDUAL_P95:-1.0}" \
  --force-scale "${TACTILE_FORCE_SCALE:-5.0}" \
  --cost-clip "${TACTILE_COST_CLIP:-1.0}" \
  "${TACTILE_GATE_ARGS[@]}" \
  --log "${CHUNK_LOG}" \
  >"${LOG_FILE}" 2>&1 &
echo "$!" >"${PID_FILE}"

for _ in $(seq 1 300); do
  if probe >/dev/null 2>&1; then
    echo "[envelope_sidecar] ready log=${LOG_FILE}"
    probe
    exit 0
  fi
  if ! is_running; then
    echo "[envelope_sidecar] startup failed; log follows" >&2
    tail -80 "${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 0.1
done

echo "[envelope_sidecar] timeout; inspect ${LOG_FILE}" >&2
exit 1
