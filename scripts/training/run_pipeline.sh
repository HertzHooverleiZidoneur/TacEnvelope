#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-$ROOT/configs/production/franka_tactile310_stable_contact.yaml}"

if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m tactile_force_envelope.cli "${1:-all}" --config "$CONFIG"
