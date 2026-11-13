#!/usr/bin/env bash
set -euo pipefail
MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${FPSRG_RUNTIME_DIR:-$MODULE_DIR/.runtime}"
export PYTHONPATH="$RUNTIME_DIR/transformers${PYTHONPATH:+:$PYTHONPATH}"
exec "$RUNTIME_DIR/venv/bin/fpsrg-vector-serve" "$@"
