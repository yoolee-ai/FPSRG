#!/usr/bin/env bash
set -euo pipefail
MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${FPSRG_RUNTIME_DIR:-$MODULE_DIR/.runtime}"
python3.11 -m venv "$RUNTIME_DIR/venv"
"$RUNTIME_DIR/venv/bin/python" -m pip install --upgrade pip
"$RUNTIME_DIR/venv/bin/python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
"$RUNTIME_DIR/venv/bin/python" -m pip install "$MODULE_DIR[gpu]"
# Keep the tested Transformers overlay separate from vLLM's <5 dependency.
"$RUNTIME_DIR/venv/bin/python" -m pip install --target "$RUNTIME_DIR/transformers" 'transformers==5.15.1'
printf 'Installed. Start with: bash %s/serve.sh --manifest <artifact/manifest.json> --gpu 0\n' "$MODULE_DIR"
