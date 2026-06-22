#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$ROOT_DIR/outputs/model_cache/paddlex}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/outputs/model_cache/huggingface}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Install curl or wget, then rerun this script." >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv sync

if [[ "${DOCMD_SETUP_ONLY:-}" == "1" ]]; then
  echo "Setup complete."
  exit 0
fi

exec uv run python scripts/run_demo.py "$@"
