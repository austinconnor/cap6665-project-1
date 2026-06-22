#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${UV_PYTHON:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
    export UV_PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    export UV_PYTHON="$ROOT_DIR/.venv/bin/python"
  fi
fi

exec uv run --no-sync streamlit run demo/app.py "$@"
