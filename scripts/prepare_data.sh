#!/usr/bin/env bash
# Download, import, or verify MapTab data without storing raw assets in this repository.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN:-python}" -m rlharness.data_process.prepare_data "$@"
