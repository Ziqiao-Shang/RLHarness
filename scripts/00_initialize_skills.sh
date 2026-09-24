#!/usr/bin/env bash
# Create a seed prompt, then optimize it with the bundled SkillOpt loop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
exec "${INITIALIZATION_PYTHON:-${PYTHON_BIN:-python}}" \
  -m rlharness.initialization.skillopt_run "$@"
