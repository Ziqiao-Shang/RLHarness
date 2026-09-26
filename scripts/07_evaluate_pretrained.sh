#!/usr/bin/env bash
# Resolve a local or released fully merged model, then evaluate fixed Test400.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${VLLM_PYTHON:-python}"
DOMAIN="${DOMAIN:-metromap}"
CHECK_ONLY=0
while (($#)); do
  case "$1" in
    --domain)
      [[ $# -ge 2 ]] || { echo "--domain requires metromap or travelmap" >&2; exit 2; }
      DOMAIN="$2"
      shift 2
      ;;
    --domain=*)
      DOMAIN="${1#*=}"
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$DOMAIN" in
  metromap|travelmap) ;;
  *) echo "Unsupported --domain: $DOMAIN" >&2; exit 2 ;;
esac

RELEASE_ROOT="${RLHARNESS_MODEL_ROOT:-$ROOT/models/release}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if [[ -n "${MODEL_PATH:-}" ]]; then
  RESOLVED_MODEL_PATH="$("$PYTHON_BIN" -m rlharness.model_release \
    --domain "$DOMAIN" \
    --local-path "$MODEL_PATH" \
    --print-local-path)"
else
  RESOLVED_MODEL_PATH="$("$PYTHON_BIN" -m rlharness.model_release \
    --domain "$DOMAIN" \
    --output-dir "$RELEASE_ROOT" \
    --print-local-path)"
fi

if ((CHECK_ONLY)); then
  [[ -s "$ROOT/prompts/student/$DOMAIN/final.txt" ]] || {
    echo "Missing final prompt for $DOMAIN" >&2
    exit 1
  }
  echo "Validated local model: $RESOLVED_MODEL_PATH"
  echo "Validated final prompt: $ROOT/prompts/student/$DOMAIN/final.txt"
  exit 0
fi

MODEL_PATH="$RESOLVED_MODEL_PATH" \
PROMPT_TEMPLATE="$ROOT/prompts/student/$DOMAIN/final.txt" \
VLLM_PYTHON="$PYTHON_BIN" \
  env -u ADAPTER_PATH bash "$ROOT/scripts/06_evaluate_test.sh" --domain "$DOMAIN"
