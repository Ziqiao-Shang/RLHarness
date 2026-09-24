#!/usr/bin/env bash
# Download a released fully merged model, then evaluate fixed Test400.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${VLLM_PYTHON:-python}"
DOMAIN="${DOMAIN:-metromap}"
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

"$PYTHON_BIN" -m rlharness.model_release \
  --domain "$DOMAIN" \
  --output-dir "$RELEASE_ROOT"

MODEL_PATH="$RELEASE_ROOT/$DOMAIN" \
PROMPT_TEMPLATE="$ROOT/prompts/student/$DOMAIN/final.txt" \
VLLM_PYTHON="$PYTHON_BIN" \
  env -u ADAPTER_PATH bash "$ROOT/scripts/06_evaluate_test.sh" --domain "$DOMAIN"
