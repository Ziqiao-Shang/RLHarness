#!/usr/bin/env bash
# Generate checkpoint-400 evidence and produce three synchronized Skill candidates.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${SKILL_EVOLUTION_PYTHON:-${VLLM_PYTHON:-python}}"
DOMAIN="${DOMAIN:-metromap}"
PIPELINE_ARGS=()
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
      PIPELINE_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$DOMAIN" in
  metromap)
    DEFAULT_MODEL="$ROOT/artifacts/sft_merged"
    DEFAULT_ADAPTER="$ROOT/artifacts/grpo_round1/checkpoints/global_step_400/adapter"
    DEFAULT_SPLITS="$ROOT/data/generated/splits"
    FIXED_SPLITS="$ROOT/data/reference/metromap/splits"
    DEFAULT_SOURCE="$ROOT/artifacts/skill_evolution_source"
    DEFAULT_RUN="$ROOT/artifacts/skill_evolution"
    ;;
  travelmap)
    DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"
    DEFAULT_ADAPTER="$ROOT/artifacts/travelmap/grpo_round1/checkpoints/global_step_400/adapter"
    DEFAULT_SPLITS="$ROOT/data/generated/travelmap/splits"
    FIXED_SPLITS="$ROOT/data/reference/travelmap/splits"
    DEFAULT_SOURCE="$ROOT/artifacts/travelmap/skill_evolution_source"
    DEFAULT_RUN="$ROOT/artifacts/travelmap/skill_evolution"
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

MODEL="${SFT_MERGED_MODEL:-$DEFAULT_MODEL}"
ADAPTER="${ROUND1_ADAPTER:-$DEFAULT_ADAPTER}"
SPLITS="${SPLITS_DIR:-$DEFAULT_SPLITS}"
SOURCE="${SKILL_EVOLUTION_SOURCE_DIR:-$DEFAULT_SOURCE}"
RUN="${SKILL_EVOLUTION_RUN_DIR:-$DEFAULT_RUN}"
DEFAULT_PROMPT="$ROOT/prompts/student/$DOMAIN/original.txt"
PROMPT="${PROMPT_TEMPLATE:-$DEFAULT_PROMPT}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PYTHON_BIN" -m rlharness.data_process.data_split \
  --domain "$DOMAIN" \
  --output-dir "$SPLITS" \
  --fixed-ids-dir "$FIXED_SPLITS"
exec "$PYTHON_BIN" -m rlharness.evolution.skill_pipeline \
  --domain "$DOMAIN" \
  --prompt "$PROMPT" \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --sample-ids "$SPLITS/train_sample_ids.json" \
  --validation-ids "$SPLITS/validation_sample_ids.json" \
  --source-dir "$SOURCE" \
  --run-dir "$RUN" \
  "${PIPELINE_ARGS[@]}"
