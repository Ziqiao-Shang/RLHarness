#!/usr/bin/env bash
# Build domain-specific original-Skill GRPO data and train through global step 400.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT/third_party/verl-modern}"
PYTHON_BIN="${VERL_PYTHON:-python}"
DOMAIN="${DOMAIN:-metromap}"
LAUNCHER_OVERRIDES=()
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
      LAUNCHER_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

case "$DOMAIN" in
  metromap)
    DEFAULT_SPLITS="$ROOT/data/generated/splits"
    FIXED_SPLITS="$ROOT/data/reference/metromap/splits"
    DEFAULT_DATA_DIR="$ROOT/data/generated/rl_round1"
    DEFAULT_MODEL="$ROOT/artifacts/sft_merged"
    DEFAULT_OUTPUT="$ROOT/artifacts/grpo_round1"
    ;;
  travelmap)
    DEFAULT_SPLITS="$ROOT/data/generated/travelmap/splits"
    FIXED_SPLITS="$ROOT/data/reference/travelmap/splits"
    DEFAULT_DATA_DIR="$ROOT/data/generated/travelmap/rl_round1"
    DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"
    DEFAULT_OUTPUT="$ROOT/artifacts/travelmap/grpo_round1"
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

SPLITS="${SPLITS_DIR:-$DEFAULT_SPLITS}"
DATA_DIR="${RL_DATA_DIR:-$DEFAULT_DATA_DIR}"
MODEL="${SFT_MERGED_MODEL:-$DEFAULT_MODEL}"
OUTPUT="${RL_OUTPUT_DIR:-$DEFAULT_OUTPUT}"
DEFAULT_PROMPT="$ROOT/prompts/student/$DOMAIN/original.txt"
PROMPT="${PROMPT_TEMPLATE:-$DEFAULT_PROMPT}"
CONFIG="${RL_CONFIG:-$ROOT/configs/rl_round1.yaml}"
export PYTHONPATH="$ROOT/src:$VERL_ROOT:${PYTHONPATH:-}"

for path in "$MODEL/config.json" "$PROMPT" "$CONFIG" "$VERL_ROOT/verl"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
"$PYTHON_BIN" -m rlharness.evolution.skill_select \
  lock-round1 --domain "$DOMAIN" --prompt "$PROMPT" --output-dir "$OUTPUT"
PROMPT="$OUTPUT/selected_prompt.txt"

"$PYTHON_BIN" -m rlharness.data_process.data_split \
  --domain "$DOMAIN" \
  --output-dir "$SPLITS" \
  --fixed-ids-dir "$FIXED_SPLITS"
"$PYTHON_BIN" -m rlharness.data_process.rl_data \
  --domain "$DOMAIN" \
  --subset-dir "$SPLITS" \
  --prompt-template "$PROMPT" \
  --output-dir "$DATA_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec "$PYTHON_BIN" -m rlharness.rl.rl_launcher \
  --config "$CONFIG" \
  --verl-root "$VERL_ROOT" \
  "data.train_files=$DATA_DIR/train.jsonl" \
  "data.val_files=$DATA_DIR/validation.jsonl" \
  "data.custom_cls.path=$ROOT/src/rlharness/rl/rl_dataset.py" \
  "actor_rollout_ref.model.path=$MODEL" \
  "actor_rollout_ref.model.tokenizer_path=$MODEL" \
  "reward.custom_reward_function.path=$ROOT/src/rlharness/rl/reward.py" \
  "trainer.default_local_dir=$OUTPUT/checkpoints" \
  "trainer.rollout_data_dir=$OUTPUT/rollouts" \
  "trainer.validation_data_dir=$OUTPUT/validation" \
  "data.prompt_transform_version=${DOMAIN}_skill_reasoning_v1" \
  "trainer.experiment_name=${DOMAIN}_grpo_round1" \
  "${LAUNCHER_OVERRIDES[@]}"
