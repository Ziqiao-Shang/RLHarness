#!/usr/bin/env bash
# Continue a domain-specific selected Skill Prompt from global step 400 to step 600.
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
    DEFAULT_DATA_DIR="$ROOT/data/generated/rl_round2"
    DEFAULT_MODEL="$ROOT/artifacts/sft_merged"
    DEFAULT_SOURCE_CKPT="$ROOT/artifacts/grpo_round1/checkpoints/global_step_400"
    DEFAULT_OUTPUT="$ROOT/artifacts/grpo_round2"
    DEFAULT_SELECTION="$ROOT/artifacts/skill_evolution/selected"
    ;;
  travelmap)
    DEFAULT_SPLITS="$ROOT/data/generated/travelmap/splits"
    FIXED_SPLITS="$ROOT/data/reference/travelmap/splits"
    DEFAULT_DATA_DIR="$ROOT/data/generated/travelmap/rl_round2"
    DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"
    DEFAULT_SOURCE_CKPT="$ROOT/artifacts/travelmap/grpo_round1/checkpoints/global_step_400"
    DEFAULT_OUTPUT="$ROOT/artifacts/travelmap/grpo_round2"
    DEFAULT_SELECTION="$ROOT/artifacts/travelmap/skill_evolution/selected"
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

SPLITS="${SPLITS_DIR:-$DEFAULT_SPLITS}"
DATA_DIR="${RL_DATA_DIR:-$DEFAULT_DATA_DIR}"
MODEL="${SFT_MERGED_MODEL:-$DEFAULT_MODEL}"
SOURCE_CKPT="${ROUND1_CHECKPOINT:-$DEFAULT_SOURCE_CKPT}"
OUTPUT="${RL_OUTPUT_DIR:-$DEFAULT_OUTPUT}"
CONFIG="${RL_CONFIG:-$ROOT/configs/rl_round2.yaml}"
export PYTHONPATH="$ROOT/src:$VERL_ROOT:${PYTHONPATH:-}"

CHECKED_PROMPT="$ROOT/prompts/student/$DOMAIN/final.txt"
SELECTION_DIR="${SELECTED_SKILL_DIR:-$DEFAULT_SELECTION}"
SELECTION_MANIFEST=""
if [[ -n "${PROMPT_TEMPLATE:-}" ]]; then
  PROMPT="$PROMPT_TEMPLATE"
  SELECTION_MANIFEST="${SELECTED_SKILL_MANIFEST:-}"
elif [[ -s "$SELECTION_DIR/full_prompt.txt" && -s "$SELECTION_DIR/selection.json" ]]; then
  PROMPT="$SELECTION_DIR/full_prompt.txt"
  SELECTION_MANIFEST="$SELECTION_DIR/selection.json"
elif [[ -e "$SELECTION_DIR/full_prompt.txt" || -e "$SELECTION_DIR/selection.json" ]]; then
  echo "Incomplete selected Skill artifact: $SELECTION_DIR" >&2
  exit 1
else
  # Reproduction path: use the reviewed Prompt committed with the release.
  PROMPT="$CHECKED_PROMPT"
fi

for path in "$MODEL/config.json" "$SOURCE_CKPT/data.pt" "$CONFIG" "$VERL_ROOT/verl" "$PROMPT"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

lock_args=(
  lock-round2 --domain "$DOMAIN" --prompt "$PROMPT" --output-dir "$OUTPUT"
)
[[ -n "$SELECTION_MANIFEST" ]] && lock_args+=(--selection-manifest "$SELECTION_MANIFEST")
"$PYTHON_BIN" -m rlharness.evolution.skill_select "${lock_args[@]}"
PROMPT="$OUTPUT/selected_prompt.txt"

"$PYTHON_BIN" -m rlharness.data_process.data_split \
  --domain "$DOMAIN" \
  --output-dir "$SPLITS" \
  --fixed-ids-dir "$FIXED_SPLITS"
"$PYTHON_BIN" -m rlharness.data_process.rl_data \
  --domain "$DOMAIN" \
  --subset-dir "$SPLITS" --prompt-template "$PROMPT" --output-dir "$DATA_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec "$PYTHON_BIN" -m rlharness.rl.rl_launcher \
  --config "$CONFIG" --verl-root "$VERL_ROOT" \
  "data.train_files=$DATA_DIR/train.jsonl" \
  "data.val_files=$DATA_DIR/validation.jsonl" \
  "data.custom_cls.path=$ROOT/src/rlharness/rl/rl_dataset.py" \
  "actor_rollout_ref.model.path=$MODEL" \
  "actor_rollout_ref.model.tokenizer_path=$MODEL" \
  "reward.custom_reward_function.path=$ROOT/src/rlharness/rl/reward.py" \
  "trainer.default_local_dir=$OUTPUT/checkpoints" \
  "trainer.rollout_data_dir=$OUTPUT/rollouts" \
  "trainer.validation_data_dir=$OUTPUT/validation" \
  "trainer.resume_from_path=$SOURCE_CKPT" \
  "data.prompt_transform_version=${DOMAIN}_skill_reasoning_v1" \
  "trainer.experiment_name=${DOMAIN}_grpo_round2" \
  "${LAUNCHER_OVERRIDES[@]}"
