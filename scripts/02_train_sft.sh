#!/usr/bin/env bash
# Train the language-side SFT adapter and merge the selected final checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-$ROOT/third_party/LlamaFactory}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"
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
  metromap)
    DEFAULT_GENERATED="$ROOT/data/generated"
    DEFAULT_OUTPUT="$ROOT/artifacts/sft"
    DEFAULT_MERGED="$ROOT/artifacts/sft_merged"
    DEFAULT_SELECTED="$ROOT/artifacts/selected_sft_checkpoint.txt"
    ;;
  travelmap)
    DEFAULT_GENERATED="$ROOT/data/generated/travelmap"
    DEFAULT_OUTPUT="$ROOT/artifacts/travelmap/sft"
    DEFAULT_MERGED="$ROOT/artifacts/travelmap/sft_merged"
    DEFAULT_SELECTED="$ROOT/artifacts/travelmap/selected_sft_checkpoint.txt"
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

# Keep stage 02 aligned with stage 01: a custom generation root implies its sft/ child.
DATA_DIR="${SFT_DATA_DIR:-${GENERATED_DATA_DIR:-$DEFAULT_GENERATED}/sft}"
REGISTRY_DIR="${SFT_REGISTRY_DIR:-$DATA_DIR/llamafactory}"
OUTPUT="${SFT_OUTPUT_DIR:-$DEFAULT_OUTPUT}"
MERGED="${SFT_MERGED_MODEL:-$DEFAULT_MERGED}"
SELECTED_CHECKPOINT_FILE="${SELECTED_SFT_CHECKPOINT_FILE:-$DEFAULT_SELECTED}"
CONFIG="${SFT_CONFIG:-$ROOT/configs/sft_train.yaml}"
MERGE_CONFIG="${SFT_MERGE_CONFIG:-$ROOT/configs/sft_merge.yaml}"
DATASET_NAME="${DOMAIN}_skill_sft"

if [[ ! -s "$DATA_DIR/train.json" ]]; then
  bash "$ROOT/scripts/01_generate_sft_data.sh" --domain "$DOMAIN"
fi
for path in "$BASE_MODEL/config.json" "$CONFIG" "$MERGE_CONFIG" "$LLAMA_FACTORY_ROOT/pyproject.toml"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
command -v "$LLAMAFACTORY_CLI" >/dev/null 2>&1 || { echo "Cannot find llamafactory-cli" >&2; exit 1; }

mkdir -p "$REGISTRY_DIR"
"$PYTHON_BIN" - "$REGISTRY_DIR/dataset_info.json" "$DATA_DIR/train.json" "$DATASET_NAME" <<'PY'
import json
import sys
from pathlib import Path

info_path, train_path = map(Path, sys.argv[1:3])
name = sys.argv[3]
info = {name: {
    "file_name": str(train_path.resolve()),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"},
    "tags": {
        "role_tag": "from", "content_tag": "value",
        "user_tag": "human", "assistant_tag": "gpt",
    },
}}
info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

mkdir -p "$OUTPUT" "$(dirname "$MERGED")" "$(dirname "$SELECTED_CHECKPOINT_FILE")"
"$LLAMAFACTORY_CLI" train "$CONFIG" \
  "model_name_or_path=$BASE_MODEL" \
  "dataset=$DATASET_NAME" \
  "dataset_dir=$REGISTRY_DIR" \
  "output_dir=$OUTPUT"

ADAPTER="${SFT_ADAPTER_PATH:-$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)}"
[[ -s "$ADAPTER/adapter_config.json" ]] || { echo "No completed SFT checkpoint found in $OUTPUT" >&2; exit 1; }

"$LLAMAFACTORY_CLI" export "$MERGE_CONFIG" \
  "model_name_or_path=$BASE_MODEL" \
  "adapter_name_or_path=$ADAPTER" \
  "export_dir=$MERGED"

printf '%s\n' "$ADAPTER" > "$SELECTED_CHECKPOINT_FILE"
echo "Merged SFT model: $MERGED"
