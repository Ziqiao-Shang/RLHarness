#!/usr/bin/env bash
# Generate, validate, and convert teacher trajectories into SFT data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
    FIXED_SPLITS="$ROOT/data/reference/metromap/splits"
    DEFAULT_TEACHER_PROMPT="$ROOT/prompts/data_process/teacher_metromap.txt"
    DEFAULT_TEACHER_MAX_TOKENS=4096
    DEFAULT_MIN_SFT_LABELS=1395
    TEACHER_DOMAIN_ARGS=(--exclude_start_transfer)
    ;;
  travelmap)
    DEFAULT_GENERATED="$ROOT/data/generated/travelmap"
    FIXED_SPLITS="$ROOT/data/reference/travelmap/splits"
    DEFAULT_TEACHER_PROMPT="$ROOT/prompts/data_process/teacher_travelmap.txt"
    DEFAULT_TEACHER_MAX_TOKENS=8192
    DEFAULT_MIN_SFT_LABELS=1233
    TEACHER_DOMAIN_ARGS=()
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

GENERATED="${GENERATED_DATA_DIR:-$DEFAULT_GENERATED}"
SPLITS="${SPLITS_DIR:-$GENERATED/splits}"
TEACHER="${SFT_TEACHER_DIR:-$GENERATED/sft_teacher}"
SFT="${SFT_DATA_DIR:-$GENERATED/sft}"
MAPTAB_ROOT="${MAPTAB_ROOT:-$ROOT/../maptab_data}"
RAW_DOMAIN="$MAPTAB_ROOT/$DOMAIN"
SKILLS="${ORIGINAL_SKILLS_JSON:-$ROOT/prompts/student/$DOMAIN/original_skills.json}"
STUDENT_PROMPT="${STUDENT_PROMPT:-$ROOT/prompts/student/$DOMAIN/original.txt}"
TEACHER_PROMPT="${TEACHER_PROMPT:-$DEFAULT_TEACHER_PROMPT}"
MODEL="${TEACHER_MODEL:-gpt-5.6-sol}"
WORKERS="${TEACHER_WORKERS:-64}"
ROUNDS="${TEACHER_ROUNDS:-3}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-$DEFAULT_TEACHER_MAX_TOKENS}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
MIN_SFT_LABELS="${MIN_SFT_LABELS:-$DEFAULT_MIN_SFT_LABELS}"

: "${OPENLUX_API_KEY:?Set OPENLUX_API_KEY before generating teacher trajectories}"
for path in \
  "$RAW_DOMAIN" "$SKILLS" "$STUDENT_PROMPT" "$TEACHER_PROMPT" \
  "$BASE_MODEL/tokenizer.json" \
  "$FIXED_SPLITS/train_sample_ids.json" \
  "$FIXED_SPLITS/validation_sample_ids.json" \
  "$FIXED_SPLITS/test_sample_ids.json"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export MAPTAB_ROOT
mkdir -p "$SPLITS" "$TEACHER" "$SFT"

"$PYTHON_BIN" -m rlharness.data_process.data_split \
  --domain "$DOMAIN" \
  --output-dir "$SPLITS" \
  --fixed-ids-dir "$FIXED_SPLITS"

LABELS="$TEACHER/labels.jsonl"
status=1
for ((round = 1; round <= ROUNDS; round++)); do
  echo "[SFT data] teacher generation round $round/$ROUNDS"
  set +e
  "$PYTHON_BIN" -u -m rlharness.data_process.sft_teacher \
    --domain "$DOMAIN" \
    --skills_json "$SKILLS" \
    --student_prompt_path "$STUDENT_PROMPT" \
    --teacher_prompt_path "$TEACHER_PROMPT" \
    --include_sample_ids_file "$SPLITS/train_sample_ids.json" \
    --output_file "$TEACHER/labels.jsonl" \
    --raw_output_file "$TEACHER/raw_attempts.jsonl" \
    --failures_file "$TEACHER/failures.jsonl" \
    --selection_file "$TEACHER/selection.json" \
    --prompt_preview "$TEACHER/prompt_preview.txt" \
    --limit 1600 \
    --sample_seed 20260908 \
    "${TEACHER_DOMAIN_ARGS[@]}" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    --temperature 0.2 \
    --teacher_max_tokens "$TEACHER_MAX_TOKENS" \
    --max_output_tokens "$MAX_OUTPUT_TOKENS" \
    --tokenizer_path "$BASE_MODEL" \
    --timeout 300 \
    --retries 2 \
    --resume
  status=$?
  set -e
  ((status == 0)) && break
  ((round < ROUNDS)) && sleep 30
done
if ((status != 0)); then
  echo "Teacher generation still has rejected rows; auditing the accepted rows collected so far." >&2
fi
[[ -s "$TEACHER/labels.jsonl" ]] || { echo "No accepted teacher labels were generated." >&2; exit 1; }

"$PYTHON_BIN" -u -m rlharness.data_process.sft_audit \
  "$LABELS" \
  --domain "$DOMAIN" \
  --split train \
  --clean_output "$TEACHER/labels.clean.jsonl" \
  --rejected_output "$TEACHER/labels.rejected.jsonl" \
  --report "$TEACHER/audit_report.json"

accepted_labels="$(wc -l < "$TEACHER/labels.clean.jsonl")"
if ((accepted_labels < MIN_SFT_LABELS)); then
  echo "Only $accepted_labels audited labels are available; need at least $MIN_SFT_LABELS. Rerun to resume failed rows." >&2
  exit 1
fi

"$PYTHON_BIN" -m rlharness.data_process.sft_build \
  --domain "$DOMAIN" \
  --labels "$TEACHER/labels.clean.jsonl" \
  --raw-training-data "$SPLITS/train_source.json" \
  --fixed-train-ids "$SPLITS/train_sample_ids.json" \
  --validation-ids "$SPLITS/validation_sample_ids.json" \
  --test-ids "$SPLITS/test_sample_ids.json" \
  --validation-file "$SPLITS/validation.json" \
  --prompt-template "$STUDENT_PROMPT" \
  --tokenizer "$BASE_MODEL" \
  --cutoff-len 24576 \
  --output-dir "$SFT"

echo "SFT data ready: $SFT/train.json"
