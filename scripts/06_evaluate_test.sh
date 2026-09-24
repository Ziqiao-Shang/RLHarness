#!/usr/bin/env bash
# Evaluate the step-600 adapter produced by round two, or an explicit merged model.
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
  metromap)
    DEFAULT_SPLITS="$ROOT/data/generated/splits"
    FIXED_SPLITS="$ROOT/data/reference/metromap/splits"
    DEFAULT_MODEL="$ROOT/artifacts/sft_merged"
    DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/grpo_round2"
    DEFAULT_OUTPUT="$ROOT/results/test400"
    ;;
  travelmap)
    DEFAULT_SPLITS="$ROOT/data/generated/travelmap/splits"
    FIXED_SPLITS="$ROOT/data/reference/travelmap/splits"
    DEFAULT_MODEL="$ROOT/artifacts/travelmap/sft_merged"
    DEFAULT_ROUND2_OUTPUT="$ROOT/artifacts/travelmap/grpo_round2"
    DEFAULT_OUTPUT="$ROOT/results/travelmap/test400"
    ;;
  *)
    echo "Unsupported --domain: $DOMAIN (expected metromap or travelmap)" >&2
    exit 2
    ;;
esac

ROUND2_OUTPUT="${RL_OUTPUT_DIR:-$DEFAULT_ROUND2_OUTPUT}"
DEFAULT_ADAPTER="$ROUND2_OUTPUT/checkpoints/global_step_600/adapter"
CHECKED_PROMPT="$ROOT/prompts/student/$DOMAIN/final.txt"
VERIFY_ROUND2_PROMPT=0
if [[ -n "${PROMPT_TEMPLATE:-}" ]]; then
  PROMPT="$PROMPT_TEMPLATE"
elif [[ -s "$ROUND2_OUTPUT/selected_prompt.txt" && ( -z "${MODEL_PATH:-}" || -n "${ADAPTER_PATH:-}" ) ]]; then
  PROMPT="$ROUND2_OUTPUT/selected_prompt.txt"
  VERIFY_ROUND2_PROMPT=1
elif [[ -n "${MODEL_PATH:-}" ]]; then
  # A separately supplied merged model has no local round-two prompt manifest.
  PROMPT="$CHECKED_PROMPT"
else
  PROMPT="$CHECKED_PROMPT"
fi
SPLITS="${SPLITS_DIR:-$DEFAULT_SPLITS}"
MODEL="${MODEL_PATH:-$DEFAULT_MODEL}"
if [[ -v ADAPTER_PATH ]]; then
  ADAPTER="$ADAPTER_PATH"
elif [[ -n "${MODEL_PATH:-}" ]]; then
  # An explicitly supplied model is treated as already merged unless an
  # adapter is also supplied explicitly.
  ADAPTER=""
else
  ADAPTER="$DEFAULT_ADAPTER"
fi
OUTPUT="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if ((VERIFY_ROUND2_PROMPT)); then
  "$PYTHON_BIN" -m rlharness.evolution.skill_select verify-round2 \
    --domain "$DOMAIN" --output-dir "$ROUND2_OUTPUT" >/dev/null
fi

"$PYTHON_BIN" -m rlharness.data_process.data_split \
  --domain "$DOMAIN" \
  --output-dir "$SPLITS" \
  --fixed-ids-dir "$FIXED_SPLITS"
[[ -s "$MODEL/config.json" ]] || { echo "Missing model: $MODEL" >&2; exit 1; }
[[ -z "$ADAPTER" || -s "$ADAPTER/adapter_config.json" ]] || {
  echo "Missing adapter: $ADAPTER" >&2
  exit 1
}
[[ -s "$PROMPT" ]] || { echo "Missing prompt: $PROMPT" >&2; exit 1; }
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "GPU_IDS is empty" >&2; exit 1; }

mkdir -p "$OUTPUT/shards" "$OUTPUT/logs"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

pids=()
shards=()
cleanup() {
  trap - INT TERM
  ((${#pids[@]})) && kill -INT "${pids[@]}" 2>/dev/null || true
  wait || true
  exit 130
}
trap cleanup INT TERM

total="$($PYTHON_BIN - "$SPLITS/test_sample_ids.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
)"
for index in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$index]}"
  start=$((index * total / ${#GPU_IDS[@]}))
  end=$(((index + 1) * total / ${#GPU_IDS[@]}))
  shard="$OUTPUT/shards/gpu_${index}.jsonl"
  shards+=("$shard")
  args=(
    --domain "$DOMAIN" --split test
    --sample-ids-from "$SPLITS/test_sample_ids.json"
    --prompt-template "$PROMPT" --model "$MODEL" --output "$shard"
    --start "$start" --limit "$((end - start))"
    --no-thinking --visible-reasoning --max-new-tokens 4096
    --max-model-len 32768 --image-max-pixels 1000000
    --table-max-chars 20000 --batch-size 16
    --gpu-memory-utilization 0.70 --skip-mm-profiling --resume
  )
  [[ -n "$ADAPTER" ]] && args+=(--adapter "$ADAPTER")
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m rlharness.eval.eval_local \
    "${args[@]}" >"$OUTPUT/logs/gpu_${index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
((failed == 0)) || { echo "Evaluation failed; inspect $OUTPUT/logs" >&2; exit 1; }

"$PYTHON_BIN" -m rlharness.eval.eval_merge \
  --domain "$DOMAIN" --split test \
  --sample-ids-from "$SPLITS/test_sample_ids.json" \
  --output "$OUTPUT/predictions.jsonl" "${shards[@]}"

echo "Test summary: $OUTPUT/predictions.summary.json"
