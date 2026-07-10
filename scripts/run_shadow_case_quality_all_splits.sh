#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/allclear_semantic_softshadow_dadigan_stage1.yaml}"
OUTPUT_DIR="${2:-outputs/allclear/shadow_case_quality_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VISUAL_SAMPLES_PER_CASE="${VISUAL_SAMPLES_PER_CASE:-3}"

cd "$(dirname "$0")/.."

python scripts/evaluate_shadow_case_quality.py \
  --config "$CONFIG" \
  --splits train val test \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --visual-samples-per-case "$VISUAL_SAMPLES_PER_CASE" \
  --visual-buckets low medium high \
  --output-dir "$OUTPUT_DIR"

echo "Shadow-case quality report written to: $OUTPUT_DIR"
