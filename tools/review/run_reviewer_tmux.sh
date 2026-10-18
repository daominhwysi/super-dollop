#!/usr/bin/env bash
set -euo pipefail

# Environment setup
export AGY_ACCOUNT="daominhwysi"
export PYTHONUNBUFFERED="1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="artifacts/manifests/unaudited_563_chunks.txt"
LOG_FILE="artifacts/ocr_logs/agy_reviewer_563_chunks.log"
OUTPUT_REPORT="data/review_report_chunks.md"

mkdir -p artifacts/ocr_logs artifacts/manifests

echo "======================================================================"
echo "🚀 Starting Antigravity Batch Reviewer on Unaudited Chunks"
echo "  Account     : $AGY_ACCOUNT"
echo "  Provider    : agy"
echo "  Model       : gemini-3.8-flash (thinking: low)"
echo "  Manifest    : $MANIFEST"
echo "  Report      : $OUTPUT_REPORT"
echo "  Log File    : $LOG_FILE"
echo "  Granularity : chunk (strict chunk-level, no auto-rollup)"
echo "  Concurrency : 2"
echo "======================================================================"

exec uv run python tools/review/batch_review_annotated.py \
  --manifest "$MANIFEST" \
  --granularity chunk \
  --no-rollup \
  --provider agy \
  --model gemini-3.8-flash \
  --thinking low \
  --concurrency 2 \
  --output-report "$OUTPUT_REPORT" 2>&1 | tee -a "$LOG_FILE"
