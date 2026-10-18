#!/usr/bin/env bash
# ==============================================================================
# Pure Chunk-Level Reviewer Session for 500 Exam Documents
# Audits individual chunk_*.xml files across data/sequence_labelling_annotated
# ==============================================================================
set -euo pipefail

REPO_DIR="/home/daominhwysi/project/vietnamese-sequence-labelling-v2"
cd "$REPO_DIR"

# Source environment
set -a
[ -f .env ] && . .env
set +a

export PYTHONUNBUFFERED=1

REVIEW_REPORT="artifacts/reports/batch_review_500_diverse_documents.md"
REVIEW_LOG="artifacts/reports/batch_review_500_diverse_documents.log"

DOC_COUNT=$(find data/sequence_labelling_annotated -maxdepth 1 -name "doc_*" | wc -l)
echo "======================================================================"
echo "🚀 [Reviewer] Launching Pure Chunk-Level Quality Reviewer Session"
echo "   Target Dataset : data/sequence_labelling_annotated ($DOC_COUNT doc directories)"
echo "   Granularity    : PURE CHUNK LEVEL (chunk_*.xml)"
echo "   Raw Input Dir  : data/sequence_labelling_input_data"
echo "   Review Engine  : Google Antigravity (agy) | gemini-3.8-flash"
echo "   Output Report  : $REVIEW_REPORT"
echo "   Log File       : $REVIEW_LOG"
echo "   Start Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

mkdir -p artifacts/reports

uv run python tools/review/batch_review_annotated.py \
  --annotated-dir data/sequence_labelling_annotated \
  --raw-dir data/sequence_labelling_input_data \
  --provider agy \
  --model gemini-3.8-flash \
  --thinking low \
  --min-score 75 \
  --concurrency 2 \
  --granularity chunk \
  --pattern "doc_*" \
  --output-report "$REVIEW_REPORT" \
  2>&1 | tee -a "$REVIEW_LOG"

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 [Reviewer] Pure chunk-level review session completed successfully!"
else
    echo "⚠️ [Reviewer] Review session exited with code $EXIT_CODE"
fi
echo "   Report exported: $REVIEW_REPORT"
echo "   Log output     : $REVIEW_LOG"
echo "   Finish Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

exec bash
