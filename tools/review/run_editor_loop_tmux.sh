#!/usr/bin/env bash
set -euo pipefail

# Environment setup
export AGY_ACCOUNT="daominhwysi"
export AGY_CONCURRENCY_LIMIT="4"
export PYTHONUNBUFFERED="1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REPORT_SOURCE="data/review_report_chunks.json"
BRANCH_DIR="data/sequence_labelling_annotated_branch"
OUT_REPORT="artifacts/reports/revision_resolution_chunks.json"
LOG_FILE="artifacts/ocr_logs/editor_reviewer_loop.log"

mkdir -p artifacts/ocr_logs artifacts/reports "$BRANCH_DIR"

echo "======================================================================"
echo "🚀 Starting EditorAgent & Reviewer Revision Loop"
echo "  Account         : $AGY_ACCOUNT"
echo "  Source Report   : $REPORT_SOURCE"
echo "  Target Decisions: NEEDS_REVISION, DISCARD"
echo "  Granularity     : chunk"
echo "  Concurrency     : 4"
echo "  Branch Dir      : $BRANCH_DIR"
echo "  Output Report   : $OUT_REPORT"
echo "  Log File        : $LOG_FILE"
echo "======================================================================"

exec uv run python tools/review/edit_review_set.py \
  --report "$REPORT_SOURCE" \
  --filter discards,needs_revision \
  --concurrency 4 \
  --granularity chunk \
  --auto-save \
  --branch-dir "$BRANCH_DIR" \
  --out-report "$OUT_REPORT" \
  --provider agy \
  --model gemini-3.8-flash \
  --editor-thinking low \
  --reviewer-provider agy \
  --reviewer-model gemini-3.8-flash \
  --reviewer-thinking low 2>&1 | tee -a "$LOG_FILE"
