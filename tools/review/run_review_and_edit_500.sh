#!/usr/bin/env bash
set -euo pipefail

# Unbuffer stdout so log tailing in tmux / tee is instantaneous
export PYTHONUNBUFFERED=1

# ==============================================================================
# All-In-One Pipeline: Reviewer (Resume or Fresh) -> Surgical Editor Loop
# Target: 500 Diverse Math Examination Documents
# Engine: Google Antigravity (agy) with Gemini 3.8 Flash Low
# Concurrency: 1 (safe sequential execution preventing burst TPM throttle)
# ==============================================================================

REVIEWER_MODEL="${1:-gemini-3.8-flash}"
REVIEWER_PROVIDER="${2:-agy}"
CONCURRENCY="${3:-1}"
REVIEWER_THINKING="${4:-low}"
MIN_SCORE="${5:-60}"
OVERWRITE="${6:-false}"

EDITOR_MODEL="${EDITOR_MODEL:-gemini-3.8-flash}"
EDITOR_PROVIDER="${EDITOR_PROVIDER:-agy}"
EDITOR_THINKING="${EDITOR_THINKING:-low}"
RUN_PHASE="${RUN_PHASE:-edit}" # options: all, review, edit (default: edit)
GRANULARITY="${GRANULARITY:-chunk}" # options: chunk, document (default: chunk)

OVERWRITE_FLAG=""
MODE_DESC="Resuming from previous progress (skipping already audited documents)"
if [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "--overwrite" ]; then
  OVERWRITE_FLAG="--overwrite"
  MODE_DESC="Full re-audit from document 1 (--overwrite enabled)"
fi

echo "======================================================================"
echo "🚀 Starting All-In-One Pipeline (Reviewer -> Surgical Editor)"
echo "  Target Dataset  : data/sequence_labelling_annotated (500 docs)"
echo "  Raw Source Dir  : data/batch_500_diverse_raw"
echo "  Run Phase       : $RUN_PHASE"
echo "  Granularity     : $GRANULARITY"
echo "  Editor Engine   : $EDITOR_PROVIDER ($EDITOR_MODEL, effort: $EDITOR_THINKING)"
echo "  Reviewer Engine : $REVIEWER_PROVIDER ($REVIEWER_MODEL, effort: $REVIEWER_THINKING)"
echo "  Min Pass Score  : $MIN_SCORE"
echo "  Concurrency     : $CONCURRENCY worker threads"
echo "  Audit Mode      : $MODE_DESC"
echo "======================================================================"

# ── Phase 1: Reviewer (Resumes unfinished docs or runs fresh) ──────────────────
if [ "$RUN_PHASE" = "all" ] || [ "$RUN_PHASE" = "review" ]; then
  echo ""
  echo "=== [Phase 1/2] Running Reviewer Audit via $REVIEWER_PROVIDER ==="
  # shellcheck disable=SC2086
  uv run python tools/review/batch_review_annotated.py \
    --annotated-dir data/sequence_labelling_annotated \
    --raw-dir data/batch_500_diverse_raw \
    --provider "$REVIEWER_PROVIDER" \
    --model "$REVIEWER_MODEL" \
    --thinking "$REVIEWER_THINKING" \
    --min-score "$MIN_SCORE" \
    --concurrency "$CONCURRENCY" \
    --granularity "$GRANULARITY" \
    --pattern "doc_*" \
    $OVERWRITE_FLAG \
    --output-report artifacts/reports/batch_review_500_diverse_documents.md
fi

# ── Phase 2: Surgical Editor Revision Loop ───────────────────────────────────
if [ "$RUN_PHASE" = "all" ] || [ "$RUN_PHASE" = "edit" ]; then
  echo ""
  echo "=== [Phase 2/2] Launching Surgical Editor Loop on Defective Documents ==="
  uv run python tools/review/edit_review_set.py \
    --report artifacts/reports/batch_review_500_diverse_documents.json \
    --raw-dir data/batch_500_diverse_raw \
    --annotated-dir data/sequence_labelling_annotated \
    --provider "$EDITOR_PROVIDER" \
    --model "$EDITOR_MODEL" \
    --thinking "$EDITOR_THINKING" \
    --reviewer-provider "$REVIEWER_PROVIDER" \
    --reviewer-model "$REVIEWER_MODEL" \
    --reviewer-thinking "$REVIEWER_THINKING" \
    --granularity "$GRANULARITY" \
    --filter discards,needs_revision \
    --skip-unrecoverable \
    --reuse-prerun-logs \
    --concurrency "$CONCURRENCY" \
    --branch-dir data/sequence_labelling_annotated_branch \
    --auto-save \
    --out-report artifacts/reports/revision_resolution_500.json
fi

echo ""
echo "======================================================================"
echo "🎉 Pipeline Completed!"
echo "  - Audit Report           : artifacts/reports/batch_review_500_diverse_documents.md"
echo "  - Repaired Ground Truth  : data/sequence_labelling_annotated_branch"
echo "  - Resolution Report      : artifacts/reports/revision_resolution_500.json"
echo "======================================================================"
