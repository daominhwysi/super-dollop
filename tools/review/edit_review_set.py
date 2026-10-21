#!/usr/bin/env python3
"""
CLI Runner for isolated Editor/Reviewer revision loops.

Discovers all documents in NEEDS_REVISION status from review_report.json,
applies deterministic and targeted LLM surgical repairs, validates score elevation,
and optionally saves repaired ground truth to merged.xml and merged.json.

This tool delegates core execution to sequence_labelling.annotator.revision_loop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sequence_labelling.config import (
    EDITOR_MODEL,
    EDITOR_PROVIDER,
    EDITOR_THINKING,
    REVIEWER_MODEL,
    REVIEWER_PROVIDER,
    REVIEWER_THINKING,
    WORKSPACE_DIR,
    get_provider_base_url,
)
# Re-export core domain functions from sequence_labelling.annotator.revision_loop
# for backwards compatibility with tests and callers
from sequence_labelling.annotator.revision_loop import (
    CHECKPOINT_SCHEMA_VERSION,
    UNRECOVERABLE_PATTERNS,
    UNRECOVERABLE_REGEX,
    EditorReviewerLoop,
    RevisionLoopResult,
    RevisionRound,
    _atomic_write_json,
    _checkpoint_path,
    _chunk_sort_key,
    _sha256_file,
    apply_branch_resume,
    is_unrecoverable_discard,
    load_revision_targets,
    process_single_revision,
    reconcile_repaired_exam,
    run_batch_revision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Revision Resolution Engine using EditorAgent (Search/Replace Diff Blocks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default="backend/logs/review_report.json",
        help="Path to review report JSON (default: backend/logs/review_report.json)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/sequence_labelling_input_data",
        help="Path to raw markdown directory",
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Path to annotated XML directory",
    )
    parser.add_argument(
        "--doc-id",
        "--target-doc",
        dest="target_doc",
        type=str,
        default=None,
        help="Filter resolution to a specific document ID or substring",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EDITOR_MODEL,
        help="Editor model name",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=EDITOR_PROVIDER,
        help="Editor LLM provider",
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=REVIEWER_MODEL,
        help="Reviewer model name",
    )
    parser.add_argument(
        "--reviewer-provider",
        type=str,
        default=REVIEWER_PROVIDER,
        help="Reviewer provider",
    )
    parser.add_argument(
        "--editor-thinking",
        "--thinking",
        type=str,
        default=EDITOR_THINKING,
        help="Editor reasoning/thinking effort",
    )
    parser.add_argument(
        "--reviewer-thinking",
        type=str,
        default=REVIEWER_THINKING,
        help="Reviewer reasoning/thinking effort",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Concurrency worker threads",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of documents to repair",
    )
    parser.add_argument(
        "--max-passes",
        "--max-rounds",
        dest="max_passes",
        type=int,
        default=2,
        help="Maximum repair passes per document",
    )
    parser.add_argument(
        "--filter",
        "--only",
        dest="filter_decision",
        type=str,
        default="needs_revision",
        help="Review decisions to repair: needs_revision, discards, or discards,needs_revision",
    )
    parser.add_argument(
        "--branch-dir",
        "--output-dir",
        dest="branch_dir",
        type=str,
        default="data/sequence_labelling_annotated_branch",
        help="Branch output directory to save repaired files",
    )
    parser.add_argument(
        "--auto-save",
        "--save-branch",
        dest="auto_save",
        action="store_true",
        default=True,
        help="Automatically save repaired files to the branch folder",
    )
    parser.add_argument(
        "--no-auto-save",
        dest="auto_save",
        action="store_false",
        help="Do not save repaired files to disk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate repairs in memory without writing to disk",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="artifacts/reports/revision_resolution_report.json",
        help="Path to output resolution report JSON",
    )
    parser.add_argument(
        "--reuse-prerun-logs",
        action="store_true",
        default=True,
        help="Directly consume prerun error logs from report on Round 0",
    )
    parser.add_argument(
        "--no-reuse-prerun-logs",
        "--fresh-review",
        dest="reuse_prerun_logs",
        action="store_false",
        help="Force fresh Reviewer LLM evaluation on Round 0",
    )
    parser.add_argument(
        "--resume-branch",
        dest="resume_branch",
        action="store_true",
        default=True,
        help="Resume from existing branch candidates",
    )
    parser.add_argument(
        "--no-resume-branch",
        dest="resume_branch",
        action="store_false",
        help="Do not resume from branch candidates",
    )
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        default=False,
        help="Retry repairing documents previously resolved in branch with non-PASS status",
    )
    parser.add_argument(
        "--auto-reconcile",
        action="store_true",
        default=True,
        help="Automatically synthesize merged.xml when all chunks of an exam pass",
    )
    parser.add_argument(
        "--no-auto-reconcile",
        dest="auto_reconcile",
        action="store_false",
        help="Disable automatic chunk synthesis",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="chunk",
        choices=["chunk", "document"],
        help="Granularity of revision: chunk-level or document-level",
    )
    parser.add_argument(
        "--skip-unrecoverable",
        action="store_true",
        default=True,
        help="Automatically skip documents with unrecoverable OCR flaws",
    )
    parser.add_argument(
        "--no-skip-unrecoverable",
        dest="skip_unrecoverable",
        action="store_false",
        help="Do not skip unrecoverable documents",
    )
    parser.add_argument(
        "--max-doc-size",
        type=int,
        default=300000,
        help="Maximum XML file size in bytes to process",
    )
    parser.add_argument(
        "--force-large",
        action="store_true",
        default=False,
        help="Force processing documents exceeding max-doc-size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = WORKSPACE_DIR / report_path
    if not report_path.exists() and (WORKSPACE_DIR / "data" / "review_report.json").exists():
        report_path = WORKSPACE_DIR / "data" / "review_report.json"

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = WORKSPACE_DIR / raw_dir

    annotated_dir = Path(args.annotated_dir)
    if not annotated_dir.is_absolute():
        annotated_dir = WORKSPACE_DIR / annotated_dir

    branch_dir = Path(args.branch_dir)
    if not branch_dir.is_absolute():
        branch_dir = WORKSPACE_DIR / branch_dir

    if branch_dir.resolve() == annotated_dir.resolve():
        raise SystemExit(
            "❌ Error: --branch-dir cannot be the same as --annotated-dir.\n"
            "Directly overwriting original files is disabled."
        )

    out_report = Path(args.out_report)
    if not out_report.is_absolute():
        out_report = WORKSPACE_DIR / out_report

    run_batch_revision(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=annotated_dir,
        branch_dir=branch_dir,
        filter_decision=args.filter_decision,
        model=args.model.strip() if args.model else EDITOR_MODEL,
        provider=args.provider.strip() if args.provider else EDITOR_PROVIDER,
        reviewer_model=args.reviewer_model.strip() if args.reviewer_model else REVIEWER_MODEL,
        reviewer_provider=args.reviewer_provider.strip() if args.reviewer_provider else REVIEWER_PROVIDER,
        editor_thinking=args.editor_thinking.strip() if args.editor_thinking else EDITOR_THINKING,
        reviewer_thinking=args.reviewer_thinking.strip() if args.reviewer_thinking else REVIEWER_THINKING,
        concurrency=args.concurrency,
        limit=args.limit,
        max_passes=args.max_passes,
        granularity=args.granularity,
        target_doc=args.target_doc,
        auto_save=args.auto_save,
        dry_run=args.dry_run,
        out_report=out_report,
        reuse_prerun_logs=args.reuse_prerun_logs,
        resume_branch=args.resume_branch,
        retry_unresolved=args.retry_unresolved,
        auto_reconcile=args.auto_reconcile,
        skip_unrecoverable=args.skip_unrecoverable,
        max_doc_size=args.max_doc_size,
        force_large=args.force_large,
    )


if __name__ == "__main__":
    main()
