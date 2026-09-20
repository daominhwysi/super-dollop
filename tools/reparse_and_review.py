#!/usr/bin/env python3
"""
End-to-End Pipeline to Reparse Discarded & Filtered Documents from Scratch
using Gemini 3.8 Flash High, followed by Quality Auditing with GPT-5.6 Luna High.

Workflow:
  1. Discovers documents from review_report.json matching failure decisions
     (DISCARD, NEEDS_REVISION).
  2. Reparses each document from scratch from raw OCR markdown using
     Gemini 3.8 Flash High (provider: agy) via greedy chunking and reconciliation.
  3. Audits and rates the newly generated ground-truth XML using
     GPT-5.6 Luna High (provider: codex) via hybrid deterministic + semantic review.
  4. Generates a comprehensive before-and-after comparison report.

Usage:
  # Reparse & review all discarded and filtered documents:
  uv run python tools/reparse_and_review.py

  # Test on 2 documents first:
  uv run python tools/reparse_and_review.py --limit 2

  # Filter to a specific document:
  uv run python tools/reparse_and_review.py --doc-id exam-007

  # Reparse only discards:
  uv run python tools/reparse_and_review.py --filter discards_only

  # Dry-run to preview targets without executing:
  uv run python tools/reparse_and_review.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure workspace root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tqdm import tqdm

from sequence_labelling.config import (
    WORKSPACE_DIR,
    ARTIFACTS_DIR,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
    REVIEWER_MIN_SCORE,
    get_provider_base_url,
)
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    ReviewReport,
    ReviewDecision,
)
from tools.annotate_sequence_labelling_dataset import process_single_document


def load_failure_targets(
    report_path: Path,
    raw_dir: Path,
    annotated_dir: Path,
    filter_decisions: Set[str],
) -> List[Dict[str, Any]]:
    """Loads target documents from review_report.json matching specified decisions."""
    json_path = report_path.with_suffix(".json") if report_path.suffix == ".md" else report_path
    if not json_path.exists():
        print(f"❌ Error: Report file '{json_path}' not found.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = []
    reports = data.get("reports", [])

    for r in reports:
        decision = r.get("decision", "")
        if decision not in filter_decisions:
            continue

        doc_id = r.get("doc_id")
        raw_p = r.get("raw_file_path")
        xml_p = r.get("file_path")

        raw_path = Path(raw_p) if raw_p else None
        if not (raw_path and raw_path.exists()):
            # Fallback relative to raw_dir
            if raw_p:
                cand = raw_dir / Path(raw_p).name
                if cand.exists():
                    raw_path = cand

            if not (raw_path and raw_path.exists()) and xml_p:
                xml_path = Path(xml_p)
                try:
                    rel_to_annot = xml_path.relative_to(annotated_dir.resolve())
                except ValueError:
                    rel_to_annot = Path(xml_p)
                    if str(rel_to_annot).startswith("data/sequence_labelling_annotated/"):
                        rel_to_annot = Path(str(rel_to_annot)[len("data/sequence_labelling_annotated/"):])

                doc_stem = rel_to_annot.parent if rel_to_annot.name in ["merged.xml", "merged.json"] else rel_to_annot.with_suffix("")
                cand_raw = raw_dir / f"{doc_stem}.md"
                if cand_raw.exists():
                    raw_path = cand_raw

        if not (raw_path and raw_path.exists()):
            print(f"⚠️ [Warning] Skipping '{doc_id}': could not locate raw OCR markdown.")
            continue

        raw_path = raw_path.resolve()
        resolved_raw_dir = raw_dir.resolve()
        try:
            rel_path = raw_path.relative_to(resolved_raw_dir)
        except ValueError:
            rel_path = Path(raw_path.name)

        targets.append({
            "doc_id": doc_id,
            "raw_path": raw_path,
            "rel_path": rel_path,
            "prior_decision": decision,
            "prior_score": float(r.get("overall_score", 0.0)),
            "prior_grade": r.get("grade", "F"),
            "discard_reasons": r.get("discard_reasons", []),
            "prior_issues_count": len(r.get("issues", [])),
        })

    return targets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reparse failed documents using Gemini 3.8 Flash High & Review with GPT-5.6 Luna High"
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default="data/review_report.json",
        help="Path to review report JSON (default: data/review_report.json)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"),
        help="Path to raw markdown input directory (default: data/sequence_labelling_input_data)",
    )
    parser.add_argument(
        "--output-dir",
        "--annotated-dir",
        dest="output_dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_annotated"),
        help="Output directory for reparsed XML (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_mode",
        choices=["all_failures", "discards_only", "revision_only"],
        default="all_failures",
        help="Target failures to reparse: all_failures (DISCARD + NEEDS_REVISION), discards_only, or revision_only",
    )
    parser.add_argument(
        "--doc-id",
        "--target-doc",
        dest="target_doc",
        type=str,
        default=None,
        help="Filter execution to a specific document ID or substring (e.g. exam-007)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "reparse-only", "review-only"],
        default="all",
        help="Pipeline stage: 'all' (reparse + review), 'reparse-only', or 'review-only' (default: all)",
    )
    # Parser Configuration (Gemini 3.8 Flash High)
    parser.add_argument(
        "--parser-model",
        type=str,
        default="gpt-5.6-sol",
        help="Parser model name ",
    )
    parser.add_argument(
        "--parser-provider",
        type=str,
        default="codex",
        help="Parser LLM provider ",
    )
    parser.add_argument(
        "--parser-thinking",
        type=str,
        default="medium",
        help="Parser reasoning/thinking effort",
    )
    # Reviewer Configuration (GPT-5.6 Luna High)
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default="gpt-5.6-sol",
        help="Reviewer model name ",
    )
    parser.add_argument(
        "--reviewer-provider",
        type=str,
        default="codex",
        help="Reviewer LLM provider",
    )
    parser.add_argument(
        "--reviewer-thinking",
        type=str,
        default="medium",
        help="Reviewer reasoning/thinking effort ",
    )
    # Operational Flags
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=2,
        help="Concurrency workers for parsing (default: 2 to match agy lock)",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="artifacts/ocr_logs/reparse_and_review_report.json",
        help="Path to output resolution report JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview target documents and exit without modifying files",
    )
    return parser.parse_args()


def reparse_document(
    target: Dict[str, Any],
    output_dir: Path,
    model: str,
    provider: str,
    thinking: str,
    concurrency: int,
) -> Dict[str, Any]:
    """Reparses a single document from raw OCR markdown from scratch."""
    raw_path: Path = target["raw_path"]
    rel_path: Path = target["rel_path"]

    start_time = time.time()
    res = process_single_document(
        file_path=raw_path,
        rel_path=rel_path,
        out_dir=output_dir,
        parser_model=model,
        parser_provider=provider,
        target_tokens=CHUNKER_TARGET_TOKENS,
        max_tokens=CHUNKER_MAX_TOKENS,
        overlap_pages=CHUNKER_OVERLAP_PAGES,
        concurrency=concurrency,
        parser_thinking=thinking,
    )
    duration = time.time() - start_time

    doc_out_dir = output_dir / rel_path.parent / rel_path.stem
    merged_xml_path = doc_out_dir / "merged.xml"
    merged_json_path = doc_out_dir / "merged.json"

    success = merged_xml_path.exists() and merged_xml_path.stat().st_size > 0

    return {
        "success": success,
        "questions_count": res.get("questions_count", 0),
        "stimuli_count": res.get("stimuli_count", 0),
        "duration_seconds": round(duration, 2),
        "merged_xml_path": merged_xml_path,
        "merged_json_path": merged_json_path,
        "merge_status": res.get("merge_status", "unknown"),
    }


def review_document(
    reviewer: AnnotationReviewerAgent,
    target: Dict[str, Any],
    xml_path: Path,
) -> ReviewReport:
    """Runs quality audit on the newly generated XML using GPT-5.6 Luna High."""
    raw_path: Path = target["raw_path"]
    doc_id = target["doc_id"]

    annotated_xml = xml_path.read_text(encoding="utf-8")
    raw_ocr_text = raw_path.read_text(encoding="utf-8")

    return reviewer.review_document(
        xml_content=annotated_xml,
        raw_ocr_text=raw_ocr_text,
        doc_id=doc_id,
        file_path=str(xml_path),
        raw_file_path=str(raw_path),
        use_llm=True,
    )


def main():
    args = parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = WORKSPACE_DIR / report_path

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = WORKSPACE_DIR / raw_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE_DIR / output_dir

    out_report_path = Path(args.out_report)
    if not out_report_path.is_absolute():
        out_report_path = WORKSPACE_DIR / out_report_path
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    filter_map = {
        "all_failures": {"DISCARD", "NEEDS_REVISION"},
        "discards_only": {"DISCARD"},
        "revision_only": {"NEEDS_REVISION"},
    }
    filter_decisions = filter_map[args.filter_mode]

    targets = load_failure_targets(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=output_dir,
        filter_decisions=filter_decisions,
    )

    if args.target_doc:
        query = args.target_doc.strip()
        targets = [
            t for t in targets
            if query in str(t["doc_id"]) or query in str(t["rel_path"])
        ]

    if args.limit:
        targets = targets[: args.limit]

    print("=" * 75)
    print("🚀 AZOZO REPARSE & REVIEW PIPELINE")
    print("=" * 75)
    print(f"  Source Report     : {report_path}")
    print(f"  Raw OCR Dir       : {raw_dir}")
    print(f"  Output XML Dir    : {output_dir}")
    print(f"  Filter Selection  : {args.filter_mode} ({', '.join(sorted(filter_decisions))})")
    print(f"  Total Targets     : {len(targets)} document(s)")
    print(f"  Pipeline Stage    : {args.stage.upper()}")
    print(f"  Parser Model      : {args.parser_model} (effort: {args.parser_thinking})")
    print(f"  Parser Provider   : {args.parser_provider}")
    print(f"  Reviewer Model    : {args.reviewer_model} (effort: {args.reviewer_thinking})")
    print(f"  Reviewer Provider : {args.reviewer_provider}")
    print(f"  Concurrency       : {args.concurrency} worker(s)")
    print(f"  Output Report     : {out_report_path}")
    print(f"  Execution Mode    : {'DRY-RUN (Preview Only)' if args.dry_run else 'LIVE EXECUTION'}")
    print("=" * 75)

    if not targets:
        print("✅ No target documents found to process.")
        sys.exit(0)

    if args.dry_run:
        print("\n📋 Target Document Preview (Dry Run):")
        for idx, t in enumerate(targets[:20], start=1):
            print(f"  {idx:2d}. {t['doc_id']} | Prior: {t['prior_decision']} ({t['prior_score']} pts, Grade {t['prior_grade']}) -> {t['rel_path']}")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more documents.")
        print("\nDry run completed successfully.")
        sys.exit(0)

    # Initialize Reviewer Agent if reviewing is enabled
    reviewer = None
    if args.stage in ("all", "review-only"):
        print(f"\n🔍 Initializing Reviewer Agent ({args.reviewer_model}, provider={args.reviewer_provider}, thinking={args.reviewer_thinking})...")
        reviewer = AnnotationReviewerAgent(
            model=args.reviewer_model,
            provider=args.reviewer_provider,
            thinking=args.reviewer_thinking,
        )

    results: List[Dict[str, Any]] = []
    pass_count = 0
    needs_revision_count = 0
    discard_count = 0
    reparse_failed_count = 0
    start_all = time.time()

    pbar = tqdm(targets, desc="Reparse & Review", unit="doc")

    for target in pbar:
        doc_id = target["doc_id"]
        rel_path = target["rel_path"]
        prior_dec = target["prior_decision"]
        prior_sc = target["prior_score"]

        pbar.set_postfix({"doc": str(rel_path.stem)[:18], "prior": prior_dec[:3]})

        doc_out_dir = output_dir / rel_path.parent / rel_path.stem
        merged_xml_path = doc_out_dir / "merged.xml"

        reparse_info: Dict[str, Any] = {}

        # ── Step 1: Reparse from Scratch ──────────────────────────────────
        if args.stage in ("all", "reparse-only"):
            try:
                reparse_info = reparse_document(
                    target=target,
                    output_dir=output_dir,
                    model=args.parser_model,
                    provider=args.parser_provider,
                    thinking=args.parser_thinking,
                    concurrency=args.concurrency,
                )
                if not reparse_info["success"]:
                    reparse_failed_count += 1
                    tqdm.write(f"❌ [Reparse Failed] {doc_id} -> XML output not generated.")
                    results.append({
                        "doc_id": doc_id,
                        "rel_path": str(rel_path),
                        "prior_decision": prior_dec,
                        "prior_score": prior_sc,
                        "reparse_success": False,
                        "new_decision": "REPARSE_FAILED",
                        "new_score": 0.0,
                    })
                    continue
            except Exception as e:
                reparse_failed_count += 1
                tqdm.write(f"❌ [Reparse Error] {doc_id}: {e}")
                results.append({
                    "doc_id": doc_id,
                    "rel_path": str(rel_path),
                    "prior_decision": prior_dec,
                    "prior_score": prior_sc,
                    "reparse_success": False,
                    "error": str(e),
                    "new_decision": "REPARSE_FAILED",
                    "new_score": 0.0,
                })
                continue

        # ── Step 2: Quality Review with GPT-5.6 Luna High ─────────────────
        if args.stage in ("all", "review-only"):
            if not merged_xml_path.exists():
                tqdm.write(f"⚠️ [Skip Review] {doc_id}: {merged_xml_path} does not exist.")
                continue

            try:
                review_rep: ReviewReport = review_document(
                    reviewer=reviewer,
                    target=target,
                    xml_path=merged_xml_path,
                )

                new_dec = review_rep.decision.value
                new_sc = review_rep.overall_score
                delta = new_sc - prior_sc

                if new_dec == ReviewDecision.PASS.value:
                    pass_count += 1
                    status_icon = "🎉 [PASSED]"
                elif new_dec == ReviewDecision.NEEDS_REVISION.value:
                    needs_revision_count += 1
                    status_icon = "⚠️ [NEEDS REVISION]"
                else:
                    discard_count += 1
                    status_icon = "❌ [DISCARD]"

                delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
                tqdm.write(
                    f"  {status_icon} {doc_id} ({rel_path.stem[:25]}): "
                    f"{prior_dec} ({prior_sc:.1f}) -> {new_dec} ({new_sc:.1f}) [{delta_str} pts, Grade {review_rep.grade}]"
                )

                results.append({
                    "doc_id": doc_id,
                    "rel_path": str(rel_path),
                    "prior_decision": prior_dec,
                    "prior_score": prior_sc,
                    "prior_grade": target["prior_grade"],
                    "reparse_success": True,
                    "reparse_duration_sec": reparse_info.get("duration_seconds", 0.0),
                    "questions_extracted": reparse_info.get("questions_count", 0),
                    "new_decision": new_dec,
                    "new_score": new_sc,
                    "new_grade": review_rep.grade,
                    "score_delta": round(delta, 1),
                    "deterministic_score": review_rep.deterministic_score,
                    "discard_reasons": review_rep.discard_reasons,
                    "issues_count": len(review_rep.issues),
                    "issues": [iss.model_dump(mode="json") for iss in review_rep.issues[:10]],
                })
            except Exception as rev_err:
                tqdm.write(f"❌ [Review Error] {doc_id}: {rev_err}")
                results.append({
                    "doc_id": doc_id,
                    "rel_path": str(rel_path),
                    "prior_decision": prior_dec,
                    "prior_score": prior_sc,
                    "reparse_success": True,
                    "review_error": str(rev_err),
                    "new_decision": "REVIEW_ERROR",
                    "new_score": 0.0,
                })
        else:
            # Reparse-only result recording
            tqdm.write(f"  ✅ [Reparsed] {doc_id}: {reparse_info.get('questions_count', 0)} questions in {reparse_info.get('duration_seconds', 0)}s")
            results.append({
                "doc_id": doc_id,
                "rel_path": str(rel_path),
                "prior_decision": prior_dec,
                "prior_score": prior_sc,
                "reparse_success": True,
                "questions_extracted": reparse_info.get("questions_count", 0),
                "duration_seconds": reparse_info.get("duration_seconds", 0),
            })

    total_duration = time.time() - start_all
    total_processed = len(results)

    # ── Summary & Report Generation ───────────────────────────────────────
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_targets": len(targets),
        "total_processed": total_processed,
        "pipeline_stage": args.stage,
        "parser_model": args.parser_model,
        "parser_provider": args.parser_provider,
        "reviewer_model": args.reviewer_model,
        "reviewer_provider": args.reviewer_provider,
        "reviewer_thinking": args.reviewer_thinking,
        "total_duration_seconds": round(total_duration, 2),
        "reparse_failures": reparse_failed_count,
        "outcomes": {
            "PASS": pass_count,
            "NEEDS_REVISION": needs_revision_count,
            "DISCARD": discard_count,
            "pass_rate_pct": round((pass_count / max(1, total_processed)) * 100, 1),
            "salvaged_rate_pct": round(((pass_count + needs_revision_count) / max(1, total_processed)) * 100, 1),
        },
        "results": results,
    }

    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write accompanying Markdown report
    md_report_path = out_report_path.with_suffix(".md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Azozo Reparse & Review Resolution Report\n\n")
        f.write(f"- **Generated At**: {summary['generated_at']}\n")
        f.write(f"- **Parser**: `{args.parser_model}` ({args.parser_provider}, effort: `{args.parser_thinking}`)\n")
        f.write(f"- **Reviewer**: `{args.reviewer_model}` ({args.reviewer_provider}, effort: `{args.reviewer_thinking}`)\n")
        f.write(f"- **Total Targets**: {len(targets)}\n")
        f.write(f"- **Duration**: {total_duration:.1f}s\n\n")

        f.write("## 📊 Summary Outcome\n\n")
        f.write("| Status | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| 🟢 **PASS** | {pass_count} | {summary['outcomes']['pass_rate_pct']}% |\n")
        f.write(f"| 🟡 **NEEDS_REVISION** | {needs_revision_count} | {round((needs_revision_count/max(1, total_processed))*100, 1)}% |\n")
        f.write(f"| 🔴 **DISCARD** | {discard_count} | {round((discard_count/max(1, total_processed))*100, 1)}% |\n")
        f.write(f"| ❌ **Reparse Failed** | {reparse_failed_count} | {round((reparse_failed_count/max(1, total_processed))*100, 1)}% |\n\n")

        f.write("## 📑 Detailed Document Results\n\n")
        f.write("| Doc ID | Prior Decision | Prior Score | Reparse | New Decision | New Score | Delta |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for res in results:
            doc = res["doc_id"]
            p_dec = res["prior_decision"]
            p_sc = res["prior_score"]
            rep_ok = "✅" if res.get("reparse_success") else "❌"
            n_dec = res.get("new_decision", "-")
            n_sc = res.get("new_score", 0.0)
            delta = res.get("score_delta", 0.0)
            d_str = f"+{delta}" if delta >= 0 else f"{delta}"
            f.write(f"| `{doc}` | {p_dec} | {p_sc:.1f} | {rep_ok} | **{n_dec}** | {n_sc:.1f} | {d_str} |\n")

    print("\n" + "=" * 75)
    print("🎉 REPARSE & REVIEW RUN COMPLETED!")
    print("=" * 75)
    print(f"  Total Processed        : {total_processed} document(s)")
    print(f"  Duration               : {total_duration:.1f}s")
    if args.stage in ("all", "review-only"):
        print(f"  🟢 Promoted to PASS    : {pass_count} ({summary['outcomes']['pass_rate_pct']}%)")
        print(f"  🟡 Promoted to REVISION: {needs_revision_count} ({round((needs_revision_count/max(1, total_processed))*100, 1)}%)")
        print(f"  🔴 Retained DISCARD    : {discard_count} ({round((discard_count/max(1, total_processed))*100, 1)}%)")
        print(f"  ✨ Total Salvage Rate  : {summary['outcomes']['salvaged_rate_pct']}% (PASS + REVISION)")
    print(f"  JSON Report Saved      : {out_report_path}")
    print(f"  Markdown Report Saved  : {md_report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
