#!/usr/bin/env python3
"""
Reparse Discarded and Needs-Revision Document Sets from Quality Review Report.
Uses Codex API Parser (or configured provider) and the Long-Context Chunker Pipeline.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
    get_provider_base_url,
)
from tools.annotate_sequence_labelling_dataset import process_single_document


def load_reparse_targets(
    report_path: Path,
    raw_dir: Path,
    annotated_dir: Path,
    filter_decisions: List[str],
) -> List[Dict[str, Any]]:
    """
    Parses review_report.json (or review_report.md) to discover documents matching filter decisions.
    Resolves raw source markdown files.
    """
    targets = []

    # If review_report.json exists, read structured reports
    json_report = report_path.with_suffix(".json") if report_path.suffix == ".md" else report_path
    if json_report.exists():
        with open(json_report, "r", encoding="utf-8") as f:
            data = json.load(f)

        reports = data.get("reports", [])
        for r in reports:
            decision = r.get("decision", "")
            if decision in filter_decisions:
                raw_file_str = r.get("raw_file_path")
                raw_path = Path(raw_file_str) if raw_file_str else None

                # Resolve if raw_file_path was missing or relative
                if not (raw_path and raw_path.exists()):
                    xml_p = r.get("file_path")
                    if xml_p:
                        xml_path = Path(xml_p)
                        if xml_path.is_absolute():
                            try:
                                rel_to_annot = xml_path.relative_to(annotated_dir)
                            except ValueError:
                                rel_to_annot = xml_path.relative_to(WORKSPACE_DIR / "data" / "sequence_labelling_annotated")
                        else:
                            rel_to_annot = Path(xml_p)
                            if str(rel_to_annot).startswith("data/sequence_labelling_annotated/"):
                                rel_to_annot = Path(str(rel_to_annot)[len("data/sequence_labelling_annotated/"):])

                        # exam_dir is e.g. DGNL_HSA/Hoa_hoc/exam_229
                        doc_stem = rel_to_annot.parent if rel_to_annot.name in ["merged.xml", "merged.json"] else rel_to_annot.with_suffix("")
                        candidate_raw = raw_dir / f"{doc_stem}.md"
                        if candidate_raw.exists():
                            raw_path = candidate_raw

                if raw_path and raw_path.exists():
                    raw_path = raw_path.resolve()
                    resolved_raw_dir = raw_dir.resolve()
                    rel_path = raw_path.relative_to(resolved_raw_dir)
                    targets.append({
                        "doc_id": r.get("doc_id"),
                        "raw_file": raw_path,
                        "rel_path": rel_path,
                        "decision": decision,
                        "overall_score": r.get("overall_score"),
                        "grade": r.get("grade"),
                        "discard_reasons": r.get("discard_reasons", []),
                    })
                else:
                    print(f"⚠️ [Warning] Cannot find raw source file for document '{r.get('doc_id')}' (decision: {decision})")

    return targets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reparse discarded and needs-revision documents from review report using Codex API parser"
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default="backend/logs/review_report.json",
        help="Path to review report JSON or Markdown (default: backend/logs/review_report.json)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"),
        help="Path to raw OCR markdown input directory (default: data/sequence_labelling_input_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_annotated"),
        help="Output directory to save re-annotated files (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_mode",
        choices=["all_failures", "discards_only", "revision_only"],
        default="all_failures",
        help="Which target sets to reparse: all_failures (DISCARD + NEEDS_REVISION), discards_only, or revision_only (default: all_failures)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PARSER_MODEL,
        help=f"Parser model name (default: {PARSER_MODEL})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=PARSER_PROVIDER,
        help=f"Parser LLM provider (default: {PARSER_PROVIDER})",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Concurrency for chunk worker threads (default: 4)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of documents to reparse",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip Role B Validator pass to run fast 1-pass Role A parsing",
    )
    parser.add_argument(
        "--doc-id",
        "--target-doc",
        dest="target_doc",
        type=str,
        default=None,
        help="Filter re-parsing to a specific document ID or substring (e.g. exam_424 or Other/Lich_su/exam_424)",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude_docs",
        type=str,
        default=None,
        help="Comma-separated list of document IDs or filenames to exclude from re-parsing (e.g. exam_424)",
    )
    return parser.parse_args()


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

    if not report_path.exists():
        # Fallback to .json if .md was passed
        if report_path.suffix == ".md" and report_path.with_suffix(".json").exists():
            report_path = report_path.with_suffix(".json")
        else:
            print(f"❌ Error: Report file '{report_path}' does not exist.")
            sys.exit(1)

    if not raw_dir.exists():
        print(f"❌ Error: Raw source directory '{raw_dir}' does not exist.")
        sys.exit(1)

    filter_map = {
        "all_failures": ["DISCARD", "NEEDS_REVISION"],
        "discards_only": ["DISCARD"],
        "revision_only": ["NEEDS_REVISION"],
    }
    filter_decisions = filter_map[args.filter_mode]

    targets = load_reparse_targets(report_path, raw_dir, output_dir, filter_decisions)

    # Filter by specific target_doc if provided
    if args.target_doc:
        doc_query = args.target_doc.strip()
        targets = [
            t for t in targets
            if doc_query in str(t.get("doc_id", "")) or doc_query in str(t.get("rel_path", ""))
        ]

    # Exclude documents if specified
    if args.exclude_docs:
        exclude_list = [e.strip() for e in args.exclude_docs.split(",") if e.strip()]
        targets = [
            t for t in targets
            if not any(
                ex in str(t.get("doc_id", "")) or ex in str(t.get("rel_path", ""))
                for ex in exclude_list
            )
        ]

    if args.limit:
        targets = targets[: args.limit]

    parser_base_url = get_provider_base_url(args.provider)

    print("=" * 70)
    print("🔄 AZOZO SEQUENCE LABELLING RE-PARSING ENGINE")
    print("=" * 70)
    print(f"  Source Report     : {report_path}")
    print(f"  Raw Input Dir     : {raw_dir}")
    print(f"  Output Annotated  : {output_dir}")
    print(f"  Filter Mode       : {args.filter_mode} ({', '.join(filter_decisions)})")
    print(f"  Total Targets     : {len(targets)} document(s)")
    print(f"  Parser Provider   : {args.provider}")
    print(f"  Parser Model      : {args.model}")
    print(f"  Base URL          : {parser_base_url if parser_base_url else '(SDK Native / Codex)'}")
    print(f"  Concurrency       : {args.concurrency} worker thread(s)")
    print(f"  Validator Mode    : {'Single-Pass (Skipped)' if args.skip_validator else 'Two-Pass (Role A + Role B)'}")
    print("=" * 70)

    if not targets:
        print("✅ No target documents found to reparse.")
        sys.exit(0)

    success_count = 0
    failed_count = 0

    pbar = tqdm(targets, desc="Reparsing Documents", unit="doc")
    for item in pbar:
        rel_path = item["rel_path"]
        raw_file = item["raw_file"]
        doc_id = item["doc_id"]
        decision = item["decision"]

        pbar.set_postfix({
            "doc": str(rel_path.stem)[:20],
            "type": decision[:3],
            "ok": success_count,
            "fail": failed_count,
        })

        try:
            res = process_single_document(
                file_path=raw_file,
                rel_path=rel_path,
                out_dir=output_dir,
                parser_model=args.model,
                parser_provider=args.provider,
                target_tokens=CHUNKER_TARGET_TOKENS,
                max_tokens=CHUNKER_MAX_TOKENS,
                concurrency=args.concurrency,
            )
            success_count += 1
            tqdm.write(
                f"  ✅ [Reparsed] {rel_path} ({decision}) -> {res['questions']} questions in {res['duration']}s"
            )
        except Exception as e:
            failed_count += 1
            tqdm.write(f"\n❌ [Failed] {rel_path} ({decision}): {e}")

    print("\n" + "=" * 70)
    print("🎉 REPARSING RUN COMPLETED!")
    print(f"  Total Targets Processed : {len(targets)}")
    print(f"  Successfully Reparsed   : {success_count} ({success_count/max(1, len(targets))*100:.1f}%)")
    print(f"  Failed                  : {failed_count}")
    print(f"  Output Directory        : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
