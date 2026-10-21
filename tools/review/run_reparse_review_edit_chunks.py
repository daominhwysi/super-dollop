#!/usr/bin/env python3
"""
run_reparse_review_edit_chunks.py — Unified 2-Shard Parser, Reviewer, and Surgical Editor Pipeline.

Processes all chunks in DISCARD and NEEDS_REVISION status in sequence_labelling_annotated:
- Model: gpt-6-luna via provider aixingialaire
- Reviewer reasoning_effort: high
- Editor reasoning_effort: medium
- Max revision retries: 5 rounds per chunk
- Automatic multi-shard partitioning for parallel tmux execution
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent
from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop
from tools.review.edit_review_set import reconcile_repaired_exam


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reparse_review_edit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunk-level Parser + Reviewer (high) + Editor (medium) loop with gpt-6-luna."
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based shard index (e.g. 0 or 1).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=2,
        help="Total number of parallel shards (default: 2).",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated",
        help="Path to sequence_labelling_annotated directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-6-luna",
        help="LLM model name (default: gpt-6-luna).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="aixingialaire",
        help="LLM provider name (default: aixingialaire).",
    )
    parser.add_argument(
        "--reviewer-thinking",
        type=str,
        default="high",
        help="Reviewer reasoning effort (default: high).",
    )
    parser.add_argument(
        "--editor-thinking",
        type=str,
        default="medium",
        help="Editor reasoning effort (default: medium).",
    )
    parser.add_argument(
        "--parser-thinking",
        type=str,
        default="low",
        help="Parser reasoning effort for discarded chunk reparse (default: low).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum revision loop rounds per chunk (default: 5).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to progress/summary JSON report.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process in this shard (for smoke testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous checkpoint if available (default: True).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force reprocess without resuming from checkpoint.",
    )
    return parser.parse_args()


def discover_defective_chunks(annotated_dir: Path) -> List[Dict[str, Any]]:
    """Discovers all chunk_*.audit.json files marked DISCARD or NEEDS_REVISION."""
    targets: List[Dict[str, Any]] = []
    audit_files = sorted(annotated_dir.glob("*/chunks/chunk_*.audit.json"))

    for a_path in audit_files:
        try:
            data = json.loads(a_path.read_text(encoding="utf-8"))
            decision = data.get("decision")
            if decision in ("DISCARD", "NEEDS_REVISION"):
                xml_path = a_path.with_name(a_path.name.replace(".audit.json", ".xml"))
                json_path = a_path.with_name(a_path.name.replace(".audit.json", ".json"))
                if json_path.exists():
                    exam_id = a_path.parent.parent.name
                    chunk_stem = a_path.name.replace(".audit.json", "")
                    targets.append({
                        "doc_id": f"{exam_id}::{chunk_stem}",
                        "exam_id": exam_id,
                        "chunk_stem": chunk_stem,
                        "audit_path": str(a_path.resolve()),
                        "xml_path": str(xml_path.resolve()),
                        "json_path": str(json_path.resolve()),
                        "initial_score": float(data.get("overall_score", 0.0)),
                        "initial_decision": decision,
                        "initial_issues_count": len(data.get("issues", [])),
                    })
        except Exception:
            continue

    # Deterministic sort order
    targets.sort(key=lambda x: (x["exam_id"], x["chunk_stem"]))
    return targets


def main() -> None:
    args = parse_args()
    annotated_dir = args.annotated_dir
    shard_id = args.shard_id
    num_shards = args.num_shards

    if shard_id < 0 or shard_id >= num_shards:
        print(f"❌ Error: shard_id ({shard_id}) must be in [0, {num_shards - 1}]")
        sys.exit(1)

    reports_dir = WORKSPACE_DIR / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_report_path = (
        args.output_report
        or reports_dir / f"reparse_review_edit_shard_{shard_id}.json"
    )

    all_targets = discover_defective_chunks(annotated_dir)
    discard_total = sum(1 for t in all_targets if t["initial_decision"] == "DISCARD")
    rev_total = sum(1 for t in all_targets if t["initial_decision"] == "NEEDS_REVISION")

    print("=" * 75)
    print(f"🚀 REPARSE + REVIEW (HIGH) + EDIT (MEDIUM) — SHARD {shard_id}/{num_shards}")
    print("=" * 75)
    print(f"  Annotated Dir       : {annotated_dir}")
    print(f"  Total Targets       : {len(all_targets)} chunks ({discard_total} DISCARD, {rev_total} NEEDS_REVISION)")
    print(f"  Shard Index         : {shard_id} of {num_shards}")
    print(f"  Model               : {args.model}")
    print(f"  Provider            : {args.provider}")
    print(f"  Reviewer Thinking   : {args.reviewer_thinking}")
    print(f"  Editor Thinking     : {args.editor_thinking}")
    print(f"  Max Revision Rounds : {args.max_rounds}")
    print(f"  Checkpoint File     : {out_report_path}")

    # Interleaved partitioning
    shard_targets = [t for i, t in enumerate(all_targets) if i % num_shards == shard_id]
    if args.limit:
        shard_targets = shard_targets[:args.limit]

    print(f"  Assigned Chunks     : {len(shard_targets)} chunks in this shard")
    print("=" * 75)

    # Load existing checkpoint if resuming
    processed_results: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_report_path.exists():
        try:
            prev_data = json.loads(out_report_path.read_text(encoding="utf-8"))
            for r in prev_data.get("results", []):
                doc_id = r.get("doc_id")
                if doc_id and r.get("status") in ("PASS", "NEEDS_REVISION", "DISCARD", "ERROR"):
                    processed_results[doc_id] = r
            print(f"  ⚡ Loaded checkpoint with {len(processed_results)} already processed chunk(s).")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not read checkpoint {out_report_path}: {e}")

    # Initialize agents
    worker = ParserAgentWorker(
        model=args.model,
        provider=args.provider,
        thinking=args.parser_thinking,
    )
    reviewer = AnnotationReviewerAgent(
        model=args.model,
        provider=args.provider,
        thinking=args.reviewer_thinking,
    )
    editor = EditorAgent(
        model=args.model,
        provider=args.provider,
        thinking=args.editor_thinking,
        reviewer=reviewer,
    )
    revision_loop = EditorReviewerLoop(
        editor=editor,
        reviewer=reviewer,
        max_rounds=args.max_rounds,
    )

    results: List[Dict[str, Any]] = list(processed_results.values())
    processed_doc_ids = set(processed_results.keys())

    def save_checkpoint() -> None:
        summary = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "provider": args.provider,
            "reviewer_thinking": args.reviewer_thinking,
            "editor_thinking": args.editor_thinking,
            "max_rounds": args.max_rounds,
            "total_assigned": len(shard_targets),
            "completed": len(results),
            "pass_count": sum(1 for r in results if r.get("new_decision") == "PASS"),
            "needs_revision_count": sum(1 for r in results if r.get("new_decision") == "NEEDS_REVISION"),
            "discard_count": sum(1 for r in results if r.get("new_decision") == "DISCARD"),
            "error_count": sum(1 for r in results if r.get("status") == "ERROR"),
            "results": results,
        }
        out_report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    start_time = time.time()
    for idx, target in enumerate(shard_targets, start=1):
        doc_id = target["doc_id"]
        exam_id = target["exam_id"]
        chunk_stem = target["chunk_stem"]
        init_dec = target["initial_decision"]
        init_score = target["initial_score"]

        if doc_id in processed_doc_ids:
            res_item = processed_results[doc_id]
            print(f"[{idx}/{len(shard_targets)}] ⚡ [Cached] {doc_id} -> {res_item.get('new_decision')} ({res_item.get('new_score', 0):.1f})")
            continue

        json_path = Path(target["json_path"])
        xml_path = Path(target["xml_path"])
        audit_path = Path(target["audit_path"])

        t0 = time.time()
        print(f"\n[{idx}/{len(shard_targets)}] 🎯 Starting {doc_id} (Initial: {init_dec}, Score: {init_score:.1f})...")

        try:
            c_data = json.loads(json_path.read_text(encoding="utf-8"))
            raw_chunk_text = c_data.get("raw_chunk_text", "")
            if not raw_chunk_text.strip():
                print(f"    ⚠️ [Empty Raw] {doc_id} has empty raw_chunk_text. Skipping.")
                continue

            # Stage 1: Candidate XML determination
            # If DISCARD, reparse from raw OCR text with ParserWorker
            if init_dec == "DISCARD":
                print(f"    🔄 [Reparsing] Running ParserAgentWorker (gpt-6-luna) on raw chunk text...")
                chunk_idx = c_data.get("chunk_index", 0)
                page_ids = c_data.get("page_ids", [])
                overlap_page_ids = c_data.get("overlap_page_ids", [])
                page_offset_ranges = c_data.get("page_offset_ranges", [])

                parse_res = worker.process_chunk(
                    raw_chunk_text,
                    chunk_index=chunk_idx,
                    page_ids=page_ids,
                    overlap_page_ids=overlap_page_ids,
                    page_offset_ranges=page_offset_ranges,
                )
                raw_xml = parse_res.get("raw_xml", "")
                if not raw_xml.strip():
                    raise ValueError("Parser returned empty raw_xml output")
                candidate_xml = XMLCleaner.clean(raw_xml).cleaned_xml
                c_data.update(parse_res)
            else:
                # NEEDS_REVISION: use existing XML cleaned
                if xml_path.exists():
                    raw_xml = xml_path.read_text(encoding="utf-8")
                    candidate_xml = XMLCleaner.clean(raw_xml).cleaned_xml
                else:
                    raise FileNotFoundError(f"XML file not found: {xml_path}")

            # Stage 2: EditorReviewerLoop with gpt-6-luna (Reviewer: high, Editor: medium)
            print(f"    🛠️ [Revision Loop] Reviewer (high) + Editor (medium), max {args.max_rounds} rounds...")
            loop_res = revision_loop.run(
                annotated_xml=candidate_xml,
                raw_ocr_text=raw_chunk_text,
                doc_id=doc_id,
                use_llm_reviewer=True,
            )

            new_xml = loop_res.repaired_xml or candidate_xml
            new_decision = loop_res.final_decision
            new_score = loop_res.final_score
            rounds_done = loop_res.rounds_completed
            duration = round(time.time() - t0, 1)

            # Stage 3: Writeback
            # Save candidate XML if it improved or passed
            if new_decision == "PASS" or new_score >= init_score:
                xml_path.write_text(new_xml, encoding="utf-8")
                if loop_res.final_report:
                    audit_path.write_text(
                        json.dumps(loop_res.final_report.model_dump(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                c_data["raw_xml"] = new_xml
                c_data["repaired_at"] = datetime.now(timezone.utc).isoformat()
                c_data["repaired_model"] = args.model
                c_data["repaired_provider"] = args.provider
                c_data["repaired_score"] = new_score
                c_data["repaired_decision"] = new_decision
                c_data["repaired_rounds"] = rounds_done
                c_data["repaired_duration_seconds"] = duration
                json_path.write_text(json.dumps(c_data, ensure_ascii=False, indent=2), encoding="utf-8")

            rec_item = {
                "doc_id": doc_id,
                "exam_id": exam_id,
                "chunk_stem": chunk_stem,
                "status": new_decision,
                "initial_decision": init_dec,
                "initial_score": init_score,
                "new_decision": new_decision,
                "new_score": new_score,
                "rounds_completed": rounds_done,
                "duration_seconds": duration,
                "xml_path": str(xml_path),
            }
            results.append(rec_item)
            processed_doc_ids.add(doc_id)

            icon = "🎉" if new_decision == "PASS" else ("⚠️" if new_decision == "NEEDS_REVISION" else "❌")
            print(
                f"    {icon} Result: {init_dec} ({init_score:.1f}) -> {new_decision} ({new_score:.1f}) "
                f"after {rounds_done} round(s) ({duration}s)"
            )

            # Check if entire exam can now be reconciled into merged.xml
            if new_decision == "PASS":
                chunks_dir = xml_path.parent
                all_sibling_audits = list(chunks_dir.glob("chunk_*.audit.json"))
                all_pass = True
                for sa in all_sibling_audits:
                    try:
                        s_data = json.loads(sa.read_text(encoding="utf-8"))
                        if s_data.get("decision") != "PASS":
                            all_pass = False
                            break
                    except Exception:
                        all_pass = False
                        break

                if all_pass and len(all_sibling_audits) > 0:
                    try:
                        rec_info = reconcile_repaired_exam(exam_id, annotated_dir, annotated_dir, reviewer)
                        if rec_info.get("status") == "reconciled":
                            print(f"    ✨ [Full Exam Reconciled] {exam_id}: All chunks PASS -> merged.xml updated!")
                    except Exception as rec_err:
                        print(f"    ⚠️ [Reconcile Warning] {exam_id}: {rec_err}")

            save_checkpoint()

        except Exception as e:
            duration = round(time.time() - t0, 1)
            print(f"    ❌ [Error] {doc_id} after {duration}s: {e}")
            results.append({
                "doc_id": doc_id,
                "exam_id": exam_id,
                "chunk_stem": chunk_stem,
                "status": "ERROR",
                "initial_decision": init_dec,
                "initial_score": init_score,
                "error": str(e),
                "duration_seconds": duration,
            })
            processed_doc_ids.add(doc_id)
            save_checkpoint()

    total_time = round(time.time() - start_time, 1)
    pass_cnt = sum(1 for r in results if r.get("new_decision") == "PASS")
    rev_cnt = sum(1 for r in results if r.get("new_decision") == "NEEDS_REVISION")
    disc_cnt = sum(1 for r in results if r.get("new_decision") == "DISCARD")

    print("\n" + "=" * 75)
    print(f"🎉 SHARD {shard_id}/{num_shards} FINISHED in {total_time}s")
    print(f"  Total Processed : {len(results)}/{len(shard_targets)}")
    print(f"  PASS            : {pass_cnt}")
    print(f"  NEEDS_REVISION  : {rev_cnt}")
    print(f"  DISCARD         : {disc_cnt}")
    print(f"  Report File     : {out_report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
