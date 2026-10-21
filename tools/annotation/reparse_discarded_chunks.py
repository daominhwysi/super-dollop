#!/usr/bin/env python3
"""
reparse_discarded_chunks.py — High-Performance Multi-Shard Discard Reparser.

Reparses discarded chunk annotations in sequence_labelling_annotated
using gpt-6-luna via the aixingialaire provider.
Supports sharded parallel execution in tmux sessions.
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

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_THINKING,
    get_provider_base_url,
    get_provider_api_key,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent
from tools.review.edit_review_set import reconcile_repaired_exam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse discarded chunk annotations with gpt-6-luna via aixingialaire."
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
        help="Path to sequence_labelling_annotated dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-6-luna",
        help="Parser model name (default: gpt-6-luna).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="aixingialaire",
        help="LLM provider name (default: aixingialaire).",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="low",
        help="Thinking effort for parser (default: low).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to progress/summary JSON report (defaults to artifacts/reports/reparse_chunks_shard_{id}.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process in this shard (for testing).",
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
        help="Force reparse all chunks from scratch without resuming.",
    )
    return parser.parse_args()


def discover_discarded_chunks(annotated_dir: Path) -> List[Dict[str, Any]]:
    """Discovers all 218 chunk_*.audit.json files marked DISCARD in canonical directory."""
    targets: List[Dict[str, Any]] = []
    audit_files = sorted(annotated_dir.glob("*/chunks/chunk_*.audit.json"))

    for a_path in audit_files:
        try:
            data = json.loads(a_path.read_text(encoding="utf-8"))
            if data.get("decision") == "DISCARD":
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
                        "initial_score": data.get("overall_score", 0.0),
                        "initial_decision": "DISCARD",
                        "discard_reasons": data.get("discard_reasons", []),
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
        or reports_dir / f"reparse_chunks_shard_{shard_id}.json"
    )

    all_targets = discover_discarded_chunks(annotated_dir)
    print("=" * 70)
    print(f"🚀 REPARSE DISCARDED CHUNKS — SHARD {shard_id}/{num_shards}")
    print("=" * 70)
    print(f"  Annotated Dir     : {annotated_dir}")
    print(f"  Total DISCARD     : {len(all_targets)} chunks across dataset")
    print(f"  Shard Index       : {shard_id} of {num_shards}")
    print(f"  Model             : {args.model}")
    print(f"  Provider          : {args.provider}")
    print(f"  Thinking          : {args.thinking}")
    print(f"  Checkpoint File   : {out_report_path}")

    # Partition targets for this shard (interleaved partitioning)
    shard_targets = [t for i, t in enumerate(all_targets) if i % num_shards == shard_id]
    if args.limit:
        shard_targets = shard_targets[:args.limit]

    print(f"  Assigned Chunks   : {len(shard_targets)} chunks in this shard")
    print("=" * 70)

    # Load existing checkpoint if resuming
    processed_results: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_report_path.exists():
        try:
            prev_data = json.loads(out_report_path.read_text(encoding="utf-8"))
            for r in prev_data.get("results", []):
                if r.get("doc_id") and r.get("status") in ("PASS", "NEEDS_REVISION", "DISCARD", "ERROR"):
                    processed_results[r["doc_id"]] = r
            print(f"  ⚡ Loaded checkpoint with {len(processed_results)} already processed chunk(s).")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not read checkpoint {out_report_path}: {e}")

    # Initialize parser worker & reviewer
    worker = ParserAgentWorker(
        model=args.model,
        provider=args.provider,
        thinking=args.thinking,
    )
    reviewer = AnnotationReviewerAgent()

    results: List[Dict[str, Any]] = list(processed_results.values())
    processed_doc_ids = set(processed_results.keys())

    def save_checkpoint():
        summary = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "provider": args.provider,
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

        if doc_id in processed_doc_ids:
            print(f"[{idx}/{len(shard_targets)}] ⚡ [Cached] {doc_id} -> {processed_results[doc_id].get('new_decision')}")
            continue

        json_path = Path(target["json_path"])
        xml_path = Path(target["xml_path"])
        audit_path = Path(target["audit_path"])

        try:
            c_data = json.loads(json_path.read_text(encoding="utf-8"))
            raw_chunk_text = c_data.get("raw_chunk_text", "")
            if not raw_chunk_text.strip():
                print(f"[{idx}/{len(shard_targets)}] ⚠️ [Empty Raw] {doc_id} has empty raw_chunk_text. Skipping.")
                continue

            chunk_idx = c_data.get("chunk_index", 0)
            page_ids = c_data.get("page_ids", [])
            overlap_page_ids = c_data.get("overlap_page_ids", [])
            page_offset_ranges = c_data.get("page_offset_ranges", [])

            t0 = time.time()
            parse_res = worker.process_chunk(
                raw_chunk_text,
                chunk_index=chunk_idx,
                page_ids=page_ids,
                overlap_page_ids=overlap_page_ids,
                page_offset_ranges=page_offset_ranges,
            )
            duration = round(time.time() - t0, 1)

            raw_xml = parse_res.get("raw_xml", "")
            if not raw_xml.strip():
                raise ValueError("Parser returned empty raw_xml output")

            clean_res = XMLCleaner.clean(raw_xml)
            cleaned_xml = clean_res.cleaned_xml

            # Write updated XML
            xml_path.write_text(cleaned_xml, encoding="utf-8")

            # Write updated chunk JSON metadata
            c_data.update(parse_res)
            c_data["raw_xml"] = cleaned_xml
            c_data["reparsed_at"] = datetime.now(timezone.utc).isoformat()
            c_data["reparse_model"] = args.model
            c_data["reparse_provider"] = args.provider
            c_data["reparse_duration_seconds"] = duration
            json_path.write_text(json.dumps(c_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # Run deterministic quality review
            audit_report = reviewer.review_document(
                cleaned_xml,
                raw_ocr_text=raw_chunk_text,
                use_llm=False,
            )
            audit_path.write_text(
                json.dumps(audit_report.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            new_decision = audit_report.decision.value if hasattr(audit_report.decision, "value") else str(audit_report.decision)
            new_score = audit_report.overall_score

            rec_item = {
                "doc_id": doc_id,
                "exam_id": exam_id,
                "chunk_stem": chunk_stem,
                "status": new_decision,
                "initial_decision": "DISCARD",
                "initial_score": target["initial_score"],
                "new_decision": new_decision,
                "new_score": new_score,
                "duration_seconds": duration,
                "questions_count": len(parse_res.get("questions", [])),
                "xml_path": str(xml_path),
            }
            results.append(rec_item)
            processed_doc_ids.add(doc_id)

            icon = "🎉" if new_decision == "PASS" else ("⚠️" if new_decision == "NEEDS_REVISION" else "❌")
            print(
                f"[{idx}/{len(shard_targets)}] {icon} [{new_decision}] {doc_id}: "
                f"Score {target['initial_score']:.1f} -> {new_score:.1f} "
                f"({len(parse_res.get('questions', []))} questions, {duration}s)"
            )

            # Check if entire exam can now be reconciled into merged.xml
            exam_dir = xml_path.parent.parent
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
            print(f"[{idx}/{len(shard_targets)}] ❌ [Error] {doc_id}: {e}")
            results.append({
                "doc_id": doc_id,
                "exam_id": exam_id,
                "chunk_stem": chunk_stem,
                "status": "ERROR",
                "error": str(e),
            })
            processed_doc_ids.add(doc_id)
            save_checkpoint()

    total_time = round(time.time() - start_time, 1)
    pass_cnt = sum(1 for r in results if r.get("new_decision") == "PASS")
    rev_cnt = sum(1 for r in results if r.get("new_decision") == "NEEDS_REVISION")
    disc_cnt = sum(1 for r in results if r.get("new_decision") == "DISCARD")

    print("\n" + "=" * 70)
    print(f"🎉 SHARD {shard_id}/{num_shards} COMPLETED in {total_time}s")
    print(f"  Processed : {len(results)}/{len(shard_targets)}")
    print(f"  PASS      : {pass_cnt}")
    print(f"  REVISION  : {rev_cnt}")
    print(f"  DISCARD   : {disc_cnt}")
    print(f"  Report    : {out_report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
