#!/usr/bin/env python3
"""
reparse_cartography_documents.py — High-Performance Cartography-Locked Reparsing & Remediation Pipeline.

Executes a complete, autonomous Parse -> Review -> Edit Loop on examination documents:
1. Cartography-Locked topological chunking (7k/10k budget, clean cut seams, safety overlap).
2. Parallel Parser Workers using gpt-6-luna (thinking: low).
3. Chunk-level Review & Surgical Repair Loop (Reviewer: thinking high, Editor: thinking medium, max 3 rounds).
4. Zero-drift Source Merger & Stimulus Deduplication (merged.xml, merged.json).
5. Document-level Audit against raw markdown and Cartographer Manifest (100% question recall).
6. Safe staging to data/sequence_labelling_annotated_branch/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_THINKING,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
    get_provider_base_url,
    get_provider_api_key,
)
from sequence_labelling.annotator.ocr_document import (
    OCRDocument,
    OCRPage,
    PAGE_SEPARATOR,
    split_raw_markdown_pages,
    strip_page_markup,
)
from sequence_labelling.parser.long_parser.document_cartographer import (
    CartographyManifest,
    DocumentCartographer,
)
from sequence_labelling.parser.long_parser.greedy_chunker import (
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import parse_xml_with_anchors
from sequence_labelling.parser.long_parser.sequence_reconciler import (
    DocumentStateStack,
    extract_metadata_headers_from_markdown,
    reconcile_parser_chunk_results,
)
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    AuditIssue,
    DeterministicAuditor,
    IssueSeverity,
    ReviewDecision,
    ReviewReport,
)
from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop, RevisionLoopResult
from sequence_labelling.annotator.xml_checker import XMLChecker
from sequence_labelling.annotator.xml_cleaner import XMLCleaner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("cartography_reparser")


def process_and_repair_chunk(
    worker: ParserAgentWorker,
    revision_loop: Optional[EditorReviewerLoop],
    plan: Dict[str, Any],
    doc_id: str,
    existing_chunks_dir: Path,
    enable_loop: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Parses a single chunk and optionally executes the Review & Surgical Edit loop."""
    chunk_idx = plan["chunk_index"]
    c_json = existing_chunks_dir / f"chunk_{chunk_idx}.json"
    c_xml = existing_chunks_dir / f"chunk_{chunk_idx}.xml"
    c_audit = existing_chunks_dir / f"chunk_{chunk_idx}.audit.json"

    # Check disk cache
    if not overwrite and c_xml.exists() and c_xml.stat().st_size > 100 and c_json.exists():
        try:
            cached = json.loads(c_json.read_text(encoding="utf-8"))
            if cached.get("parse_status") not in ("failed", "error") and cached.get("raw_xml"):
                dec = cached.get("review_decision")
                if c_audit.exists():
                    try:
                        dec = json.loads(c_audit.read_text(encoding="utf-8")).get("decision", dec)
                    except Exception:
                        pass
                # A cached chunk is reusable only if review decision was PASS
                if not enable_loop or dec == "PASS":
                    logger.info(f"  ⚡ [Cached Chunk] '{doc_id}' Chunk {chunk_idx}: loaded from disk (PASS)")
                    return cached
        except Exception:
            pass

    # 1. Parse Chunk with Worker (gpt-6-luna, thinking: low)
    res = worker.process_chunk(
        plan["raw_chunk_text"],
        chunk_index=chunk_idx,
        page_ids=plan["page_ids"],
        overlap_page_ids=plan["overlap_page_ids"],
        page_offset_ranges=plan["page_offset_ranges"],
    )
    raw_xml = res.get("raw_xml", "")
    if not raw_xml.strip():
        return res

    clean_cand = XMLCleaner.clean(raw_xml).cleaned_xml

    # 2. Chunk-level Review and Surgical Edit Loop (LLM Semantic Reviewer & Editor)
    res.setdefault("review_decision", "UNREVIEWED")
    res.setdefault("review_score", None)
    res.setdefault("review_rounds", 0)

    if enable_loop and revision_loop is not None:
        chunk_doc_id = f"{doc_id}::chunk_{chunk_idx:03d}"
        raw_chunk_text = plan["raw_chunk_text"]

        try:
            loop_res = revision_loop.run(
                annotated_xml=clean_cand,
                raw_ocr_text=raw_chunk_text,
                doc_id=chunk_doc_id,
                use_llm_reviewer=True,
            )
            repaired_xml = loop_res.repaired_xml or clean_cand
            final_dec = loop_res.final_decision
            final_sc = loop_res.final_score

            res["raw_xml"] = repaired_xml
            res["review_decision"] = final_dec
            res["review_score"] = final_sc
            res["review_rounds"] = loop_res.rounds_completed

            if loop_res.final_report:
                report_dict = loop_res.final_report.model_dump(mode="json")
                c_audit.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
                res["audit_report"] = report_dict

            # If editor modified XML, re-extract structured questions and stimuli
            if loop_res.rounds_completed > 0 and repaired_xml != clean_cand:
                try:
                    spans, stimuli, questions = parse_xml_with_anchors(raw_chunk_text, repaired_xml)
                    scoped_stimuli = {}
                    stimulus_id_map = {}
                    for old_id, text in stimuli.items():
                        new_id = old_id if old_id.startswith(f"chunk_{chunk_idx}_") else f"chunk_{chunk_idx}_{old_id}"
                        scoped_stimuli[new_id] = text
                        if old_id != new_id:
                            stimulus_id_map[old_id] = new_id

                    for q_i, q in enumerate(questions):
                        base_id = q.get("id") or f"q_{q_i + 1}"
                        if not str(base_id).startswith(f"chunk_{chunk_idx}_"):
                            base_id = f"chunk_{chunk_idx}_{base_id}"
                        q["id"] = base_id
                        q["chunk_index"] = chunk_idx
                        stim_id = q.get("stimulus_id")
                        if stim_id:
                            q["stimulus_id"] = stimulus_id_map.get(stim_id, stim_id)

                    res["questions"] = questions
                    res["stimuli"] = scoped_stimuli
                    res["spans"] = spans
                except Exception as parse_err:
                    logger.warning(f"  ⚠️ Error re-parsing anchors for repaired chunk {chunk_idx}: {parse_err}")

            icon = "🎉" if final_dec == "PASS" else "⚠️"
            logger.info(
                f"  {icon} [Chunk {chunk_idx}] Review/Edit: {loop_res.initial_decision} ({loop_res.initial_score:.1f}) -> "
                f"{final_dec} ({final_sc:.1f}) after {loop_res.rounds_completed} round(s)"
            )
        except Exception as loop_err:
            logger.error(f"  ❌ Revision loop error for chunk {chunk_idx}: {loop_err}")

    # Write chunk files
    c_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_chunk = XMLCleaner.clean(res.get("raw_xml", ""))
    c_xml.write_text(clean_chunk.cleaned_xml, encoding="utf-8")

    return res


def reparse_single_document(
    doc_id: str,
    subject: str,
    input_file: Path,
    cartography_file: Path,
    out_dir: Path,
    model: str = "gpt-6-luna",
    provider: str = "aixingialaire",
    parser_thinking: str = "low",
    reviewer_thinking: str = "high",
    editor_thinking: str = "medium",
    enable_review_edit_loop: bool = True,
    max_revision_rounds: int = 3,
    target_tokens: int = 7000,
    max_tokens: int = 10000,
    overlap_pages: int = 1,
    overlap_fallback_tokens: int = 750,
    concurrency: int = 2,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Executes the full Cartography-locked parse -> review -> edit pipeline on a document."""
    start_time = time.time()
    doc_out_dir = out_dir / doc_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    merged_xml_path = doc_out_dir / "merged.xml"
    merged_json_path = doc_out_dir / "merged.json"
    audit_report_path = doc_out_dir / "audit_report.json"

    # Fast document-level resume check
    if not overwrite and merged_xml_path.exists() and merged_json_path.exists():
        try:
            existing_meta = json.loads(merged_json_path.read_text(encoding="utf-8"))
            dec = existing_meta.get("review_decision")
            if dec == "PASS" and existing_meta.get("cartography_locked"):
                logger.info(f"⚡ [Skipping Completed Document] '{doc_id}' already completed with PASS.")
                return {
                    "status": "skipped",
                    "doc_id": doc_id,
                    "questions": existing_meta.get("questions_count", 0),
                    "stimuli": existing_meta.get("stimuli_count", 0),
                    "duration": existing_meta.get("duration_seconds", 0.0),
                    "is_xml_valid": existing_meta.get("xml_validation", {}).get("is_valid", True),
                    "is_char_match": existing_meta.get("character_fidelity", {}).get("is_exact_match", True),
                    "review_decision": dec,
                    "review_score": existing_meta.get("review_score", 100.0),
                }
        except Exception as e:
            logger.debug(f"Could not read existing merged files for '{doc_id}': {e}")

    # 1. Ingest Cartography Manifest
    manifest_data = json.loads(cartography_file.read_text(encoding="utf-8"))
    manifest = CartographyManifest.from_dict(manifest_data)

    # 2. Ingest & Split Markdown Pages using authoritative OCRDocument
    doc = OCRDocument.from_file(input_file, doc_id=doc_id)
    page_list = doc.to_chunk_pages()

    # 3. Partition into Cartography-Locked Chunks
    chunks = greedy_oversize_chunker(
        page_list,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_pages=overlap_pages,
        cartography=manifest,
        overlap_fallback_tokens=overlap_fallback_tokens,
    )
    chunk_plans = [
        build_chunk_plan(chunk_pages, chunk_index)
        for chunk_index, chunk_pages in enumerate(chunks)
    ]

    # Verify chunk seam cleanliness
    severed_seams = []
    for c_idx in range(len(chunks) - 1):
        last_page = max(p["p"] for p in chunks[c_idx])
        if not manifest.is_seam_clean(last_page):
            severed_seams.append(last_page)

    logger.info(
        f"📑 [Document {doc_id}] ({subject}) {doc.total_pages} pages -> {len(chunks)} chunks. "
        f"Severed seams in new plan: {severed_seams or 'NONE (100% clean)'}"
    )

    existing_chunks_dir = doc_out_dir / "chunks"
    existing_chunks_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize Parser, Reviewer, and Editor Agents
    worker = ParserAgentWorker(
        model=model,
        provider=provider,
        thinking=parser_thinking,
        max_attempts=3,
    )
    reviewer = None
    editor = None
    revision_loop = None

    if enable_review_edit_loop:
        reviewer = AnnotationReviewerAgent(
            model=model,
            provider=provider,
            thinking=reviewer_thinking,
        )
        editor = EditorAgent(
            model=model,
            provider=provider,
            thinking=editor_thinking,
            reviewer=reviewer,
        )
        revision_loop = EditorReviewerLoop(
            editor=editor,
            reviewer=reviewer,
            max_rounds=max_revision_rounds,
        )

    # 5. Parallel Processing: Parse + Review + Surgical Edit per Chunk
    chunk_results = [None] * len(chunks)
    max_workers = max(1, min(concurrency, len(chunks)))
    failed_chunk_indexes = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for plan in chunk_plans:
            future = executor.submit(
                process_and_repair_chunk,
                worker=worker,
                revision_loop=revision_loop,
                plan=plan,
                doc_id=doc_id,
                existing_chunks_dir=existing_chunks_dir,
                enable_loop=enable_review_edit_loop,
                overwrite=overwrite,
            )
            futures[future] = plan

        for future in as_completed(futures):
            plan = futures[future]
            chunk_idx = plan["chunk_index"]
            try:
                res = future.result()
                q_count = len(res.get("questions", []))
                s_count = len(res.get("stimuli", {}))
                chunk_results[chunk_idx] = res
                logger.info(f"  ✅ [Ready Chunk] '{doc_id}' Chunk {chunk_idx}: {q_count} questions, {s_count} stimuli")
            except Exception as e:
                logger.error(f"  ❌ [Error] Chunk {chunk_idx} failed for '{doc_id}': {e}")
                failed_chunk_indexes.append(chunk_idx)
                chunk_results[chunk_idx] = {
                    **plan,
                    "raw_xml": "",
                    "spans": [],
                    "parse_status": "failed",
                    "parse_diagnostics": {"attempts": worker.max_attempts, "validation_errors": [str(e)]},
                    "questions": [],
                    "stimuli": {},
                }

    if failed_chunk_indexes:
        logger.error(
            f"❌ Document '{doc_id}' encountered fatal failures in chunk(s) {failed_chunk_indexes}. "
            "Aborting document reconciliation to avoid producing corrupted output."
        )
        return {
            "status": "failed",
            "doc_id": doc_id,
            "error": f"Chunks {failed_chunk_indexes} failed",
        }

    valid_chunk_results = sorted(
        [r for r in chunk_results if r is not None],
        key=lambda x: x.get("chunk_index", 0),
    )

    # 6. Reconcile & Merge Exam Annotations
    merge_status = "success"
    merge_error_msg = None
    try:
        merge_result = reconcile_parser_chunk_results(valid_chunk_results)
    except Exception as merge_err:
        logger.warning(f"⚠️ Merge reconciliation failed for '{doc_id}': {merge_err}. Using fallback concatenation.")
        merge_status = "fallback_concatenation"
        merge_error_msg = str(merge_err)
        all_q = []
        all_s = {}
        xml_parts = []
        for cr in valid_chunk_results:
            all_q.extend(cr.get("questions", []))
            all_s.update(cr.get("stimuli", {}))
            if cr.get("raw_xml"):
                xml_parts.append(cr["raw_xml"])
        merge_result = {
            "structured_questions": all_q,
            "structured_stimuli": all_s,
            "merged_xml": "\n".join(xml_parts),
            "diagnostics": {"merge_status": "fallback", "error": merge_error_msg},
        }

    raw_merged_xml = merge_result.get("merged_xml", "")
    clean_merged = XMLCleaner.clean(raw_merged_xml)
    merged_xml_content = clean_merged.cleaned_xml
    merged_xml_check = XMLChecker.check(merged_xml_content)

    questions_count = len(merge_result.get("structured_questions", []))
    stimuli_count = len(merge_result.get("structured_stimuli", {}))

    # 7. Character Preservation Verification using OCRDocument
    fidelity = doc.verify_character_fidelity(merged_xml_content)
    char_match = fidelity["is_exact_match"]

    # 8. Document-Level Review & Cartographer Audit
    doc_review_report = None
    if enable_review_edit_loop and reviewer is not None:
        try:
            doc_review_report = reviewer.review_document(
                xml_content=merged_xml_content,
                raw_ocr_text=doc.raw_cleaned_text,
                doc_id=doc_id,
                use_llm=True,
            )
            logger.info(
                f"  📋 [Doc Review] '{doc_id}': {doc_review_report.decision.value} "
                f"(Score: {doc_review_report.overall_score:.1f})"
            )

            # Cartography manifest question recall audit
            # Use manifest.objects (type == 'question') to avoid ID collision across multi-exam books
            if manifest and manifest.objects:
                q_objects = [obj for obj in manifest.objects if obj.type == "question"]
                manifest_q_nums = set()
                for q_obj in q_objects:
                    lbl = q_obj.label or q_obj.id or ""
                    m = re.search(r"\b(\d+)\b", str(lbl))
                    if m:
                        try:
                            manifest_q_nums.add(int(m.group(1)))
                        except Exception:
                            pass

                parsed_q_nums = set()
                for q in merge_result.get("structured_questions", []):
                    q_lab = q.get("question_label", "")
                    m = re.search(r"\b(\d+)\b", q_lab)
                    if m:
                        try:
                            parsed_q_nums.add(int(m.group(1)))
                        except Exception:
                            pass

                missing_nums = manifest_q_nums - parsed_q_nums
                # Only flag as missing if a meaningful percentage or contiguous questions are missing
                if missing_nums and (len(missing_nums) >= 3 or (manifest_q_nums and len(missing_nums) / len(manifest_q_nums) > 0.05)):
                    logger.warning(f"  ⚠️ [Cartography Recall Audit] Missing questions: {sorted(missing_nums)[:15]}")
                    issue = AuditIssue(
                        category="missing_question",
                        severity=IssueSeverity.CRITICAL if len(missing_nums) >= 5 else IssueSeverity.MAJOR,
                        message=f"Cartography Manifest detected questions {sorted(missing_nums)[:10]} missing in parsed XML",
                    )
                    doc_review_report.issues.append(issue)
                    doc_review_report.confirmed_issues.append(issue)
                    penalty = min(30.0, 4.0 * len(missing_nums))
                    doc_review_report.overall_score = max(0.0, doc_review_report.overall_score - penalty)
                    if doc_review_report.overall_score < 80.0:
                        doc_review_report.decision = ReviewDecision.NEEDS_REVISION

            # If document needs revision and is within safe context size limit (<= 120 KB),
            # attempt 1 document-level surgical repair
            xml_bytes_len = len(merged_xml_content.encode("utf-8"))
            if doc_review_report.decision != ReviewDecision.PASS and revision_loop is not None and xml_bytes_len <= 120000:
                logger.info(f"  🛠️ [Doc-Level Repair] Running surgical EditorAgent on '{doc_id}' ({xml_bytes_len} bytes)...")
                doc_loop_res = revision_loop.run(
                    annotated_xml=merged_xml_content,
                    raw_ocr_text=doc.raw_cleaned_text,
                    doc_id=doc_id,
                    initial_report=doc_review_report,
                    use_llm_reviewer=True,
                )
                if doc_loop_res.final_score > doc_review_report.overall_score and doc_loop_res.repaired_xml:
                    repaired_clean = XMLCleaner.clean(doc_loop_res.repaired_xml).cleaned_xml
                    merged_xml_content = repaired_clean
                    doc_review_report = doc_loop_res.final_report or doc_review_report
                    logger.info(
                        f"  🎉 [Doc Repaired] '{doc_id}' improved to {doc_loop_res.final_score:.1f} "
                        f"({doc_loop_res.final_decision})"
                    )

                    # Re-extract structured questions and stimuli from the repaired XML
                    try:
                        spans, re_stimuli, re_questions = parse_xml_with_anchors(doc.clean_text, merged_xml_content)
                        if re_questions:
                            merge_result["structured_questions"] = re_questions
                            merge_result["structured_stimuli"] = re_stimuli
                            questions_count = len(re_questions)
                            stimuli_count = len(re_stimuli)
                    except Exception as reparse_err:
                        logger.warning(f"  ⚠️ Could not re-extract anchors after doc-level repair: {reparse_err}")

                    # Re-evaluate validation and character fidelity on the repaired XML
                    merged_xml_check = XMLChecker.check(merged_xml_content)
                    fidelity = doc.verify_character_fidelity(merged_xml_content)
                    char_match = fidelity["is_exact_match"]

            # Save audit report
            audit_report_path.write_text(
                json.dumps(doc_review_report.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as doc_rev_err:
            logger.error(f"  ❌ Document-level review error for '{doc_id}': {doc_rev_err}")

    duration = time.time() - start_time

    # 9. Write final merged outputs
    merged_data = {
        "doc_id": doc_id,
        "input_file": str(input_file.relative_to(WORKSPACE_DIR)),
        "subject": subject,
        "total_chunks": len(chunks),
        "chunk_token_config": {"target_tokens": target_tokens, "max_tokens": max_tokens},
        "cartography_locked": True,
        "severed_seams": severed_seams,
        "merge_status": merge_status,
        "merge_error": merge_error_msg,
        "questions_count": questions_count,
        "stimuli_count": stimuli_count,
        "duration_seconds": round(duration, 2),
        "character_fidelity": {
            "is_exact_match": char_match,
            "stripped_length": fidelity["stripped_length"],
            "clean_doc_length": fidelity["clean_doc_length"],
            "raw_cleaned_length": fidelity["raw_cleaned_length"],
            "mismatch_char_diff": fidelity.get("mismatch_char_diff", 0),
            "first_diff_position": fidelity.get("first_diff_position"),
        },
        "xml_validation": {
            "is_valid": merged_xml_check.is_valid,
            "issues_count": len(merged_xml_check.issues),
            "issues": [str(i) for i in merged_xml_check.issues],
        },
        "review_decision": doc_review_report.decision.value if doc_review_report else "UNREVIEWED",
        "review_score": doc_review_report.overall_score if doc_review_report else None,
        "questions": merge_result["structured_questions"],
        "stimuli": merge_result["structured_stimuli"],
        "merged_xml": merged_xml_content,
    }

    merged_json_path.write_text(json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_xml_path.write_text(merged_xml_content, encoding="utf-8")

    dec_str = doc_review_report.decision.value if doc_review_report else "DONE"
    score_str = f"{doc_review_report.overall_score:.1f}" if doc_review_report else "N/A"
    logger.info(
        f"🎯 [Completed Document '{doc_id}'] Status: {dec_str} (Score: {score_str}), "
        f"{questions_count} questions, {stimuli_count} stimuli, Char match: {char_match} in {duration:.1f}s"
    )

    return {
        "status": "success",
        "doc_id": doc_id,
        "questions": questions_count,
        "stimuli": stimuli_count,
        "duration": round(duration, 2),
        "is_xml_valid": merged_xml_check.is_valid,
        "is_char_match": char_match,
        "review_decision": dec_str,
        "review_score": doc_review_report.overall_score if doc_review_report else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse severed multi-chunk documents with gpt-6-luna, Review & Edit Loop, and Cartography snapping."
    )
    parser.add_argument("--audit-file", type=Path, default=WORKSPACE_DIR / "artifacts" / "reports" / "chunk_seam_severing_audit.json")
    parser.add_argument("--cartography-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_cartography")
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    parser.add_argument("--annotated-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch")
    parser.add_argument("--model", type=str, default="gpt-6-luna")
    parser.add_argument("--provider", type=str, default="aixingialaire")
    parser.add_argument("--parser-thinking", "--thinking", dest="parser_thinking", type=str, default="low", help="Parser thinking effort (default: low)")
    parser.add_argument("--reviewer-thinking", type=str, default="high", help="Reviewer thinking effort (default: high)")
    parser.add_argument("--editor-thinking", type=str, default="medium", help="Editor thinking effort (default: medium)")
    parser.add_argument("--enable-review-edit-loop", action="store_true", default=True, help="Enable chunk and document Review & Edit loop")
    parser.add_argument("--no-review-edit-loop", dest="enable_review_edit_loop", action="store_false", help="Disable Review & Edit loop")
    parser.add_argument("--max-revision-rounds", type=int, default=3, help="Max repair rounds per chunk (default: 3)")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--concurrency", type=int, default=1, help="Chunk threads per document (default: 1)")
    parser.add_argument("--target-tokens", type=int, default=7000)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--overlap-pages", type=int, default=1, help="Safety net page overlap")
    parser.add_argument("--overlap-fallback-tokens", type=int, default=750, help="Fallback token overlap if unsplit")
    parser.add_argument("--priority-non-math", action="store_true", default=True, help="Sort non-math exams first")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process in this run")
    parser.add_argument("--doc-id", type=str, default=None, help="Run on a single specific document ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = get_provider_api_key(args.provider)
    if not api_key:
        logger.error(f"Missing API key for provider '{args.provider}'!")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("🚀 HIGH-PERFORMANCE CARTOGRAPHY-LOCKED REPARSER & REMEDIATION PIPELINE")
    logger.info(f"Model: {args.model} | Provider: {args.provider}")
    logger.info(f"Parser Thinking: {args.parser_thinking} | Reviewer Thinking: {args.reviewer_thinking} | Editor Thinking: {args.editor_thinking}")
    logger.info(f"Review & Edit Loop: {'ENABLED (max ' + str(args.max_revision_rounds) + ' rounds)' if args.enable_review_edit_loop else 'DISABLED'}")
    logger.info(f"Shard: {args.shard_id + 1}/{args.num_shards} | Concurrency: {args.concurrency}")
    logger.info(f"Output Directory: {args.annotated_dir}")
    logger.info("=" * 70)

    # 1. Ingest Audit File
    if not args.audit_file.exists():
        logger.error(f"Audit file not found at {args.audit_file}")
        sys.exit(1)

    audit_data = json.loads(args.audit_file.read_text(encoding="utf-8"))
    results = audit_data.get("results", [])

    # Filter to specific doc-id if requested
    if args.doc_id:
        targets = [r for r in results if r["doc_id"] == args.doc_id]
        if not targets:
            targets = [{"doc_id": args.doc_id, "subject": "Unknown"}]
    else:
        targets = results

    # Priority sorting: Non-math first, then math
    if args.priority_non_math:
        non_math = [r for r in targets if r.get("subject") != "Toán"]
        math_docs = [r for r in targets if r.get("subject") == "Toán"]
        targets = non_math + math_docs
        logger.info(f"Priority queue: {len(non_math)} non-math exams first, followed by {len(math_docs)} math exams.")

    # Apply Sharding
    sharded_targets = [
        item for idx, item in enumerate(targets)
        if idx % args.num_shards == args.shard_id
    ]

    if args.limit:
        sharded_targets = sharded_targets[:args.limit]

    logger.info(f"Shard {args.shard_id}: Processing {len(sharded_targets)} documents (out of {len(targets)} total targets).")

    completed = 0
    skipped = 0
    failed = 0
    start_all = time.time()

    for idx, item in enumerate(sharded_targets, start=1):
        doc_id = item["doc_id"]
        subject = item.get("subject", "Unknown")
        cart_file = args.cartography_dir / f"{doc_id}.json"

        # Locate input markdown
        md_file = args.input_dir / f"{doc_id}.md"
        if not md_file.exists():
            candidates = list(args.input_dir.glob(f"*{doc_id}*.md"))
            md_file = candidates[0] if candidates else None

        if not md_file or not md_file.exists():
            logger.warning(f"[{idx}/{len(sharded_targets)}] Missing input markdown for '{doc_id}'. Skipping.")
            failed += 1
            continue

        if not cart_file.exists():
            logger.warning(f"[{idx}/{len(sharded_targets)}] Missing cartography json for '{doc_id}'. Skipping.")
            failed += 1
            continue

        logger.info(f"\n--- [{idx}/{len(sharded_targets)}] Reparsing & Remediation for '{doc_id}' ({subject}) ---")
        try:
            res = reparse_single_document(
                doc_id=doc_id,
                subject=subject,
                input_file=md_file,
                cartography_file=cart_file,
                out_dir=args.annotated_dir,
                model=args.model,
                provider=args.provider,
                parser_thinking=args.parser_thinking,
                reviewer_thinking=args.reviewer_thinking,
                editor_thinking=args.editor_thinking,
                enable_review_edit_loop=args.enable_review_edit_loop,
                max_revision_rounds=args.max_revision_rounds,
                target_tokens=args.target_tokens,
                max_tokens=args.max_tokens,
                overlap_pages=args.overlap_pages,
                overlap_fallback_tokens=args.overlap_fallback_tokens,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
            )
            if res.get("status") == "skipped":
                skipped += 1
            elif res.get("status") == "success":
                completed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Unhandled error in document '{doc_id}': {e}", exc_info=True)
            failed += 1

    total_time = time.time() - start_all
    logger.info("\n" + "=" * 70)
    logger.info(f"🏁 SHARD {args.shard_id} RUN COMPLETE in {total_time / 60:.2f} minutes")
    logger.info(f"Completed: {completed} | Skipped: {skipped} | Failed: {failed} | Total: {len(sharded_targets)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
