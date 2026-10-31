#!/usr/bin/env python3
"""
run_repair_chunks_agy.py — Full Chunk-as-Samples Surgical Repair & Review Pipeline.

Features:
- Target: All 82 remaining unpassed exams from data/pipeline_420_state.json.
- Chunk Granularity: Partitions documents into clean ~3K-4K token chunks.
- Surgical Focus: Reviews and patches each chunk_X.xml independently using EditorReviewerLoop.
- Models: gemini-3.8-flash-low for both Editor and Reviewer via Google Antigravity (agy).
- Account: kien.duet (AGY_ACCOUNT=kien.duet).
- Concurrency: 4 parallel worker threads.
- Reconciliation: Stitches repaired chunks into merged.xml, merged.json, and audit_report.json.
- Live State Sync: Atomically updates data/pipeline_420_state.json on each completed document.
- Dedicated Reporting: artifacts/reports/repair_chunk_level_82_progress.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enforce agy account and concurrency in environment
os.environ["AGY_ACCOUNT"] = "kien.duet"
os.environ["AGY_CONCURRENCY_LIMIT"] = "4"

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.ocr_document import OCRDocument
from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop, RevisionLoopResult
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import parse_xml_with_anchors
from sequence_labelling.parser.long_parser.greedy_chunker import (
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.parser.long_parser.sequence_reconciler import reconcile_parser_chunk_results

DEFAULT_STATE_FILE = Path("data/repair_chunk_level_82_state.json")
DEFAULT_PIPELINE_420_STATE = Path("data/pipeline_420_state.json")
DEFAULT_REPORT_FILE = Path("artifacts/reports/repair_chunk_level_82_progress.md")
DEFAULT_LOG_FILE = Path("artifacts/repair_chunk_level_82.log")

Path("artifacts/reports").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("repair_chunks")


def setup_logger(log_file: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class StateManager:
    def __init__(self, state_path: Path, p420_path: Path, total_targets: int):
        self.state_path = state_path
        self.p420_path = p420_path
        self.total_targets = total_targets
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed loading state {self.state_path}: {e}")
        return {
            "total_target_files": self.total_targets,
            "completed_count": 0,
            "pass_count": 0,
            "needs_revision_count": 0,
            "discard_count": 0,
            "total_duration": 0.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "docs": {},
        }

    def save_state(self) -> None:
        with self.lock:
            self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
            tmp = self.state_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            tmp.replace(self.state_path)

    def is_done(self, stem: str) -> bool:
        with self.lock:
            doc = self.data["docs"].get(stem)
            return doc is not None and doc.get("status") == "DONE"

    def record_completed(self, stem: str, record: Dict[str, Any]) -> None:
        with self.lock:
            self.data["docs"][stem] = record
            self.data["completed_count"] = sum(
                1 for d in self.data["docs"].values() if d.get("status") == "DONE"
            )
            self.data["pass_count"] = sum(
                1 for d in self.data["docs"].values() if d.get("final_decision") == "PASS"
            )
            self.data["needs_revision_count"] = sum(
                1 for d in self.data["docs"].values() if d.get("final_decision") == "NEEDS_REVISION"
            )
            self.data["discard_count"] = sum(
                1 for d in self.data["docs"].values() if d.get("final_decision") == "DISCARD"
            )
            self.data["total_duration"] += record.get("duration", 0.0)
            self._sync_to_pipeline_420(stem, record)
        self.save_state()

    def _sync_to_pipeline_420(self, stem: str, record: Dict[str, Any]) -> None:
        """Syncs the repaired exam record into data/pipeline_420_state.json."""
        if not self.p420_path.exists():
            return
        try:
            with open(self.p420_path, "r", encoding="utf-8") as f:
                p420_data = json.load(f)

            if "docs" in p420_data and stem in p420_data["docs"]:
                prev = p420_data["docs"][stem]
                prev["status"] = record.get("status", "DONE")
                prev["final_score"] = record.get("final_score", prev.get("final_score"))
                prev["final_decision"] = record.get("final_decision", prev.get("final_decision"))
                prev["chunks_count"] = record.get("chunks_count", 1)
                prev["patches_applied"] = prev.get("patches_applied", 0) + record.get("patches_applied", 0)
                prev["repaired_with"] = "gemini-3.8-flash-low chunk-level"
                prev["repaired_at"] = datetime.now(timezone.utc).isoformat()
                p420_data["docs"][stem] = prev

                # Recalculate global 420 totals
                p420_data["completed_count"] = sum(
                    1 for d in p420_data["docs"].values() if d.get("status") == "DONE"
                )
                p420_data["pass_count"] = sum(
                    1 for d in p420_data["docs"].values() if d.get("final_decision") == "PASS"
                )
                p420_data["needs_revision_count"] = sum(
                    1 for d in p420_data["docs"].values() if d.get("final_decision") == "NEEDS_REVISION"
                )
                p420_data["discard_count"] = sum(
                    1 for d in p420_data["docs"].values() if d.get("final_decision") == "DISCARD"
                )
                p420_data["last_updated"] = datetime.now(timezone.utc).isoformat()

                tmp = self.p420_path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(p420_data, f, ensure_ascii=False, indent=2)
                tmp.replace(self.p420_path)
                logger.info(
                    f"🔄 Synced '{stem[:45]}...' ({record.get('final_decision')} {record.get('final_score'):.1f}, {record.get('chunks_count')} chunks) to pipeline_420_state.json"
                )
        except Exception as e:
            logger.warning(f"Could not sync to {self.p420_path}: {e}")


def generate_markdown_report(manager: StateManager, report_path: Path) -> None:
    data = manager.data
    total = data.get("total_target_files", manager.total_targets)
    done = data.get("completed_count", 0)
    passed = data.get("pass_count", 0)
    rev = data.get("needs_revision_count", 0)
    discard = data.get("discard_count", 0)
    pct = (done / total * 100) if total > 0 else 0.0
    pass_pct = (passed / done * 100) if done > 0 else 0.0
    usable_count = passed + rev
    usable_pct = (usable_count / done * 100) if done > 0 else 0.0

    scores = [
        d.get("final_score")
        for d in data.get("docs", {}).values()
        if d.get("final_score") is not None
    ]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    lines = [
        "# Báo Cáo Tiến Độ Phẫu Thuật Sửa Lỗi Cấp Độ Chunk (Chunk-Level Repair)",
        "",
        f"> **Thời gian cập nhật:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"> **Tài khoản AGY:** `kien.duet` | **Editor & Reviewer:** `gemini-3.8-flash-low` | **Concurrency:** 4  ",
        "",
        "## 1. Tổng Quan Tiến Độ",
        "",
        f"- **Tiến độ xử lý:** **`{done} / {total}`** đề (**`{pct:.1f}%`**)",
        f"- **Điểm trung bình:** 🚀 **`{avg_score:.2f} / 100`**",
        f"- **🏆 PASS (≥ 75 điểm & 0 critical):** **`{passed}`** đề (**`{pass_pct:.1f}%`**)",
        f"- **🌟 Sử dụng tốt (PASS + NEEDS_REVISION):** **`{usable_count}`** đề (**`{usable_pct:.1f}%`**)",
        f"- **📉 DISCARD (Hỏng):** **`{discard}`** đề",
        "",
        "## 2. Bảng Thống Kê Chi Tiết Từng Đề",
        "",
        "| STT | Môn Học | Tên Đề Thi | Chunks | Trước Sửa | Sau Sửa | Patches | Kết Quả |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    sorted_docs = sorted(
        data.get("docs", {}).items(),
        key=lambda x: x[1].get("completed_at", ""),
        reverse=True,
    )

    for idx, (stem, d) in enumerate(sorted_docs, start=1):
        cat = d.get("category", "Chung")
        chunks = d.get("chunks_count", 1)
        sc_init = d.get("initial_score", 0.0)
        sc_fin = d.get("final_score", 0.0)
        patches = d.get("patches_applied", 0)
        dec = d.get("final_decision", "UNKNOWN")
        icon = "🏆 PASS" if dec == "PASS" else ("⚠️ REVISION" if dec == "NEEDS_REVISION" else "❌ DISCARD")
        short_stem = stem[:50] + "..." if len(stem) > 50 else stem
        short_cat = cat.split("(")[0].strip()
        lines.append(f"| {idx} | {short_cat} | {short_stem} | {chunks} | `{sc_init:.1f}` | **`{sc_fin:.1f}`** | {patches} | {icon} |")

    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def process_single_exam_chunk_level(
    stem: str,
    category: str,
    raw_path: Path,
    out_dir: Path,
    initial_score: float,
    initial_decision: str,
    worker: ParserAgentWorker,
    reviewer: AnnotationReviewerAgent,
    editor: EditorAgent,
    editor_loop: EditorReviewerLoop,
) -> Dict[str, Any]:
    t0 = time.time()
    doc_dir = out_dir / stem
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    ocr_doc = OCRDocument.from_file(raw_path, doc_id=stem)
    pages = ocr_doc.to_chunk_pages()

    # Step 1: Partition into clean ~3K-4K token chunks
    raw_text = raw_path.read_text(encoding="utf-8")
    words_count = len(raw_text.split())

    if words_count <= 3500 and ocr_doc.total_pages <= 4:
        chunk_plans = [{
            "chunk_index": 0,
            "raw_chunk_text": raw_text,
            "page_ids": [p["p"] for p in pages] if pages else [1],
            "overlap_page_ids": [],
            "page_offset_ranges": [],
        }]
    else:
        chunks = greedy_oversize_chunker(
            pages,
            target_tokens=3000,
            max_tokens=4500,
            overlap_pages=0,
        )
        chunk_plans = [build_chunk_plan(cp, idx) for idx, cp in enumerate(chunks)]

    chunk_results = []
    total_patches = 0
    chunk_scores = []
    chunk_decisions = []

    # Step 2: Review and Repair EACH chunk independently
    for plan in chunk_plans:
        c_idx = plan["chunk_index"]
        c_text = plan["raw_chunk_text"]
        c_xml_path = chunks_dir / f"chunk_{c_idx}.xml"
        c_json_path = chunks_dir / f"chunk_{c_idx}.json"
        c_audit_path = chunks_dir / f"chunk_{c_idx}.audit.json"

        # Obtain candidate XML for this chunk
        c_xml = None
        # Check if existing chunk_X.xml is valid and matches length
        if c_xml_path.exists() and c_xml_path.stat().st_size > 50 and len(chunk_plans) == 1:
            c_xml = c_xml_path.read_text(encoding="utf-8")
        elif (doc_dir / "merged.xml").exists() and len(chunk_plans) == 1:
            c_xml = (doc_dir / "merged.xml").read_text(encoding="utf-8")

        # If newly split multi-chunk or no valid XML, parse with ParserAgentWorker
        if not c_xml or len(c_xml.strip()) < 50:
            parse_res = worker.process_chunk(
                c_text,
                chunk_index=c_idx,
                page_ids=plan.get("page_ids", []),
            )
            c_xml = XMLCleaner.clean(parse_res.get("raw_xml", "")).cleaned_xml

        clean_c_xml = XMLCleaner.clean(c_xml).cleaned_xml

        # Run chunk-level Review & Surgical Repair loop
        chunk_doc_id = f"{stem}::chunk_{c_idx}"
        loop_res: RevisionLoopResult = editor_loop.run(
            annotated_xml=clean_c_xml,
            raw_ocr_text=c_text,
            doc_id=chunk_doc_id,
            use_llm_reviewer=True,
        )

        repaired_c_xml = loop_res.repaired_xml if loop_res.repaired_xml else clean_c_xml
        repaired_c_xml_clean = XMLCleaner.clean(repaired_c_xml).cleaned_xml
        c_final_decision = loop_res.final_decision
        c_final_score = loop_res.final_score

        total_patches += loop_res.applied_patches_count
        chunk_scores.append(c_final_score)
        chunk_decisions.append(c_final_decision)

        # Write per-chunk outputs
        c_xml_path.write_text(repaired_c_xml_clean, encoding="utf-8")
        c_audit_path.write_text(
            json.dumps({
                "chunk_index": c_idx,
                "score": c_final_score,
                "decision": c_final_decision,
                "applied_patches": loop_res.applied_patches_count,
                "rounds": loop_res.rounds_completed,
            }, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Extract structured items for chunk
        spans, stimuli, questions = parse_xml_with_anchors(c_text, repaired_c_xml_clean)
        c_data = {
            "chunk_index": c_idx,
            "page_ids": plan.get("page_ids", []),
            "raw_xml": repaired_c_xml_clean,
            "questions": questions,
            "stimuli": stimuli,
            "review_decision": c_final_decision,
            "review_score": c_final_score,
        }
        c_json_path.write_text(json.dumps(c_data, ensure_ascii=False, indent=2), encoding="utf-8")
        chunk_results.append(c_data)

    # Step 3: Reconcile all repaired chunks into document level
    if len(chunk_results) == 1:
        merged_xml = chunk_results[0]["raw_xml"]
        spans, stimuli, questions = parse_xml_with_anchors(raw_text, merged_xml)
    else:
        try:
            mr = reconcile_parser_chunk_results(chunk_results)
            merged_xml = mr.get("merged_xml", "")
            questions = mr.get("structured_questions", [])
            stimuli = mr.get("structured_stimuli", {})
        except Exception as e:
            logger.warning(f"[{stem}] Chunk reconciliation error: {e}. Fallback concat.")
            merged_xml = "\n\n".join(cr.get("raw_xml", "") for cr in chunk_results)
            spans, stimuli, questions = parse_xml_with_anchors(raw_text, merged_xml)

    clean_merged = XMLCleaner.clean(merged_xml).cleaned_xml
    (doc_dir / "merged.xml").write_text(clean_merged, encoding="utf-8")

    # Step 4: Calculate Document Rollup Decision & Score
    avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
    if all(d == "PASS" for d in chunk_decisions):
        doc_final_decision = "PASS"
    elif any(d == "DISCARD" for d in chunk_decisions) or avg_score < 75.0:
        doc_final_decision = "DISCARD"
    else:
        doc_final_decision = "NEEDS_REVISION"

    merged_data = {
        "document_id": stem,
        "repaired_at": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-3.8-flash-low",
        "provider": "agy",
        "account": "kien.duet",
        "chunks_count": len(chunk_plans),
        "chunk_scores": chunk_scores,
        "chunk_decisions": chunk_decisions,
        "final_decision": doc_final_decision,
        "final_score": round(avg_score, 1),
        "questions_count": len(questions),
        "stimuli_count": len(stimuli),
        "questions": questions,
        "stimuli": stimuli,
    }
    (doc_dir / "merged.json").write_text(
        json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit_info = {
        "doc_id": stem,
        "overall_score": round(avg_score, 1),
        "decision": doc_final_decision,
        "chunks_count": len(chunk_plans),
        "chunk_scores": chunk_scores,
        "chunk_decisions": chunk_decisions,
        "patches_applied": total_patches,
        "initial_score": initial_score,
        "initial_decision": initial_decision,
    }
    (doc_dir / "audit_report.json").write_text(
        json.dumps(audit_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    return {
        "doc_stem": stem,
        "category": category,
        "status": "DONE",
        "initial_score": initial_score,
        "initial_decision": initial_decision,
        "final_score": round(avg_score, 1),
        "final_decision": doc_final_decision,
        "score_delta": round(avg_score - initial_score, 2),
        "chunks_count": len(chunk_plans),
        "patches_applied": total_patches,
        "questions_count": len(questions),
        "stimuli_count": len(stimuli),
        "duration": round(elapsed, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Chunk-Level Surgical Repair of Remaining Unpassed Exams")
    parser.add_argument("--concurrency", type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to repair")
    parser.add_argument("--doc-id", type=str, default=None, help="Process a specific document stem")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    setup_logger(args.log_file)

    logger.info("=" * 70)
    logger.info("🚀 CHUNK-LEVEL SURGICAL REPAIR & REVIEW PIPELINE (AGY)")
    logger.info("Provider: Google Antigravity (agy) | Account: kien.duet | Concurrency: %d", args.concurrency)
    logger.info("Reviewer: gemini-3.8-flash-low | Editor: gemini-3.8-flash-low | Mode: Chunk-as-Samples")
    logger.info("=" * 70)

    p420_state_path = DEFAULT_PIPELINE_420_STATE
    if not p420_state_path.exists():
        logger.error(f"Cannot find {p420_state_path}!")
        sys.exit(1)

    with open(p420_state_path, "r", encoding="utf-8") as f:
        p420 = json.load(f)

    docs = p420.get("docs", {})
    unpassed_targets = []
    for stem, d in docs.items():
        dec = d.get("final_decision")
        st = d.get("status")
        if dec != "PASS" or st != "DONE":
            sc = d.get("final_score") or d.get("initial_score") or 0.0
            cat = d.get("category", "Chung")
            unpassed_targets.append((stem, cat, sc, dec or "UNKNOWN"))

    if args.doc_id:
        unpassed_targets = [item for item in unpassed_targets if item[0] == args.doc_id]

    logger.info(f"Identified {len(unpassed_targets)} unpassed documents from {p420_state_path.name}.")

    raw_dir = Path("data/sequence_labelling_input_data")
    out_dir = Path("data/sequence_labelling_annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_mgr = StateManager(args.state_file, p420_state_path, total_targets=len(unpassed_targets))

    # Filter pending
    pending_items = []
    for stem, cat, init_sc, init_dec in unpassed_targets:
        if state_mgr.is_done(stem):
            continue
        raw_p = raw_dir / f"{stem}.md"
        if not raw_p.exists():
            logger.warning(f"Missing raw markdown for {stem}: {raw_p}")
            continue
        pending_items.append((stem, cat, raw_p, init_sc, init_dec))

    if args.limit:
        pending_items = pending_items[:args.limit]

    logger.info(f"Total pending to repair: {len(pending_items)} / {len(unpassed_targets)}")

    if not pending_items:
        logger.info("All targeted documents already repaired! Exiting.")
        generate_markdown_report(state_mgr, args.report_file)
        return

    def make_instances():
        rev = AnnotationReviewerAgent(
            model="gemini-3.8-flash-low",
            provider="agy",
            thinking="low",
        )
        edt = EditorAgent(
            model="gemini-3.8-flash-low",
            provider="agy",
            thinking="low",
            reviewer=rev,
        )
        loop = EditorReviewerLoop(editor=edt, reviewer=rev, max_rounds=2)
        wrk = ParserAgentWorker(
            model="gemini-3.8-flash-low",
            provider="agy",
            thinking="low",
        )
        return rev, edt, loop, wrk

    completed_count = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="chunk_worker") as executor:
        futures = {}
        for stem, cat, raw_p, init_sc, init_dec in pending_items:
            rev, edt, loop, wrk = make_instances()
            fut = executor.submit(
                process_single_exam_chunk_level,
                stem=stem,
                category=cat,
                raw_path=raw_p,
                out_dir=out_dir,
                initial_score=init_sc,
                initial_decision=init_dec,
                worker=wrk,
                reviewer=rev,
                editor=edt,
                editor_loop=loop,
            )
            futures[fut] = stem

        for fut in as_completed(futures):
            stem = futures[fut]
            try:
                result = fut.result()
                state_mgr.record_completed(stem, result)
                completed_count += 1

                dec = result.get("final_decision", "UNKNOWN")
                sc = result.get("final_score", 0.0)
                init_sc = result.get("initial_score", 0.0)
                dur = result.get("duration", 0.0)
                patches = result.get("patches_applied", 0)
                chunks_n = result.get("chunks_count", 1)
                icon = "🏆" if dec == "PASS" else ("⚠️" if dec == "NEEDS_REVISION" else "❌")

                logger.info(
                    f"{icon} [{completed_count}/{len(pending_items)}] '{stem[:45]}...': "
                    f"{dec} ({init_sc:.1f} -> {sc:.1f}, {chunks_n} chunks, {patches} patches) in {dur:.1f}s"
                )
                generate_markdown_report(state_mgr, args.report_file)

            except Exception as e:
                logger.error(f"❌ Failed chunk repair for '{stem}': {e}", exc_info=True)
                state_mgr.record_completed(stem, {
                    "doc_stem": stem,
                    "status": "ERROR",
                    "error": str(e),
                    "duration": 0.0,
                })

    total_time = time.time() - start_all
    logger.info("=" * 70)
    logger.info(f"🏁 ALL REMAINING EXAMS CHUNK REPAIRED IN {total_time / 60:.2f} MINUTES")
    logger.info(f"PASS: {state_mgr.data['pass_count']} | NEEDS_REVISION: {state_mgr.data['needs_revision_count']} | DISCARD: {state_mgr.data['discard_count']}")
    logger.info("=" * 70)
    generate_markdown_report(state_mgr, args.report_file)


if __name__ == "__main__":
    main()
