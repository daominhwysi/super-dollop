#!/usr/bin/env python3
"""
reparse_english_49_agy.py — Parallel Reparsing & Review Pipeline for 49 English Exams.

Features:
- Target: 49 English exams previously discarded in batch 420.
- LLM Provider: Google Antigravity (agy) under quota account 'kien.duet'.
- Parser: gemini-3.8-flash-low (thinking: low).
- Reviewer: gemini-3.8-flash-low (thinking: low).
- Concurrency: 4 workers.
- Storage: Canonical directories in data/sequence_labelling_annotated/<doc_stem>/
- State & Sync: Atomic checkpointing in data/reparse_english_49_state.json and
  automatic live sync to data/pipeline_420_state.json.
- Progress Reporting: artifacts/reports/reparse_english_49_progress.md.
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
from typing import Any, Dict, List, Optional

# Enforce agy account and concurrency in environment
os.environ["AGY_ACCOUNT"] = "kien.duet"
os.environ["AGY_CONCURRENCY_LIMIT"] = "4"

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.ocr_document import OCRDocument
from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.llm.agy_rate_limiter import Agy5HourCapError
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import parse_xml_with_anchors
from sequence_labelling.parser.long_parser.greedy_chunker import (
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.parser.long_parser.sequence_reconciler import reconcile_parser_chunk_results

DEFAULT_STATE_FILE = Path("data/reparse_english_49_state.json")
DEFAULT_PIPELINE_420_STATE = Path("data/pipeline_420_state.json")
DEFAULT_REPORT_FILE = Path("artifacts/reports/reparse_english_49_progress.md")
DEFAULT_LOG_FILE = Path("artifacts/reparse_english_49.log")

Path("artifacts/reports").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("reparse_english_49")


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
    def __init__(self, state_path: Path, p420_path: Path):
        self.state_path = state_path
        self.p420_path = p420_path
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
            "total_target_files": 49,
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
        """Syncs the updated document record into data/pipeline_420_state.json."""
        if not self.p420_path.exists():
            return
        try:
            with open(self.p420_path, "r", encoding="utf-8") as f:
                p420_data = json.load(f)

            if "docs" in p420_data and stem in p420_data["docs"]:
                prev_record = p420_data["docs"][stem]
                prev_record["status"] = record.get("status", "DONE")
                prev_record["initial_score"] = record.get("initial_score", prev_record.get("initial_score"))
                prev_record["final_score"] = record.get("final_score")
                prev_record["final_decision"] = record.get("final_decision")
                prev_record["score_delta"] = round(
                    record.get("final_score", 0.0) - prev_record.get("initial_score", 0.0), 2
                )
                prev_record["reparsed_with"] = "gemini-3.8-flash-low"
                prev_record["reparsed_at"] = datetime.now(timezone.utc).isoformat()
                p420_data["docs"][stem] = prev_record

                # Recalculate totals
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
                logger.info(f"🔄 Synced '{stem}' ({record.get('final_decision')} {record.get('final_score')}) to {self.p420_path.name}")
        except Exception as e:
            logger.warning(f"Could not sync to {self.p420_path}: {e}")


def generate_markdown_report(manager: StateManager, report_path: Path) -> None:
    data = manager.data
    total = data.get("total_target_files", 49)
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
        "# Báo Cáo Tiến Độ Reparse 49 Đề Tiếng Anh (Gemini 3.8 Flash Low)",
        "",
        f"> **Thời gian cập nhật:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"> **Tài khoản AGY:** `kien.duet` | **Mô hình Parser & Reviewer:** `gemini-3.8-flash-low` | **Concurrency:** 4  ",
        "",
        "## 1. Tổng Quan Tiến Độ",
        "",
        f"- **Tiến độ xử lý:** **`{done} / {total}`** đề (**`{pct:.1f}%`**)",
        f"- **Điểm trung bình:** 🚀 **`{avg_score:.2f} / 100`**",
        f"- **🏆 PASS (≥ 75 điểm & 0 critical):** **`{passed}`** đề (**`{pass_pct:.1f}%`**)",
        f"- **🌟 Dùng tốt (PASS + NEEDS_REVISION):** **`{usable_count}`** đề (**`{usable_pct:.1f}%`**)",
        f"- **📉 DISCARD (Hỏng):** **`{discard}`** đề",
        "",
        "## 2. Bảng Thống Kê Chi Tiết Từng Đề",
        "",
        "| STT | Tên Đề Thi | Initial Score | Reparse Score | Qs | Stims | Decision |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    sorted_docs = sorted(
        data.get("docs", {}).items(),
        key=lambda x: x[1].get("completed_at", ""),
        reverse=True,
    )

    for idx, (stem, d) in enumerate(sorted_docs, start=1):
        sc_init = d.get("initial_score", "N/A")
        sc_fin = d.get("final_score", 0.0)
        qs = d.get("questions_count", 0)
        stims = d.get("stimuli_count", 0)
        dec = d.get("final_decision", "UNKNOWN")
        icon = "🏆 PASS" if dec == "PASS" else ("⚠️ REVISION" if dec == "NEEDS_REVISION" else "❌ DISCARD")
        short_stem = stem[:60] + "..." if len(stem) > 60 else stem
        lines.append(f"| {idx} | {short_stem} | `{sc_init}` | **`{sc_fin:.1f}`** | {qs} | {stims} | {icon} |")

    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def process_single_exam(
    stem: str,
    raw_path: Path,
    out_dir: Path,
    initial_score: float,
    worker: ParserAgentWorker,
    reviewer: AnnotationReviewerAgent,
) -> Dict[str, Any]:
    t0 = time.time()
    doc_dir = out_dir / stem
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    raw_text = raw_path.read_text(encoding="utf-8")
    words_count = len(raw_text.split())

    # Strategy: If exam is <= 4500 words (~6000 tokens), process as 1 chunk for perfect global coherence
    # If longer (> 4500 words), partition into clean-seam chunks using greedy_oversize_chunker
    if words_count <= 4500:
        chunk_plans = [{
            "chunk_index": 0,
            "raw_chunk_text": raw_text,
            "page_ids": [1],
            "overlap_page_ids": [],
            "page_offset_ranges": [],
        }]
    else:
        ocr_doc = OCRDocument.from_markdown(raw_text, doc_id=stem)
        page_list = ocr_doc.to_chunk_pages()
        chunks = greedy_oversize_chunker(
            page_list,
            target_tokens=6000,
            max_tokens=8500,
            overlap_pages=0,
        )
        chunk_plans = [
            build_chunk_plan(cp, idx) for idx, cp in enumerate(chunks)
        ]

    chunk_results = []
    for plan in chunk_plans:
        c_idx = plan["chunk_index"]
        c_text = plan["raw_chunk_text"]
        res = worker.process_chunk(
            c_text,
            chunk_index=c_idx,
            page_ids=plan.get("page_ids", []),
            overlap_page_ids=plan.get("overlap_page_ids", []),
        )
        c_raw_xml = res.get("raw_xml", "")
        clean_c_xml = XMLCleaner.clean(c_raw_xml).cleaned_xml
        res["raw_xml"] = clean_c_xml

        # Write chunk files
        (chunks_dir / f"chunk_{c_idx}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (chunks_dir / f"chunk_{c_idx}.xml").write_text(clean_c_xml, encoding="utf-8")
        chunk_results.append(res)

    # Merge chunks
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

    merged_data = {
        "document_id": stem,
        "reparsed_at": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-3.8-flash-low",
        "provider": "agy",
        "account": "kien.duet",
        "questions_count": len(questions),
        "stimuli_count": len(stimuli),
        "questions": questions,
        "stimuli": stimuli,
    }
    (doc_dir / "merged.json").write_text(
        json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Review with AnnotationReviewerAgent
    t_rev = time.time()
    review_report = reviewer.review_document(
        xml_content=clean_merged,
        raw_ocr_text=raw_text,
        doc_id=stem,
        use_llm=True,
    )
    rev_duration = time.time() - t_rev

    report_dict = review_report.model_dump(mode="json")
    (doc_dir / "audit_report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (chunks_dir / "chunk_0.audit.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    final_score = review_report.overall_score
    final_decision = review_report.decision.value

    return {
        "doc_stem": stem,
        "category": "Tiếng Anh",
        "status": "DONE",
        "initial_score": initial_score,
        "final_score": final_score,
        "final_decision": final_decision,
        "score_delta": round(final_score - initial_score, 2),
        "questions_count": len(questions),
        "stimuli_count": len(stimuli),
        "chunks_count": len(chunk_plans),
        "duration": round(elapsed, 2),
        "review_duration": round(rev_duration, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Reparse 49 English Exams with Gemini 3.8 Flash Low")
    parser.add_argument("--concurrency", type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to reparse")
    parser.add_argument("--doc-id", type=str, default=None, help="Process a specific document stem")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    setup_logger(args.log_file)

    logger.info("=" * 70)
    logger.info("🚀 REPARSE PIPELINE: 49 ENGLISH EXAMS (GEMINI 3.8 FLASH LOW)")
    logger.info("AGY Account: kien.duet | Concurrency: %d", args.concurrency)
    logger.info("Parser: gemini-3.8-flash-low | Reviewer: gemini-3.8-flash-low")
    logger.info("=" * 70)

    # 1. Load targets from data/pipeline_420_state.json
    p420_state_path = DEFAULT_PIPELINE_420_STATE
    if not p420_state_path.exists():
        logger.error(f"Cannot find {p420_state_path}!")
        sys.exit(1)

    with open(p420_state_path, "r", encoding="utf-8") as f:
        p420 = json.load(f)

    docs = p420.get("docs", {})
    english_discards = [
        (k, v.get("final_score", 0.0))
        for k, v in docs.items()
        if "Tiếng Anh" in v.get("category", "") and v.get("final_decision") == "DISCARD"
    ]

    if args.doc_id:
        english_discards = [(k, sc) for k, sc in english_discards if k == args.doc_id]

    logger.info(f"Identified {len(english_discards)} target English DISCARD exams.")

    raw_dir = Path("data/sequence_labelling_input_data")
    out_dir = Path("data/sequence_labelling_annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_mgr = StateManager(args.state_file, p420_state_path)

    # Filter pending
    pending_items = []
    for stem, sc in english_discards:
        if state_mgr.is_done(stem):
            continue
        raw_p = raw_dir / f"{stem}.md"
        if not raw_p.exists():
            logger.warning(f"Missing raw markdown for {stem}: {raw_p}")
            continue
        pending_items.append((stem, raw_p, sc))

    if args.limit:
        pending_items = pending_items[:args.limit]

    logger.info(f"Total pending: {len(pending_items)} / {len(english_discards)}")

    if not pending_items:
        logger.info("All documents already completed! Exiting.")
        generate_markdown_report(state_mgr, args.report_file)
        return

    # Worker & Reviewer instance factory per thread
    def make_worker():
        return ParserAgentWorker(
            model="gemini-3.8-flash-low",
            provider="agy",
            thinking="low",
            max_attempts=2,
        )

    def make_reviewer():
        return AnnotationReviewerAgent(
            model="gemini-3.8-flash-low",
            provider="agy",
            thinking="low",
        )

    completed_count = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="reparse_worker") as executor:
        futures = {}
        for stem, raw_p, init_sc in pending_items:
            w = make_worker()
            r = make_reviewer()
            fut = executor.submit(
                process_single_exam,
                stem=stem,
                raw_path=raw_p,
                out_dir=out_dir,
                initial_score=init_sc,
                worker=w,
                reviewer=r,
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
                icon = "🏆" if dec == "PASS" else ("⚠️" if dec == "NEEDS_REVISION" else "❌")

                logger.info(
                    f"{icon} [{completed_count}/{len(pending_items)}] '{stem[:50]}...': "
                    f"{dec} ({init_sc:.1f} -> {sc:.1f}) in {dur:.1f}s"
                )
                generate_markdown_report(state_mgr, args.report_file)

            except Exception as e:
                logger.error(f"❌ Failed processing '{stem}': {e}", exc_info=True)
                state_mgr.record_completed(stem, {
                    "doc_stem": stem,
                    "category": "Tiếng Anh",
                    "status": "ERROR",
                    "error": str(e),
                    "duration": 0.0,
                })

    total_time = time.time() - start_all
    logger.info("=" * 70)
    logger.info(f"🏁 ALL 49 ENGLISH EXAMS REPARSED IN {total_time / 60:.2f} MINUTES")
    logger.info(f"Pass: {state_mgr.data['pass_count']} | Revision: {state_mgr.data['needs_revision_count']} | Discard: {state_mgr.data['discard_count']}")
    logger.info("=" * 70)
    generate_markdown_report(state_mgr, args.report_file)


if __name__ == "__main__":
    main()
