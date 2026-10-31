"""
run_pipeline_420_editor.py — Batch Reviewer & Editor Repair Orchestrator for 420 Exams.

Features:
- Models: gpt-6-luna via aixingialaire provider (reviewer & editor).
- Concurrency: 2 worker threads (thread-safe).
- Granularity: chunk-level staging in data/sequence_labelling_annotated/<doc_stem>/chunks/chunk_0.xml + merged.xml.
- Full Resumption: Atomic checkpoint in data/pipeline_420_state.json.
- Pass-Through: Automatically promotes previously PASS-audited documents without redundant API calls.
- Real-time Markdown Report: artifacts/reports/pipeline_420_progress.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision
from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop, RevisionLoopResult
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import parse_xml_with_anchors

DEFAULT_STATE_FILE = Path("data/pipeline_420_state.json")
DEFAULT_REPORT_FILE = Path("artifacts/reports/pipeline_420_progress.md")
DEFAULT_LOG_FILE = Path("artifacts/pipeline_420.log")

logger = logging.getLogger("pipeline_420")


def clean_page_tags(text: str) -> str:
    cleaned = re.sub(r'<\/?(?:pages?|page_metadata)[^>]*>', '', text)
    cleaned = re.sub(r'\{"p":\s*\d+,\s*"w":\s*\d+,\s*"h":\s*\d+\}', '', cleaned)
    return cleaned.strip()


class PipelineStateTracker:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file {self.state_path}: {e}")
        return {
            "total_target_files": 420,
            "completed_count": 0,
            "pass_count": 0,
            "needs_revision_count": 0,
            "discard_count": 0,
            "total_duration": 0.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "docs": {},
        }

    def save(self) -> None:
        with self.lock:
            self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
            tmp_path = self.state_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self.state_path)

    def is_done(self, stem: str) -> bool:
        with self.lock:
            doc = self.data["docs"].get(stem)
            return doc is not None and doc.get("status") == "DONE"

    def record_completed(self, stem: str, record: Dict[str, Any]) -> None:
        with self.lock:
            self.data["docs"][stem] = record
            self.data["completed_count"] = sum(1 for d in self.data["docs"].values() if d.get("status") == "DONE")
            self.data["pass_count"] = sum(1 for d in self.data["docs"].values() if d.get("final_decision") == "PASS")
            self.data["needs_revision_count"] = sum(1 for d in self.data["docs"].values() if d.get("final_decision") == "NEEDS_REVISION")
            self.data["discard_count"] = sum(1 for d in self.data["docs"].values() if d.get("final_decision") == "DISCARD")
            self.data["total_duration"] += record.get("duration", 0.0)
        self.save()


def generate_markdown_report(tracker: PipelineStateTracker, report_path: Path) -> None:
    data = tracker.data
    total = data.get("total_target_files", 420)
    done = data.get("completed_count", 0)
    passes = data.get("pass_count", 0)
    needs_rev = data.get("needs_revision_count", 0)
    discards = data.get("discard_count", 0)
    total_dur = data.get("total_duration", 0.0)
    avg_dur = (total_dur / done) if done > 0 else 0.0
    pass_rate = (passes / done * 100.0) if done > 0 else 0.0

    scores = [d.get("final_score", 0.0) for d in data["docs"].values() if d.get("status") == "DONE"]
    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    lines = [
        "# Batch 420 Exams — Reviewer & Editor Repair Report",
        "",
        f"- **Model**: `gpt-6-luna` (Provider: `aixingialaire`)",
        f"- **Last Updated**: `{data.get('last_updated', '')}`",
        f"- **Progress**: `{done}/{total}` ({done/total*100:.1f}%)",
        f"- **Current Pass Rate**: **`{pass_rate:.1f}%`** ({passes} PASS)",
        f"- **Needs Revision**: `{needs_rev}` ({needs_rev/total*100:.1f}%)",
        f"- **Discard**: `{discards}` ({discards/total*100:.1f}%)",
        f"- **Average Composite Score**: **`{avg_score:.2f} / 100`**",
        f"- **Avg Duration per Doc**: `{avg_dur:.1f}s`",
        "",
        "## Subject Breakdown",
        "",
        "| Subject / Category | Total | Completed | Pass | Needs Revision | Discard | Avg Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Aggregate by category
    by_cat: Dict[str, Dict[str, Any]] = {}
    for d in data["docs"].values():
        if d.get("status") != "DONE":
            continue
        cat = d.get("category", "Unknown")
        if cat not in by_cat:
            by_cat[cat] = {"count": 0, "pass": 0, "rev": 0, "disc": 0, "scores": []}
        by_cat[cat]["count"] += 1
        dec = d.get("final_decision", "")
        if dec == "PASS":
            by_cat[cat]["pass"] += 1
        elif dec == "NEEDS_REVISION":
            by_cat[cat]["rev"] += 1
        elif dec == "DISCARD":
            by_cat[cat]["disc"] += 1
        by_cat[cat]["scores"].append(d.get("final_score", 0.0))

    for cat, stats in sorted(by_cat.items(), key=lambda x: x[0]):
        c_tot = stats["count"]
        c_pass = stats["pass"]
        c_rev = stats["rev"]
        c_disc = stats["disc"]
        c_avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        lines.append(f"| {cat} | {c_tot} | {c_tot} | {c_pass} | {c_rev} | {c_disc} | **{c_avg:.1f}** |")

    lines.extend([
        "",
        "## Recent Completed Files",
        "",
        "| Document | Category | Initial | Final | Delta | Decision | Time |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    recent = list(data["docs"].values())[-25:]
    for r in reversed(recent):
        d_name = r.get("doc_stem", "")[:45]
        cat = r.get("category", "")[:15]
        i_sc = r.get("initial_score", 0.0)
        f_sc = r.get("final_score", 0.0)
        delta = r.get("score_delta", 0.0)
        dec = r.get("final_decision", "")
        dur = r.get("duration", 0.0)
        lines.append(f"| {d_name}... | {cat} | {i_sc:.1f} | {f_sc:.1f} | `{delta:+0.1f}` | **{dec}** | {dur:.1f}s |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def process_single_exam(
    stem: str,
    category: str,
    raw_path: Path,
    pred_path: Path,
    out_dir: Path,
    eval_cache: Optional[Dict[str, Any]],
    editor_loop: EditorReviewerLoop,
    clean_fn=clean_page_tags
) -> Dict[str, Any]:
    t0 = time.time()
    doc_dir = out_dir / stem
    chunks_dir = doc_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    raw_text = clean_fn(raw_path.read_text(encoding="utf-8"))
    xml_text = clean_fn(pred_path.read_text(encoding="utf-8"))

    init_score = eval_cache.get("score", 70.0) if eval_cache else 70.0
    init_decision = eval_cache.get("decision", "NEEDS_REVISION") if eval_cache else "NEEDS_REVISION"

    # Fast-promote if already PASS with score >= 75 and 0 critical defects
    if eval_cache and init_decision == "PASS" and init_score >= 75.0:
        defects = eval_cache.get("defects", {})
        if defects.get("critical", 0) == 0:
            final_xml = xml_text
            repaired_path = doc_dir / "merged.xml"
            chunk_0_path = chunks_dir / "chunk_0.xml"
            repaired_path.write_text(final_xml, encoding="utf-8")
            chunk_0_path.write_text(final_xml, encoding="utf-8")

            # Parse structured questions
            try:
                spans, stimuli, questions = parse_xml_with_anchors(raw_text, final_xml)
                json_data = {
                    "document_id": stem,
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                    "questions_count": len(questions),
                    "questions": questions,
                    "stimuli": stimuli,
                }
                (doc_dir / "merged.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            audit_info = {
                "doc_id": stem,
                "overall_score": init_score,
                "decision": "PASS",
                "promoted_from_cache": True,
            }
            (doc_dir / "audit_report.json").write_text(json.dumps(audit_info, ensure_ascii=False, indent=2), encoding="utf-8")
            (chunks_dir / "chunk_0.audit.json").write_text(json.dumps(audit_info, ensure_ascii=False, indent=2), encoding="utf-8")

            elapsed = time.time() - t0
            return {
                "doc_stem": stem,
                "category": category,
                "status": "DONE",
                "initial_score": init_score,
                "initial_decision": "PASS",
                "final_score": init_score,
                "final_decision": "PASS",
                "score_delta": 0.0,
                "patches_applied": 0,
                "duration": round(elapsed, 2),
                "promoted": True,
            }

    # Run surgical editor & reviewer loop
    try:
        loop_res: RevisionLoopResult = editor_loop.run(
            annotated_xml=xml_text,
            raw_ocr_text=raw_text,
            doc_id=stem
        )
        final_xml = loop_res.repaired_xml if loop_res.repaired_xml else xml_text
        repaired_path = doc_dir / "merged.xml"
        chunk_0_path = chunks_dir / "chunk_0.xml"
        repaired_path.write_text(final_xml, encoding="utf-8")
        chunk_0_path.write_text(final_xml, encoding="utf-8")

        # Parse structured questions
        try:
            spans, stimuli, questions = parse_xml_with_anchors(raw_text, final_xml)
            json_data = {
                "document_id": stem,
                "repaired_at": datetime.now(timezone.utc).isoformat(),
                "questions_count": len(questions),
                "questions": questions,
                "stimuli": stimuli,
            }
            (doc_dir / "merged.json").write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        audit_info = {
            "doc_id": stem,
            "overall_score": loop_res.final_score,
            "decision": loop_res.final_decision,
            "rounds_completed": loop_res.rounds_completed,
            "applied_patches_count": loop_res.applied_patches_count,
        }
        (doc_dir / "audit_report.json").write_text(json.dumps(audit_info, ensure_ascii=False, indent=2), encoding="utf-8")
        (chunks_dir / "chunk_0.audit.json").write_text(json.dumps(audit_info, ensure_ascii=False, indent=2), encoding="utf-8")

        elapsed = time.time() - t0
        delta = round(loop_res.final_score - loop_res.initial_score, 2)
        return {
            "doc_stem": stem,
            "category": category,
            "status": "DONE",
            "initial_score": loop_res.initial_score,
            "initial_decision": loop_res.initial_decision,
            "final_score": loop_res.final_score,
            "final_decision": loop_res.final_decision,
            "score_delta": delta,
            "patches_applied": loop_res.applied_patches_count,
            "duration": round(elapsed, 2),
            "promoted": False,
        }

    except Exception as err:
        logger.exception(f"Error repairing {stem}: {err}")
        elapsed = time.time() - t0
        return {
            "doc_stem": stem,
            "category": category,
            "status": "ERROR",
            "error": str(err),
            "duration": round(elapsed, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch 420 Exams Reviewer & Editor Repair Pipeline")
    parser.add_argument("--concurrency", "-c", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-6-luna")
    parser.add_argument("--provider", type=str, default="aixingialaire")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--limit", "-l", type=int, default=None)
    args = parser.parse_args()

    # Setup logger
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("STARTING BATCH 420 REVIEWER & EDITOR PIPELINE")
    logger.info(f"Model: {args.model} | Provider: {args.provider} | Concurrency: {args.concurrency}")
    logger.info("=" * 70)

    # Load inputs
    manifest_path = Path("data/batch_420_exams_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    eval_v3_path = Path("data/eval_v3_state.json")
    eval_cache: Dict[str, Any] = {}
    if eval_v3_path.exists():
        with open(eval_v3_path, "r", encoding="utf-8") as f:
            eval_cache = json.load(f).get("evaluated_docs", {})

    raw_dir = Path("data/sequence_labelling_input_data")
    pred_dir = Path("data/predictions_model_v3")
    out_dir = Path("data/sequence_labelling_annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = PipelineStateTracker(args.state_file)

    # Build queue of pending items
    items_to_process = []
    for item in manifest:
        stem = Path(item["pdf_path"]).stem
        cat = item.get("category", "General")
        raw_p = raw_dir / f"{stem}.md"
        pred_p = pred_dir / f"{stem}.xml"

        if not raw_p.exists() or not pred_p.exists():
            logger.warning(f"Skipping {stem}: missing markdown or predicted xml")
            continue

        if tracker.is_done(stem):
            continue

        items_to_process.append((stem, cat, raw_p, pred_p, eval_cache.get(stem)))

    if args.limit:
        items_to_process = items_to_process[:args.limit]

    total_pending = len(items_to_process)
    logger.info(f"Total target files: {len(manifest)} | Pending files to process: {total_pending}")

    if total_pending == 0:
        logger.info("All files already completed in state! Exiting.")
        generate_markdown_report(tracker, args.report_file)
        return

    # Thread-local loop initializer
    def get_editor_loop():
        rev = AnnotationReviewerAgent(
            model=args.model,
            provider=args.provider,
            thinking="low"
        )
        edt = EditorAgent(
            model=args.model,
            provider=args.provider,
            thinking="low",
            reviewer=rev
        )
        return EditorReviewerLoop(editor=edt, reviewer=rev, max_rounds=2)

    completed_in_session = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for stem, cat, raw_p, pred_p, ev in items_to_process:
            loop = get_editor_loop()
            fut = executor.submit(
                process_single_exam,
                stem=stem,
                category=cat,
                raw_path=raw_p,
                pred_path=pred_p,
                out_dir=out_dir,
                eval_cache=ev,
                editor_loop=loop,
            )
            futures[fut] = stem

        for future in as_completed(futures):
            stem = futures[future]
            try:
                res = future.result()
                tracker.record_completed(stem, res)
                completed_in_session += 1

                dec = res.get("final_decision", "UNKNOWN")
                sc = res.get("final_score", 0.0)
                delta = res.get("score_delta", 0.0)
                dur = res.get("duration", 0.0)
                promoted = " [PROMOTED]" if res.get("promoted") else ""

                logger.info(
                    f"[{tracker.data['completed_count']}/{tracker.data['total_target_files']}] "
                    f"{dec} ({sc:.1f} pts, delta {delta:+0.1f}) | {stem[:40]}... "
                    f"({dur:.1f}s){promoted}"
                )

                if completed_in_session % 5 == 0 or completed_in_session == total_pending:
                    generate_markdown_report(tracker, args.report_file)

            except Exception as e:
                logger.error(f"Worker exception for {stem}: {e}")

    generate_markdown_report(tracker, args.report_file)
    logger.info("=" * 70)
    logger.info("BATCH PIPELINE FINISHED SUCCESSFULLY!")
    logger.info(f"Report saved to {args.report_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
