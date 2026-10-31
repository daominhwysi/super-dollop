"""
Experiment: Testing Reviewer-Editor Loop on 5 Model v3 Predicted Exams.
Uses gpt-6-luna via aixingialaire provider with max_rounds=2.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision
from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop, RevisionLoopResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/revision_experiment_5.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("revision_experiment")


def clean_page_tags(text: str) -> str:
    cleaned = re.sub(r'<\/?(?:pages?|page_metadata)[^>]*>', '', text)
    cleaned = re.sub(r'\{"p":\s*\d+,\s*"w":\s*\d+,\s*"h":\s*\d+\}', '', cleaned)
    return cleaned.strip()


def run_experiment():
    target_names = [
        "Đề Kiểm Tra Giữa Học Kỳ 1 GDCD 9 Năm 2025-2026 Có Đáp Án Ma Trận Đặc Tả",
        "Đề Ôn Tập Giữa Học Kỳ 2 Tin 8 Kết Nối Tri Thức Có Đáp Án Ma Trận",
        "Bài Tập Tiếng Anh 9 Global Success Review 2 Có Đáp Án",
        "Đề Kiểm Tra Cuối Kì 1 Ngữ Văn 6 Chân Trời Sáng Tạo 21-22 Có Đáp Án Và Ma Trận Đặc Tả",
        "Đề Thi Olimpic Môn Địa 11 Sở GD&ĐT Quảng Nam 2019 Có Đáp Án"
    ]

    raw_dir = Path("data/sequence_labelling_input_data")
    pred_dir = Path("data/predictions_model_v3")
    out_dir = Path("data/repaired_experiment_5")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    logger.info("Initializing gpt-6-luna Reviewer & Editor via aixingialaire...")
    reviewer = AnnotationReviewerAgent(
        model="gpt-6-luna",
        provider="aixingialaire",
        thinking="low"
    )
    editor = EditorAgent(
        model="gpt-6-luna",
        provider="aixingialaire",
        thinking="low",
        reviewer=reviewer
    )
    loop = EditorReviewerLoop(
        editor=editor,
        reviewer=reviewer,
        max_rounds=2
    )

    results = []

    for idx, name in enumerate(target_names, 1):
        raw_path = raw_dir / f"{name}.md"
        pred_path = pred_dir / f"{name}.xml"

        if not raw_path.exists() or not pred_path.exists():
            logger.error(f"Missing files for {name}")
            continue

        raw_text = clean_page_tags(raw_path.read_text(encoding="utf-8"))
        xml_text = clean_page_tags(pred_path.read_text(encoding="utf-8"))

        logger.info("=" * 70)
        logger.info(f"[{idx}/5] Processing: {name}")
        logger.info(f"Source length: {len(raw_text)} chars | XML length: {len(xml_text)} chars")

        t0 = time.time()
        try:
            loop_res: RevisionLoopResult = loop.run(
                annotated_xml=xml_text,
                raw_ocr_text=raw_text,
                doc_id=name
            )
            elapsed = time.time() - t0

            # Save repaired XML
            repaired_path = out_dir / f"{name}.xml"
            repaired_path.write_text(loop_res.repaired_xml, encoding="utf-8")

            # Collect round metrics
            rounds_summary = []
            for r in loop_res.rounds:
                rounds_summary.append({
                    "round_number": r.round_number,
                    "applied_patches": r.applied_patches_count,
                    "failed_reasons": r.failed_patches_reasons,
                })

            item = {
                "doc_name": name,
                "initial_score": loop_res.initial_score,
                "initial_decision": loop_res.initial_decision,
                "final_score": loop_res.final_score,
                "final_decision": loop_res.final_decision,
                "score_delta": round(loop_res.final_score - loop_res.initial_score, 2),
                "success": loop_res.success,
                "rounds_count": len(loop_res.rounds),
                "rounds_details": rounds_summary,
                "duration_seconds": round(elapsed, 1),
                "repaired_xml_path": str(repaired_path),
            }
            results.append(item)

            logger.info(
                f"==> RESULT: {loop_res.initial_decision} ({loop_res.initial_score} pts) -> "
                f"{loop_res.final_decision} ({loop_res.final_score} pts) | "
                f"Delta: {item['score_delta']:+0.1f} | Success: {loop_res.success} | Time: {elapsed:.1f}s"
            )

        except Exception as err:
            logger.exception(f"Error repairing {name}: {err}")
            results.append({
                "doc_name": name,
                "error": str(err)
            })

    # Save summary report
    out_json = Path("artifacts/revision_experiment_5_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT FINISHED. Saved to {out_json}")


if __name__ == "__main__":
    run_experiment()
