#!/usr/bin/env python3
"""
Comprehensive End-to-End Evaluation of ModernBERT v3 (sealior-modernbert-vi-exam-seq-labeling)
Across 420 Real Exam Benchmarks with GPT-6 Luna LLM Reviewer.

Features:
- Fix-cứng: Automatically skips <section> tags in model predictions (ignore_section=True).
- Reviewer prompt calibrated to strictly exempt / ignore <section> tags.
- Full atomic checkpointing & resume via data/eval_v3_state.json.
- Real-time score aggregation & live Markdown report generation.
"""

import os
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import time
import json
import logging
import argparse
from typing import Dict, Any, List
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoConfig
from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping
from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent

# Logging setup
LOG_FILE = WORKSPACE_DIR / "artifacts" / "eval_model_v3.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    ]
)
logger = logging.getLogger("eval_v3")


def clean_page_tags(text: str) -> str:
    """Strips raw OCR page boundary markup before sequence labelling inference."""
    import re
    text = re.sub(r'<\/?page\b[^>]*>', '', text)
    text = re.sub(r'<\/?page_metadata\b[^>]*>', '', text)
    text = re.sub(r'<\/?pages\b[^>]*>', '', text)
    text = re.sub(r'\{\s*"p":\s*\d+,\s*"w":\s*\d+,\s*"h":\s*\d+\s*\}', '', text)
    return text.strip()


class EvalState:
    """Thread-safe and atomic evaluation state tracker."""
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.data: Dict[str, Any] = {
            "model_id": "daominhwysi/sealior-modernbert-vi-exam-seq-labeling",
            "total_target_files": 420,
            "completed_count": 0,
            "total_inference_time": 0.0,
            "total_review_time": 0.0,
            "average_composite_score": 0.0,
            "decision_counts": {"PASS": 0, "NEEDS_REVISION": 0, "DISCARD": 0},
            "category_metrics": {},
            "evaluated_docs": {}
        }
        self.load()

    def load(self):
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
            except Exception as e:
                logger.warning(f"Could not load existing state file ({e}), starting fresh.")

    def save(self):
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.state_path)

    def record_doc(self, stem: str, category: str, score: float, decision: str,
                   defects: Dict[str, int], summary: str, t_infer: float, t_rev: float,
                   spans_count: int, xml_path: str):
        self.data["evaluated_docs"][stem] = {
            "category": category,
            "score": round(score, 2),
            "decision": decision,
            "defects": defects,
            "summary": summary,
            "inference_duration": round(t_infer, 2),
            "review_duration": round(t_rev, 2),
            "spans_count": spans_count,
            "xml_path": xml_path
        }
        self.data["completed_count"] = len(self.data["evaluated_docs"])
        self.data["total_inference_time"] += t_infer
        self.data["total_review_time"] += t_rev

        # Recompute totals
        scores = [d["score"] for d in self.data["evaluated_docs"].values()]
        self.data["average_composite_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0

        dec_counts = Counter(d["decision"] for d in self.data["evaluated_docs"].values())
        self.data["decision_counts"] = {
            "PASS": dec_counts.get("PASS", 0),
            "NEEDS_REVISION": dec_counts.get("NEEDS_REVISION", 0),
            "DISCARD": dec_counts.get("DISCARD", 0)
        }

        # Category metrics
        cat_metrics = {}
        for d in self.data["evaluated_docs"].values():
            cat = d.get("category", "Unknown")
            if cat not in cat_metrics:
                cat_metrics[cat] = {"count": 0, "total_score": 0.0, "decisions": Counter()}
            cat_metrics[cat]["count"] += 1
            cat_metrics[cat]["total_score"] += d["score"]
            cat_metrics[cat]["decisions"][d["decision"]] += 1

        formatted_cats = {}
        for cat, val in cat_metrics.items():
            formatted_cats[cat] = {
                "count": val["count"],
                "avg_score": round(val["total_score"] / val["count"], 2),
                "pass_rate": round(val["decisions"]["PASS"] / val["count"] * 100, 1),
                "pass_count": val["decisions"]["PASS"],
                "needs_revision_count": val["decisions"]["NEEDS_REVISION"],
                "discard_count": val["decisions"]["DISCARD"]
            }
        self.data["category_metrics"] = formatted_cats
        self.save()


def generate_markdown_report(state: EvalState, report_path: Path):
    """Generates comprehensive Github-flavored markdown report."""
    data = state.data
    total = data["total_target_files"]
    done = data["completed_count"]
    pct = (done / total * 100) if total else 0.0
    avg_score = data["average_composite_score"]
    dec_counts = data["decision_counts"]
    pass_pct = (dec_counts["PASS"] / done * 100) if done else 0.0

    lines = [
        "# 📊 Đánh Giá Benchmark Toàn Diện: ModernBERT v3 (sealior-modernbert-vi-exam-seq-labeling)",
        "",
        f"> **Thời điểm cập nhật:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **Backbone Model:** `{data['model_id']}` (312M params, 8192 context)",
        "> **Reviewer Agent:** `gpt-6-luna` via provider `aixingialaire`",
        "> **Cấu hình đặc biệt:** **Fix cứng bỏ qua `<section>` tag** (không dự đoán và reviewer không chấm điểm section).",
        "",
        "---",
        "",
        "## 1. Kết Quả Tổng Quan",
        "",
        f"- **Tiến độ hoàn thành:** **`{done} / {total}` đề** (**{pct:.1f}%**)",
        f"- **Điểm Composite Trung Bình:** **`{avg_score} / 100`**",
        f"- **Tỷ lệ PASS:** **`{pass_pct:.1f}%`** (`{dec_counts['PASS']}` PASS / `{dec_counts['NEEDS_REVISION']}` NEEDS_REVISION / `{dec_counts['DISCARD']}` DISCARD)",
        f"- **Thời gian trung bình mỗi đề:** Inference: ~{data['total_inference_time']/(done or 1):.1f}s | Reviewer: ~{data['total_review_time']/(done or 1):.1f}s",
        "",
        "---",
        "",
        "## 2. Bảng Điểm Chi Tiết Theo Từng Môn Học",
        "",
        "| Môn học | Số đề đã chấm | Điểm TB (/100) | Tỷ lệ PASS | PASS | NEEDS_REVISION | DISCARD |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cat, m in sorted(data.get("category_metrics", {}).items(), key=lambda x: -x[1]["avg_score"]):
        lines.append(
            f"| **{cat}** | {m['count']} | **{m['avg_score']}** | {m['pass_rate']}% | {m['pass_count']} | {m['needs_revision_count']} | {m['discard_count']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Phân Phối Điểm Số & Quyết Định Reviewer",
        "",
        "```mermaid",
        "xychart-beta",
        '    title "Phân Bố Quyết Định Đánh Giá Của Reviewer (GPT-6 Luna)"',
        '    x-axis ["PASS", "NEEDS_REVISION", "DISCARD"]',
        f'    y-axis "Số lượng đề" 0 --> {max(total, 50)}',
        f'    bar [{dec_counts["PASS"]}, {dec_counts["NEEDS_REVISION"]}, {dec_counts["DISCARD"]}]',
        "```",
        "",
        "---",
        "",
        "## 4. Danh Sách 15 Đề Đánh Giá Gần Nhất",
        "",
        "| Tên đề thi | Môn | Điểm | Quyết định | Spans | Tóm tắt đánh giá của Reviewer |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |"
    ])

    recent_items = list(data["evaluated_docs"].items())[-15:]
    recent_items.reverse()
    for stem, d in recent_items:
        clean_name = stem[:35] + ("..." if len(stem) > 35 else "")
        summary_snip = d.get("summary", "").replace("\n", " ")[:60] + "..."
        lines.append(
            f"| `{clean_name}` | {d.get('category', 'N/A')} | **{d['score']}** | `{d['decision']}` | {d.get('spans_count', 0)} | {summary_snip} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Model v3 Evaluation on 420 Exams with GPT-6 Luna")
    parser.add_argument("--model-id", type=str, default="daominhwysi/sealior-modernbert-vi-exam-seq-labeling")
    parser.add_argument("--manifest", type=str, default="data/batch_420_exams_manifest.json")
    parser.add_argument("--input-dir", type=str, default="data/sequence_labelling_input_data")
    parser.add_argument("--output-dir", type=str, default="data/predictions_model_v3")
    parser.add_argument("--state-file", type=str, default="data/eval_v3_state.json")
    parser.add_argument("--report-file", type=str, default="artifacts/reports/evaluation_model_v3_sealior_420.md")
    parser.add_argument("--reviewer-model", type=str, default="gpt-6-luna")
    parser.add_argument("--reviewer-provider", type=str, default="aixingialaire")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride (default 256)")
    parser.add_argument("--max-length", type=int, default=1024, help="Window size (default 1024)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = Path(args.state_file)
    report_file = Path(args.report_file)

    state = EvalState(state_file)

    # Load manifest
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_entries = json.load(f)

    # Build lookup map: stem -> entry
    manifest_map = {}
    for e in manifest_entries:
        stem = Path(e["filename"]).stem
        manifest_map[stem] = e

    state.data["total_target_files"] = len(manifest_entries)
    state.save()

    logger.info("=" * 60)
    logger.info("STARTING MODEL v3 EVALUATION PIPELINE")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Target Exams: {len(manifest_entries)}")
    logger.info(f"Already Evaluated: {len(state.data['evaluated_docs'])}")
    logger.info(f"Reviewer: {args.reviewer_model} ({args.reviewer_provider})")
    logger.info("Section Policy: STRICTLY EXEMPT / IGNORED")
    logger.info("=" * 60)

    # Load Model onto CPU
    logger.info("Loading ModernBERT tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tag_to_id, id_to_tag = load_label_mapping(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id)

    model = EnhancedBertForTokenClassification.from_pretrained(
        args.model_id,
        config=config,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    model.eval()
    logger.info("Model loaded successfully!")

    # Initialize Reviewer Agent
    reviewer = AnnotationReviewerAgent(
        model=args.reviewer_model,
        provider=args.reviewer_provider
    )

    entries_to_process = []
    for e in manifest_entries:
        stem = Path(e["filename"]).stem
        if stem not in state.data["evaluated_docs"]:
            entries_to_process.append(e)

    if args.limit:
        entries_to_process = entries_to_process[:args.limit]

    logger.info(f"Remaining exams to evaluate: {len(entries_to_process)}")

    for idx, e in enumerate(entries_to_process, start=1):
        stem = Path(e["filename"]).stem
        category = e.get("category", "Unknown")
        md_file = input_dir / f"{stem}.md"

        if not md_file.exists():
            logger.warning(f"Markdown file missing: {md_file.name}, skipping.")
            continue

        raw_text = clean_page_tags(md_file.read_text(encoding="utf-8"))
        xml_out = output_dir / f"{stem}.xml"

        # 1. Model Inference
        t0 = time.time()
        try:
            pred = predict_text(
                raw_text,
                model=model,
                tokenizer=tokenizer,
                id_to_tag=id_to_tag,
                device="cpu",
                max_length=args.max_length,
                stride=args.stride,
                batch_size=4,
                use_viterbi=True,
                ignore_section=True
            )
            t_infer = time.time() - t0
            xml_content = pred["xml_text"]
            xml_out.write_text(xml_content, encoding="utf-8")
        except Exception as err:
            logger.error(f"Inference error on {stem}: {err}")
            continue

        # 2. LLM Reviewer
        t0_rev = time.time()
        try:
            review_res = reviewer.review_document(
                xml_content=xml_content,
                raw_ocr_text=raw_text,
                doc_id=stem,
                use_llm=True
            )
            t_rev = time.time() - t0_rev

            score = review_res.overall_score
            decision = review_res.decision.value if hasattr(review_res.decision, "value") else str(review_res.decision)
            defects = {
                "critical": sum(1 for i in review_res.issues if "CRITICAL" in str(getattr(i, "severity", "")).upper()),
                "major": sum(1 for i in review_res.issues if "MAJOR" in str(getattr(i, "severity", "")).upper()),
                "minor": sum(1 for i in review_res.issues if "MINOR" in str(getattr(i, "severity", "")).upper())
            }
            summary = review_res.summary

            state.record_doc(
                stem=stem,
                category=category,
                score=score,
                decision=decision,
                defects=defects,
                summary=summary,
                t_infer=t_infer,
                t_rev=t_rev,
                spans_count=len(pred.get("spans", [])),
                xml_path=str(xml_out)
            )

            done_count = state.data["completed_count"]
            total_count = state.data["total_target_files"]
            logger.info(
                f"[{done_count}/{total_count}] {decision} ({score:.1f} pts) | {stem[:40]}... "
                f"({len(pred['spans'])} spans, infer {t_infer:.1f}s, rev {t_rev:.1f}s)"
            )

            # Update live report
            if done_count % 3 == 0 or done_count == total_count:
                generate_markdown_report(state, report_file)

        except Exception as err:
            logger.error(f"Reviewer error on {stem}: {err}")

    # Final report generation
    generate_markdown_report(state, report_file)
    logger.info("=" * 60)
    logger.info("EVALUATION RUN COMPLETE!")
    logger.info(f"Total Evaluated: {state.data['completed_count']}")
    logger.info(f"Average Composite Score: {state.data['average_composite_score']}")
    logger.info(f"Pass Rate: {state.data['decision_counts']['PASS']} / {state.data['completed_count']}")
    logger.info(f"Report saved to: {report_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
