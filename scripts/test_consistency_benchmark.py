"""
Benchmark script to test reviewer consistency between:
1. Codex (gpt-5.6-luna)
2. Antigravity Agent (gemini-3.8-flash with medium thinking)
across 12 discarded and revision documents.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision

TARGET_DOCS = [
    # 6 Discarded documents
    "exam-007--333e58e1279b",
    "exam-019--6bf349fe480f",
    "exam-018--0ccd581501db",
    "exam-025--4a19980384f5",
    "exam-035--2eb29d0b34d2",
    "exam-052--42b985ae82c1",
    # 6 Revision documents
    "exam-003--7c16cef8c9ae",
    "exam-008--b9f38d223463",
    "exam-013--a15575867816",
    "exam-023--95801fdfe9b3",
    "exam-030--c7d0ff6da9d2",
    "exam-038--b18813c0d039",
]

OUTPUT_JSON = Path("backend/logs/reviewer_consistency_benchmark.json")


def load_doc_data(doc_id: str) -> tuple[str, str]:
    ann_dir = Path("data/sequence_labelling_annotated")
    raw_dir = Path("data/sequence_labelling_input_data")

    xml_path = ann_dir / doc_id / "merged.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")

    raw_path = raw_dir / f"{doc_id}.md"
    if not raw_path.exists():
        raw_path = raw_dir / f"{doc_id}.txt"
    if not raw_path.exists():
        raw_path = raw_dir / doc_id / "raw_ocr.md"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw OCR for {doc_id}")

    return xml_path.read_text(encoding="utf-8"), raw_path.read_text(encoding="utf-8")


def run_benchmark():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Load previous review report for reference baseline
    orig_map = {}
    orig_report_path = Path("data/review_report.json")
    if orig_report_path.exists():
        orig_data = json.loads(orig_report_path.read_text(encoding="utf-8"))
        for rep in orig_data.get("reports", []):
            orig_map[rep["doc_id"]] = {
                "orig_decision": rep.get("decision"),
                "orig_score": rep.get("overall_score"),
                "orig_det_score": rep.get("deterministic_score"),
                "orig_llm_score": rep.get("llm_score"),
                "orig_issues_count": len(rep.get("issues", [])),
            }

    rev_codex = AnnotationReviewerAgent(
        model="gpt-5.6-luna", provider="codex", thinking="low"
    )
    rev_agy = AnnotationReviewerAgent(
        model="gemini-3.8-flash", provider="agy", thinking="medium"
    )

    results: List[Dict[str, Any]] = []
    # Resume support if file exists
    if OUTPUT_JSON.exists():
        try:
            cached = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                results = cached
        except Exception:
            results = []

    evaluated_doc_ids = {r["doc_id"] for r in results}

    print("=" * 80)
    print("STARTING REVIEWER CONSISTENCY BENCHMARK")
    print(f"Total target documents : {len(TARGET_DOCS)}")
    print(f"Codex Model            : gpt-5.6-luna (thinking: low)")
    print(f"AGY Model              : gemini-3.8-flash (thinking: medium)")
    print(f"Already evaluated      : {len(evaluated_doc_ids)}")
    print("=" * 80)

    for idx, doc_id in enumerate(TARGET_DOCS, 1):
        if doc_id in evaluated_doc_ids:
            print(f"[{idx}/{len(TARGET_DOCS)}] Skipping {doc_id} (already evaluated)")
            continue

        print(f"\n[{idx}/{len(TARGET_DOCS)}] Evaluating {doc_id}...")
        try:
            xml_text, raw_text = load_doc_data(doc_id)
        except Exception as exc:
            print(f"   [ERROR] Failed to load data: {exc}")
            continue

        orig_info = orig_map.get(doc_id, {})

        # 1. Evaluate with Codex
        print(f"   -> Running Codex (gpt-5.6-luna)...", end="", flush=True)
        t0 = time.time()
        rep_codex = None
        err_codex = None
        try:
            rep_codex = rev_codex.review_document(
                xml_content=xml_text,
                raw_ocr_text=raw_text,
                doc_id=doc_id,
                use_llm=True,
            )
            print(
                f" Done ({time.time() - t0:.1f}s) => {rep_codex.decision.value}, Score: {rep_codex.overall_score}"
            )
        except Exception as exc:
            err_codex = str(exc)
            print(f" FAILED: {exc}")

        # 2. Evaluate with AGY
        print(f"   -> Running AGY (gemini-3.8-flash)...", end="", flush=True)
        t0 = time.time()
        rep_agy = None
        err_agy = None
        try:
            rep_agy = rev_agy.review_document(
                xml_content=xml_text,
                raw_ocr_text=raw_text,
                doc_id=doc_id,
                use_llm=True,
            )
            print(
                f" Done ({time.time() - t0:.1f}s) => {rep_agy.decision.value}, Score: {rep_agy.overall_score}"
            )
        except Exception as exc:
            err_agy = str(exc)
            print(f" FAILED: {exc}")

        doc_result = {
            "doc_id": doc_id,
            "orig_decision": orig_info.get("orig_decision"),
            "orig_score": orig_info.get("orig_score"),
            "orig_issues_count": orig_info.get("orig_issues_count"),
            "codex": {
                "decision": rep_codex.decision.value if rep_codex else "ERROR",
                "overall_score": rep_codex.overall_score if rep_codex else None,
                "deterministic_score": rep_codex.deterministic_score if rep_codex else None,
                "llm_score": rep_codex.llm_score if rep_codex else None,
                "confirmed_issues_count": len(rep_codex.confirmed_issues) if rep_codex else 0,
                "issues_count": len(rep_codex.issues) if rep_codex else 0,
                "discard_reasons": rep_codex.discard_reasons if rep_codex else [],
                "issues": [
                    {
                        "category": iss.category,
                        "severity": iss.severity.value,
                        "message": iss.message,
                    }
                    for iss in (rep_codex.issues if rep_codex else [])
                ],
                "error": err_codex,
            },
            "agy": {
                "decision": rep_agy.decision.value if rep_agy else "ERROR",
                "overall_score": rep_agy.overall_score if rep_agy else None,
                "deterministic_score": rep_agy.deterministic_score if rep_agy else None,
                "llm_score": rep_agy.llm_score if rep_agy else None,
                "confirmed_issues_count": len(rep_agy.confirmed_issues) if rep_agy else 0,
                "issues_count": len(rep_agy.issues) if rep_agy else 0,
                "discard_reasons": rep_agy.discard_reasons if rep_agy else [],
                "issues": [
                    {
                        "category": iss.category,
                        "severity": iss.severity.value,
                        "message": iss.message,
                    }
                    for iss in (rep_agy.issues if rep_agy else [])
                ],
                "error": err_agy,
            },
            "decision_match": (
                (rep_codex.decision == rep_agy.decision)
                if (rep_codex and rep_agy)
                else False
            ),
            "score_diff": (
                abs(rep_codex.overall_score - rep_agy.overall_score)
                if (rep_codex and rep_agy)
                else None
            ),
        }
        results.append(doc_result)

        # Save checkpoint after each document
        OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print(f"Results saved to: {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
