"""
Unit tests for Dual-Granularity Editor and Revision Engine.
"""

import json
from pathlib import Path
import pytest

from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision
from tools.review.edit_review_set import (
    load_revision_targets,
    reconcile_repaired_exam,
    _chunk_sort_key,
)


def test_chunk_sort_key():
    p1 = Path("/path/to/chunk_0.xml")
    p2 = Path("/path/to/chunk_1.xml")
    p10 = Path("/path/to/chunk_10.xml")
    assert _chunk_sort_key(p1) < _chunk_sort_key(p2) < _chunk_sort_key(p10)


def test_load_revision_targets_chunk_decomposition(tmp_path):
    annotated_dir = tmp_path / "annotated"
    exam_dir = annotated_dir / "exam_multi"
    chunks_dir = exam_dir / "chunks"
    chunks_dir.mkdir(parents=True)

    c0_xml = "<question_label>Câu 1.</question_label> <stem>Stem 1</stem>"
    c1_xml = "<question_label>Câu 2.</question_label> <stem>Stem 2</stem>"

    (chunks_dir / "chunk_0.xml").write_text(c0_xml, encoding="utf-8")
    (chunks_dir / "chunk_0.json").write_text(json.dumps({"chunk_index": 0, "raw_chunk_text": "Câu 1. Stem 1"}), encoding="utf-8")
    (chunks_dir / "chunk_0.audit.json").write_text(json.dumps({"decision": "PASS", "overall_score": 100.0}), encoding="utf-8")

    (chunks_dir / "chunk_1.xml").write_text(c1_xml, encoding="utf-8")
    (chunks_dir / "chunk_1.json").write_text(json.dumps({"chunk_index": 1, "raw_chunk_text": "Câu 2. Stem 2"}), encoding="utf-8")
    (chunks_dir / "chunk_1.audit.json").write_text(
        json.dumps({
            "decision": "NEEDS_REVISION",
            "overall_score": 70.0,
            "issues": [{"category": "missing_stem", "severity": "MAJOR", "message": "Missing stem", "context_snippet": "Câu 2."}]
        }),
        encoding="utf-8"
    )

    merged_xml = f"{c0_xml}\n{c1_xml}"
    (exam_dir / "merged.xml").write_text(merged_xml, encoding="utf-8")

    report_path = tmp_path / "review_report.json"
    report_data = {
        "results": [
            {
                "doc_id": "exam_multi",
                "decision": "NEEDS_REVISION",
                "overall_score": 75.0,
                "file_path": str(exam_dir / "merged.xml"),
            }
        ]
    }
    report_path.write_text(json.dumps(report_data), encoding="utf-8")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "exam_multi.md").write_text("Câu 1. Stem 1\nCâu 2. Stem 2", encoding="utf-8")

    # 1. Chunk granularity -> should skip chunk_0 (already PASS) and target only chunk_1
    targets = load_revision_targets(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=annotated_dir,
        decisions={"NEEDS_REVISION"},
        granularity="chunk",
    )
    assert len(targets) == 1
    assert targets[0]["doc_id"] == "exam_multi::chunk_1"
    assert targets[0]["xml_path"].name == "chunk_1.xml"
    assert targets[0]["raw_path"].name == "chunk_1.json"

    # 2. Document granularity -> should target merged.xml directly
    targets_doc = load_revision_targets(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=annotated_dir,
        decisions={"NEEDS_REVISION"},
        granularity="document",
    )
    assert len(targets_doc) == 1
    assert targets_doc[0]["doc_id"] == "exam_multi"
    assert targets_doc[0]["xml_path"].name == "merged.xml"


def test_reconcile_repaired_exam_flow(tmp_path):
    annotated_dir = tmp_path / "annotated"
    exam_dir = annotated_dir / "exam_303"
    chunks_dir = exam_dir / "chunks"
    chunks_dir.mkdir(parents=True)

    c0_xml = "<question_label>Câu 1.</question_label> <stem>T1</stem>"
    c0_raw = "Câu 1. T1"
    c1_xml = "\n<question_label>Câu 2.</question_label> <stem>T2</stem>"
    c1_raw = "\nCâu 2. T2"

    (chunks_dir / "chunk_0.xml").write_text(c0_xml, encoding="utf-8")
    (chunks_dir / "chunk_0.json").write_text(json.dumps({
        "chunk_index": 0,
        "raw_chunk_text": c0_raw,
        "page_offset_ranges": [{"page_id": 1, "local_start": 0, "local_end": len(c0_raw), "global_start": 0, "global_end": len(c0_raw)}]
    }), encoding="utf-8")
    (chunks_dir / "chunk_0.audit.json").write_text(json.dumps({"decision": "PASS", "overall_score": 100.0}), encoding="utf-8")

    (chunks_dir / "chunk_1.xml").write_text(c1_xml, encoding="utf-8")
    (chunks_dir / "chunk_1.json").write_text(json.dumps({
        "chunk_index": 1,
        "raw_chunk_text": c1_raw,
        "page_offset_ranges": [{"page_id": 2, "local_start": 0, "local_end": len(c1_raw), "global_start": len(c0_raw), "global_end": len(c0_raw) + len(c1_raw)}]
    }), encoding="utf-8")
    (chunks_dir / "chunk_1.audit.json").write_text(json.dumps({"decision": "NEEDS_REVISION", "overall_score": 70.0}), encoding="utf-8")

    branch_dir = tmp_path / "branch"
    branch_chunks = branch_dir / "exam_303" / "chunks"
    branch_chunks.mkdir(parents=True)

    # In branch, chunk_1 is repaired and marked PASS
    (branch_chunks / "chunk_1.xml").write_text(c1_xml, encoding="utf-8")
    (branch_chunks / "chunk_1_revision_state.json").write_text(json.dumps({"status": "PASS", "doc_id": "exam_303::chunk_1"}), encoding="utf-8")
    (branch_chunks / "chunk_1.audit.json").write_text(json.dumps({"decision": "PASS", "overall_score": 95.0}), encoding="utf-8")

    reviewer = AnnotationReviewerAgent()
    res = reconcile_repaired_exam(
        exam_id="exam_303",
        annotated_dir=annotated_dir,
        branch_dir=branch_dir,
        reviewer=reviewer,
    )

    assert res["status"] == "reconciled"
    assert res["exam_id"] == "exam_303"
    assert res["decision"] == "PASS"

    branch_exam = branch_dir / "exam_303"
    assert (branch_exam / "merged.xml").exists()
    assert (branch_exam / "merged.json").exists()
    assert (branch_exam / "audit_report.json").exists()

    audit_rep = json.loads((branch_exam / "audit_report.json").read_text(encoding="utf-8"))
    assert audit_rep["decision"] == "PASS"
    assert audit_rep["metrics"]["reconciled"] is True
