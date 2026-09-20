"""
Review & Editor Studio Backend for Vietnamese Exam Sequence Labelling Dataset.

Provides fast, in-memory caching and REST endpoints for auditing, interactive editing,
auto-anchoring, and saving ground-truth XML annotations across the 468 exam papers.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sequence_labelling.annotator.anchor_helper import AnchorHelper
from sequence_labelling.annotator.reviewer import (
    AuditIssue,
    DeterministicAuditor,
    IssueSeverity,
    ReviewDecision,
    compute_grade,
)
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import (
    parse_xml_with_anchors,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SEQUENCE_LABEL_DATA_DIR", PROJECT_DIR / "data")).expanduser().resolve()
REAL_ANNOTATED_DIR = DATA_DIR / "sequence_labelling_annotated"
RAW_DIR = DATA_DIR / "sequence_labelling_input_data"
REPORT_PATH = Path(os.environ.get("SEQUENCE_LABEL_REVIEW_REPORT", PROJECT_DIR / "artifacts" / "review_report.json")).expanduser().resolve()

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Azozo Sequence Labelling Review & Editor Studio")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# In-memory document registry
class DocumentRegistry:
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.report_data: Dict[str, Any] = {}
        self.raw_lookup: Dict[str, Path] = {}
        self.loaded = False

    def initialize(self):
        if self.loaded:
            return

        # 1. Index raw OCR files
        if RAW_DIR.exists():
            for p in RAW_DIR.glob("**/*.md"):
                self.raw_lookup[p.stem] = p

        # 2. Load report if available
        if REPORT_PATH.exists():
            try:
                self.report_data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
                for r in self.report_data.get("reports", []):
                    doc_id = r.get("doc_id")
                    if not doc_id:
                        continue
                    fp = Path(r.get("file_path", ""))
                    if not fp.is_absolute():
                        fp = REAL_ANNOTATED_DIR / fp
                    
                    # Category from rel_path
                    rel_parts = fp.relative_to(REAL_ANNOTATED_DIR).parts if fp.exists() and REAL_ANNOTATED_DIR in fp.parents else fp.parts
                    category = "/".join(rel_parts[:-2]) if len(rel_parts) > 2 else "Other"
                    subject = rel_parts[-2] if len(rel_parts) > 2 else "Unknown"

                    raw_path = self.raw_lookup.get(doc_id)

                    self.documents[doc_id] = {
                        "doc_id": doc_id,
                        "file_path": str(fp),
                        "raw_path": str(raw_path) if raw_path else None,
                        "category": category,
                        "subject": subject,
                        "overall_score": r.get("overall_score", 0.0),
                        "deterministic_score": r.get("deterministic_score", 0.0),
                        "grade": r.get("grade", "F"),
                        "decision": r.get("decision", "DISCARD"),
                        "is_malfunctioned": r.get("is_malfunctioned", True),
                        "issues": r.get("issues", []),
                        "discard_reasons": r.get("discard_reasons", []),
                        "metrics": r.get("metrics", {}),
                        "summary": r.get("summary", ""),
                    }
            except Exception as e:
                print(f"[Studio Warning] Could not parse report: {e}")

        # 3. Fallback discovery if report was empty
        if not self.documents and REAL_ANNOTATED_DIR.exists():
            for xml_p in REAL_ANNOTATED_DIR.glob("**/merged.xml"):
                doc_id = xml_p.parent.name
                rel_parts = xml_p.relative_to(REAL_ANNOTATED_DIR).parts
                category = "/".join(rel_parts[:-2]) if len(rel_parts) > 2 else "Other"
                subject = rel_parts[-2] if len(rel_parts) > 2 else "Unknown"
                raw_path = self.raw_lookup.get(doc_id)
                self.documents[doc_id] = {
                    "doc_id": doc_id,
                    "file_path": str(xml_p),
                    "raw_path": str(raw_path) if raw_path else None,
                    "category": category,
                    "subject": subject,
                    "overall_score": 0.0,
                    "deterministic_score": 0.0,
                    "grade": "F",
                    "decision": "DISCARD",
                    "is_malfunctioned": True,
                    "issues": [],
                    "discard_reasons": [],
                    "metrics": {},
                    "summary": "Pending audit",
                }

        self.loaded = True
        print(f"[Studio] Loaded {len(self.documents)} exam documents into registry.")

    def update_doc_report(self, doc_id: str, new_report: Dict[str, Any]):
        if doc_id in self.documents:
            self.documents[doc_id].update(new_report)

        # Sync back to master report
        if "reports" in self.report_data:
            for idx, r in enumerate(self.report_data["reports"]):
                if r.get("doc_id") == doc_id:
                    self.report_data["reports"][idx].update(new_report)
                    break
            else:
                self.report_data["reports"].append(new_report)

            try:
                REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                REPORT_PATH.write_text(
                    json.dumps(self.report_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[Studio Warning] Could not persist report update: {e}")


registry = DocumentRegistry()


@app.on_event("startup")
def on_startup():
    registry.initialize()


class AuditRequest(BaseModel):
    xml_content: str


class SaveRequest(BaseModel):
    xml_content: str


@app.get("/", response_class=HTMLResponse)
def get_studio_page(request: Request):
    registry.initialize()
    return templates.TemplateResponse(
        request=request,
        name="review_studio.html",
        context={"request": request},
    )


@app.get("/api/documents")
def list_documents(
    filter_status: Optional[str] = None,
    subject: Optional[str] = None,
    search: Optional[str] = None,
):
    registry.initialize()
    docs = list(registry.documents.values())

    # Summary counts
    total = len(docs)
    needs_revision_cnt = sum(1 for d in docs if d["decision"] == "NEEDS_REVISION")
    discard_cnt = sum(1 for d in docs if d["decision"] == "DISCARD")
    pass_cnt = sum(1 for d in docs if d["decision"] == "PASS")

    # Filter
    if filter_status and filter_status.upper() != "ALL":
        target = filter_status.upper()
        docs = [d for d in docs if d["decision"].upper() == target]

    if subject and subject.lower() != "all":
        docs = [d for d in docs if d["subject"].lower() == subject.lower()]

    if search:
        s = search.lower().strip()
        docs = [
            d
            for d in docs
            if s in d["doc_id"].lower()
            or s in d["category"].lower()
            or s in d["subject"].lower()
        ]

    # Sort: DISCARD first, then NEEDS_REVISION, then PASS; within group sort by score ascending
    tier_order = {"DISCARD": 0, "NEEDS_REVISION": 1, "PASS": 2}
    docs.sort(key=lambda x: (tier_order.get(x["decision"], 3), x["overall_score"]))

    return {
        "total": total,
        "counts": {
            "all": total,
            "needs_revision": needs_revision_cnt,
            "discard": discard_cnt,
            "pass": pass_cnt,
        },
        "filtered_count": len(docs),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "category": d["category"],
                "subject": d["subject"],
                "score": d["overall_score"],
                "grade": d["grade"],
                "decision": d["decision"],
                "questions_count": d.get("metrics", {}).get("questions_count", 0),
                "stimuli_count": d.get("metrics", {}).get("stimuli_count", 0),
                "issues_count": len(d.get("issues", [])),
            }
            for d in docs
        ],
    }


@app.get("/api/documents/{doc_id}")
def get_document_detail(doc_id: str):
    registry.initialize()
    doc = registry.documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    xml_path = Path(doc["file_path"])
    raw_path = Path(doc["raw_path"]) if doc.get("raw_path") else None

    annotated_xml = ""
    if xml_path.exists():
        annotated_xml = xml_path.read_text(encoding="utf-8")

    raw_ocr_text = ""
    if raw_path and raw_path.exists():
        raw_ocr_text = raw_path.read_text(encoding="utf-8")
    elif xml_path.exists():
        # Fallback raw search in same directory
        cand_raw = list(xml_path.parent.glob("*.md"))
        if cand_raw:
            raw_ocr_text = cand_raw[0].read_text(encoding="utf-8")

    # Run live audit to get current state
    report = run_deterministic_audit(doc_id, annotated_xml, raw_ocr_text)

    return {
        "doc_id": doc_id,
        "category": doc["category"],
        "subject": doc["subject"],
        "xml_path": str(xml_path),
        "raw_path": str(raw_path) if raw_path else None,
        "raw_ocr_text": raw_ocr_text,
        "annotated_xml": annotated_xml,
        "audit_report": report,
    }


def run_deterministic_audit(
    doc_id: str, xml_content: str, raw_ocr_text: Optional[str] = None
) -> Dict[str, Any]:
    """Runs high-speed deterministic evaluation on XML content."""
    pure_text = DeterministicAuditor.strip_xml_tags(xml_content)

    issues: List[AuditIssue] = []
    discard_reasons: List[str] = []

    # 1. Syntax check
    syn_issues, syn_score = DeterministicAuditor.check_xml_syntax(xml_content)
    issues.extend(syn_issues)

    # 2. Prohibited tags
    proh_issues, proh_score = DeterministicAuditor.check_prohibited_tags(xml_content)
    issues.extend(proh_issues)
    s_syntax = round(syn_score * 0.70 + proh_score * 0.30, 1)

    # 3. Stimulus wrapping system tags & anchor attributes
    stim_wrap_issues, stim_wrap_score = (
        DeterministicAuditor.check_stimulus_wrapping_system_tags(xml_content)
    )
    issues.extend(stim_wrap_issues)

    stim_anchor_issues, stim_anchor_score = (
        DeterministicAuditor.check_stimulus_anchors(
            xml_content, pure_text, raw_ocr_text
        )
    )
    issues.extend(stim_anchor_issues)
    stim_score = min(stim_wrap_score, stim_anchor_score)

    # 4. Question and option structure
    q_issues, q_score, q_metrics = (
        DeterministicAuditor.check_question_and_option_structure(
            xml_content, raw_ocr_text=raw_ocr_text
        )
    )
    issues.extend(q_issues)
    s_structure = round(q_score * 0.70 + stim_score * 0.30, 1)

    # 5. DET Question Coverage (Recall)
    det_cov_issues, det_cov_score, det_cov_metrics = (
        DeterministicAuditor.check_question_coverage_via_det(
            xml_content, raw_ocr_text
        )
    )
    issues.extend(det_cov_issues)

    # 6. Verbatim Content Fidelity (Precision)
    verb_issues, verb_score, verb_metrics = (
        DeterministicAuditor.check_verbatim_alignment(
            xml_content, raw_ocr_text
        )
    )
    issues.extend(verb_issues)

    # Generalized deterministic total score
    if raw_ocr_text and raw_ocr_text.strip():
        det_score = (
            s_syntax * 0.25
            + det_cov_score * 0.35
            + verb_score * 0.25
            + s_structure * 0.15
        )
    else:
        det_score = s_syntax * 0.60 + s_structure * 0.40

    det_score = round(max(0.0, min(100.0, det_score)), 1)
    grade = compute_grade(det_score)

    # Critical issues
    critical_issues = [iss for iss in issues if iss.severity == IssueSeverity.CRITICAL]
    for c_iss in critical_issues:
        discard_reasons.append(f"[{c_iss.category.upper()}] {c_iss.message}")

    if det_score < 75.0:
        discard_reasons.append(
            f"Overall score {det_score}/100 is below minimum threshold 75."
        )

    is_malfunctioned = len(discard_reasons) > 0
    if is_malfunctioned:
        decision = ReviewDecision.DISCARD
    elif det_score >= 85.0:
        decision = ReviewDecision.PASS
    else:
        decision = ReviewDecision.NEEDS_REVISION

    all_metrics = {**q_metrics, **det_cov_metrics, **verb_metrics}

    return {
        "doc_id": doc_id,
        "overall_score": det_score,
        "deterministic_score": det_score,
        "grade": grade,
        "decision": decision.value,
        "is_malfunctioned": is_malfunctioned,
        "discard_reasons": discard_reasons,
        "rubric_breakdown": {
            "syntax": syn_score,
            "prohibited": proh_score,
            "det_question_coverage": det_cov_score,
            "verbatim": verb_score,
            "questions": q_score,
            "stimulus": stim_score,
        },
        "issues": [
            {
                "category": iss.category,
                "severity": iss.severity.value,
                "message": iss.message,
                "context_snippet": iss.context_snippet,
            }
            for iss in issues
        ],
        "metrics": all_metrics,
    }


@app.post("/api/documents/{doc_id}/audit")
def audit_document_endpoint(doc_id: str, req: AuditRequest):
    registry.initialize()
    doc = registry.documents.get(doc_id)
    raw_text = None
    if doc and doc.get("raw_path"):
        rp = Path(doc["raw_path"])
        if rp.exists():
            raw_text = rp.read_text(encoding="utf-8")

    report = run_deterministic_audit(doc_id, req.xml_content, raw_text)
    return report


@app.post("/api/documents/{doc_id}/auto-fix")
def auto_fix_endpoint(doc_id: str, req: AuditRequest):
    registry.initialize()
    doc = registry.documents.get(doc_id)
    raw_text = None
    if doc and doc.get("raw_path"):
        rp = Path(doc["raw_path"])
        if rp.exists():
            raw_text = rp.read_text(encoding="utf-8")

    all_fixes: List[str] = []
    current_xml = req.xml_content

    # 1. Auto-anchor stimulus tags
    current_xml, stim_fixes = AnchorHelper.auto_anchor_stimuli(current_xml, raw_text)
    all_fixes.extend(stim_fixes)

    # 2. Auto-clean XML tags (mismatches, unclosed stems, etc.)
    clean_res = XMLCleaner.clean(current_xml)
    current_xml = clean_res.cleaned_xml
    all_fixes.extend(clean_res.fixes_applied)

    # 3. Re-audit
    report = run_deterministic_audit(doc_id, current_xml, raw_text)

    return {
        "cleaned_xml": current_xml,
        "fixes": all_fixes,
        "audit_report": report,
    }


@app.post("/api/documents/{doc_id}/save")
def save_document_endpoint(doc_id: str, req: SaveRequest):
    registry.initialize()
    doc = registry.documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    xml_path = Path(doc["file_path"])
    raw_path = Path(doc["raw_path"]) if doc.get("raw_path") else None
    raw_text = raw_path.read_text(encoding="utf-8") if raw_path and raw_path.exists() else None

    # Write merged.xml
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(req.xml_content, encoding="utf-8")

    # Update merged.json if present
    json_path = xml_path.with_name("merged.json")
    if json_path.exists() and raw_text:
        try:
            spans, stimuli, questions = parse_xml_with_anchors(raw_text, req.xml_content)
            json_data = {
                "document_id": doc_id,
                "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "questions_count": len(questions),
                "questions": questions,
                "stimuli": stimuli,
            }
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[Studio Warning] Could not update {json_path}: {e}")

    # Re-audit and sync to master report
    new_report = run_deterministic_audit(doc_id, req.xml_content, raw_text)
    registry.update_doc_report(doc_id, new_report)

    return {
        "success": True,
        "doc_id": doc_id,
        "score": new_report["overall_score"],
        "grade": new_report["grade"],
        "decision": new_report["decision"],
        "audit_report": new_report,
    }
