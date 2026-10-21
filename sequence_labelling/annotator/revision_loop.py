"""Isolated editor/reviewer loop for XML annotation repair.

The editor keeps a per-document conversation.  The reviewer is deliberately
stateless: every review receives only the original OCR text and the current XML
candidate, preventing the editor's rationale from influencing the audit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.annotator.editor import (
    EditorAgent,
    EditorIssueAssessment,
    SearchReplacePatcher,
    parse_issue_assessments,
)
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    AuditIssue,
    IssueSeverity,
    ReviewDecision,
    ReviewReport,
    PARSER_ERROR_PENALTIES,
)
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import (
    parse_xml_with_anchors,
)
from sequence_labelling.llm.agy_rate_limiter import is_quota_or_cap_error


class RevisionRound(BaseModel):
    """Inspectable result of one edit/review exchange."""

    round_number: int
    editor_output: str = ""
    applied_patches_count: int = 0
    failed_patches_reasons: List[str] = Field(default_factory=list)
    score: float = 0.0
    decision: str = ReviewDecision.NEEDS_REVISION.value
    issues: List[AuditIssue] = Field(default_factory=list)
    confirmed_issues: List[AuditIssue] = Field(default_factory=list)
    issue_assessments: List[EditorIssueAssessment] = Field(default_factory=list)
    false_positive_issue_ids: List[int] = Field(default_factory=list)


class RevisionLoopResult(BaseModel):
    """Final result from a bounded editor/reviewer loop."""

    doc_id: str
    success: bool = False
    initial_score: float = 0.0
    initial_decision: str = ReviewDecision.NEEDS_REVISION.value
    final_score: float = 0.0
    final_decision: str = ReviewDecision.NEEDS_REVISION.value
    rounds_completed: int = 0
    applied_patches_count: int = 0
    failed_patches_count: int = 0
    confirmed_issues_count: int = 0
    false_positive_issue_ids: List[int] = Field(default_factory=list)
    false_positive_assessments: List[EditorIssueAssessment] = Field(default_factory=list)
    repaired_xml: str = ""
    diff_summary: str = ""
    duration_seconds: float = 0.0
    rounds: List[RevisionRound] = Field(default_factory=list)
    final_report: Optional[ReviewReport] = None


class EditorReviewerLoop:
    """Run isolated editor and reviewer agents until PASS or a round limit.

    A new ``editor_messages`` list is created for every call to ``run``.  It is
    retained across edit rounds for that document only.  Reviewer calls do not
    receive this list or any previous review response.
    """

    def __init__(
        self,
        editor: Optional[EditorAgent] = None,
        reviewer: Optional[AnnotationReviewerAgent] = None,
        max_rounds: int = 3,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.editor = editor or EditorAgent()
        self.reviewer = reviewer or AnnotationReviewerAgent()
        self.max_rounds = max_rounds

    def _editor_prompt(
        self,
        raw_ocr_text: str,
        current_xml: str,
        issues: List[AuditIssue],
        round_number: int,
    ) -> str:
        diagnostics = self.editor._format_diagnostics_block(issues)
        return (
            f"### EDIT ROUND {round_number}\n"
            "Repair the current XML using only the reviewer-confirmed errors. The XML "
            "below is the only version to patch in this round.\n\n"
            "<<<CONFIRMED_REVIEWER_ERRORS>>>\n"
            f"{diagnostics}\n"
            "<<<END_CONFIRMED_REVIEWER_ERRORS>>>\n\n"
            "<<<CURRENT_ANNOTATED_XML>>>\n"
            f"{current_xml}\n"
            "<<<END_CURRENT_XML>>>\n\n"
            "<<<RAW_SOURCE_TEXT>>>\n"
            f"{raw_ocr_text}\n"
            "<<<END_RAW_SOURCE_TEXT>>>\n\n"
            "Output Search/Replace diff blocks in the required format. If a supplied "
            "confirmed error is a false positive, do not patch for it and optionally "
            "return an ISSUE_ASSESSMENTS JSON block as described in the system prompt."
        )

    @staticmethod
    def _issue_key(issue: AuditIssue) -> tuple[str, str, str]:
        """Stable identity for carrying an editor false-positive claim across rounds."""
        severity = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
        return (issue.category.strip().lower(), severity.strip().lower(), issue.message.strip().lower())

    @staticmethod
    def _issue_id(issue: AuditIssue, index: int) -> int:
        return issue.issue_id or index

    @staticmethod
    def _diff_summary(original_xml: str, repaired_xml: str) -> str:
        lines = list(
            difflib.unified_diff(
                original_xml.splitlines(keepends=True),
                repaired_xml.splitlines(keepends=True),
                fromfile="before_edit.xml",
                tofile="after_edit.xml",
                n=2,
            )
        )
        summary = "".join(lines[:40])
        if len(lines) > 40:
            summary += f"\n... [{len(lines) - 40} more diff lines]"
        return summary

    @staticmethod
    def _candidate_rank(report: ReviewReport) -> tuple[int, float]:
        """Rank candidates by decision tier first, then overall score.

        PASS (tier 2) > NEEDS_REVISION (tier 1) > DISCARD (tier 0).
        Prevents a candidate marked DISCARD (e.g. falling back to an unreviewed
        det_score of 87.0 due to a critical error) from overwriting a valid repair.
        """
        tier_map = {
            ReviewDecision.PASS.value: 2,
            ReviewDecision.NEEDS_REVISION.value: 1,
            ReviewDecision.DISCARD.value: 0,
        }
        dec_val = (
            report.decision.value
            if hasattr(report.decision, "value")
            else str(report.decision)
        )
        tier = tier_map.get(dec_val, 0)
        return (tier, report.overall_score)

    def run(
        self,
        annotated_xml: str,
        raw_ocr_text: str,
        issues: Optional[List[AuditIssue]] = None,
        doc_id: str = "doc",
        editor_completion_fn: Optional[Callable[..., str]] = None,
        use_llm_reviewer: bool = True,
        initial_report: Optional[ReviewReport] = None,
        cached_llm_result: Optional[Dict[str, Any]] = None,
        checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> RevisionLoopResult:
        """Execute the bounded loop and return the highest-scoring candidate.

        If ``initial_report`` or ``cached_llm_result`` is provided, Round 0 directly
        consumes the prerun audit findings without triggering an unneeded duplicate
        LLM review API call. Subsequent rounds (Round 1+) continue to execute live
        against newly generated candidate XMLs.
        """
        started_at = time.time()
        complete = editor_completion_fn or chat

        if not annotated_xml or not annotated_xml.strip():
            return RevisionLoopResult(
                doc_id=doc_id,
                repaired_xml=annotated_xml,
                duration_seconds=time.time() - started_at,
            )

        if initial_report is not None and (initial_report.confirmed_issues or initial_report.decision == ReviewDecision.PASS):
            resolved_initial_report = initial_report
        elif cached_llm_result is not None:
            resolved_initial_report = self.reviewer.review_document(
                xml_content=annotated_xml,
                raw_ocr_text=raw_ocr_text,
                doc_id=doc_id,
                use_llm=use_llm_reviewer,
                cached_llm_result=cached_llm_result,
            )
        else:
            resolved_initial_report = self.reviewer.review_document(
                xml_content=annotated_xml,
                raw_ocr_text=raw_ocr_text,
                doc_id=doc_id,
                use_llm=use_llm_reviewer,
            )

        initial_report = resolved_initial_report
        if checkpoint_callback is not None:
            checkpoint_callback(
                {
                    "stage": "initial_review",
                    "round_number": 0,
                    "candidate_xml": annotated_xml,
                    "review": initial_report.model_dump(mode="json"),
                }
            )
        init_decision_val = (
            initial_report.decision.value
            if hasattr(initial_report.decision, "value")
            else str(initial_report.decision)
        )
        if initial_report.decision == ReviewDecision.PASS:
            return RevisionLoopResult(
                doc_id=doc_id,
                success=True,
                initial_score=initial_report.overall_score,
                initial_decision=init_decision_val,
                final_score=initial_report.overall_score,
                final_decision=init_decision_val,
                confirmed_issues_count=len(initial_report.confirmed_issues),
                repaired_xml=annotated_xml,
                duration_seconds=time.time() - started_at,
                final_report=initial_report,
            )

        # Never pass the review's complete issue list to the editor.  It may
        # contain deterministic candidates that the semantic reviewer marked
        # false-positive, or findings whose truth status is still unknown.
        current_xml = XMLCleaner.clean(annotated_xml).cleaned_xml
        current_issues = list(initial_report.confirmed_issues)

        # A reviewer run without a confirmation gate is not actionable by an
        # editor.  Return the audit result so a human can inspect or rerun the
        # reviewer, but do not let an unconfirmed finding trigger a patch.
        if not current_issues:
            return RevisionLoopResult(
                doc_id=doc_id,
                success=False,
                initial_score=initial_report.overall_score,
                initial_decision=init_decision_val,
                final_score=initial_report.overall_score,
                final_decision=init_decision_val,
                confirmed_issues_count=0,
                repaired_xml=annotated_xml,
                duration_seconds=time.time() - started_at,
                final_report=initial_report,
            )

        best_xml = annotated_xml
        best_report = initial_report
        rounds: List[RevisionRound] = []
        editor_messages = [{"role": "system", "content": self.editor.system_prompt}]
        total_applied = 0
        total_failed = 0
        false_positive_keys: set[tuple[str, str, str]] = set()
        false_positive_assessments: List[EditorIssueAssessment] = []

        for round_number in range(1, self.max_rounds + 1):
            prompt = self._editor_prompt(
                raw_ocr_text=raw_ocr_text,
                current_xml=current_xml,
                issues=current_issues,
                round_number=round_number,
            )
            editor_messages.append({"role": "user", "content": prompt})
            editor_output = complete(
                messages=list(editor_messages),
                model=self.editor.model,
                provider=self.editor.provider,
                thinking=self.editor.thinking,
            ) or ""
            editor_messages.append({"role": "assistant", "content": editor_output})

            assessments = parse_issue_assessments(editor_output)
            issue_by_id = {
                self._issue_id(issue, index): issue
                for index, issue in enumerate(current_issues, start=1)
            }
            valid_assessments = [
                assessment
                for assessment in assessments
                if assessment.issue_id in issue_by_id
            ]
            for assessment in valid_assessments:
                if assessment.is_false_positive:
                    false_positive_keys.add(
                        self._issue_key(issue_by_id[assessment.issue_id])
                    )
                    false_positive_assessments.append(assessment)

            blocks = SearchReplacePatcher.parse_blocks(editor_output)
            candidate_xml, applied, failed = SearchReplacePatcher.apply_blocks(
                current_xml, blocks
            )
            candidate_xml = XMLCleaner.clean(candidate_xml).cleaned_xml
            total_applied += applied
            total_failed += len(failed)

            # This is intentionally a fresh, stateless review.  The reviewer
            # receives source + candidate only; editor_messages never cross over.
            report = self.reviewer.review_document(
                xml_content=candidate_xml,
                raw_ocr_text=raw_ocr_text,
                doc_id=doc_id,
                use_llm=use_llm_reviewer,
            )
            rounds.append(
                RevisionRound(
                    round_number=round_number,
                    editor_output=editor_output,
                    applied_patches_count=applied,
                    failed_patches_reasons=failed,
                    score=report.overall_score,
                    decision=report.decision.value,
                    issues=report.issues,
                    confirmed_issues=report.confirmed_issues,
                    issue_assessments=valid_assessments,
                    false_positive_issue_ids=[
                        assessment.issue_id
                        for assessment in valid_assessments
                        if assessment.is_false_positive
                    ],
                )
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "stage": "round",
                        "round_number": round_number,
                        "candidate_xml": candidate_xml,
                        "review": report.model_dump(mode="json"),
                        "editor_output": editor_output,
                        "applied_patches_count": applied,
                        "failed_patches_reasons": failed,
                    }
                )

            if self._candidate_rank(report) > self._candidate_rank(best_report):
                best_xml = candidate_xml
                best_report = report

            current_xml = candidate_xml
            current_issues = [
                issue
                for issue in report.confirmed_issues
                if self._issue_key(issue) not in false_positive_keys
            ]

            if report.decision == ReviewDecision.PASS:
                best_xml = candidate_xml
                best_report = report
                break

            # No parseable/applicable patch means another identical request is
            # unlikely to improve the document, even with retained context.
            if not blocks or applied == 0:
                break

            # There is no actionable reviewer-confirmed finding left.  Keep
            # the candidate and its audit report, but leave the final decision
            # unchanged so a false-positive dispute cannot silently become a
            # PASS without a reviewer recheck.
            if not current_issues:
                break

        return RevisionLoopResult(
            doc_id=doc_id,
            success=best_report.decision == ReviewDecision.PASS,
            initial_score=initial_report.overall_score,
            initial_decision=initial_report.decision.value,
            final_score=best_report.overall_score,
            final_decision=best_report.decision.value,
            rounds_completed=len(rounds),
            applied_patches_count=total_applied,
            failed_patches_count=total_failed,
            confirmed_issues_count=len(initial_report.confirmed_issues),
            false_positive_issue_ids=sorted(
                {assessment.issue_id for assessment in false_positive_assessments}
            ),
            false_positive_assessments=false_positive_assessments,
            repaired_xml=best_xml,
            diff_summary=self._diff_summary(annotated_xml, best_xml),
            duration_seconds=time.time() - started_at,
            rounds=rounds,
            final_report=best_report,
        )


# ==============================================================================
# Batch Revision & Isolated Branch Staging Subsystem
# (Consolidated from tools/review/edit_review_set.py into core library)
# ==============================================================================

CHECKPOINT_SCHEMA_VERSION = 1

UNRECOVERABLE_PATTERNS = [
    r"two[- ]column",
    r"multi[- ]column",
    r"column.*interleav",
    r"interleav",
    r"scrambl",
    r"jumbled",
    r"cross[- ]reading",
    r"truncat",
    r"zero questions",
    r"no questions detected",
    r"garbled",
    r"horizontally interleaved",
    r"severe ocr corruption",
]
UNRECOVERABLE_REGEX = re.compile("|".join(UNRECOVERABLE_PATTERNS), re.IGNORECASE)


def is_unrecoverable_discard(report_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Determines if a discarded document has unrecoverable OCR flaws."""
    metrics = report_dict.get("metrics") or {}
    retention = metrics.get("retention_ratio")
    if retention is not None and retention < 0.75:
        return True, f"Low text retention ratio ({retention:.1%}): output severely incomplete"

    unannotated_q = metrics.get("unannotated_questions_count")
    if unannotated_q is not None and unannotated_q >= 10:
        return True, f"Excessive unannotated questions ({unannotated_q} omitted): exceeds surgical repair capability"

    discard_reasons = report_dict.get("discard_reasons", [])
    summary = report_dict.get("summary", "")

    for reason in discard_reasons:
        match = UNRECOVERABLE_REGEX.search(reason)
        if match:
            return True, f"Discard reason: {reason.strip()}"

    if summary:
        match = UNRECOVERABLE_REGEX.search(summary)
        if match:
            return True, f"Summary: {summary.strip()}"

    for issue in report_dict.get("issues", []):
        cat = issue.get("category", "")
        if cat in ("truncated_output", "unrecoverable_ocr"):
            return True, f"Unrecoverable issue category: {cat}"
        if str(issue.get("severity", "")).upper() == "CRITICAL":
            msg = issue.get("message", "")
            match = UNRECOVERABLE_REGEX.search(msg)
            if match:
                return True, f"Critical issue: {msg.strip()}"

    return False, ""


def _chunk_sort_key(p: Path) -> int:
    m = re.search(r"chunk_(\d+)", p.name)
    return int(m.group(1)) if m else 999


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_path(branch_dir: Path, target: Dict[str, Any]) -> Path:
    xml_stem = target["rel_xml_path"].stem
    if xml_stem.startswith("chunk_"):
        return branch_dir / target["rel_xml_path"].parent / f"{xml_stem}_revision_state.json"
    return branch_dir / target["rel_xml_path"].parent / "revision_state.json"


def load_revision_targets(
    report_path: Path,
    raw_dir: Path,
    annotated_dir: Path,
    decisions: Set[str],
    skip_unrecoverable: bool = True,
    granularity: str = "chunk",
) -> List[Dict[str, Any]]:
    """Load documents matching the requested review decisions with dual granularity support (chunk or document)."""
    granularity = (granularity or "chunk").strip().lower()
    if granularity not in {"chunk", "document"}:
        raise ValueError(f"Invalid granularity: {granularity}. Must be 'chunk' or 'document'.")

    json_path = report_path.with_suffix(".json") if report_path.suffix == ".md" else report_path
    if not json_path.exists():
        print(f"❌ Error: Report file '{json_path}' not found.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = []
    if isinstance(data, list):
        reports = data
    else:
        reports = data.get("reports") or data.get("results") or []
    skipped_unrec_count = 0

    for r in reports:
        decision = r.get("decision")
        if decision not in decisions:
            continue

        doc_id = r.get("doc_id")

        if decision == "DISCARD" and skip_unrecoverable:
            is_unrec, reason = is_unrecoverable_discard(r)
            if is_unrec:
                skipped_unrec_count += 1
                continue

        is_chunk_target = "::" in doc_id
        exam_id = doc_id.split("::", 1)[0]
        chunks_dir = annotated_dir / exam_id / "chunks"

        should_decompose = (
            not is_chunk_target
            and chunks_dir.exists()
            and granularity == "chunk"
        )

        if should_decompose:
            chunk_xml_files = sorted(
                [c for c in chunks_dir.glob("chunk_*.xml") if not c.name.endswith(".audit.xml")],
                key=_chunk_sort_key,
            )
            for c_xml in chunk_xml_files:
                c_stem = c_xml.stem
                c_doc_id = f"{exam_id}::{c_stem}"
                c_json = c_xml.with_suffix(".json")
                c_audit = c_xml.with_suffix(".audit.json")

                c_decision = decision
                c_score = r.get("overall_score", 0.0)
                c_report = None
                c_issues = []
                c_cached_llm = None

                if c_audit.exists():
                    try:
                        aud_data = json.loads(c_audit.read_text(encoding="utf-8"))
                        if aud_data.get("decision") == "PASS":
                            continue
                        c_decision = aud_data.get("decision", decision)
                        if c_decision == "DISCARD" and skip_unrecoverable:
                            is_unrec, _ = is_unrecoverable_discard(aud_data)
                            if is_unrec:
                                skipped_unrec_count += 1
                                continue
                        c_score = aud_data.get("overall_score", c_score)
                        c_report = ReviewReport.model_validate(aud_data)
                        c_issues = [
                            AuditIssue(
                                category=iss.get("category", "general"),
                                severity=IssueSeverity(iss.get("severity", "MAJOR")),
                                message=iss.get("message", ""),
                                line_number=iss.get("line_number"),
                                context_snippet=iss.get("context_snippet"),
                            )
                            for iss in aud_data.get("issues", [])
                        ]
                        if aud_data.get("llm_score") is not None:
                            c_cached_llm = {
                                "score": aud_data.get("llm_score"),
                                "rubric_scores": aud_data.get("rubric_scores"),
                                "issues": aud_data.get("issues", []),
                                "parser_error_confirmations": aud_data.get("llm_confirmations", []),
                                "summary": aud_data.get("summary", ""),
                                "is_malfunctioned": aud_data.get("is_malfunctioned", False),
                                "discard_reasons": aud_data.get("discard_reasons", []),
                            }
                    except Exception:
                        pass

                raw_p_chunk = c_json if c_json.exists() else raw_dir / f"{exam_id}.md"
                rel_xml = Path(exam_id) / "chunks" / c_xml.name

                targets.append({
                    "doc_id": c_doc_id,
                    "source_decision": c_decision,
                    "xml_path": c_xml.resolve(),
                    "raw_path": raw_p_chunk.resolve(),
                    "rel_path": rel_xml,
                    "rel_xml_path": rel_xml,
                    "initial_score": c_score,
                    "issues": c_issues,
                    "initial_report": c_report,
                    "cached_llm_result": c_cached_llm,
                })
            continue

        xml_p = r.get("file_path")
        raw_p = r.get("raw_file_path")

        if is_chunk_target:
            exam_folder, chunk_stem = doc_id.split("::", 1)
            cand_chunk = annotated_dir / exam_folder / "chunks" / f"{chunk_stem}.xml"
            xml_path = cand_chunk if cand_chunk.exists() else Path(xml_p) if xml_p else None
            cand_json = xml_path.with_suffix(".json") if xml_path else None
            raw_path = cand_json if (cand_json and cand_json.exists()) else Path(raw_p) if raw_p else None
        else:
            cand_annot = None
            for cand in [
                annotated_dir / f"{doc_id}/merged.xml",
                annotated_dir / f"{doc_id}.xml",
            ]:
                if cand.exists():
                    cand_annot = cand
                    break
            xml_path = cand_annot if cand_annot else Path(xml_p) if xml_p else None
            raw_path = Path(raw_p) if raw_p else None

        if not (raw_path and raw_path.exists()):
            if xml_path:
                try:
                    rel_to_annot = xml_path.relative_to(annotated_dir.resolve())
                except ValueError:
                    rel_to_annot = Path(xml_p)
                    if str(rel_to_annot).startswith("data/sequence_labelling_annotated/"):
                        rel_to_annot = Path(str(rel_to_annot)[len("data/sequence_labelling_annotated/"):])

                doc_stem = rel_to_annot.parent if rel_to_annot.name in ["merged.xml", "merged.json"] else rel_to_annot.with_suffix("")
                cand_raw = raw_dir / f"{doc_stem}.md"
                if cand_raw.exists():
                    raw_path = cand_raw

        if xml_path and xml_path.exists() and raw_path and raw_path.exists():
            issues_raw = r.get("issues", [])
            parsed_issues = []
            for iss in issues_raw:
                parsed_issues.append(
                    AuditIssue(
                        category=iss.get("category", "general"),
                        severity=IssueSeverity(iss.get("severity", "MAJOR")),
                        message=iss.get("message", ""),
                        line_number=iss.get("line_number"),
                        context_snippet=iss.get("context_snippet"),
                    )
                )

            try:
                rel_xml = xml_path.resolve().relative_to(annotated_dir.resolve())
            except ValueError:
                if xml_path.name in ["merged.xml", "merged.json"]:
                    rel_xml = Path(doc_id) / xml_path.name
                else:
                    rel_xml = Path(xml_path.name)

            prerun_report = None
            try:
                prerun_report = ReviewReport.model_validate(r)
            except Exception:
                pass

            cached_llm = None
            if r.get("llm_score") is not None or r.get("llm_confirmations") or r.get("issues"):
                cached_llm = {
                    "score": r.get("llm_score"),
                    "rubric_scores": r.get("rubric_scores"),
                    "issues": [
                        iss for iss in r.get("issues", [])
                        if (
                            iss.get("source") == "llm"
                            or (
                                "source" not in iss
                                and (
                                    iss.get("category") in PARSER_ERROR_PENALTIES
                                    or iss.get("category") == "llm_semantic"
                                )
                            )
                        )
                    ],
                    "parser_error_confirmations": r.get("llm_confirmations", []),
                    "summary": r.get("summary", ""),
                    "is_malfunctioned": r.get("is_malfunctioned", False),
                    "discard_reasons": [
                        disc for disc in r.get("discard_reasons", [])
                        if "[MALFUNCTION]" in disc
                    ],
                }

            if prerun_report is None or not cached_llm:
                audit_file = (
                    xml_path.parent / "audit_report.json"
                    if xml_path.name in ["merged.xml", "merged.json"]
                    else xml_path.with_suffix(".audit.json")
                )
                if audit_file.exists():
                    try:
                        with open(audit_file, "r", encoding="utf-8") as f_aud:
                            audit_data = json.load(f_aud)
                            if prerun_report is None:
                                prerun_report = ReviewReport.model_validate(audit_data)
                            if not cached_llm and audit_data.get("llm_score") is not None:
                                cached_llm = {
                                    "score": audit_data.get("llm_score"),
                                    "rubric_scores": audit_data.get("rubric_scores"),
                                    "issues": audit_data.get("issues", []),
                                    "parser_error_confirmations": audit_data.get("llm_confirmations", []),
                                    "summary": audit_data.get("summary", ""),
                                    "is_malfunctioned": audit_data.get("is_malfunctioned", False),
                                    "discard_reasons": audit_data.get("discard_reasons", []),
                                }
                    except Exception:
                        pass

            try:
                rel_path = raw_path.resolve().relative_to(raw_dir.resolve())
            except Exception:
                rel_path = rel_xml

            targets.append({
                "doc_id": doc_id,
                "source_decision": r.get("decision"),
                "xml_path": xml_path.resolve(),
                "raw_path": raw_path.resolve(),
                "rel_path": rel_path,
                "rel_xml_path": rel_xml,
                "initial_score": r.get("overall_score", 0.0),
                "issues": parsed_issues,
                "initial_report": prerun_report,
                "cached_llm_result": cached_llm,
            })
        else:
            print(f"⚠️ [Warning] Skipping '{doc_id}': could not locate XML ({xml_path}) or Raw ({raw_path})")

    if skipped_unrec_count > 0:
        print(f"ℹ️  [Filter] Skipped {skipped_unrec_count} unrecoverable DISCARD document(s) (severe multi-column/scrambled OCR).")

    return targets


def apply_branch_resume(
    targets: List[Dict[str, Any]],
    branch_dir: Path,
    resume: bool,
    retry_unresolved: bool = False,
) -> List[Dict[str, Any]]:
    """Use compatible branch candidates as inputs for a subsequent run."""
    if not resume:
        return targets

    resumed: List[Dict[str, Any]] = []
    for target in targets:
        state_path = _checkpoint_path(branch_dir, target)
        if not state_path.exists():
            resumed.append(target)
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                resumed.append(target)
                continue
            if (
                state.get("original_xml_sha256") != _sha256_file(target["xml_path"])
                or state.get("raw_sha256") != _sha256_file(target["raw_path"])
            ):
                resumed.append(target)
                continue
            status = state.get("status")
            if status == "PASS" and not retry_unresolved:
                target["resume_skip_state"] = state
                resumed.append(target)
                continue
            candidate = branch_dir / target["rel_xml_path"]
            if candidate.exists() and status in {"NEEDS_REVISION", "DISCARD", "ERROR"}:
                target = dict(target)
                target["xml_path"] = candidate.resolve()
                target["resume_from_branch"] = True
            resumed.append(target)
        except (OSError, json.JSONDecodeError):
            resumed.append(target)
    return resumed


def reconcile_repaired_exam(
    exam_id: str,
    annotated_dir: Path,
    branch_dir: Path,
    reviewer: AnnotationReviewerAgent,
) -> Dict[str, Any]:
    """
    Checks if all chunks for an examination are in PASS state.
    If so, merges them into branch/.../merged.xml and synthesizes audit_report.json.
    """
    exam_orig = annotated_dir / exam_id
    exam_branch = branch_dir / exam_id
    chunks_dir = exam_orig / "chunks"
    if not chunks_dir.exists():
        return {"status": "no_chunks", "exam_id": exam_id}

    chunk_xmls = sorted(
        [c for c in chunks_dir.glob("chunk_*.xml") if not c.name.endswith(".audit.xml")],
        key=_chunk_sort_key,
    )
    if not chunk_xmls:
        return {"status": "no_chunks", "exam_id": exam_id}

    for c_xml in chunk_xmls:
        c_stem = c_xml.stem
        branch_state_file = exam_branch / "chunks" / f"{c_stem}_revision_state.json"
        branch_audit_file = exam_branch / "chunks" / f"{c_stem}.audit.json"
        orig_audit_file = chunks_dir / f"{c_stem}.audit.json"

        is_chunk_pass = False
        if branch_state_file.exists():
            try:
                st = json.loads(branch_state_file.read_text(encoding="utf-8"))
                if st.get("status") == "PASS":
                    is_chunk_pass = True
            except Exception:
                pass
        if not is_chunk_pass and branch_audit_file.exists():
            try:
                aud = json.loads(branch_audit_file.read_text(encoding="utf-8"))
                if aud.get("decision") == "PASS":
                    is_chunk_pass = True
            except Exception:
                pass
        if not is_chunk_pass and orig_audit_file.exists():
            branch_chunk_xml = exam_branch / "chunks" / c_xml.name
            if not branch_chunk_xml.exists():
                try:
                    aud = json.loads(orig_audit_file.read_text(encoding="utf-8"))
                    if aud.get("decision") == "PASS":
                        is_chunk_pass = True
                except Exception:
                    pass

        if not is_chunk_pass:
            return {
                "status": "pending_chunks",
                "exam_id": exam_id,
                "pending_chunk": c_stem,
            }

    rollup_rep = reviewer.rollup_chunk_audits(
        exam_dir=exam_orig,
        branch_dir=branch_dir,
        save_audit_json=True,
        reconcile_merged=True,
    )
    dec_val = (
        rollup_rep.decision.value
        if (rollup_rep and hasattr(rollup_rep.decision, "value"))
        else ("PASS" if rollup_rep else "FAILED")
    )
    return {
        "status": "reconciled",
        "exam_id": exam_id,
        "decision": dec_val,
        "rollup_score": rollup_rep.overall_score if rollup_rep else None,
        "questions_count": rollup_rep.metrics.get("questions_count") if rollup_rep else None,
    }


def process_single_revision(
    target: Dict[str, Any],
    revision_loop: EditorReviewerLoop,
    auto_save: bool,
    dry_run: bool,
    branch_dir: Path,
    annotated_dir: Path,
    reuse_prerun_logs: bool = True,
) -> Dict[str, Any]:
    """Processes repair for a single document target, saving to branch_dir if enabled."""
    doc_id = target["doc_id"]
    xml_path: Path = target["xml_path"]
    raw_path: Path = target["raw_path"]
    annotated_xml = xml_path.read_text(encoding="utf-8")
    if raw_path.suffix == ".json":
        try:
            raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
            raw_ocr_text = raw_data.get("raw_chunk_text", "")
        except Exception:
            raw_ocr_text = ""
        if not raw_ocr_text:
            raw_ocr_text = raw_path.read_text(encoding="utf-8")
    else:
        raw_ocr_text = raw_path.read_text(encoding="utf-8")

    use_cached_review = reuse_prerun_logs and not target.get("resume_from_branch")
    initial_report = target.get("initial_report") if use_cached_review else None
    cached_llm_result = target.get("cached_llm_result") if use_cached_review else None

    rel_xml = target.get("rel_xml_path")
    if not rel_xml:
        try:
            rel_xml = xml_path.resolve().relative_to(annotated_dir.resolve())
        except ValueError:
            rel_xml = Path(doc_id) / xml_path.name

    branch_xml_path = branch_dir / rel_xml
    if branch_xml_path.resolve() == xml_path.resolve():
        cand_in_annot = annotated_dir / rel_xml
        if cand_in_annot.exists() and cand_in_annot.resolve() != branch_xml_path.resolve():
            xml_path = cand_in_annot
        else:
            raise RuntimeError(
                f"Branch path {branch_xml_path} matches original path {xml_path}! "
                "Direct overwriting of original files is strictly prohibited."
            )

    if xml_path.name.startswith("chunk_"):
        state_path = branch_xml_path.with_name(f"{branch_xml_path.stem}_revision_state.json")
    else:
        state_path = branch_xml_path.with_name("revision_state.json")
    state: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "doc_id": doc_id,
        "status": "IN_PROGRESS",
        "original_xml_sha256": _sha256_file(target["original_xml_path"] if target.get("original_xml_path") else xml_path),
        "raw_sha256": _sha256_file(raw_path),
        "source_xml_path": str(target.get("original_xml_path", xml_path)),
        "raw_path": str(raw_path),
        "branch_xml_path": str(branch_xml_path),
        "rounds": [],
    }

    def checkpoint(event: Dict[str, Any]) -> None:
        if not auto_save or dry_run:
            return
        branch_xml_path.parent.mkdir(parents=True, exist_ok=True)
        branch_xml_path.write_text(event["candidate_xml"], encoding="utf-8")
        round_number = int(event.get("round_number", 0))
        event_name = "initial_review.json" if event.get("stage") == "initial_review" else f"round_{round_number:03d}.json"
        event_path = branch_xml_path.parent / "revisions" / event_name
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        state["latest_round"] = round_number
        state["latest_event"] = str(event_path)
        state["rounds"] = [
            item for item in state["rounds"]
            if item.get("round_number") != round_number or item.get("stage") != event.get("stage")
        ]
        state["rounds"].append({
            "stage": event.get("stage"),
            "round_number": round_number,
            "path": str(event_path),
            "decision": event.get("review", {}).get("decision"),
            "score": event.get("review", {}).get("overall_score"),
        })
        _atomic_write_json(state_path, state)

    run_kwargs = {
        "annotated_xml": annotated_xml,
        "raw_ocr_text": raw_ocr_text,
        "doc_id": doc_id,
        "initial_report": initial_report,
        "cached_llm_result": cached_llm_result,
    }
    if auto_save and not dry_run:
        run_kwargs["checkpoint_callback"] = checkpoint
    res = revision_loop.run(**run_kwargs)

    already_passed = (
        res.success
        and res.initial_decision == "PASS"
        and res.final_decision == "PASS"
        and res.applied_patches_count == 0
        and res.rounds_completed == 0
    )

    saved = False
    if auto_save and not dry_run and not already_passed and res.repaired_xml:
        branch_xml_path.parent.mkdir(parents=True, exist_ok=True)
        branch_xml_path.write_text(res.repaired_xml, encoding="utf-8")

        if xml_path.name == "merged.xml" or xml_path.with_name("merged.json").exists():
            branch_json_path = branch_xml_path.with_name("merged.json")
            try:
                spans, stimuli, questions = parse_xml_with_anchors(
                    raw_ocr_text, res.repaired_xml
                )
                json_data = {
                    "document_id": doc_id,
                    "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "questions_count": len(questions),
                    "questions": questions,
                    "stimuli": stimuli,
                }
                branch_json_path.write_text(
                    json.dumps(json_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                state["merged_json_error"] = str(e)
                print(f"⚠️ [Warning] Could not update {branch_json_path}: {e}")

        if xml_path.name.startswith("chunk_"):
            chunk_audit_path = branch_xml_path.with_suffix(".audit.json")
            final_rep = getattr(res, "final_report", None)
            if final_rep:
                chunk_audit_path.write_text(
                    json.dumps(final_rep.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        saved = True
        state["status"] = res.final_decision
        state["result"] = {
            "doc_id": doc_id,
            "initial_score": res.initial_score,
            "final_score": res.final_score,
            "initial_decision": res.initial_decision,
            "final_decision": res.final_decision,
            "success": res.success,
            "applied_patches": res.applied_patches_count,
            "failed_patches": res.failed_patches_count,
            "rounds_completed": res.rounds_completed,
            "rounds": [round_result.model_dump(mode="json") for round_result in res.rounds],
            "diff_summary": res.diff_summary,
            "duration_seconds": res.duration_seconds,
        }
        _atomic_write_json(state_path, state)
    elif auto_save and not dry_run and already_passed:
        if xml_path.name.startswith("chunk_"):
            chunk_audit_path = branch_xml_path.with_suffix(".audit.json")
            final_rep = getattr(res, "final_report", None)
            if final_rep:
                chunk_audit_path.write_text(
                    json.dumps(final_rep.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        state["status"] = "PASS"
        state["result"] = {
            "doc_id": doc_id,
            "initial_score": res.initial_score,
            "final_score": res.final_score,
            "initial_decision": res.initial_decision,
            "final_decision": res.final_decision,
            "success": res.success,
            "already_passed": True,
            "rounds_completed": res.rounds_completed,
        }
        _atomic_write_json(state_path, state)

    return {
        "doc_id": doc_id,
        "source_decision": target["source_decision"],
        "rel_path": str(target["rel_path"]),
        "initial_score": res.initial_score,
        "final_score": res.final_score,
        "initial_decision": res.initial_decision,
        "final_decision": res.final_decision,
        "success": res.success,
        "already_passed": already_passed,
        "applied_patches": res.applied_patches_count,
        "failed_patches": res.failed_patches_count,
        "rounds_completed": res.rounds_completed,
        "rounds": [round_result.model_dump(mode="json") for round_result in res.rounds],
        "saved_to_disk": saved,
        "saved_to_branch": saved,
        "branch_xml_path": str(branch_xml_path) if saved else None,
        "duration_seconds": res.duration_seconds,
        "diff_summary": res.diff_summary,
    }


def run_batch_revision(
    report_path: Path,
    raw_dir: Path,
    annotated_dir: Path,
    branch_dir: Path,
    filter_decision: str = "needs_revision",
    model: str = "gemini-3.8-flash",
    provider: str = "agy",
    reviewer_model: str = "gpt-5.6-luna",
    reviewer_provider: str = "codex",
    editor_thinking: str = "high",
    reviewer_thinking: str = "high",
    concurrency: int = 4,
    limit: Optional[int] = None,
    max_passes: int = 2,
    granularity: str = "chunk",
    target_doc: Optional[str] = None,
    auto_save: bool = True,
    dry_run: bool = False,
    out_report: Optional[Path] = None,
    reuse_prerun_logs: bool = True,
    resume_branch: bool = True,
    retry_unresolved: bool = False,
    auto_reconcile: bool = True,
    skip_unrecoverable: bool = True,
    max_doc_size: int = 300_000,
    force_large: bool = False,
) -> Dict[str, Any]:
    """Executes the batch revision loop over targeted documents."""
    decision_aliases = {
        "needs_revision": "NEEDS_REVISION",
        "revision": "NEEDS_REVISION",
        "discards": "DISCARD",
        "discard": "DISCARD",
    }
    requested_decisions = {
        decision_aliases.get(value.strip().lower(), value.strip().upper())
        for value in filter_decision.split(",")
        if value.strip()
    }
    allowed_decisions = {"NEEDS_REVISION", "DISCARD"}
    if not requested_decisions or not requested_decisions <= allowed_decisions:
        raise ValueError(
            "filter_decision must be needs_revision, discards, or discards,needs_revision"
        )

    targets = load_revision_targets(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=annotated_dir,
        decisions=requested_decisions,
        skip_unrecoverable=skip_unrecoverable,
        granularity=granularity,
    )

    if target_doc:
        query = target_doc.strip()
        targets = [
            t for t in targets
            if query in t["doc_id"] or query in str(t["rel_path"])
        ]

    if max_doc_size and not force_large:
        normal_targets = []
        for t in targets:
            size = t["xml_path"].stat().st_size
            if size > max_doc_size:
                print(
                    f"⚠️  Skipping oversized document {t['doc_id']} "
                    f"({size/1024:.1f} KB > {max_doc_size/1024:.1f} KB). "
                    f"Use force_large=True to include."
                )
            else:
                normal_targets.append(t)
        targets = normal_targets

    if limit:
        targets = targets[:limit]

    for target in targets:
        target["original_xml_path"] = target["xml_path"]

    targets = apply_branch_resume(
        targets,
        branch_dir=branch_dir,
        resume=resume_branch,
        retry_unresolved=retry_unresolved,
    )
    resumed_passes = [t for t in targets if t.get("resume_skip_state")]
    targets = [t for t in targets if not t.get("resume_skip_state")]
    total_target_count = len(targets) + len(resumed_passes)

    results: List[Dict[str, Any]] = []
    for target in resumed_passes:
        state = target["resume_skip_state"]
        saved_result = dict(state.get("result") or {})
        saved_result.setdefault("doc_id", target["doc_id"])
        saved_result.setdefault("source_decision", target["source_decision"])
        saved_result.setdefault("rel_path", str(target["rel_path"]))
        saved_result.setdefault("success", True)
        saved_result["already_passed"] = True
        saved_result["saved_to_branch"] = True
        saved_result["saved_to_disk"] = True
        saved_result["branch_xml_path"] = str(branch_dir / target["rel_xml_path"])
        results.append(saved_result)

    success_count = len(resumed_passes)
    fail_count = 0
    already_passed_count = len(resumed_passes)

    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)

    def save_partial_report(status: str = "IN_PROGRESS") -> None:
        if not out_report:
            return
        _atomic_write_json(
            out_report,
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": status,
                "total_targets": total_target_count,
                "completed_targets": len(results),
                "success_count": success_count,
                "already_passed_count": already_passed_count,
                "fail_count": fail_count,
                "success_rate": round(success_count / max(1, total_target_count) * 100, 1),
                "annotated_dir": str(annotated_dir),
                "branch_dir": str(branch_dir),
                "auto_save": auto_save,
                "reuse_prerun_logs": reuse_prerun_logs,
                "resume_branch": resume_branch,
                "retry_unresolved": retry_unresolved,
                "results": results,
            },
        )

    save_partial_report()

    if not targets and not resumed_passes:
        print("✅ No target documents found to repair.")
        return {"total_targets": 0, "success_count": 0, "results": []}

    reviewer = AnnotationReviewerAgent(
        model=reviewer_model,
        provider=reviewer_provider,
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
        max_rounds=max_passes,
    )

    max_workers = max(1, min(concurrency, len(targets)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_revision,
                target,
                revision_loop,
                auto_save,
                dry_run,
                branch_dir,
                annotated_dir,
                reuse_prerun_logs,
            ): target
            for target in targets
        }

        for future in as_completed(futures):
            target = futures[future]
            doc_id = target["doc_id"]
            try:
                res = future.result()
                results.append(res)
                if res.get("already_passed"):
                    success_count += 1
                    already_passed_count += 1
                    print(
                        f"  ✅ [Already PASS after review] {doc_id}: {res['initial_score']:.1f} ({res['initial_decision']}) — no edit needed"
                    )
                elif res["success"]:
                    success_count += 1
                    saved_info = f" [Saved to branch: {res['branch_xml_path']}]" if res.get("saved_to_branch") else ""
                    print(
                        f"  ✅ [Repaired] {doc_id}: {res['initial_score']:.1f} ({res['initial_decision']}) -> {res['final_score']:.1f} ({res['final_decision']}) [{res['applied_patches']} patches in {res['duration_seconds']:.1f}s]{saved_info}"
                    )
                else:
                    fail_count += 1
                    print(
                        f"  ⚠️ [Partial/Unresolved] {doc_id}: {res['initial_score']:.1f} -> {res['final_score']:.1f} ({res['final_decision']})"
                    )

                if auto_reconcile and (res.get("success") or res.get("already_passed")):
                    exam_id = target["doc_id"].split("::")[0]
                    try:
                        rec_info = reconcile_repaired_exam(exam_id, annotated_dir, branch_dir, reviewer)
                        if rec_info.get("status") == "reconciled":
                            sc_str = f"{rec_info.get('rollup_score'):.1f}" if rec_info.get("rollup_score") is not None else "N/A"
                            print(
                                f"  🎉 [Reconciled Full Exam] {exam_id}: All chunks PASS -> Synthesized branch/merged.xml (Rollup Score: {sc_str})"
                            )
                    except Exception as err:
                        print(f"  ⚠️ [Reconciliation Warning] Could not reconcile {exam_id}: {err}")

                save_partial_report()
            except Exception as e:
                if is_quota_or_cap_error(e):
                    print("\n" + "=" * 70)
                    print(f"🛑 [FATAL QUOTA LIMIT DETECTED] Provider quota/rate limit exhausted while processing {doc_id}!")
                    print(f"   Reason: {e}")
                    print("=" * 70)
                    save_partial_report(status="ABORTED_QUOTA_EXHAUSTED")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                else:
                    fail_count += 1
                    print(f"  ❌ [Error] Failed repairing {doc_id}: {e}")
                    results.append({
                        "doc_id": doc_id,
                        "source_decision": target["source_decision"],
                        "error": str(e),
                        "success": False,
                    })
                    save_partial_report()

    save_partial_report(status="COMPLETED")
    return {
        "total_targets": total_target_count,
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
    }

