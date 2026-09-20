"""Isolated editor/reviewer loop for XML annotation repair.

The editor keeps a per-document conversation.  The reviewer is deliberately
stateless: every review receives only the original OCR text and the current XML
candidate, preventing the editor's rationale from influencing the audit.
"""

from __future__ import annotations

import difflib
import time
from typing import Any, Callable, Dict, List, Optional, Set

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
    ReviewDecision,
    ReviewReport,
)
from sequence_labelling.annotator.xml_cleaner import XMLCleaner


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
        )
