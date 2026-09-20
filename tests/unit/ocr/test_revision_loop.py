"""Tests for isolated editor/reviewer revision orchestration."""

from typing import Any, Dict, List
from unittest.mock import Mock

from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    AuditIssue,
    IssueSeverity,
    ReviewDecision,
    ReviewReport,
)
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop


def _report(
    score: float,
    decision: ReviewDecision,
    issue: str = "",
    confirmed: bool = True,
) -> ReviewReport:
    issues = []
    if issue:
        issues.append(
            AuditIssue(
                category="option_structure",
                severity=IssueSeverity.MAJOR,
                message=issue,
            )
        )
    return ReviewReport(
        doc_id="exam_1",
        overall_score=score,
        deterministic_score=score,
        grade="A" if score >= 90 else "B",
        decision=decision,
        is_malfunctioned=False,
        issues=issues,
        confirmed_issues=issues if confirmed else [],
        confirmation_status="complete",
    )


def test_editor_keeps_history_and_reviewer_is_stateless() -> None:
    editor_calls: List[List[Dict[str, str]]] = []

    responses = iter(
        [
            """<<<<<<< SEARCH
<stem>Bad</stem>
=======
<stem>Better</stem>
>>>>>>> REPLACE""",
            """<<<<<<< SEARCH
<stem>Better</stem>
=======
<stem>Correct</stem>
>>>>>>> REPLACE""",
        ]
    )

    def complete(messages: List[Dict[str, str]], **_: Any) -> str:
        editor_calls.append(messages)
        return next(responses)

    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(60, ReviewDecision.NEEDS_REVISION, "Initial error"),
        _report(75, ReviewDecision.NEEDS_REVISION, "Still incorrect"),
        _report(95, ReviewDecision.PASS),
    ]

    loop = EditorReviewerLoop(
        editor=EditorAgent(), reviewer=reviewer, max_rounds=3
    )
    result = loop.run(
        annotated_xml="<stem>Bad</stem>",
        raw_ocr_text="Correct",
        doc_id="exam_1",
        editor_completion_fn=complete,
    )

    assert result.success is True
    assert result.rounds_completed == 2
    assert result.repaired_xml == "<stem>Correct</stem>"
    assert len(editor_calls[0]) == 2
    assert len(editor_calls[1]) == 4
    assert "Still incorrect" in editor_calls[1][-1]["content"]

    review_calls = reviewer.review_document.call_args_list
    assert len(review_calls) == 3
    for call in review_calls:
        assert set(call.kwargs) == {
            "xml_content",
            "raw_ocr_text",
            "doc_id",
            "use_llm",
        }
        assert call.kwargs["raw_ocr_text"] == "Correct"


def test_loop_returns_highest_scoring_version_at_round_limit() -> None:
    responses = iter(
        [
            """<<<<<<< SEARCH
<stem>Bad</stem>
=======
<stem>Best</stem>
>>>>>>> REPLACE""",
            """<<<<<<< SEARCH
<stem>Best</stem>
=======
<stem>Worse</stem>
>>>>>>> REPLACE""",
        ]
    )

    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(60, ReviewDecision.NEEDS_REVISION, "Initial error"),
        _report(82, ReviewDecision.NEEDS_REVISION, "One issue remains"),
        _report(70, ReviewDecision.NEEDS_REVISION, "Regression"),
    ]
    loop = EditorReviewerLoop(
        editor=EditorAgent(), reviewer=reviewer, max_rounds=2
    )

    result = loop.run(
        annotated_xml="<stem>Bad</stem>",
        raw_ocr_text="Best",
        editor_completion_fn=lambda **_: next(responses),
    )

    assert result.success is False
    assert result.final_score == 82
    assert result.repaired_xml == "<stem>Best</stem>"
    assert result.rounds_completed == 2


def test_loop_prefers_valid_candidate_over_high_score_discard() -> None:
    responses = iter(
        [
            """<<<<<<< SEARCH
<stem>Bad</stem>
=======
<stem>Critical Broken</stem>
>>>>>>> REPLACE""",
            """<<<<<<< SEARCH
<stem>Critical Broken</stem>
=======
<stem>Fixed Quality</stem>
>>>>>>> REPLACE""",
        ]
    )

    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(75, ReviewDecision.NEEDS_REVISION, "Initial issue"),
        # Round 1: triggers a critical issue, falls back to unreviewed det_score 87.0 (DISCARD)
        _report(87, ReviewDecision.DISCARD, "Critical stimulus nesting"),
        # Round 2: properly fixed, gets valid NEEDS_REVISION score 80.0
        _report(80, ReviewDecision.NEEDS_REVISION, "Minor note"),
    ]
    loop = EditorReviewerLoop(
        editor=EditorAgent(), reviewer=reviewer, max_rounds=2
    )

    result = loop.run(
        annotated_xml="<stem>Bad</stem>",
        raw_ocr_text="Fixed Quality",
        editor_completion_fn=lambda **_: next(responses),
    )

    # Must prefer Round 2 (NEEDS_REVISION at 80.0) over Round 1 (DISCARD at 87.0)
    assert result.final_decision == ReviewDecision.NEEDS_REVISION.value
    assert result.final_score == 80
    assert result.repaired_xml == "<stem>Fixed Quality</stem>"


def test_loop_does_not_send_unconfirmed_findings_to_editor() -> None:
    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.return_value = _report(
        75,
        ReviewDecision.NEEDS_REVISION,
        "Candidate finding",
        confirmed=False,
    )

    def should_not_run(**_: Any) -> str:
        raise AssertionError("editor must not receive unconfirmed findings")

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer)
    result = loop.run(
        annotated_xml="<stem>Original</stem>",
        raw_ocr_text="Original",
        editor_completion_fn=should_not_run,
    )

    assert result.rounds_completed == 0
    assert result.repaired_xml == "<stem>Original</stem>"
    assert result.confirmed_issues_count == 0


def test_editor_can_record_false_positive_assessment_without_patching() -> None:
    responses = iter(
        [
            """<<<ISSUE_ASSESSMENTS>>>
[{"issue_id": 1, "is_false_positive": true, "reason": "The source supports the existing tag."}]
<<<END_ISSUE_ASSESSMENTS>>>"""
        ]
    )
    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(75, ReviewDecision.NEEDS_REVISION, "Confirmed finding"),
        _report(75, ReviewDecision.NEEDS_REVISION, "Confirmed finding"),
    ]

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer)
    result = loop.run(
        annotated_xml="<stem>Original</stem>",
        raw_ocr_text="Original",
        editor_completion_fn=lambda **_: next(responses),
    )

    assert result.repaired_xml == "<stem>Original</stem>"
    assert result.applied_patches_count == 0
    assert result.false_positive_issue_ids == [1]
    assert result.false_positive_assessments[0].is_false_positive is True


def test_loop_directly_consumes_prerun_initial_report() -> None:
    """Verify that when initial_report is provided, Round 0 does not call reviewer."""
    prerun_report = _report(65, ReviewDecision.NEEDS_REVISION, "Prerun error finding", confirmed=True)

    patch_response = """<<<<<<< SEARCH
<stem>Original</stem>
=======
<stem>Repaired</stem>
>>>>>>> REPLACE"""

    reviewer = Mock(spec=AnnotationReviewerAgent)
    # Reviewer should only be called once, for candidate evaluation on Round 1
    reviewer.review_document.return_value = _report(95, ReviewDecision.PASS)

    editor_prompts: List[str] = []

    def complete(messages: List[Dict[str, str]], **_: Any) -> str:
        for m in messages:
            if m.get("role") == "user":
                editor_prompts.append(m.get("content", ""))
        return patch_response

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer, max_rounds=2)
    result = loop.run(
        annotated_xml="<stem>Original</stem>",
        raw_ocr_text="Original",
        doc_id="exam_prerun",
        editor_completion_fn=complete,
        initial_report=prerun_report,
    )

    # Reviewer must only be called for Round 1 candidate check, NOT Round 0
    assert reviewer.review_document.call_count == 1
    call_args = reviewer.review_document.call_args
    assert call_args.kwargs.get("xml_content") == "<stem>Repaired</stem>"

    # Editor must have received the prerun error finding
    assert any("Prerun error finding" in p for p in editor_prompts)
    assert result.success is True
    assert result.initial_score == 65.0
    assert result.final_score == 95.0
    assert result.repaired_xml == "<stem>Repaired</stem>"


def test_loop_initial_report_already_pass_skips_all_calls() -> None:
    """Verify that an initial_report with PASS terminates immediately with 0 reviewer calls."""
    prerun_report = _report(95, ReviewDecision.PASS)
    reviewer = Mock(spec=AnnotationReviewerAgent)

    def should_not_run(**_: Any) -> str:
        raise AssertionError("editor must not run when initial_report is PASS")

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer)
    result = loop.run(
        annotated_xml="<stem>Clean</stem>",
        raw_ocr_text="Clean",
        editor_completion_fn=should_not_run,
        initial_report=prerun_report,
    )

    assert reviewer.review_document.call_count == 0
    assert result.success is True
    assert result.initial_decision == "PASS"
    assert result.final_decision == "PASS"
    assert result.rounds_completed == 0


def test_loop_consumes_cached_llm_result_on_round_0() -> None:
    """Verify that cached_llm_result is passed to review_document on Round 0."""
    cached_llm = {"score": 70.0, "parser_error_confirmations": []}

    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(70, ReviewDecision.NEEDS_REVISION, "Cached error", confirmed=True),
        _report(92, ReviewDecision.PASS),
    ]

    patch_response = """<<<<<<< SEARCH
<stem>Original</stem>
=======
<stem>Fixed</stem>
>>>>>>> REPLACE"""

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer, max_rounds=2)
    result = loop.run(
        annotated_xml="<stem>Original</stem>",
        raw_ocr_text="Original",
        editor_completion_fn=lambda **_: patch_response,
        cached_llm_result=cached_llm,
    )

    assert reviewer.review_document.call_count == 2
    # Round 0 call must pass cached_llm_result
    round0_call = reviewer.review_document.call_args_list[0]
    assert round0_call.kwargs.get("cached_llm_result") == cached_llm

    # Round 1 candidate call must NOT pass cached_llm_result (live review)
    round1_call = reviewer.review_document.call_args_list[1]
    assert round1_call.kwargs.get("cached_llm_result") is None
    assert result.success is True
    assert result.final_score == 92.0


def test_loop_emits_initial_and_round_checkpoints() -> None:
    checkpoints: List[Dict[str, Any]] = []
    reviewer = Mock(spec=AnnotationReviewerAgent)
    reviewer.review_document.side_effect = [
        _report(70, ReviewDecision.NEEDS_REVISION, "Initial error"),
        _report(95, ReviewDecision.PASS),
    ]
    patch_response = """<<<<<<< SEARCH
<stem>Original</stem>
=======
<stem>Fixed</stem>
>>>>>>> REPLACE"""

    loop = EditorReviewerLoop(editor=EditorAgent(), reviewer=reviewer, max_rounds=1)
    result = loop.run(
        annotated_xml="<stem>Original</stem>",
        raw_ocr_text="Original",
        doc_id="exam_checkpoint",
        editor_completion_fn=lambda **_: patch_response,
        checkpoint_callback=checkpoints.append,
    )

    assert result.success is True
    assert [item["stage"] for item in checkpoints] == ["initial_review", "round"]
    assert checkpoints[0]["candidate_xml"] == "<stem>Original</stem>"
    assert checkpoints[1]["candidate_xml"] == "<stem>Fixed</stem>"
    assert checkpoints[1]["round_number"] == 1
    assert checkpoints[1]["review"]["decision"] == "PASS"
