"""Unit tests for the experimental adaptive scoring system."""

import pytest
from tools.benchmarks.eval_adaptive_scoring_experiment import (
    DocumentScaleContext,
    ScoringEvaluator,
    CATEGORY_PENALTY_CAPS,
)


def test_document_scale_context():
    ctx_small = DocumentScaleContext(total_questions=10)
    assert ctx_small.scale_factor == 1.0

    ctx_medium = DocumentScaleContext(total_questions=40)
    assert ctx_medium.scale_factor == 2.0  # sqrt(40/10) = 2.0

    ctx_large = DocumentScaleContext(total_questions=90)
    assert ctx_large.scale_factor == 3.0  # sqrt(90/10) = 3.0


def test_adaptive_diminishing_saturation_bounds():
    # 5 defects of mislabelled_element (MAJOR = 15 each -> raw sum = 75)
    # Cap for mislabelled_element is 35.0
    report_large = {
        "doc_id": "test_saturation",
        "deterministic_score": 100.0,
        "metrics": {"questions_count": 50},
        "confirmed_issues": [
            {"category": "mislabelled_element", "severity": "MAJOR", "message": f"err {i}"}
            for i in range(10)
        ],
        "discard_reasons": [],
    }

    legacy_res = ScoringEvaluator.score_legacy(report_large)
    # In legacy: 10 * 15 = 150 deduction -> LLM score = 0.0 -> Overall = 40.0 -> DISCARD
    assert legacy_res.llm_score == 0.0
    assert legacy_res.overall_score == 40.0
    assert legacy_res.decision == "DISCARD"

    adaptive_res = ScoringEvaluator.score_full_adaptive(report_large)
    # In adaptive: penalty saturates asymptotically near cap 35.0
    # Overall score will be significantly higher than 40.0
    assert adaptive_res.overall_score >= 70.0
    # Because there are 10 errors (systemic), decision remains NEEDS_REVISION, NOT DISCARD
    assert adaptive_res.decision == "NEEDS_REVISION"


def test_isolated_error_in_large_document_adaptive():
    # 1 mislabelled_element in a 50-question document (2% defect rate)
    report_single_error = {
        "doc_id": "test_isolated",
        "deterministic_score": 100.0,
        "metrics": {"questions_count": 50},
        "confirmed_issues": [
            {"category": "mislabelled_element", "severity": "MAJOR", "message": "Isolated heading tag"}
        ],
        "discard_reasons": [],
    }

    legacy_res = ScoringEvaluator.score_legacy(report_single_error)
    # In legacy, any MAJOR issue forces NEEDS_REVISION
    assert legacy_res.decision == "NEEDS_REVISION"

    adaptive_res = ScoringEvaluator.score_full_adaptive(report_single_error)
    # In adaptive, 1 isolated error in 50 Qs is downgraded to MINOR and passes with high score
    assert adaptive_res.decision == "PASS"
    assert adaptive_res.overall_score >= 90.0


def test_critical_defect_invariants_are_preserved():
    # Stimulus nesting is a fatal structural defect
    report_fatal = {
        "doc_id": "test_fatal",
        "deterministic_score": 95.0,
        "metrics": {"questions_count": 100},
        "confirmed_issues": [
            {"category": "stimulus_nesting", "severity": "CRITICAL", "message": "Stimulus illegally wraps question tags"}
        ],
        "discard_reasons": ["[STIMULUS_NESTING] fatal nesting"],
    }

    adaptive_res = ScoringEvaluator.score_full_adaptive(report_fatal)
    assert adaptive_res.decision == "DISCARD"

    minimal_res = ScoringEvaluator.score_minimal_heuristic(report_fatal)
    assert minimal_res.decision == "DISCARD"


def test_minimal_heuristic_rescues_large_doc_from_arbitrary_floor():
    # 452-question document scoring 72.5 with NO critical issues
    report_large_discard = {
        "doc_id": "exam_452_q",
        "deterministic_score": 80.0,
        "metrics": {"questions_count": 452},
        "confirmed_issues": [
            {"category": "mislabelled_element", "severity": "MAJOR", "message": f"err {i}"}
            for i in range(2)
        ],
        "discard_reasons": ["Overall score 72.5/100 is below minimum threshold 75."],
    }

    legacy_res = ScoringEvaluator.score_legacy(report_large_discard)
    # Legacy discards due to the string reason / floor < 75
    assert legacy_res.decision == "DISCARD"

    minimal_res = ScoringEvaluator.score_minimal_heuristic(report_large_discard)
    # Minimal heuristic recognises Q >= 30 and no criticals -> routes to NEEDS_REVISION
    assert minimal_res.decision == "NEEDS_REVISION"
