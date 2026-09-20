import re
import pytest

pytest.importorskip("sequence_labelling.parser.long_parser.old.deterministic_parser")
from sequence_labelling.parser.long_parser.old.deterministic_parser import parse_chunk_deterministic
from sequence_labelling.parser.long_parser.old.review_payload import (
    GAP_HEAD,
    build_review_payload,
    collect_review_regions,
)


def _exam(n_questions: int = 12) -> str:
    blocks = []
    for i in range(1, n_questions + 1):
        blocks.append(
            f"**{i}.** Stem for question {i}.\n"
            f"A. alpha {i}\nB. beta {i}\nC. gamma {i}\nD. delta {i}"
        )
    return "\n\n".join(blocks)


def test_confident_questions_are_pruned_from_payload():
    raw = _exam()
    result = parse_chunk_deterministic(raw)
    payload = build_review_payload(raw, result)

    assert not result.escalation_intervals
    # Every question was parsed deterministically, so none should reach the LLM.
    assert payload.questions_elided == payload.questions_total
    assert payload.retention_ratio < 0.10
    assert "Stem for question 5." not in payload.text


def test_elision_marker_reports_what_was_removed():
    raw = _exam()
    payload = build_review_payload(raw, parse_chunk_deterministic(raw))

    markers = re.findall(
        r"\[offset \d+-\d+ elided: (\d+) chars(?:, (\d+) questions? already parsed)?\]",
        payload.text,
    )
    assert markers, "payload must state that text was removed"
    reported_questions = sum(int(q) for _, q in markers if q)
    assert reported_questions == payload.questions_elided


def test_retained_regions_carry_absolute_offsets():
    raw = (
        "Read the passage and answer questions from 1 to 2.\n\n"
        + "The passage body runs on for a while. " * 12
        + "\n\n**1.** First stem.\nA. a\nB. b\nC. c\nD. d\n\n"
        + "**2.** Second stem.\nA. a\nB. b\nC. c\nD. d"
    )
    result = parse_chunk_deterministic(raw)
    payload = build_review_payload(raw, result, include_resolved_gaps=True)

    for region in payload.regions:
        assert f"[offset {region.start}]" in payload.text
        # The offset must index the true source, so a returned anchor resolves
        # without any alignment step.
        assert raw[region.start:region.end] in payload.text


def test_stimulus_gap_is_truncated_to_its_head():
    body = "Passage sentence that keeps going and going. " * 120
    raw = (
        "Read the passage and answer the questions.\n\n"
        + body
        + "\n\n**1.** First stem.\nA. a\nB. b\nC. c\nD. d"
    )
    result = parse_chunk_deterministic(raw)
    regions = collect_review_regions(
        raw,
        result.escalation_intervals,
        result.stimulus_gaps,
        include_resolved_gaps=True,
    )

    assert regions, "a long unclaimed passage must be offered for review"
    # Only the head is needed: the passage end is the next question marker.
    assert all(region.length <= GAP_HEAD for region in regions)


def test_empty_parse_yields_no_regions():
    result = parse_chunk_deterministic("")
    payload = build_review_payload("", result)

    assert payload.regions == []
    assert payload.retention_ratio == 0.0
