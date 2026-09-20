from tests.regex_question_parser.question_parser import parse_question_candidates


def test_extracts_common_question_and_choice_formats():
    text = """
Question 1: What is the capital of France?
A. Berlin
B. Madrid
C. Paris
D. Rome

2) Choose the correct value.
(A) 10 (B) 20 (C) 30 (D) 40
"""

    result = parse_question_candidates(text)

    assert [question.number for question in result.questions] == [1, 2]
    assert result.questions[0].stem == "What is the capital of France?"
    assert [choice.label for choice in result.questions[0].choices] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert result.questions[1].choices[2].text == "30"


def test_handles_vietnamese_and_markdown_ocr_text():
    text = """
**Câu 12.** Tại sao tác giả nhắc đến chi tiết này?
> A) Để giải thích bối cảnh
> B) Để phản bác ý kiến
> C) Để đưa ra kết luận
> D) Để mô tả nhân vật

> **13.** The word "brief" is closest in meaning to:
> (A) short
> (B) hidden
> (C) old
> (D) exact
"""

    result = parse_question_candidates(text)

    assert [question.number for question in result.questions] == [12, 13]
    assert result.questions[0].label.lower() == "câu 12"
    assert len(result.questions[1].choices) == 4


def test_rejects_numbered_procedure_without_question_signals():
    text = """
Account setup
1. Open the employee portal.
2. Enter your identification number.
3. Select two security questions.
"""

    result = parse_question_candidates(text)

    assert result.questions == []
    assert [candidate.number for candidate in result.rejected] == [1, 2, 3]
    assert all("sequential-number" in item.signals for item in result.rejected)


def test_retains_ambiguous_candidates_for_llm_validation():
    text = """
7. Discuss the effects of urbanization

Appendix
2025. Annual performance report
"""

    strict_result = parse_question_candidates(text)
    high_recall_result = parse_question_candidates(text, min_score=2)

    assert [candidate.number for candidate in strict_result.rejected] == [7, 2025]
    assert [candidate.number for candidate in high_recall_result.questions] == [7]
    assert [candidate.number for candidate in high_recall_result.rejected] == [2025]


def test_preserves_source_offsets_for_grounding():
    text = "Header\n\n1. What happened?\nA. Rain\nB. Snow\n\nFooter"

    question = parse_question_candidates(text).questions[0]

    assert text[question.start_char : question.end_char].startswith("1.")
    assert question.raw_text.endswith("Footer")
