from sequence_labelling.annotator.editor import SearchReplaceBlock, SearchReplacePatcher
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic
from sequence_labelling.parser.long_parser.greedy_chunker import split_unpaged_document


def _spans(text):
    return parse_chunk_deterministic(text).spans


def test_heading_marker_belongs_to_question_label_not_previous_span():
    text = "## Câu 1. Cho\nA. 1\nB. 2\nC. 3\nD. 4\n\n## Câu 2. Tính\nA. 1\nB. 2\nC. 3\nD. 4\n"
    spans = _spans(text)
    labels = [s["text"] for s in spans if s["label"] == "question_label"]
    assert labels == ["## Câu 1.", "## Câu 2."]
    assert all("#" not in s["text"] for s in spans if s["label"] in ("stem", "option_text"))


def test_heading_marker_not_swallowed_by_optionless_stem():
    spans = _spans("## Câu 1. Cho hàm số.\n\n## Câu 2. Tính\nA. 1\nB. 2\n")
    stems = [s["text"] for s in spans if s["label"] == "stem"]
    assert stems[0] == "Cho hàm số."


def test_cloze_question_without_stem_still_valid():
    text = "## Câu 631\nA. a\nB. b\nC. c\nD. d\n\n## Câu 632\nA. a\nB. b\nC. c\nD. d\n"
    labels = [s["label"] for s in _spans(text)]
    assert "stem" not in labels
    assert labels.count("question_label") == 2


def test_unpaged_split_global_offsets_match_text():
    body = "\n\n".join(f"para{i} " + "w " * 40 for i in range(30))
    pages = split_unpaged_document(
        {"text": body, "global_start": 100}, target_tokens=200, max_tokens=300, overlap_fallback_tokens=50
    )
    assert len(pages) > 2
    for page in pages:
        assert body[page["global_start"] - 100 : page["global_end"] - 100] == page["text"]
    assert pages[-1]["global_end"] - 100 == len(body)


def test_ambiguous_search_block_is_rejected_not_first_match():
    doc = "<q1>$x = 1$</q1>\n<q2>$x = 1$</q2>"
    out, applied, failed = SearchReplacePatcher.apply_blocks(doc, [SearchReplaceBlock("$x = 1$", "$x = 2$")])
    assert out == doc and applied == 0 and "ambiguous" in failed[0]


def test_unique_search_block_with_context_applies():
    doc = "<q1>$x = 1$</q1>\n<q2>$x = 1$</q2>"
    out, applied, failed = SearchReplacePatcher.apply_blocks(
        doc, [SearchReplaceBlock("<q2>$x = 1$", "<q2>$x = 2$")]
    )
    assert out == "<q1>$x = 1$</q1>\n<q2>$x = 2$</q2>" and applied == 1 and not failed


def test_stimulus_not_bound_across_intervening_section():
    from sequence_labelling.parser.long_parser.source_merger.serializer_validator import build_structured_questions
    from sequence_labelling.parser.long_parser.source_merger.types import GlobalSpan

    canonical_text = (
        "Đoạn văn đọc hiểu tiếng Anh.\n"  # 0..28
        "Câu 1. What is this?\n"          # 29..49
        "## PHẦN II. TOÁN HỌC\n"         # 50..71
        "Câu 2. Tính 1+1.\n"              # 72..89
    )
    spans = [
        GlobalSpan(start=0, end=28, label="stimulus", chunk_index=0),
        GlobalSpan(start=29, end=35, label="question_label", chunk_index=0),
        GlobalSpan(start=36, end=49, label="stem", chunk_index=0),
        GlobalSpan(start=50, end=71, label="section", chunk_index=0),
        GlobalSpan(start=72, end=78, label="question_label", chunk_index=0),
        GlobalSpan(start=79, end=89, label="stem", chunk_index=0),
    ]
    structured_stimuli = {
        "stim_reading": {
            "text": "Đoạn văn đọc hiểu tiếng Anh.",
            "source_range": {"start": 0, "end": 28},
        }
    }
    questions = build_structured_questions(canonical_text, spans, structured_stimuli)
    assert len(questions) == 2
    # Question 1 belongs to stim_reading
    assert questions[0]["stimulus_id"] == "stim_reading"
    # Question 2 is separated by section header -> stimulus_id must be None
    assert questions[1]["stimulus_id"] is None


def test_greedy_chunker_splits_oversized_page_in_multipage():
    from sequence_labelling.parser.long_parser.greedy_chunker import greedy_oversize_chunker

    body = "\n\n".join(f"para{i} " + "w " * 50 for i in range(20))
    page_list = [
        {"p": 1, "text": "Page 1 intro", "estimated_tokens": 10},
        {"p": 2, "text": body, "estimated_tokens": 1200},  # dense oversized page
        {"p": 3, "text": "Page 3 outro", "estimated_tokens": 10},
    ]
    chunks = greedy_oversize_chunker(page_list, target_tokens=300, max_tokens=500)
    assert len(chunks) >= 3
    # Ensure no individual page in chunks exceeds max_tokens
    for chunk in chunks:
        assert sum(p.get("estimated_tokens", 0) for p in chunk) <= 700


def test_metadata_extractor_grade_and_date_disambiguation():
    from sequence_labelling.metadata.extractor import heuristic_metadata_fallback

    # Day of month should NOT match as grade
    m1 = heuristic_metadata_fallback("de-thi-thu-ngay-12-thang-5.pdf")
    assert m1.grade is None

    # Context header with grade should match
    m2 = heuristic_metadata_fallback("de-thi-hoc-ky-1.pdf", context="SỞ GD&ĐT HÀ NỘI - ĐỀ THI LỚP 12 MÔN TOÁN")
    assert m2.grade == "12"
    assert m2.subject == "Toán"

    # GDCD full title detection
    m3 = heuristic_metadata_fallback("de-thi.pdf", context="ĐỀ KIỂM TRA MÔN GIÁO DỤC CÔNG DÂN LỚP 11")
    assert m3.grade == "11"
    assert m3.subject == "GDCD / GDKT&PL"

