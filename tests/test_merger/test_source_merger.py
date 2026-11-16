import re
from pathlib import Path
from typing import Tuple
from sequence_labelling.parser.long_parser.source_merger.types import ChunkInput
from sequence_labelling.parser.long_parser.source_merger.merger import merge_chunks
from sequence_labelling.parser.long_parser.source_merger.serializer_validator import remove_annotation_tags


def _load_fixture(*parts: str) -> str:
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    return (fixture_root.joinpath(*parts)).read_text(encoding="utf-8").strip()


def _load_chunk_fixture(*parts: str) -> Tuple[str, str]:
    content = _load_fixture(*parts)
    input_match = re.search(r"<input>(.*?)</input>", content, re.DOTALL)
    parsed_match = re.search(r"<parsed>(.*?)</parsed>", content, re.DOTALL)
    raw = input_match.group(1).strip() if input_match else ""
    parsed = parsed_match.group(1).strip() if parsed_match else content.strip()
    return raw, parsed


def test_sample_regression_chunk_1_and_chunk_2():
    """
    Test sample regression using artifact/chunk_1.xml and artifact/chunk_2.xml.
    Verifies Section 12 & Section 15 specifications:
    - Questions 172-179 appear once.
    - Shared source text appears once.
    - Question 179 has a valid option D annotation.
    - Malformed split tag <option_la\\nbel> does not survive in merged tags.
    - De-tagged merged XML equals canonical original text exactly.
    """
    raw1, xml1 = _load_chunk_fixture("sample_regression_chunk_1_and_chunk_2", "chunk_a.xml")
    raw2, xml2 = _load_chunk_fixture("sample_regression_chunk_1_and_chunk_2", "chunk_b.xml")

    chunks = [
        ChunkInput(index=0, original_text=raw1, parsed_xml=xml1),
        ChunkInput(index=1, original_text=raw2, parsed_xml=xml2),
    ]

    result = merge_chunks(chunks)

    # 1. Source fidelity invariant
    detagged = remove_annotation_tags(result.merged_xml)
    assert detagged == result.original_text

    # 2. Shared text & questions appear once
    assert result.merged_xml.count("Questions 176-180") == 1
    assert result.merged_xml.count("179.") == 1

    # 3. Question 179 has valid option D tag
    assert "<option_label>(D)</option_label>" in result.merged_xml
    assert "<option_la\n" not in result.merged_xml

    # 4. Total structured questions = 8 (172, 173, 174, 175, 176, 177, 178, 179)
    q_nums = [q["question_number"] for q in result.structured_questions]
    assert "172" in q_nums
    assert "179" in q_nums
    assert len(set(q_nums)) == 8


def test_parser_transformation_preserves_original_text():
    """
    Test that LLM parser insertions/deletions/substitutions/reflows are NEVER copied into final output.
    Original text is authoritative.
    """
    raw1, xml1 = _load_chunk_fixture("parser_transformation_preserves_original_text", "chunk_a.xml")

    chunks = [
        ChunkInput(index=0, original_text=raw1, parsed_xml=xml1)
    ]

    result = merge_chunks(chunks)

    assert result.original_text == raw1
    assert remove_annotation_tags(result.merged_xml) == raw1
    assert "originaal" not in result.merged_xml
    assert "extra hallucinated word" not in result.merged_xml
    assert "<stem>" in result.merged_xml
    assert "</stem>" in result.merged_xml


def test_overlap_variations():
    """
    Test exact overlap, normalized overlap, no overlap.
    """
    raw1, xml1 = _load_chunk_fixture("overlap_variations", "chunk_a.xml")
    raw2, xml2 = _load_chunk_fixture("overlap_variations", "chunk_b.xml")

    chunks = [
        ChunkInput(index=0, original_text=raw1, parsed_xml=xml1),
        ChunkInput(index=1, original_text=raw2, parsed_xml=xml2),
    ]

    res = merge_chunks(chunks)

    assert res.merged_xml.count("Overlap passage line 1.") == 1
    assert res.merged_xml.count("Overlap passage line 2.") == 1
    assert remove_annotation_tags(res.merged_xml) == res.original_text


def test_malformed_and_split_tags_reconciliation():
    """
    Test handling of broken tags in first chunk resolved by valid tags in second chunk.
    """
    raw1, xml1 = _load_chunk_fixture("malformed_and_split_tags_reconciliation", "chunk_a.xml")
    raw2, xml2 = _load_chunk_fixture("malformed_and_split_tags_reconciliation", "chunk_b.xml")

    chunks = [
        ChunkInput(index=0, original_text=raw1, parsed_xml=xml1),
        ChunkInput(index=1, original_text=raw2, parsed_xml=xml2),
    ]

    res = merge_chunks(chunks)

    assert remove_annotation_tags(res.merged_xml) == res.original_text
    assert res.merged_xml.count("<option_label>") == 2
    assert "<option_la\n" not in res.merged_xml


def test_repeated_prefix_does_not_expand_stimulus_across_mapping_seam():
    raw1, xml1 = _load_chunk_fixture(
        "simulates_real_time_overlap_window",
        "chunk_a.xml",
    )
    raw2, xml2 = _load_chunk_fixture(
        "simulates_real_time_overlap_window",
        "chunk_b.xml",
    )

    result = merge_chunks([
        ChunkInput(index=0, original_text=raw1, parsed_xml=xml1),
        ChunkInput(index=1, original_text=raw2, parsed_xml=xml2),
    ])

    assert result.merged_xml.count(
        "<stimulus>Shared passage appears in both chunks.</stimulus>"
    ) == 1
    stimulus_spans = [span for span in result.annotations if span.label == "stimulus"]
    assert len(stimulus_spans) == 1
    assert stimulus_spans[0].text == "Shared passage appears in both chunks."


def test_no_overlap_concatenates_authoritative_sources():
    result = merge_chunks([
        ChunkInput(
            index=0,
            original_text="Alpha source.",
            parsed_xml="<stem>Alpha source.</stem>",
        ),
        ChunkInput(
            index=1,
            original_text="Beta source.",
            parsed_xml="<stem>Beta source.</stem>",
        ),
    ])

    assert result.original_text == "Alpha source.Beta source."
    assert remove_annotation_tags(result.merged_xml) == result.original_text
    assert result.diagnostics.overlaps[0]["status"] == "NO_OVERLAP"


def test_same_question_number_at_distinct_source_positions_survives():
    raw = "Part A\n**1.** First question.\nPart B\n**1.** Second question."
    parsed = (
        "<section>Part A</section>\n"
        "<question_label>**1.**</question_label> <stem>First question.</stem>\n"
        "<section>Part B</section>\n"
        "<question_label>**1.**</question_label> <stem>Second question.</stem>"
    )

    result = merge_chunks([
        ChunkInput(index=0, original_text=raw, parsed_xml=parsed),
    ])

    assert len(result.structured_questions) == 2
    assert len({question["id"] for question in result.structured_questions}) == 2
    assert [question["question_number"] for question in result.structured_questions] == [
        "1",
        "1",
    ]


def test_literal_split_tag_is_replaced_by_valid_overlap_candidate():
    raw = "**5.**\nOption A."
    malformed = (
        "<question_label>**5.**</question_label>\n"
        "<option_la\nbel>Option A.</option_label>"
    )
    valid = (
        "<question_label>**5.**</question_label>\n"
        "<option_label>Option A.</option_label>"
    )

    result = merge_chunks([
        ChunkInput(index=0, original_text=raw, parsed_xml=malformed),
        ChunkInput(index=1, original_text=raw, parsed_xml=valid),
    ])

    assert remove_annotation_tags(result.merged_xml) == raw
    assert result.merged_xml.count("<option_label>Option A.</option_label>") == 1
    assert "<option_la\n" not in result.merged_xml


def test_paired_stimulus_tag_merging():
    raw = (
        "Dựa vào thông tin sau đây để trả lời câu hỏi 1 và 2:\n"
        "Đoạn văn đọc hiểu chung.\n\n"
        "**1.** Câu hỏi 1?\n"
        "A. Lựa chọn A\n\n"
        "**2.** Câu hỏi 2?\n"
        "A. Lựa chọn B"
    )
    parsed = (
        "<stimulus>Dựa vào thông tin sau đây để trả lời câu hỏi 1 và 2:\n"
        "Đoạn văn đọc hiểu chung.</stimulus>\n\n"
        "<question_label>**1.**</question_label> <stem>Câu hỏi 1?</stem>\n"
        "<option_label>A.</option_label> <option_text>Lựa chọn A</option_text>\n\n"
        "<question_label>**2.**</question_label> <stem>Câu hỏi 2?</stem>\n"
        "<option_label>A.</option_label> <option_text>Lựa chọn B</option_text>"
    )

    result = merge_chunks([
        ChunkInput(index=0, original_text=raw, parsed_xml=parsed),
    ])

    assert remove_annotation_tags(result.merged_xml) == raw
    assert "<stimulus>Dựa vào thông tin sau đây để trả lời câu hỏi 1 và 2:\nĐoạn văn đọc hiểu chung.</stimulus>" in result.merged_xml
    assert len(result.structured_questions) == 2
    assert result.structured_questions[0]["stimulus_id"] is not None
    assert "Đoạn văn đọc hiểu chung." in result.structured_questions[0]["stimulus_text"]
    assert result.structured_questions[1]["stimulus_id"] == result.structured_questions[0]["stimulus_id"]


def test_passthrough_tags_and_figures_not_flagged_as_malformed():
    """
    Ensures <figure ... /> and HTML tables/formatting in parsed XML pass through
    as valid source text and do not trigger lexical errors or false hallucinations.
    """
    from sequence_labelling.parser.long_parser.source_merger.xml_lexer import lex_annotations

    xml = (
        '<figure id="fig_1" description="Exam cover" bbox="10,20,100,200" />\n'
        '<question_label>Câu 1.</question_label>\n'
        '<stem>Bảng sau đây mô tả số liệu:\n'
        '<table><tr><th>Tên</th><th>Giá trị</th></tr><tr><td>A</td><td>10</td></tr></table>\n'
        'Chọn phát biểu đúng:</stem>\n'
        '<option_label>A.</option_label> <option_text>Giá trị là 10.</option_text>\n'
        '<option_la\nbel>B.</option_la\nbel> <option_text>Giá trị là 20.</option_text>'
    )

    parsed_text, spans, report = lex_annotations(xml)
    errors = report["errors"]

    # Figures and table tags must NOT be flagged as unrecognized errors
    for err in errors:
        assert "figure" not in err
        assert "table" not in err
        assert "<tr>" not in err
        assert "<th>" not in err
        assert "<td>" not in err

    # The actual malformed split tag <option_la\nbel> MUST be flagged
    assert any("option_la" in err for err in errors)


def test_math_inequalities_do_not_swallow_closing_tags_in_lexer():
    from sequence_labelling.parser.long_parser.source_merger.xml_lexer import lex_annotations

    xml = (
        "<question_label>Bài 11:</question_label> <stem>Cho $2\\le x<5$.</stem>\n"
        "<explanation>Ta có $(2;5)=\\{x\\in\\mathbb{R}\\mid 2<x<5\\}$ và $[2;5)=\\{x\\mid 2\\le x<5\\}$.</explanation>\n"
        "<question_label>Bài 12:</question_label> <stem>Hỏi $x$?</stem>"
    )
    _, spans, report = lex_annotations(xml)
    assert report["errors"] == []
    expl = [s for s in spans if s.label == "explanation"]
    assert len(expl) == 1
    assert expl[0].is_valid is True
    assert "Bài 12:" not in expl[0].text


def test_snap_boundaries_protects_html_markup_and_rejects_huge_fuzzy_stretch():
    raw = "<table><tr><td>**3.1**</td><td>Bạn An có một tấm tôn đều.</td></tr></table>\n" + ("Câu hỏi khác.\n" * 80) + "Đoạn cuối."
    parsed = (
        "<table><tr><td><explanation>**3.1**</explanation></td>"
        "<td><explanation>> Bạn An có một tấm tôn đều.</explanation></td></tr></table>\n"
        "<stem>Câu hỏi khác.</stem>"
    )
    res = merge_chunks([ChunkInput(index=0, original_text=raw, parsed_xml=parsed)])
    assert remove_annotation_tags(res.merged_xml) == raw
    assert "<td<" not in res.merged_xml
    assert "</ce<" not in res.merged_xml


def test_reconciler_prevents_explanation_overlap_and_nesting():
    raw = (
        "<question_label>Câu 1.</question_label> <stem>Đề 1.</stem>\n"
        "Lời giải câu 1.\n"
        "Câu 2. Đề 2.\n"
        "Lời giải câu 2."
    )
    raw_plain = remove_annotation_tags(raw)
    c0 = (
        "<question_label>Câu 1.</question_label> <stem>Đề 1.</stem>\n"
        "<explanation>Lời giải câu 1.\n"
        "Câu 2. Đề 2.\n"
        "Lời giải câu 2."  # intentionally unclosed
    )
    c1 = (
        "<question_label>Câu 1.</question_label> <stem>Đề 1.</stem>\n"
        "<explanation>Lời giải câu 1.</explanation>\n"
        "<question_label>Câu 2.</question_label> <stem>Đề 2.</stem>\n"
        "<explanation>Lời giải câu 2.</explanation>"
    )
    res = merge_chunks([
        ChunkInput(index=0, original_text=raw_plain, parsed_xml=c0),
        ChunkInput(index=1, original_text=raw_plain, parsed_xml=c1),
    ])
    assert remove_annotation_tags(res.merged_xml) == raw_plain
    assert res.merged_xml.count("<explanation>") == 2
    assert res.merged_xml.count("</explanation>") == 2
    assert "<explanation>Lời giải câu 1.</explanation>" in res.merged_xml
    assert "<question_label>Câu 2.</question_label>" in res.merged_xml



