from sequence_labelling.parser.long_parser.source_merger.canonical_source import (
    _align_clean,
    build_canonical_source,
    strip_canonical_markup,
)
from sequence_labelling.parser.long_parser.source_merger.types import ChunkInput
from sequence_labelling.parser.long_parser.sequence_reconciler import merge_chunk_xmls


def _fidelity_mismatches(result):
    return result["diagnostics"]["validation"].get("source_fidelity_mismatches", 0)


def test_comparison_operators_are_source_content():
    clean, keep = strip_canonical_markup("Cho a < b và c > d.")
    assert clean == "Cho a < b và c > d."
    assert keep == list(range(len("Cho a < b và c > d.")))


def test_annotation_markup_is_removed_but_not_other_angle_brackets():
    clean, _ = strip_canonical_markup("<stem>Tính x<5</stem> với y>2")
    assert clean == "Tính x<5 với y>2"


def test_comparisons_survive_merge_without_loss():
    raw1 = "Câu 1: Cho a < b và c > d. Chọn đáp án đúng.\n## Câu 2\nA. đúng"
    raw2 = "## Câu 2\nA. đúng\nB. sai\n## Câu 3\nTính x<5 và y>2 rồi chọn."
    x1 = "<question_label>## Câu 2</question_label>\n<option_label>A.</option_label> <option_text>đúng</option_text>"
    x2 = (
        "<question_label>## Câu 2</question_label>\n"
        "<option_label>A.</option_label> <option_text>đúng</option_text>\n"
        "<question_label>## Câu 3</question_label>\nTính x<5 và y>2 rồi chọn."
    )
    result = merge_chunk_xmls([x1, x2], raw_chunk_inputs=[raw1, raw2])
    assert "Cho a < b và c > d." in result["original_text"]
    assert "Tính x<5 và y>2 rồi chọn." in result["original_text"]
    assert "A. đúngB. sai" not in result["original_text"]
    assert "đúng\nB. sai" in result["original_text"]
    assert _fidelity_mismatches(result) == 0


def test_text_before_overlap_is_preserved():
    prev = "Đoạn văn về khí hậu miền Bắc có nhiều mưa vào mùa hè và lạnh vào mùa đông.\n3. Chọn đáp án đúng nhất cho câu này"
    curr = "Ghi chú bổ sung quan trọng về địa lý vùng núi phía Tây.\n3. Chọn đáp án đúng nhất cho câu này\n4. Câu tiếp theo nói về sông ngòi"
    canonical, _, reports = build_canonical_source([
        ChunkInput(index=0, original_text=prev, parsed_xml=""),
        ChunkInput(index=1, original_text=curr, parsed_xml=""),
    ])
    assert canonical.count("Ghi chú bổ sung quan trọng về địa lý vùng núi phía Tây") == 1
    assert canonical.endswith("4. Câu tiếp theo nói về sông ngòi")
    assert not any(r.get("status") == "FIDELITY_MISMATCH" for r in reports)


def test_repeated_header_prefix_is_not_duplicated():
    prev = "Trang 1 Môn Ngữ văn đề thi chính thức.\nCâu 5. Đọc đoạn văn sau và trả lời các câu hỏi bên dưới"
    curr = "Trang 1 Môn Ngữ văn đề thi chính thức.\nCâu 5. Đọc đoạn văn sau và trả lời các câu hỏi bên dưới\nCâu 6. Nội dung mới hoàn toàn"
    canonical, _, _ = build_canonical_source([
        ChunkInput(index=0, original_text=prev, parsed_xml=""),
        ChunkInput(index=1, original_text=curr, parsed_xml=""),
    ])
    assert canonical.count("Trang 1 Môn Ngữ văn") == 1
    assert canonical.count("Câu 5. Đọc đoạn văn") == 1
    assert canonical.endswith("Câu 6. Nội dung mới hoàn toàn")


def test_single_shared_question_number_is_not_an_overlap():
    prev = "Đoạn văn kết thúc bằng một câu dài không liên quan đến đề thi nào khác.\n3."
    curr = "3. Chọn đáp án đúng nhất cho câu hỏi tiếp theo trong đề thi này"
    assert _align_clean(prev, curr)[4] == 0.0


def test_every_chunk_character_maps_to_matching_canonical_character():
    raw1 = "Cho a < b. ## Câu 2\nA. đúng nhất trong các đáp án"
    raw2 = "## Câu 2\nA. đúng nhất trong các đáp án\nB. sai hoàn toàn với đề bài"
    canonical, _, reports = build_canonical_source([
        ChunkInput(index=0, original_text=raw1, parsed_xml=""),
        ChunkInput(index=1, original_text=raw2, parsed_xml=""),
    ])
    assert not any(r.get("status") == "FIDELITY_MISMATCH" for r in reports)
    assert "B. sai hoàn toàn với đề bài" in canonical
