"""Unit tests for OCRDocument and robust page parsing."""

import json
from pathlib import Path
import pytest

from sequence_labelling.annotator.ocr_document import (
    OCRDocument,
    OCRPage,
    PAGE_SEPARATOR,
    split_raw_markdown_pages,
    strip_page_markup,
    extract_page_metadata_stream,
)


def test_standard_document_parsing():
    raw_md = """<pages>
<page>
# ĐỀ THI TOÁN HỌC

Câu 1: Cho hàm số $y=f(x)$.

<page_metadata>
{"p": 1, "w": 595, "h": 842, "head": "CLEAN", "tail": "CLEAN"}
</page_metadata>
</page>

<page>
Câu 2: Tính tích phân $\\int_0^1 x dx$.
A. 1/2
B. 1

<page_metadata>
{"p": 2, "w": 595, "h": 842, "head": "CLEAN", "tail": "CLEAN"}
</page_metadata>
</page>
</pages>"""

    doc = OCRDocument.from_markdown(raw_md, doc_id="exam_test_01")
    assert doc.doc_id == "exam_test_01"
    assert doc.total_pages == 2
    assert doc.page_numbers == [1, 2]

    # Verify Page 1
    p1 = doc.get_page(1)
    assert p1 is not None
    assert "# ĐỀ THI TOÁN HỌC" in p1.text
    assert "Câu 1: Cho hàm số" in p1.text
    assert "<page_metadata>" not in p1.text
    assert "</page>" not in p1.text
    assert p1.meta["p"] == 1
    assert p1.meta["w"] == 595

    # Verify Page 2
    p2 = doc.get_page(2)
    assert p2 is not None
    assert "Câu 2: Tính tích phân" in p2.text
    assert p2.meta["p"] == 2

    # Check offsets alignment with PAGE_SEPARATOR
    assert doc.clean_text[p1.global_start : p1.global_end] == p1.text
    assert doc.clean_text[p2.global_start : p2.global_end] == p2.text
    assert p2.global_start == p1.global_end + len(PAGE_SEPARATOR)


def test_mismatched_unclosed_page_tags():
    """Simulates real-world OCR defect where </page> tags are omitted between pages."""
    raw_md = """<pages>
<page>
Page 1 Content
<page_metadata>{"p": 1}</page_metadata>
<page>
Page 2 Content without prior closing page tag
<page_metadata>{"p": 2}</page_metadata>
<page>
Page 3 Content also unclosed
</page>
</pages>"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 3
    assert doc.page_numbers == [1, 2, 3]

    p1 = doc.get_page(1)
    p2 = doc.get_page(2)
    p3 = doc.get_page(3)

    assert p1.text == "Page 1 Content"
    assert p2.text == "Page 2 Content without prior closing page tag"
    assert p3.text == "Page 3 Content also unclosed"


def test_extra_and_duplicate_closing_tags():
    """Simulates real-world OCR defect where LLM emits duplicate </page> tags."""
    raw_md = """<pages>
<page>
Page 1 Content
<page_metadata>{"p": 1}</page_metadata>
</page>
</page>

<page>
Page 2 Content
<page_metadata>{"p": 2}</page_metadata>
</page>
</pages>"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 2
    assert doc.get_page(1).text == "Page 1 Content"
    assert doc.get_page(2).text == "Page 2 Content"


def test_unclosed_page_metadata_blocks():
    """Simulates real-world OCR defect where <page_metadata> is terminated directly by </page>."""
    raw_md = """<page>
Question 1: Hello
<page_metadata>
{"p": 1, "head": "CLEAN"}
</page>

<page>
Question 2: World
<page_metadata>
{"p": 2, "head": "CLEAN"}
<page>
Question 3: End of exam
</page>"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 3

    p1 = doc.get_page(1)
    assert p1.text == "Question 1: Hello"
    assert p1.meta.get("p") == 1
    assert p1.meta.get("head") == "CLEAN"

    p2 = doc.get_page(2)
    assert p2.text == "Question 2: World"
    assert p2.meta.get("p") == 2

    p3 = doc.get_page(3)
    assert p3.text == "Question 3: End of exam"


def test_preamble_outside_page_tags():
    """Verifies that preamble text outside <page> tags is preserved on page 1."""
    raw_md = """# HEADER DOCUMENT TITLE OUTSIDE PAGES

<pages>
<page>
Page 1 content here.
<page_metadata>{"p": 1}</page_metadata>
</page>
</pages>"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 1
    p1 = doc.get_page(1)
    assert "# HEADER DOCUMENT TITLE OUTSIDE PAGES" in p1.text
    assert "Page 1 content here." in p1.text


def test_malformed_page_openers():
    """Verifies handling of unclosed opening tag like `<page\\n`."""
    raw_md = """<page
Page 1 starts here
<page_metadata>{"p": 1}</page_metadata>
</page>

<page
<page>
Page 2 with double opening tag
<page_metadata>{"p": 2}</page_metadata>
</page>"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 2
    assert "Page 1 starts here" in doc.get_page(1).text
    assert "Page 2 with double opening tag" in doc.get_page(2).text


def test_legacy_comment_metadata():
    raw_md = """<!-- PAGE_METADATA_START
{"p": 1, "head": "CLEAN", "tail": "CLEAN"}
PAGE_METADATA_END -->
Page 1 Legacy Content

<!-- PAGE_METADATA_START
{"p": 2, "head": "CLEAN", "tail": "CLEAN"}
PAGE_METADATA_END -->
Page 2 Legacy Content"""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 2
    assert doc.get_page(1).text == "Page 1 Legacy Content"
    assert doc.get_page(1).meta["p"] == 1
    assert doc.get_page(2).text == "Page 2 Legacy Content"
    assert doc.get_page(2).meta["p"] == 2


def test_unpaged_markdown_fallback():
    raw_md = """# Unpaged Exam Document
Câu 1: Câu hỏi thứ nhất.
Câu 2: Câu hỏi thứ hai."""

    doc = OCRDocument.from_markdown(raw_md)
    assert doc.total_pages == 1
    p1 = doc.get_page(1)
    assert p1.p == 1
    assert p1.text == raw_md.strip()
    assert p1.meta["p"] == 1


def test_chunk_pages_compatibility():
    raw_md = """<page>
Page 1
<page_metadata>{"p": 1, "head": "CONT_GROUP", "tail": "OPEN_STIM"}</page_metadata>
</page>
<page>
Page 2
<page_metadata>{"p": 2, "head": "CONT_STIM", "tail": "CLEAN"}</page_metadata>
</page>"""

    doc = OCRDocument.from_markdown(raw_md)
    chunk_pages = doc.to_chunk_pages()

    assert len(chunk_pages) == 2
    assert chunk_pages[0]["p"] == 1
    assert chunk_pages[0]["text"] == "Page 1"
    assert chunk_pages[0]["head"] == "CONT_GROUP"
    assert chunk_pages[0]["tail"] == "OPEN_STIM"
    assert chunk_pages[0]["estimated_tokens"] >= 50
    assert chunk_pages[0]["global_start"] == 0
    assert chunk_pages[0]["global_end"] == len("Page 1")

    assert chunk_pages[1]["p"] == 2
    assert chunk_pages[1]["head"] == "CONT_STIM"
    assert chunk_pages[1]["global_start"] == len("Page 1") + len(PAGE_SEPARATOR)


def test_character_fidelity_verification():
    raw_md = """<page>
Câu 1: Nội dung câu 1.
A. Đáp án A
B. Đáp án B
<page_metadata>{"p": 1}</page_metadata>
</page>"""

    doc = OCRDocument.from_markdown(raw_md)

    # Exact annotation
    exact_xml = "<question_label>Câu 1:</question_label> <stem>Nội dung câu 1.</stem>\n<option_label>A.</option_label> <option_text>Đáp án A</option_text>\n<option_label>B.</option_label> <option_text>Đáp án B</option_text>"
    res_exact = doc.verify_character_fidelity(exact_xml)
    assert res_exact["is_exact_match"] is True

    # Hallucinated text in XML
    corrupt_xml = "<question_label>Câu 1:</question_label> <stem>Nội dung câu 1 đã bị biến đổi.</stem>"
    res_corrupt = doc.verify_character_fidelity(corrupt_xml)
    assert res_corrupt["is_exact_match"] is False
    assert "first_diff_position" in res_corrupt


def test_standard_markdown_roundtrip(tmp_path):
    raw_md = """<page>
Page 1 Content
<page_metadata>{"p": 1, "w": 600, "h": 800}</page_metadata>
</page>
<page>
Page 2 Content
<page_metadata>{"p": 2, "w": 600, "h": 800}</page_metadata>
</page>"""

    doc = OCRDocument.from_markdown(raw_md, doc_id="roundtrip")
    out_file = tmp_path / "roundtrip.md"
    doc.save(out_file)

    reloaded = OCRDocument.from_file(out_file)
    assert reloaded.total_pages == 2
    assert reloaded.get_page(1).text == "Page 1 Content"
    assert reloaded.get_page(2).text == "Page 2 Content"
    assert reloaded.get_page(1).meta["w"] == 600


def test_drop_in_helpers():
    raw_md = """<page>
Content
<page_metadata>{"p": 1}</page_metadata>
</page>"""

    chunk_pages = split_raw_markdown_pages(raw_md)
    assert len(chunk_pages) == 1
    assert chunk_pages[0]["p"] == 1
    assert chunk_pages[0]["text"] == "Content"

    clean = strip_page_markup(raw_md)
    assert clean == "Content"

    meta_stream = extract_page_metadata_stream(raw_md)
    assert len(meta_stream) == 1
    assert meta_stream[0]["p"] == 1


def test_real_world_problem_documents_from_dataset():
    """Runs tests on real-world files from dataset that previously caused crashes."""
    input_dir = Path("data/sequence_labelling_input_data")
    if not input_dir.exists():
        pytest.skip("Dataset directory not present")

    candidates = [
        "doc_356_Đề thi thử THPT Quốc gia 2019 môn Toán sở GD&ĐT Quảng Bình.md",
        "exam-411--4b2a065cb8e5.md",
        "exam-319--ced1add5b859.md",
        "doc_090_Đề giữa kỳ 1 Toán 10 năm 2024 – 2025 trường THPT Tây Thạnh – TP HCM.md",
    ]

    for fname in candidates:
        fpath = input_dir / fname
        if not fpath.exists():
            continue
        doc = OCRDocument.from_file(fpath)
        assert doc.total_pages > 0, f"Failed on {fname}: 0 pages extracted"
        assert len(doc.clean_text) > 100, f"Failed on {fname}: empty clean text"
        # Verify offsets
        for p in doc.pages:
            if not p.is_blank:
                assert doc.clean_text[p.global_start : p.global_end] == p.text
