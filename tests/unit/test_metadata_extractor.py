"""
test_metadata_extractor.py — Unit Tests for Document Metadata Extraction.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sequence_labelling.metadata.schema import ExamMetadata
from sequence_labelling.metadata.extractor import (
    clean_filename_for_context,
    extract_window_context,
    heuristic_metadata_fallback,
    extract_document_metadata,
    truncate_text_to_tokens,
)


def test_clean_filename_for_context():
    # Hashed slug
    fn1 = "bo-e-kiem-tra-cuoi-ky-2-toan-10-ctst-nam-hoc-2024-2025--a7b59d6b9788.md"
    assert clean_filename_for_context(fn1) == "bo e kiem tra cuoi ky 2 toan 10 ctst nam hoc 2024 2025"

    # doc_001_ prefix with Vietnamese text
    fn2 = "doc_001_Đề giữa học kỳ 2 Toán 10 năm 2023 – 2024 trường THPT Đô Lương 3 – Nghệ An.md"
    assert clean_filename_for_context(fn2) == "Đề giữa học kỳ 2 Toán 10 năm 2023 – 2024 trường THPT Đô Lương 3 – Nghệ An"


def test_extract_window_context_page_selection():
    # Build a simulated 10-page document
    pages_content = []
    for i in range(1, 11):
        pages_content.append(f"<page>\nPage {i} content text here.\n<figure id=\"fig_{i}\" />\n<page_metadata>{{\"p\": {i}}}</page_metadata>\n</page>")
    full_markdown = "\n".join(pages_content)

    context, total_pages, selected_indices = extract_window_context(full_markdown, max_tokens_per_page=750)
    assert total_pages == 10
    # For 10 pages: mid = 5. Indices should be [0, 4, 5] (Pages 1, 5, 6)
    assert selected_indices == [0, 4, 5]
    assert "--- TRANG 1/10 ---" in context
    assert "--- TRANG 5/10 ---" in context
    assert "--- TRANG 6/10 ---" in context
    # Figures and page_metadata should be stripped
    assert "<figure" not in context
    assert "<page_metadata>" not in context


def test_extract_window_context_small_document():
    # 2-page document
    md = "<page>Page 1</page>\n<page>Page 2</page>"
    context, total, selected = extract_window_context(md, max_tokens_per_page=750)
    assert total == 2
    assert selected == [0, 1]


def test_truncate_text_to_tokens():
    long_text = "Từ " * 1000  # 1000 tokens
    truncated = truncate_text_to_tokens(long_text, max_tokens=750)
    assert "[truncated at 750 tokens]" in truncated


def test_heuristic_metadata_fallback():
    fn = "bo-e-kiem-tra-cuoi-ky-2-toan-10-ctst-nam-hoc-2024-2025--a7b59d6b9788.md"
    meta = heuristic_metadata_fallback(fn)
    assert meta.subject == "Toán"
    assert meta.grade == "10"
    assert meta.examination == "Kiểm tra cuối học kỳ 2"
    assert meta.academic_year == "2024-2025"
    assert meta.curriculum == "GDPT 2018"
    assert meta.authority_or_school is None


def test_extract_document_metadata_with_mocked_llm():
    mock_reply = json.dumps({
        "subject": "Toán",
        "examination": "Kiểm tra cuối học kỳ 2",
        "grade": "10",
        "academic_year": "2024-2025",
        "curriculum": "GDPT 2018",
        "authority_or_school": None,
        "raw_title": "BỘ ĐỀ KIỂM TRA CUỐI KỲ 2 - KHỐI 10"
    })

    with patch("sequence_labelling.metadata.extractor.chat", return_value=mock_reply):
        meta = extract_document_metadata(
            ocr_markdown="<page>Header</page>",
            filename="sample.md",
            doc_id="sample_doc",
        )
        assert meta.subject == "Toán"
        assert meta.grade == "10"
        assert meta.examination == "Kiểm tra cuối học kỳ 2"
        assert meta.academic_year == "2024-2025"
        assert meta.curriculum == "GDPT 2018"
        assert meta.authority_or_school is None
        assert meta.raw_title == "BỘ ĐỀ KIỂM TRA CUỐI KỲ 2 - KHỐI 10"
        assert meta.metadata_extractor.method == "llm"


def test_extract_document_metadata_with_none_for_unknown():
    mock_reply = json.dumps({
        "subject": "Tiếng Anh",
        "examination": None,
        "grade": None,
        "academic_year": None,
        "curriculum": None,
        "authority_or_school": None,
        "raw_title": None
    })

    with patch("sequence_labelling.metadata.extractor.chat", return_value=mock_reply):
        meta = extract_document_metadata(
            ocr_markdown="<page>English Exam</page>",
            filename="unknown.md",
        )
        assert meta.subject == "Tiếng Anh"
        assert meta.grade is None
        assert meta.examination is None
        assert meta.academic_year is None
        assert meta.curriculum is None
        assert meta.authority_or_school is None
