"""
Unit tests for DigitalChunkExtractor and expand_chunks_in_markdown.
"""

from pathlib import Path
import fitz
import pytest

from tools.experiments.chunk_extractor import (
    ChunkItem,
    DigitalChunkExtractor,
    expand_chunks_in_markdown,
)


def test_is_valid_chunk_text():
    extractor = DigitalChunkExtractor(min_chars=5, min_words=2)

    # Valid chunks
    assert extractor.is_valid_chunk_text("Câu 1: Cho hàm số")
    assert extractor.is_valid_chunk_text("TRƯỜNG THPT CHUYÊN BẮC NINH")
    assert extractor.is_valid_chunk_text("Thí sinh chọn một phương án")

    # Invalid chunks: too short
    assert not extractor.is_valid_chunk_text("A.")
    assert not extractor.is_valid_chunk_text("1.")
    assert not extractor.is_valid_chunk_text(".")
    assert not extractor.is_valid_chunk_text("là")

    # Invalid chunks: math/noise characters
    assert not extractor.is_valid_chunk_text("!\"#")
    assert not extractor.is_valid_chunk_text("%&'()")

    # Invalid chunks: shared footer
    assert not extractor.is_valid_chunk_text("Tài liệu chia sẻ miễn phí tại: TailieuOnthi.org")
    assert not extractor.is_valid_chunk_text("Fanpage: Hậu học tập")


def test_expand_chunks_in_markdown():
    chunks = [
        ChunkItem(id=1, chunk_key="CHUNK_1", text="TRƯỜNG THPT CHUYÊN BẮC NINH", bbox=(0, 0, 10, 10)),
        ChunkItem(id=2, chunk_key="CHUNK_2", text="Câu 1: Cho hàm số", bbox=(0, 0, 10, 10)),
        ChunkItem(id=3, chunk_key="CHUNK_3", text="Giá trị bằng", bbox=(0, 0, 10, 10)),
    ]

    raw_markdown = (
        "# [CHUNK_1]\n\n"
        "**[CHUNK_2]** $y = \\frac{2x - 1}{x + 1}$. [CHUNK_3] 5.\n\n"
        "Also supports mustache: {{CHUNK_2}}"
    )

    expanded, count = expand_chunks_in_markdown(raw_markdown, chunks)
    assert count == 4
    assert "# TRƯỜNG THPT CHUYÊN BẮC NINH" in expanded
    assert "**Câu 1: Cho hàm số** $y = \\frac{2x - 1}{x + 1}$. Giá trị bằng 5." in expanded
    assert "Also supports mustache: Câu 1: Cho hàm số" in expanded


def test_chunk_extractor_on_real_pdf():
    pdf_path = Path("data/sequence_labelling_input_data_raw/exam-001--8982d5d00fcd.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    assert DigitalChunkExtractor.is_digital_pdf(pdf_path)

    doc = fitz.open(pdf_path)
    extractor = DigitalChunkExtractor(min_chars=5, min_words=2)
    chunks = extractor.extract_page_chunks(doc[0], 1)

    assert len(chunks) >= 10
    # Verify no noise or 1-word tiny chunks
    for c in chunks:
        assert len(c.text) >= 5
        assert c.chunk_key.startswith("CHUNK_")
        assert not c.text.startswith("TailieuOnthi")
