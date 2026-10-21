"""Unit tests for DocumentCartographer and Topological Manifest."""

import json
from typing import Any, Dict, List
import pytest

from sequence_labelling.parser.long_parser.document_cartographer import (
    CartographyManifest,
    CartographyObject,
    DocumentCartographer,
)


def test_cartography_object_serialization():
    obj = CartographyObject(
        type="question",
        id="3",
        page_span=(1, 2),
        label="Câu 3",
        stimulus_id="stim_1",
    )
    d = obj.to_dict()
    assert d["type"] == "question"
    assert d["id"] == "3"
    assert d["page_span"] == [1, 2]
    assert d["stimulus_id"] == "stim_1"


def test_clean_cut_seams_no_overlap():
    """When all questions are confined to single pages, all seams should be clean."""
    objects = [
        CartographyObject(type="question", id="1", page_span=(1, 1)),
        CartographyObject(type="question", id="2", page_span=(2, 2)),
        CartographyObject(type="question", id="3", page_span=(3, 3)),
    ]
    seams = DocumentCartographer.compute_clean_cut_seams(
        total_pages=3,
        objects=objects,
        stimuli={},
    )
    assert seams == [1, 2]


def test_clean_cut_seams_with_cross_page_question():
    """Q(3) [1, 2] straddles seam 1 -> seam 1 must be BLOCKED, seam 2 is clean."""
    objects = [
        CartographyObject(type="question", id="1", page_span=(1, 1)),
        CartographyObject(type="question", id="2", page_span=(1, 1)),
        CartographyObject(type="question", id="3", page_span=(1, 2)),  # Straddles seam 1
        CartographyObject(type="question", id="4", page_span=(2, 2)),
        CartographyObject(type="question", id="5", page_span=(3, 3)),
    ]
    seams = DocumentCartographer.compute_clean_cut_seams(
        total_pages=3,
        objects=objects,
        stimuli={},
    )
    assert 1 not in seams
    assert 2 in seams


def test_clean_cut_seams_with_stimulus_group():
    """
    Stimulus on [2, 3] serving Q(15) on p2 and Q(16) on p4.
    The entire stimulus group spans pages 2 through 4!
    Seams 2 and 3 must be blocked; seam 1 and seam 4 must be clean.
    """
    stim = CartographyObject(
        type="stimulus",
        id="stim_1",
        page_span=(2, 3),
        serves_questions=["15", "16"],
    )
    objects = [
        CartographyObject(type="question", id="1", page_span=(1, 1)),
        stim,
        CartographyObject(type="question", id="15", page_span=(2, 2), stimulus_id="stim_1"),
        CartographyObject(type="question", id="16", page_span=(3, 4), stimulus_id="stim_1"),
        CartographyObject(type="question", id="17", page_span=(5, 5)),
    ]
    seams = DocumentCartographer.compute_clean_cut_seams(
        total_pages=5,
        objects=objects,
        stimuli={"stim_1": stim},
    )
    assert 1 in seams
    assert 2 not in seams
    assert 3 not in seams
    assert 4 in seams


def test_clean_cut_seams_with_multi_page_section():
    """
    A section spans [1, 5]. Questions are all single-page.
    Sections should NOT block cuts between questions.
    All seams [1, 2, 3, 4] must be clean.
    """
    objects = [
        CartographyObject(type="section", id="sec_1", page_span=(1, 5)),
        CartographyObject(type="question", id="1", page_span=(1, 1)),
        CartographyObject(type="question", id="2", page_span=(2, 2)),
        CartographyObject(type="question", id="3", page_span=(3, 3)),
        CartographyObject(type="question", id="4", page_span=(4, 4)),
        CartographyObject(type="question", id="5", page_span=(5, 5)),
    ]
    seams = DocumentCartographer.compute_clean_cut_seams(
        total_pages=5,
        objects=objects,
        stimuli={},
    )
    assert seams == [1, 2, 3, 4]


def test_cartography_manifest_from_dict():
    """Verifies that CartographyManifest.from_dict properly reconstructs manifest and recalculates clean seams."""
    d = {
        "doc_id": "test_doc",
        "total_pages": 3,
        "subject": "Toán",
        "objects": [
            {"type": "question", "id": "1", "page_span": [1, 1]},
            {"type": "question", "id": "2", "page_span": [2, 2]},
            {"type": "question", "id": "3", "page_span": [3, 3]},
        ],
    }
    manifest = CartographyManifest.from_dict(d)
    assert manifest.doc_id == "test_doc"
    assert manifest.total_pages == 3
    assert len(manifest.questions) == 3
    assert manifest.clean_cut_seams == [1, 2]
    assert manifest.is_seam_clean(1) is True
    assert manifest.is_seam_clean(2) is True



def test_json_stream_parsing_variants():
    cartographer = DocumentCartographer()

    # 1. Plain JSON string
    raw_json = json.dumps([{"type": "question", "id": "1", "page_span": [1, 1]}])
    parsed = cartographer._parse_json_stream(raw_json)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "1"

    # 2. Markdown fenced JSON
    fenced_json = f"```json\n{raw_json}\n```"
    parsed_fenced = cartographer._parse_json_stream(fenced_json)
    assert len(parsed_fenced) == 1
    assert parsed_fenced[0]["id"] == "1"

    # 3. Conversational wrapper text
    wrapper_text = f"Here is the topological manifest:\n\n{raw_json}\n\nHope this helps!"
    parsed_wrapped = cartographer._parse_json_stream(wrapper_text)
    assert len(parsed_wrapped) == 1
    assert parsed_wrapped[0]["id"] == "1"


def test_deterministic_fallback_scan():
    cartographer = DocumentCartographer()
    pages = [
        {"p": 1, "text": "SỞ GD&ĐT HÀ NỘI\n# PHẦN I: TRẮC NGHIỆM\nCâu 1. Tính tích phân...\nCâu 2. Cho hình chóp..."},
        {"p": 2, "text": "Câu 3. Tìm nguyên hàm...\n## PHẦN II: TỰ LUẬN\nCâu 4. Giải phương trình..."},
    ]
    fallback_objs = cartographer._deterministic_fallback_scan(pages)
    q_ids = [o["id"] for o in fallback_objs if o["type"] == "question"]
    sec_labels = [o["label"] for o in fallback_objs if o["type"] == "section"]

    assert "1" in q_ids
    assert "2" in q_ids
    assert "3" in q_ids
    assert "4" in q_ids
    assert any("TRẮC NGHIỆM" in s for s in sec_labels)


def test_survey_document_end_to_end_with_mock():
    mock_llm_response = json.dumps([
        {"type": "section", "id": "sec_1", "label": "PHẦN I. TRẮC NGHIỆM", "page_span": [1, 2]},
        {"type": "question", "id": "1", "page_span": [1, 1]},
        {"type": "question", "id": "2", "page_span": [1, 2]},
        {
            "type": "stimulus",
            "id": "stim_1",
            "page_span": [2, 3],
            "start_snippet": "Read the following passage",
            "end_snippet": "for marine life.",
            "serves_questions": ["3", "4"],
        },
        {"type": "question", "id": "3", "page_span": [2, 2], "stimulus_id": "stim_1"},
        {"type": "question", "id": "4", "page_span": [3, 3], "stimulus_id": "stim_1"},
        {"type": "question", "id": "5", "page_span": [4, 4]},
    ])

    def mock_complete(**kwargs):
        return mock_llm_response

    cartographer = DocumentCartographer()
    mock_ocr = (
        "<page p='1'>Content page 1</page>\n"
        "<page p='2'>Content page 2</page>\n"
        "<page p='3'>Content page 3</page>\n"
        "<page p='4'>Content page 4</page>"
    )

    manifest = cartographer.survey_document(
        ocr_markdown=mock_ocr,
        doc_id="test_exam_01",
        subject="Tiếng Anh",
        completion_fn=mock_complete,
    )

    assert manifest.total_pages == 4
    assert len(manifest.questions) == 5
    assert len(manifest.stimuli) == 1
    assert "stim_1" in manifest.stimuli
    assert manifest.stimuli["stim_1"].serves_questions == ["3", "4"]

    # Check clean seams:
    # Seam 1 is blocked by Q(2) [1, 2]
    # Seam 2 is blocked by stim_1 [2, 3]
    # Seam 3 is clean!
    assert 1 not in manifest.clean_cut_seams
    assert 2 not in manifest.clean_cut_seams
    assert 3 in manifest.clean_cut_seams
    assert manifest.is_seam_clean(3) is True
    assert manifest.is_seam_clean(1) is False


def test_greedy_chunker_with_cartography_snapping():
    """
    Verify that greedy_oversize_chunker uses CartographyManifest to prevent
    severing cross-page questions or stimulus groups.
    """
    from sequence_labelling.parser.long_parser.greedy_chunker import greedy_oversize_chunker

    # 4 pages, each 1000 tokens
    pages = [
        {"p": 1, "text": "Page 1", "estimated_tokens": 1000},
        {"p": 2, "text": "Page 2", "estimated_tokens": 1000},
        {"p": 3, "text": "Page 3", "estimated_tokens": 1000},
        {"p": 4, "text": "Page 4", "estimated_tokens": 1000},
    ]

    # Create a manifest where stim_1 spans [2, 3] and serves questions on p2 and p3.
    # Clean seams: only seam 3 is clean (between p3 and p4).
    # Seam 1: clean (p1 to p2). Seam 2: BLOCKED (inside stim_1 [2, 3]).
    manifest = CartographyManifest(
        doc_id="doc_snap_test",
        total_pages=4,
        objects=[
            CartographyObject(type="question", id="1", page_span=(1, 1)),
            CartographyObject(type="stimulus", id="stim_1", page_span=(2, 3), serves_questions=["2", "3"]),
            CartographyObject(type="question", id="2", page_span=(2, 2), stimulus_id="stim_1"),
            CartographyObject(type="question", id="3", page_span=(3, 3), stimulus_id="stim_1"),
            CartographyObject(type="question", id="4", page_span=(4, 4)),
        ],
        stimuli={
            "stim_1": CartographyObject(type="stimulus", id="stim_1", page_span=(2, 3), serves_questions=["2", "3"]),
        },
        questions={},
        sections=[],
        clean_cut_seams=[1, 3],  # Seam 2 is blocked!
    )

    # If target_tokens = 1500:
    # After p1 (1000 tokens) -> < target (1500) -> continue
    # After p2 (2000 tokens) -> >= target (1500). BUT seam 2 is NOT in clean_cut_seams!
    # Chunker is FORBIDDEN to cut after p2! It MUST keep p3 in the same chunk.
    # After p3 (3000 tokens) -> >= target (1500). Seam 3 IS clean! Cuts after p3.
    # Chunk 0 has [p1, p2, p3], Chunk 1 has [p4].
    chunks = greedy_oversize_chunker(
        pages,
        target_tokens=1500,
        max_tokens=4000,
        cartography=manifest,
    )

    assert len(chunks) == 2
    chunk_0_pages = [p["p"] for p in chunks[0]]
    chunk_1_pages = [p["p"] for p in chunks[1]]

    assert chunk_0_pages == [1, 2, 3]
    # With default overlap_pages = 1, page 3 is carried over as overlap context
    assert chunk_1_pages == [3, 4]
    assert chunks[1][0].get("is_overlap_context") is True
    assert chunks[1][1].get("is_overlap_context") is not True

    # With explicit overlap_pages = 0
    chunks_no_overlap = greedy_oversize_chunker(
        pages,
        target_tokens=1500,
        max_tokens=4000,
        overlap_pages=0,
        cartography=manifest,
    )
    assert [p["p"] for p in chunks_no_overlap[1]] == [4]
    # Stimulus on [2, 3] remained 100% unified in Chunk 0!


def test_greedy_chunker_unpaged_document_fallback():
    """
    Verify that an unpaged single-page document exceeding max_tokens
    is partitioned with 750-token fallback overlap.
    """
    from sequence_labelling.parser.long_parser.greedy_chunker import greedy_oversize_chunker

    # 1 giant page with 15,000 tokens (150 paragraphs of 100 words each)
    paras = [f"Paragraph {i}: " + "sample word " * 45 for i in range(150)]
    giant_text = "\n\n".join(paras)
    pages = [
        {
            "p": 1,
            "text": giant_text,
            "estimated_tokens": len(giant_text.split()),
            "global_start": 0,
            "global_end": len(giant_text),
        }
    ]

    chunks = greedy_oversize_chunker(
        pages,
        target_tokens=7000,
        max_tokens=10000,
        overlap_pages=1,
        overlap_fallback_tokens=750,
    )

    assert len(chunks) > 1
    # Check that Chunk 1 has overlap from Chunk 0
    chunk_1 = chunks[1]
    has_overlap = any(p.get("is_overlap_context") for p in chunk_1)
    assert has_overlap is True


