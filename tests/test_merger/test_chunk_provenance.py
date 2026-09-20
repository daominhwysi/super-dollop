from sequence_labelling.parser.long_parser.greedy_chunker import (
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.sequence_reconciler import (
    reconcile_parser_chunk_results,
)


def test_chunk_plan_preserves_page_offsets_and_overlap_provenance():
    pages = [
        {
            "p": 1,
            "text": "Page one.",
            "estimated_tokens": 10,
            "tail": "CLEAN",
            "global_start": 0,
            "global_end": 9,
        },
        {
            "p": 2,
            "text": "Page two.",
            "estimated_tokens": 10,
            "tail": "CLEAN",
            "global_start": 11,
            "global_end": 20,
        },
        {
            "p": 3,
            "text": "Page three.",
            "estimated_tokens": 10,
            "tail": "CLEAN",
            "global_start": 22,
            "global_end": 33,
        },
    ]

    chunks = greedy_oversize_chunker(
        pages,
        target_tokens=20,
        max_tokens=30,
        overlap_pages=1,
    )
    plans = [build_chunk_plan(chunk, index) for index, chunk in enumerate(chunks)]

    assert plans[0]["page_ids"] == [1, 2]
    assert plans[1]["page_ids"] == [2, 3]
    assert plans[1]["overlap_page_ids"] == [2]
    for plan in plans:
        for page_range in plan["page_offset_ranges"]:
            source_slice = plan["raw_chunk_text"][
                page_range["local_start"]:page_range["local_end"]
            ]
            assert source_slice.startswith("Page ")


def test_reconcile_parser_results_uses_page_provenance():
    results = [
        {
            "chunk_index": 0,
            "raw_chunk_text": "Page one.\n\nPage two.",
            "raw_xml": "<stem>Page one.</stem>\n\n<stem>Page two.</stem>",
            "page_ids": [1, 2],
            "overlap_page_ids": [],
            "page_offset_ranges": [
                {"page_id": 1, "local_start": 0, "local_end": 9, "global_start": 0, "global_end": 9, "is_overlap": False},
                {"page_id": 2, "local_start": 11, "local_end": 20, "global_start": 11, "global_end": 20, "is_overlap": False},
            ],
        },
        {
            "chunk_index": 1,
            "raw_chunk_text": "Page two.\n\nPage three.",
            "raw_xml": "<stem>Page two.</stem>\n\n<stem>Page three.</stem>",
            "page_ids": [2, 3],
            "overlap_page_ids": [2],
            "page_offset_ranges": [
                {"page_id": 2, "local_start": 0, "local_end": 9, "global_start": 11, "global_end": 20, "is_overlap": True},
                {"page_id": 3, "local_start": 11, "local_end": 22, "global_start": 22, "global_end": 33, "is_overlap": False},
            ],
        },
    ]

    merged = reconcile_parser_chunk_results(results)

    assert merged["original_text"] == "Page one.\n\nPage two.\n\nPage three."
    assert merged["diagnostics"]["overlaps"][0]["status"] == "PROVENANCE"
    assert merged["diagnostics"]["validation"]["source_fidelity_ok"] is True
