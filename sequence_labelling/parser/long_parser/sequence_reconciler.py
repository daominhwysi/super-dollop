import json
import re
from dataclasses import asdict
from typing import List, Dict, Any, Optional

class DocumentStateStack:
    """
    Formal stack-based state machine resolving nested document hierarchies
    (CHAPTER -> SECTION -> STIMULUS -> QUESTION) with validation rules.
    """
    def __init__(self, doc_id: str = "doc_global"):
        self.doc_id = doc_id
        self.stack: List[Dict[str, Any]] = []
        self.completed_groups: List[Dict[str, Any]] = []
        self.current_section = "global"

    def process_metadata_stream(self, page_metadata_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for meta in sorted(page_metadata_stream, key=lambda x: x.get("p", 0)):
            page_num = meta.get("p", 1)

            # Handle top boundary continuity
            if meta.get("head") in ("CONT_GROUP", "CONT_STIM", "CONT_THEORY") and self.stack:
                if "pages" in self.stack[-1]:
                    if page_num not in self.stack[-1]["pages"]:
                        self.stack[-1]["pages"].append(page_num)

            for event in meta.get("seq", []):
                if not isinstance(event, list) or not event:
                    continue
                event_type = event[0]

                if event_type == "SECTION_START":
                    self.current_section = str(event[1]) if len(event) > 1 else "global"

                elif event_type == "STIM_START":
                    stim_name = str(event[1]) if len(event) > 1 else "stim_1"
                    stim_id = f"{self.doc_id}:{self.current_section}:p{page_num}:{stim_name}"
                    group_node = {
                        "group_id": stim_id,
                        "type": "CONTEXT_QUESTION_GROUP",
                        "pages": [page_num],
                        "questions": []
                    }
                    self.stack.append(group_node)

                elif event_type == "THEORY_START":
                    theory_name = str(event[1]) if len(event) > 1 else "theory_1"
                    theory_id = f"{self.doc_id}:{self.current_section}:p{page_num}:{theory_name}"
                    group_node = {
                        "group_id": theory_id,
                        "type": "LECTURE_THEORY",
                        "pages": [page_num],
                        "questions": []
                    }
                    self.stack.append(group_node)

                elif event_type == "Q_START":
                    q_num = str(event[1]) if len(event) > 1 else "1"
                    q_id = f"{self.doc_id}:{self.current_section}:p{page_num}:q{q_num}"
                    if self.stack:
                        self.stack[-1]["questions"].append(q_id)

            # Handle bottom boundary clean closure
            if meta.get("tail") == "CLEAN" and self.stack:
                closed_group = self.stack.pop()
                closed_group["status"] = "CLOSED"
                self.completed_groups.append(closed_group)

        # Flush residual open stack
        while self.stack:
            group = self.stack.pop()
            group["status"] = "PARTIAL_CLOSED"
            self.completed_groups.append(group)

        return self.completed_groups


def extract_metadata_headers_from_markdown(full_markdown_text: str) -> List[Dict[str, Any]]:
    """
    Extracts <page_metadata> or <|page_metadata|> JSON blocks embedded in OCR markdown text.
    """
    pattern = re.compile(r"(?:<\|page_metadata\|>|<page_metadata>)\s*(\{.*?\})\s*(?:<\|end_metadata\|>|</page_metadata>)", re.DOTALL)
    headers = []

    for match in pattern.finditer(full_markdown_text):
        json_str = match.group(1).strip()
        try:
            headers.append(json.loads(json_str))
        except Exception:
            pass

    return headers


def merge_chunk_xmls(
    chunk_xml_contents: List[str],
    raw_chunk_inputs: Optional[List[str]] = None,
    *,
    allow_parser_text_as_source: bool = False,
) -> Dict[str, Any]:
    """
    Stitches multiple per-chunk XML sequence-annotated outputs into a single,
    unified 100% annotated XML document using the Source-Grounded Parsed Chunk Merger algorithm.
    """
    from sequence_labelling.parser.long_parser.source_merger.types import ChunkInput
    from sequence_labelling.parser.long_parser.source_merger.merger import merge_chunks
    from sequence_labelling.parser.long_parser.source_merger.serializer_validator import remove_annotation_tags

    if raw_chunk_inputs is None and not allow_parser_text_as_source:
        raise ValueError(
            "raw_chunk_inputs are required for source-grounded merging; "
            "set allow_parser_text_as_source=True only for legacy diagnostics"
        )
    if raw_chunk_inputs is not None and len(raw_chunk_inputs) != len(chunk_xml_contents):
        raise ValueError("chunk_xml_contents and raw_chunk_inputs must have equal length")

    chunks = []
    for idx, xml in enumerate(chunk_xml_contents):
        if raw_chunk_inputs is not None:
            orig_text = raw_chunk_inputs[idx]
        else:
            orig_text = remove_annotation_tags(xml)
        chunks.append(ChunkInput(index=idx, original_text=orig_text, parsed_xml=xml))

    result = merge_chunks(chunks)

    q_nums = []
    for q in result.structured_questions:
        q_str = str(q.get("question_number", "")).strip()
        if q_str.isdigit():
            q_nums.append(int(q_str))

    all_questions_sorted = sorted(list(set(q_nums)))

    total_raw_questions = sum(len(re.findall(r"<question_label>", xml)) for xml in chunk_xml_contents)
    deduplicated_count = max(0, total_raw_questions - len(result.structured_questions))

    return {
        "original_text": result.original_text,
        "merged_xml": result.merged_xml,
        "total_questions": len(result.structured_questions),
        "deduplicated_count": deduplicated_count,
        "chunks_processed": len(chunk_xml_contents),
        "question_range": f"{all_questions_sorted[0]} - {all_questions_sorted[-1]}" if all_questions_sorted else "N/A",
        "structured_questions": result.structured_questions,
        "structured_stimuli": result.structured_stimuli,
        "annotations": [asdict(span) for span in result.annotations],
        "diagnostics": asdict(result.diagnostics),
        "source_authority": (
            "original_chunk_text"
            if raw_chunk_inputs is not None
            else "parser_fallback"
        ),
    }


def reconcile_parser_chunk_results(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from sequence_labelling.parser.long_parser.source_merger.types import (
            ChunkInput,
            PageOffsetRange,
        )
        from sequence_labelling.parser.long_parser.source_merger.merger import merge_chunks
    except ImportError:
        from sequence_labelling.parser.long_parser.old.source_merger.types import (
            ChunkInput,
            PageOffsetRange,
        )
        from sequence_labelling.parser.long_parser.old.source_merger.merger import merge_chunks

    ordered = sorted(chunk_results, key=lambda item: int(item["chunk_index"]))
    expected = list(range(len(ordered)))
    actual = [int(item["chunk_index"]) for item in ordered]
    if actual != expected:
        raise ValueError(f"Chunk indexes must be contiguous and zero-based: {actual}")

    chunks = []
    for item in ordered:
        ranges = [PageOffsetRange(**value) for value in item.get("page_offset_ranges", [])]
        chunks.append(ChunkInput(
            index=int(item["chunk_index"]),
            original_text=item["raw_chunk_text"],
            parsed_xml=item.get("raw_xml", ""),
            page_ids=item.get("page_ids") or [],
            overlap_page_ids=item.get("overlap_page_ids") or [],
            page_offset_ranges=ranges or None,
        ))

    result = merge_chunks(chunks)
    return {
        "original_text": result.original_text,
        "merged_xml": result.merged_xml,
        "annotations": [asdict(span) for span in result.annotations],
        "structured_questions": result.structured_questions,
        "structured_stimuli": result.structured_stimuli,
        "diagnostics": asdict(result.diagnostics),
    }

