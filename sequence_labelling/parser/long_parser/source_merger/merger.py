from typing import List
from sequence_labelling.parser.long_parser.source_merger.types import (
    ChunkInput,
    MergeResult,
    MergeDiagnostics,
    MergeInvariantError,
)
from sequence_labelling.parser.long_parser.source_merger.canonical_source import build_canonical_source
from sequence_labelling.parser.long_parser.source_merger.xml_lexer import lex_annotations
from sequence_labelling.parser.long_parser.source_merger.source_aligner import SourceAligner
from sequence_labelling.parser.long_parser.source_merger.span_projector import project_local_spans_to_global
from sequence_labelling.parser.long_parser.source_merger.reconciler import reconcile_candidates
from sequence_labelling.parser.long_parser.source_merger.serializer_validator import (
    serialize_annotations,
    validate_result,
    build_structured_questions,
    build_structured_stimuli,
)


def merge_chunks(chunks: List[ChunkInput]) -> MergeResult:
    """
    Source-Grounded Parsed Chunk Merger.
    Merges an ordered sequence of overlapping parsed chunks without duplicating or losing source text.
    Enforces invariant: remove_annotation_tags(merged_xml) == canonical_original_text.
    """
    if not chunks:
        return MergeResult(
            original_text="",
            merged_xml="",
            annotations=[],
            structured_questions=[],
            structured_stimuli={},
            diagnostics=MergeDiagnostics(),
        )

    # Stage A: Build canonical original source & piecewise mapping
    canonical_text, chunk_maps, overlap_reports = build_canonical_source(chunks)

    candidates = []
    parse_reports = []

    # Stages B, C, D: Lex, Align, Project
    for chunk, local_to_global in zip(chunks, chunk_maps):
        parsed_text, local_spans, lexical_report = lex_annotations(chunk.parsed_xml)
        aligner = SourceAligner(parsed_text, chunk.original_text)

        global_spans = project_local_spans_to_global(
            local_spans=local_spans,
            aligner=aligner,
            piecewise_map=local_to_global,
            chunk_input=chunk,
            canonical_text=canonical_text,
        )
        candidates.extend(global_spans)
        parse_reports.append({
            "chunk_index": chunk.index,
            "lexical": lexical_report,
            "alignment_kind": aligner.alignment_kind,
            "spans_projected": len(global_spans),
            "spans_rejected": max(0, len(local_spans) - len(global_spans)),
        })

    # Stage E: Reconcile candidate global spans
    selected, conflict_report = reconcile_candidates(candidates, canonical_text)

    # Stage F: Serialize & Validate
    #
    # A violated invariant means the annotations are untrustworthy, not that the
    # source text is. Degrading to unannotated canonical text preserves the
    # document; raising would discard a whole parse because of one bad span.
    merged_xml = serialize_annotations(canonical_text, selected)
    try:
        validation_report = validate_result(canonical_text, merged_xml, selected)
    except MergeInvariantError as exc:
        conflict_report.append({
            "type": "MERGE_INVARIANT_VIOLATION",
            "reason": str(exc),
            "spans_discarded": len(selected),
        })
        selected = []
        merged_xml = canonical_text
        validation_report = validate_result(canonical_text, merged_xml, selected)
        validation_report["degraded"] = True
        validation_report["degradation_reason"] = str(exc)

    stimuli = build_structured_stimuli(canonical_text, selected)
    questions = build_structured_questions(canonical_text, selected, stimuli)

    diagnostics = MergeDiagnostics(
        overlaps=overlap_reports,
        parses=parse_reports,
        conflicts=conflict_report,
        validation=validation_report,
    )

    return MergeResult(
        original_text=canonical_text,
        merged_xml=merged_xml,
        annotations=selected,
        structured_questions=questions,
        structured_stimuli=stimuli,
        diagnostics=diagnostics,
    )
