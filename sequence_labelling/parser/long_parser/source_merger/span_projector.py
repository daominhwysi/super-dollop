import re
from typing import List, Dict, Any, Optional, Tuple
from sequence_labelling.parser.long_parser.source_merger.types import GlobalSpan, ChunkInput
from sequence_labelling.parser.long_parser.source_merger.canonical_source import PiecewiseMap
from sequence_labelling.parser.long_parser.source_merger.xml_lexer import LocalSpan
from sequence_labelling.parser.long_parser.source_merger.source_aligner import SourceAligner

_ATOMIC_HTML_TAG_RE = re.compile(
    r"</?(?:figure|table|tr|th|td|tbody|thead|tfoot|caption|colgroup|col|"
    r"b|i|u|s|strong|em|sub|sup|small|mark|p|br|hr|span|div|a|img|code|pre|"
    r"ul|ol|li|center|font|del|strike|h[1-6]|blockquote)\b"
    r"(?:\s+(?:[^<>\"']|\"[^\"]*\"|'[^']*')*|\s*/?\s*)>"
)


def _snap_boundaries_out_of_tags(
    g_start: int,
    g_end: int,
    p_sub: str,
    tag_intervals: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Never allow a projected span boundary to slice inside an atomic HTML/figure tag."""
    p_stripped = p_sub.strip()
    for f_start, f_end in tag_intervals:
        if f_start < g_start < f_end:
            if p_stripped.startswith("<") and len(p_stripped) > 1 and (p_stripped[1].isalpha() or p_stripped[1] == "/"):
                g_start = f_start
            else:
                g_start = f_end
        if f_start < g_end < f_end:
            if not p_stripped.endswith((">", "/>")):
                g_end = f_start
            else:
                g_end = f_end
    if g_end < g_start:
        g_end = g_start
    return g_start, g_end


def score_candidate(
    source_alignment: float,
    is_structurally_valid: bool,
    is_complete_entity: bool,
    chunk_offset: int,
    chunk_length: int,
    cross_agreements: int = 0
) -> float:
    """
    Score quality of a candidate span based on Section 9.1 formula:
    quality = 0.40 * source_alignment
            + 0.25 * structural_validity
            + 0.15 * entity_completeness
            + 0.10 * distance_from_chunk_edge
            + 0.10 * cross_chunk_agreement
    """
    s_align = max(0.0, min(1.0, source_alignment))
    s_valid = 1.0 if is_structurally_valid else 0.0
    e_comp = 1.0 if is_complete_entity else 0.5

    # Distance from chunk edge (higher score near middle of chunk)
    if chunk_length > 0:
        edge_dist = min(chunk_offset, chunk_length - chunk_offset)
        dist_score = min(1.0, edge_dist / (chunk_length * 0.2 + 1e-5))
    else:
        dist_score = 0.5

    cross_score = min(1.0, cross_agreements * 0.5)

    quality = (
        0.40 * s_align +
        0.25 * s_valid +
        0.15 * e_comp +
        0.10 * dist_score +
        0.10 * cross_score
    )
    return float(round(quality, 4))


def project_local_spans_to_global(
    local_spans: List[LocalSpan],
    aligner: SourceAligner,
    piecewise_map: PiecewiseMap,
    chunk_input: ChunkInput,
    canonical_text: str,
) -> List[GlobalSpan]:
    """
    Project local parsed XML spans -> original chunk text -> global canonical text coordinates.
    """
    global_spans: List[GlobalSpan] = []
    figure_intervals = [(m.start(), m.end()) for m in _ATOMIC_HTML_TAG_RE.finditer(canonical_text)]

    for l_span in local_spans:
        if l_span.is_self_closing:
            s_start, s_end, quality, alignment_kind = aligner.map_span_to_source(l_span.p_start, l_span.p_start)
            g_start = piecewise_map.map_local_to_global(s_start)
            g_start_clamped = max(0, min(len(canonical_text), g_start))
            global_spans.append(GlobalSpan(
                start=g_start_clamped,
                end=g_start_clamped,
                label=l_span.label,
                chunk_index=chunk_input.index,
                confidence=1.0,
                alignment_kind=alignment_kind,
                structural_quality=1.0,
                text="",
                question_num=l_span.question_num,
                exam_code=l_span.exam_code,
                is_self_closing=True,
                raw_tag=l_span.raw_tag,
            ))
            continue

        # Step 1: Map parsed text offset P -> original chunk text offset S
        s_start, s_end, quality, alignment_kind = aligner.map_span_to_source(l_span.p_start, l_span.p_end)

        if alignment_kind == "rejected" or quality < 0.90:
            continue

        # Step 2: Map original chunk offset S -> global canonical text offset G
        mapped_span = piecewise_map.map_span_to_global(
            s_start,
            s_end,
            chunk_input.original_text,
        )
        if mapped_span is None:
            continue
        g_start, g_end = mapped_span

        g_start_clamped = max(0, min(len(canonical_text), g_start))
        g_end_clamped = max(g_start_clamped, min(len(canonical_text), g_end))

        # Trim trailing whitespace if parsed span text did not end with whitespace
        p_sub = aligner.parsed_text[l_span.p_start:l_span.p_end]
        if p_sub and not p_sub[-1].isspace():
            while g_end_clamped > g_start_clamped and canonical_text[g_end_clamped - 1].isspace():
                g_end_clamped -= 1

        # Include trailing sentence punctuation inside span if parsed text ends with punctuation
        if p_sub and p_sub[-1] in ".!?:" and g_end_clamped < len(canonical_text) and canonical_text[g_end_clamped] in ".!?:":
            if l_span.label in ("stem", "option_text"):
                g_end_clamped += 1

        if figure_intervals:
            g_start_clamped, g_end_clamped = _snap_boundaries_out_of_tags(
                g_start_clamped, g_end_clamped, p_sub, figure_intervals
            )

        # Guard against fuzzy alignment stretching a short span over an omitted chunk gap
        p_len = l_span.p_end - l_span.p_start
        g_len = g_end_clamped - g_start_clamped
        if alignment_kind == "fuzzy" and p_len > 0 and (
            g_len > max(p_len * 3, p_len + 500) or p_len > max(g_len * 3, g_len + 500)
        ):
            continue

        # Extract authoritative text from canonical text
        span_text = canonical_text[g_start_clamped:g_end_clamped]

        conf = score_candidate(
            source_alignment=quality,
            is_structurally_valid=l_span.is_valid,
            is_complete_entity=True,
            chunk_offset=s_start,
            chunk_length=len(chunk_input.original_text),
        )

        global_spans.append(GlobalSpan(
            start=g_start_clamped,
            end=g_end_clamped,
            label=l_span.label,
            chunk_index=chunk_input.index,
            confidence=conf,
            alignment_kind=alignment_kind,
            structural_quality=1.0 if l_span.is_valid else 0.5,
            text=span_text,
            question_num=l_span.question_num,
            exam_code=l_span.exam_code,
            is_self_closing=False,
            raw_tag=l_span.raw_tag,
        ))

    return global_spans
