from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional
from sequence_labelling.parser.long_parser.source_merger.types import GlobalSpan


def reconcile_candidates(
    candidates: List[GlobalSpan],
    canonical_text: str,
) -> Tuple[List[GlobalSpan], List[Dict[str, Any]]]:
    """
    Reconcile candidate global spans using deterministic quality ranking and schema constraints.
    Returns (selected_spans, conflict_reports).
    """
    if not candidates:
        return [], []

    # 1. Coalesce identical global spans (same label, start, end, self-closing)
    grouped: Dict[Tuple[str, int, int, bool, Optional[str]], List[GlobalSpan]] = {}
    for span in candidates:
        key = (span.label, span.start, span.end, span.is_self_closing, span.raw_tag)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(span)

    coalesced_candidates: List[GlobalSpan] = []
    for key, span_list in grouped.items():
        best_span = max(
            span_list,
            key=lambda s: (
                s.structural_quality,
                1.0 if s.alignment_kind == "exact" else (0.95 if s.alignment_kind == "normalized" else 0.8),
                s.confidence,
                -s.chunk_index,  # prefer lower chunk index on tie
            )
        )
        # Boost confidence for cross-chunk agreement
        if len(span_list) > 1:
            best_span.confidence = min(1.0, best_span.confidence + 0.05 * (len(span_list) - 1))
        coalesced_candidates.append(best_span)

    # Sort candidates by start ascending, end descending (outer before inner), then label priority
    label_priority = {
        "section": 1,
        "stimulus": 2,
        "question": 3,
        "question_label": 4,
        "stem": 5,
        "option_label": 6,
        "option_text": 7,
        "explanation": 8,
    }

    coalesced_candidates.sort(
        key=lambda s: (s.start, -s.end, label_priority.get(s.label, 99))
    )

    conflict_reports: List[Dict[str, Any]] = []
    selected: List[GlobalSpan] = []

    # Atomic question component labels are mutually exclusive (cannot overlap each other)
    atomic_labels = {"question_label", "stem", "option_label", "option_text"}

    # 1. Resolve atomic element overlaps across all atomic labels together
    atomic_spans = [s for s in coalesced_candidates if s.label in atomic_labels]
    atomic_spans.sort(
        key=lambda s: (
            -s.confidence,
            -s.structural_quality,
            s.start,
            s.chunk_index,
        )
    )

    accepted_atomic_spans: List[GlobalSpan] = []
    for candidate in atomic_spans:
        overlaps = False
        for accepted in accepted_atomic_spans:
            if candidate.overlaps_with(accepted):
                overlaps = True
                conflict_reports.append({
                    "type": "ATOMIC_ELEMENT_OVERLAP_CONFLICT",
                    "label": candidate.label,
                    "rejected": candidate,
                    "accepted": accepted,
                    "reason": f"Atomic element {candidate.label} overlaps accepted {accepted.label}",
                })
                break
        if not overlaps:
            accepted_atomic_spans.append(candidate)

    selected.extend(accepted_atomic_spans)

    # 2. Resolve other labels (section, stimulus, explanation)
    other_labels = [lbl for lbl in label_priority.keys() if lbl not in atomic_labels]
    for label in other_labels:
        label_spans = [s for s in coalesced_candidates if s.label == label]
        if not label_spans:
            continue

        if label in ("section", "stimulus"):
            label_spans.sort(
                key=lambda s: (
                    -s.confidence,
                    -s.structural_quality,
                    s.start,
                    s.chunk_index,
                )
            )

            accepted_label_spans: List[GlobalSpan] = []
            for candidate in label_spans:
                overlaps = False
                for accepted in accepted_label_spans:
                    if candidate.overlaps_with(accepted):
                        overlaps = True
                        conflict_reports.append({
                            "type": "INTERVAL_OVERLAP_CONFLICT",
                            "label": label,
                            "rejected": candidate,
                            "accepted": accepted,
                            "reason": "Lower quality candidate over overlapping source interval",
                        })
                        break
                if not overlaps:
                    accepted_label_spans.append(candidate)

            selected.extend(accepted_label_spans)
        else:
            selected.extend(label_spans)

    # Sort final selected spans by start position
    selected.sort(
        key=lambda s: (s.start, -s.end, label_priority.get(s.label, 99))
    )

    selected = enforce_well_nesting(selected, canonical_text, conflict_reports)

    return selected, conflict_reports


def enforce_well_nesting(
    spans: List[GlobalSpan],
    canonical_text: str,
    conflict_reports: List[Dict[str, Any]],
) -> List[GlobalSpan]:
    """
    Clip or drop spans that cross an enclosing span, so the selection forms a
    forest and can always be serialized as balanced tags.
    Preserves self-closing point spans (e.g. <stimulus ... />).
    """
    ordered: List[GlobalSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if span.is_self_closing:
            ordered.append(span)
            continue
        if span.end <= span.start:
            conflict_reports.append({
                "type": "EMPTY_SPAN_DROPPED",
                "label": span.label,
                "rejected": span,
                "reason": f"{span.label} collapsed to zero width at {span.start}",
            })
            continue
        ordered.append(span)

    kept: List[GlobalSpan] = []
    stack: List[GlobalSpan] = []

    for span in ordered:
        if span.is_self_closing:
            kept.append(span)
            continue

        while stack and stack[-1].end <= span.start:
            stack.pop()

        if stack and span.end > stack[-1].end:
            enclosing = stack[-1]
            clipped_end = enclosing.end
            if clipped_end <= span.start:
                conflict_reports.append({
                    "type": "CROSSING_SPAN_DROPPED",
                    "label": span.label,
                    "rejected": span,
                    "accepted": enclosing,
                    "reason": (
                        f"{span.label}[{span.start},{span.end}] crosses "
                        f"{enclosing.label}[{enclosing.start},{enclosing.end}]"
                    ),
                })
                continue

            conflict_reports.append({
                "type": "CROSSING_SPAN_CLIPPED",
                "label": span.label,
                "rejected": span,
                "accepted": enclosing,
                "reason": (
                    f"{span.label} end {span.end} clipped to {clipped_end} to nest "
                    f"inside {enclosing.label}[{enclosing.start},{enclosing.end}]"
                ),
            })
            span = replace(
                span,
                end=clipped_end,
                text=canonical_text[span.start:clipped_end],
            )

        kept.append(span)
        stack.append(span)

    kept.sort(key=lambda s: (s.start, -s.end))
    return kept
