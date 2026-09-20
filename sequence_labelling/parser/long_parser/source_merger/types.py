from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PageOffsetRange:
    page_id: int
    local_start: int
    local_end: int
    global_start: int
    global_end: int
    is_overlap: bool = False


@dataclass
class ChunkInput:
    index: int
    original_text: str
    parsed_xml: str
    page_ids: Optional[List[int]] = None
    overlap_page_ids: Optional[List[int]] = None
    page_offset_ranges: Optional[List[PageOffsetRange]] = None


@dataclass
class GlobalSpan:
    start: int
    end: int
    label: str
    chunk_index: int
    confidence: float = 1.0
    alignment_kind: str = "exact"  # exact, normalized, fuzzy, rejected
    structural_quality: float = 1.0
    text: Optional[str] = None
    question_num: Optional[str] = None
    exam_code: Optional[str] = None
    is_self_closing: bool = False
    raw_tag: Optional[str] = None

    def overlaps_with(self, other: "GlobalSpan") -> bool:
        if self.is_self_closing or other.is_self_closing:
            return False
        return max(self.start, other.start) < min(self.end, other.end)


@dataclass
class MergeDiagnostics:
    overlaps: List[Dict[str, Any]] = field(default_factory=list)
    parses: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeResult:
    original_text: str
    merged_xml: str
    annotations: List[GlobalSpan]
    structured_questions: List[Dict[str, Any]]
    structured_stimuli: Dict[str, Dict[str, Any]]
    diagnostics: MergeDiagnostics


class MergeInvariantError(ValueError):
    """Raised when remove_annotation_tags(merged_xml) != canonical_original_text."""
    pass
