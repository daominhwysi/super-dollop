"""
Source-Grounded Parsed Chunk Merging Package.
"""

from sequence_labelling.parser.long_parser.source_merger.types import (
    ChunkInput,
    GlobalSpan,
    MergeDiagnostics,
    MergeResult,
    MergeInvariantError,
)
from sequence_labelling.parser.long_parser.source_merger.merger import merge_chunks

__all__ = [
    "ChunkInput",
    "GlobalSpan",
    "MergeDiagnostics",
    "MergeResult",
    "MergeInvariantError",
    "merge_chunks",
]
