"""
Document metadata extraction package.
"""

from sequence_labelling.metadata.schema import ExamMetadata, ExtractorProvenance
from sequence_labelling.metadata.extractor import (
    clean_filename_for_context,
    extract_window_context,
    heuristic_metadata_fallback,
    extract_document_metadata,
    extract_and_save_metadata,
)

__all__ = [
    "ExamMetadata",
    "ExtractorProvenance",
    "clean_filename_for_context",
    "extract_window_context",
    "heuristic_metadata_fallback",
    "extract_document_metadata",
    "extract_and_save_metadata",
]
