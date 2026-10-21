"""
Long-Context Parser Package (Azozo Engine v2.0)
High-precision decoupled document parsing pipeline for multi-thousand page exams and textbooks.
"""

from sequence_labelling.parser.long_parser.document_cartographer import (
    CartographyManifest,
    CartographyObject,
    DocumentCartographer,
)

__all__ = [
    "DocumentCartographer",
    "CartographyObject",
    "CartographyManifest",
]

