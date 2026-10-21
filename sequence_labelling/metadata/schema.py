"""
schema.py — Structured Data Contracts for Document Metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExtractorProvenance(BaseModel):
    model: str = "gpt-6-luna"
    provider: str = "aixingialaire"
    extracted_at: str = ""
    method: str = "llm"  # "llm" or "heuristic_fallback"


class ExamMetadata(BaseModel):
    schema_version: int = 2
    document_id: str = ""
    filename: str = ""
    original_filename: Optional[str] = None
    
    # Core educational dimensions requested by user (None if unknown)
    subject: Optional[str] = Field(
        default=None,
        description="Standard subject name (e.g., 'Toán', 'Tiếng Anh', 'Vật lí', 'Hóa học', 'Sinh học', 'Ngữ văn', etc.) or None if unknown."
    )
    examination: Optional[str] = Field(
        default=None,
        description="Standardized examination type (e.g., 'Kiểm tra cuối học kỳ 2', 'Thi tốt nghiệp THPT', 'Kiểm tra giữa học kỳ 1', etc.) or None if unknown."
    )
    grade: Optional[str] = Field(
        default=None,
        description="Standard grade level (e.g., '10', '11', '12', '9', 'THPT') or None if unknown."
    )
    
    # Extended structural metadata
    academic_year: Optional[str] = Field(
        default=None,
        description="Academic school year (e.g., '2024-2025', '2023-2024') or None if unknown."
    )
    curriculum: Optional[str] = Field(
        default=None,
        description="Educational curriculum standard: 'GDPT 2018' (CT mới: KNTT, CTST, Cánh Diều) | 'GDPT 2006' (CT cũ) | None if unknown."
    )
    authority_or_school: Optional[str] = Field(
        default=None,
        description="Educational authority or school name (e.g., 'Sở GD&ĐT Hà Tĩnh', 'THPT Chuyên Hà Nội - Amsterdam') or None if unknown."
    )
    raw_title: Optional[str] = Field(
        default=None,
        description="Verbatim examination title line from document header, or None if unknown."
    )
    
    # Provenance tracking
    metadata_extractor: Optional[ExtractorProvenance] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
