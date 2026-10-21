"""
extractor.py — LLM-Powered Document Metadata Extractor using GPT-6 Luna.

Core Capabilities:
1. Input Window Strategy:
   - Page-aware extraction: Page 1 (Header/Cover) + two pages in the middle (Question Content).
   - Strict token cap: 750 tokens maximum per page.
   - Cleans figure badges and page metadata headers.
2. Structured LLM Call:
   - Uses gpt-6-luna via provider aixingialaire.
   - Enforces schema contracts with None / null for unknown fields.
3. Resilient Fallback:
   - Deterministic regex/heuristic fallback on API failure or malformed output.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENC = None

from sequence_labelling.annotator.ocr_document import OCRDocument
from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.metadata.schema import ExamMetadata, ExtractorProvenance

logger = logging.getLogger(__name__)


def clean_filename_for_context(filename: str) -> str:
    """
    Sanitize raw filename into clean, readable Vietnamese context.
    Removes random hash suffixes (e.g. --a7b59d6b9788), extensions (.pdf, .md),
    and converts hyphens/underscores to spaces while preserving accented characters.
    """
    name = Path(filename).stem
    # Remove --hash suffix if present
    name = re.sub(r"--[a-f0-9]{8,64}$", "", name, flags=re.IGNORECASE)
    # Remove doc_XXX_ prefix if present
    name = re.sub(r"^doc_\d+_", "", name)
    # Normalize hyphens and underscores to spaces if it looks like a slug
    if "-" in name and " " not in name:
        name = name.replace("-", " ")
    return name.strip()


def truncate_text_to_tokens(text: str, max_tokens: int = 750) -> str:
    """
    Truncate text to at most max_tokens tokens.
    Uses tiktoken if available, falling back to ~3.5 chars per token.
    """
    text = text.strip()
    if not text:
        return ""

    if _TIKTOKEN_AVAILABLE and _ENC is not None:
        try:
            tokens = _ENC.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return _ENC.decode(tokens[:max_tokens]).strip() + "\n... [truncated at 750 tokens]"
        except Exception:
            pass

    # Fallback character-based truncation
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    return text[:char_limit].rsplit(" ", 1)[0].strip() + "\n... [truncated at 750 tokens]"


def extract_window_context(
    ocr_markdown: str,
    max_tokens_per_page: int = 750,
) -> Tuple[str, int, List[int]]:
    """
    Extracts the first page and two pages in the middle of the document.
    Enforces maximum 750 tokens per selected page.
    
    Returns:
        (formatted_context_string, total_pages_in_doc, selected_page_indices_0_based)
    """
    ocr_doc = OCRDocument.from_markdown(ocr_markdown)
    
    if not ocr_doc.pages:
        clean_text = ocr_doc.raw_cleaned_text
        clean_text = re.sub(r"<figure[^>]*/>", "", clean_text)
        clean_text = re.sub(r"<figure.*?</figure>", "", clean_text, flags=re.DOTALL)
        truncated = truncate_text_to_tokens(clean_text, max_tokens=max_tokens_per_page * 3)
        return (f"--- DOCUMENT START ---\n{truncated}", 1, [0])

    total_pages = len(ocr_doc.pages)
    
    # 2. Select first page + two middle pages
    if total_pages == 1:
        selected_indices = [0]
    elif total_pages == 2:
        selected_indices = [0, 1]
    elif total_pages == 3:
        selected_indices = [0, 1, 2]
    else:
        # total_pages >= 4
        mid = total_pages // 2
        # Middle two pages around midpoint (1-based: mid and mid+1)
        selected_indices = sorted(list(set([0, max(1, mid - 1), min(total_pages - 1, mid)])))

    formatted_pages: List[str] = []
    for idx in selected_indices:
        p_obj = ocr_doc.pages[idx]
        
        # Clean figure badges (metadata wrappers are already cleanly stripped in p_obj.cleaned_text)
        p_clean = re.sub(r"<figure[^>]*/>", "", p_obj.cleaned_text)
        p_clean = re.sub(r"<figure.*?</figure>", "", p_clean, flags=re.DOTALL).strip()
        
        # Apply 750 token limit
        p_truncated = truncate_text_to_tokens(p_clean, max_tokens=max_tokens_per_page)
        
        header = f"--- TRANG {p_obj.page_number}/{total_pages} ---"
        formatted_pages.append(f"{header}\n{p_truncated}")

    context_str = "\n\n".join(formatted_pages)
    return (context_str, total_pages, selected_indices)


def heuristic_metadata_fallback(
    filename: str,
    context: str = "",
    doc_id: str = "",
) -> ExamMetadata:
    """
    Deterministic regex fallback to extract partial metadata from filename and context
    when LLM call fails or times out.
    """
    clean_name = clean_filename_for_context(filename).lower()
    combined = (clean_name + " " + context[:2000]).lower()

    # 1. Subject detection
    subject = None
    if re.search(r"\b(toán|toan)\b", combined):
        subject = "Toán"
    elif re.search(r"\b(tiếng anh|tieng anh|tiếng-anh|english|ngoại ngữ|ngoai ngu)\b", combined):
        subject = "Tiếng Anh"
    elif re.search(r"\b(vật lí|vật lý|vat ly|vat li|vật-lí|vật-lý)\b", combined):
        subject = "Vật lí"
    elif re.search(r"\b(hóa học|hóa|hoa hoc|hoa)\b", combined):
        subject = "Hóa học"
    elif re.search(r"\b(sinh học|sinh|sinh hoc)\b", combined):
        subject = "Sinh học"
    elif re.search(r"\b(ngữ văn|văn|ngu van|van)\b", combined):
        subject = "Ngữ văn"
    elif re.search(r"\b(lịch sử|sử|lich su|su)\b", combined):
        subject = "Lịch sử"
    elif re.search(r"\b(địa lí|địa lý|dia ly|dia li)\b", combined):
        subject = "Địa lí"
    elif re.search(r"\b(tin học|tin hoc)\b", combined):
        subject = "Tin học"
    elif re.search(r"\b(gdcd|kinh tế và pháp luật|gdkt&pl|gdkt)\b", combined):
        subject = "GDCD / GDKT&PL"

    # 2. Grade detection
    grade = None
    grade_match = re.search(r"\b(?:khối|lớp|k|toán|toan|tiếng anh|hóa|lí|sinh)?\s*(10|11|12|9|8|7|6)\b", clean_name)
    if grade_match:
        grade = grade_match.group(1)
    elif "thpt quốc gia" in combined or "tốt nghiệp thpt" in combined or "thpt quoc gia" in combined:
        grade = "12"

    # 3. Examination detection
    examination = None
    if re.search(r"\b(?:cuối học kỳ 2|cuối kì 2|cuối kỳ 2|cuoi hoc ky 2|cuoi ki 2|cuoi ky 2|hk2|học kỳ 2|học kì 2|hoc ky 2)\b", combined):
        examination = "Kiểm tra cuối học kỳ 2"
    elif re.search(r"\b(?:giữa học kỳ 2|giữa kì 2|giữa kỳ 2|giua hoc ky 2|giua ki 2|giua ky 2)\b", combined):
        examination = "Kiểm tra giữa học kỳ 2"
    elif re.search(r"\b(?:cuối học kỳ 1|cuối kì 1|cuối kỳ 1|cuoi hoc ky 1|cuoi ki 1|cuoi ky 1|hk1|học kỳ 1|học kì 1|hoc ky 1)\b", combined):
        examination = "Kiểm tra cuối học kỳ 1"
    elif re.search(r"\b(?:giữa học kỳ 1|giữa kì 1|giữa kỳ 1|giua hoc ky 1|giua ki 1|giua ky 1)\b", combined):
        examination = "Kiểm tra giữa học kỳ 1"
    elif re.search(r"\b(?:tốt nghiệp thpt|thpt quốc gia|thptqg|tot nghiep thpt|thpt quoc gia)\b", combined):
        examination = "Thi tốt nghiệp THPT"
    elif re.search(r"\b(?:học sinh giỏi|hsg|hoc sinh gioi)\b", combined):
        examination = "Chọn học sinh giỏi"
    elif re.search(r"\b(?:đánh giá năng lực|đgnl|đgtd|danh gia nang luc)\b", combined):
        examination = "Đánh giá năng lực"
    elif re.search(r"\b(?:khảo sát|kscl|thi thử|khao sat|thi thu)\b", combined):
        examination = "Khảo sát chất lượng / Thi thử"
    elif re.search(r"\b(?:chuyên đề|bài tập|ôn tập|chuyen de|bai tap|on tap)\b", combined):
        examination = "Chuyên đề / Ôn tập"

    # 4. Academic year
    academic_year = None
    year_match = re.search(r"\b(20\d\d)\s*[-–_/\s]\s*(20\d\d)\b", combined)
    if year_match:
        academic_year = f"{year_match.group(1)}-{year_match.group(2)}"
    else:
        single_year = re.search(r"\b(?:năm|nam)?\s*(20\d\d)\b", combined)
        if single_year:
            academic_year = single_year.group(1)

    # 5. Curriculum
    curriculum = None
    if re.search(r"\b(ctst|chân trời sáng tạo|kntt|kết nối tri thức|cánh diều|chương trình mới|gdpt 2018)\b", combined):
        curriculum = "GDPT 2018"
    elif academic_year and (academic_year.startswith("201") or academic_year.startswith("2020") or academic_year.startswith("2021")):
        curriculum = "GDPT 2006"

    return ExamMetadata(
        document_id=doc_id or Path(filename).stem,
        filename=filename,
        subject=subject,
        examination=examination,
        grade=grade,
        academic_year=academic_year,
        curriculum=curriculum,
        authority_or_school=None,
        raw_title=clean_filename_for_context(filename),
        metadata_extractor=ExtractorProvenance(
            model="heuristic-fallback",
            provider="regex",
            extracted_at=datetime.now(timezone.utc).isoformat(),
            method="heuristic_fallback",
        ),
    )


METADATA_EXTRACTION_SYSTEM_PROMPT = """You are an expert document metadata extraction system for Vietnamese educational examination documents.
Analyze the provided document excerpt (first page and middle pages) and filename to extract essential educational dimensions.

CRITICAL CONTRACTS:
1. STRICT HONESTY & NULL RULE:
   - If any field is unknown, unmentioned, or cannot be determined with certainty, output null.
   - NEVER invent, extrapolate, or guess values.
2. STANDARDIZED CANONICAL OUTPUTS:
   - "subject": Standard Vietnamese subject name ("Toán", "Tiếng Anh", "Vật lí", "Hóa học", "Sinh học", "Ngữ văn", "Lịch sử", "Địa lí", "GDCD / GDKT&PL", "Tin học", "Khoa học tự nhiên", "Công nghệ") or null.
   - "examination": Standard examination type (e.g., "Kiểm tra cuối học kỳ 2", "Kiểm tra giữa học kỳ 2", "Kiểm tra cuối học kỳ 1", "Kiểm tra giữa học kỳ 1", "Thi tốt nghiệp THPT", "Chọn học sinh giỏi", "Đánh giá năng lực", "Khảo sát chất lượng / Thi thử", "Chuyên đề / Ôn tập") or null.
   - "grade": Grade level ("10", "11", "12", "9", "8", "7", "6", "THPT") or null.
   - "academic_year": Standard academic year (e.g. "2024-2025", "2023-2024") or single year ("2025") or null.
   - "curriculum": "GDPT 2018" (for new 2018 curriculum: KNTT, CTST, Cánh Diều, or Part I/II/III structure) | "GDPT 2006" (legacy program) | null.
   - "authority_or_school": Department of education, province, or high school name (e.g. "Sở GD&ĐT Hà Tĩnh", "THPT Chuyên Hà Nội - Amsterdam") or null.
   - "raw_title": Exact verbatim title line extracted from document header, or null.

3. STRICT JSON ONLY:
   - Output MUST be valid JSON adhering strictly to the schema. No markdown outside ```json...``` fences.
"""


def extract_document_metadata(
    ocr_markdown: str,
    filename: str,
    doc_id: str = "",
    original_filename: Optional[str] = None,
    model: str = "gpt-6-luna",
    provider: str = "aixingialaire",
    max_tokens_per_page: int = 750,
) -> ExamMetadata:
    """
    Extracts metadata from document OCR markdown and filename using gpt-6-luna.
    Falls back to heuristic extraction on error.
    """
    if not doc_id:
        doc_id = Path(filename).stem

    context, total_pages, selected_indices = extract_window_context(
        ocr_markdown, max_tokens_per_page=max_tokens_per_page
    )
    cleaned_name = clean_filename_for_context(filename)

    user_prompt = f"""Document Filename: {filename}
Cleaned Title Clue: {cleaned_name}
Original Filename: {original_filename or 'None'}
Total Document Pages: {total_pages}
Inspected Pages: {', '.join(str(idx + 1) for idx in selected_indices)}

Document Excerpt:
{context}

Extract the metadata strictly according to the rules. Remember: output null for any unknown field.

Required JSON Structure:
{{
  "subject": "string or null",
  "examination": "string or null",
  "grade": "string or null",
  "academic_year": "string or null",
  "curriculum": "string or null",
  "authority_or_school": "string or null",
  "raw_title": "string or null"
}}
"""

    try:
        reply = chat(
            prompt=user_prompt,
            system=METADATA_EXTRACTION_SYSTEM_PROMPT,
            model=model,
            provider=provider,
            max_tokens=1000,
        )

        # Extract JSON from reply
        json_str = reply.strip()
        # Strip ```json ... ``` code fence if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, flags=re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            first_brace = json_str.find("{")
            last_brace = json_str.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = json_str[first_brace : last_brace + 1].strip()

        data = json.loads(json_str)

        # Helper to ensure empty strings or "unknown" are mapped to None
        def clean_val(val: Any) -> Optional[str]:
            if val is None:
                return None
            s = str(val).strip()
            if not s or s.lower() in ["null", "none", "unknown", "n/a", "không xác định"]:
                return None
            return s

        return ExamMetadata(
            document_id=doc_id,
            filename=filename,
            original_filename=original_filename,
            subject=clean_val(data.get("subject")),
            examination=clean_val(data.get("examination")),
            grade=clean_val(data.get("grade")),
            academic_year=clean_val(data.get("academic_year")),
            curriculum=clean_val(data.get("curriculum")),
            authority_or_school=clean_val(data.get("authority_or_school")),
            raw_title=clean_val(data.get("raw_title")),
            metadata_extractor=ExtractorProvenance(
                model=model,
                provider=provider,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                method="llm",
            ),
        )

    except Exception as e:
        logger.warning(
            f"LLM metadata extraction failed for '{doc_id}' ({e}). Using heuristic fallback."
        )
        return heuristic_metadata_fallback(
            filename=filename, context=context, doc_id=doc_id
        )


def extract_and_save_metadata(
    ocr_markdown: str,
    filename: str,
    doc_id: str = "",
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    original_filename: Optional[str] = None,
    model: str = "gpt-6-luna",
    provider: str = "aixingialaire",
) -> Path:
    """
    High-level convenience API: extracts metadata and writes to <output_dir>/<doc_id>.metadata.json.
    Preserves original_filename from existing metadata file if available.
    """
    if not doc_id:
        doc_id = Path(filename).stem

    if output_dir is None:
        workspace_dir = Path(__file__).resolve().parents[2]
        output_dir = workspace_dir / "data" / "sequence_labelling_metadata"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{doc_id}.metadata.json"

    # If file exists and not overwrite, skip
    if target_path.exists() and not overwrite:
        try:
            existing_data = json.loads(target_path.read_text(encoding="utf-8"))
            # Only skip if it's already v2 and has subject populated
            if existing_data.get("schema_version") == 2 and existing_data.get("metadata_extractor", {}).get("method") == "llm":
                logger.info(f"Skipping existing up-to-date metadata for: {doc_id}")
                return target_path
            # If v1 or incomplete, we preserve original_filename
            if not original_filename and existing_data.get("original_filename"):
                original_filename = existing_data.get("original_filename")
        except Exception:
            pass

    metadata = extract_document_metadata(
        ocr_markdown=ocr_markdown,
        filename=filename,
        doc_id=doc_id,
        original_filename=original_filename,
        model=model,
        provider=provider,
    )

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

    return target_path
