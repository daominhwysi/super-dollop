"""
Digital Chunk Extractor.

Extracts clean, authentic Vietnamese digital text chunks from digital exam PDFs,
filtering out embedded MathType symbols, legacy symbol fonts, and noise.
Applies length and word thresholds to ignore short fragments, producing
high-value, referenceable semantic chunks for LLM injection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import fitz  # PyMuPDF


@dataclass
class ChunkItem:
    id: int
    chunk_key: str
    text: str
    bbox: Tuple[float, float, float, float]

    def to_prompt_line(self) -> str:
        # Escape quotes inside text for clean prompt presentation
        escaped = self.text.replace('"', '\\"')
        return f'[{self.chunk_key}]: "{escaped}"'


class DigitalChunkExtractor:
    """
    Extracts and filters digital text chunks from PDF pages.
    """

    def __init__(
        self,
        min_chars: int = 5,
        min_words: int = 2,
        margin_top: float = 35.0,
        margin_bottom: float = 35.0,
    ):
        self.min_chars = min_chars
        self.min_words = min_words
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom

    @staticmethod
    def is_digital_pdf(pdf_path: Union[str, Path], min_chars_per_page: int = 80) -> bool:
        """Determines if a PDF has a usable digital text layer."""
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                doc.close()
                return False
            sample_pages = min(3, len(doc))
            total_chars = sum(len(doc[i].get_text().strip()) for i in range(sample_pages))
            doc.close()
            return (total_chars / sample_pages) >= min_chars_per_page
        except Exception:
            return False

    def is_valid_chunk_text(self, text: str) -> bool:
        """
        Filters out short fragments, pure punctuation, isolated option labels,
        and MathType noise characters.
        """
        stripped = text.strip()
        if len(stripped) < self.min_chars:
            return False

        # Reject isolated choice labels (A., B., C., D.) or question numbers alone (e.g. "Câu 1.")
        if re.match(r"^(?:[A-D]\.|\d+\.)$", stripped):
            return False

        # Reject noise consisting mainly of non-word ASCII math artifacts (e.g., !"#, %&')
        alnum_chars = sum(1 for c in stripped if c.isalnum() or c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ")
        if alnum_chars < 3:
            return False

        # Word count check
        words = stripped.split()
        if len(words) < self.min_words:
            # Allow single-word headings if sufficiently long and meaningful (e.g., "TRƯỜNG")
            if len(stripped) < 8 or not stripped.isupper():
                return False

        # Filter out common shared footer/watermark phrases
        if "TailieuOnthi" in stripped or "Fanpage:" in stripped:
            return False

        return True

    def extract_page_chunks(self, page: fitz.Page, page_num: int = 1) -> List[ChunkItem]:
        """
        Extracts semantic text chunks from a page, filtering out embedded formulas,
        tables, and headers/footers.
        """
        h_page, w_page = page.rect.height, page.rect.width

        # 1. Detect Tables to suppress raw flattened cell fragments
        table_rects = [fitz.Rect(t.bbox) for t in page.find_tables()]

        # 2. Extract structured spans
        data = page.get_text("dict")
        raw_chunks: List[Tuple[str, fitz.Rect]] = []
        current_chunk_words: List[str] = []
        current_rect: Optional[fitz.Rect] = None

        def flush_current():
            nonlocal current_chunk_words, current_rect
            if current_chunk_words and current_rect is not None:
                combined_text = " ".join(current_chunk_words)
                combined_text = re.sub(r"\s+([.,:;!?])", r"\1", combined_text).strip()
                if self.is_valid_chunk_text(combined_text):
                    raw_chunks.append((combined_text, current_rect))
            current_chunk_words = []
            current_rect = None

        for b in data.get("blocks", []):
            if "lines" not in b:
                continue
            for l in b["lines"]:
                spans = l.get("spans", [])
                for i, s in enumerate(spans):
                    font = s.get("font", "")
                    text = s.get("text", "").strip()
                    sr = fitz.Rect(s["bbox"])

                    # Margin check
                    if sr.y0 < self.margin_top or sr.y1 > (h_page - self.margin_bottom):
                        flush_current()
                        continue

                    # Table check
                    center = fitz.Point((sr.x0 + sr.x1) / 2, (sr.y0 + sr.y1) / 2)
                    if any(tr.contains(center) for tr in table_rects):
                        flush_current()
                        continue

                    # Check for math fonts or single-character italic math symbols
                    is_math_span = any(
                        mf in font
                        for mf in ["Symbol", "MT-Extra", "CambriaMath", "Wingdings"]
                    ) or (font == "TimesNewRomanPS-ItalicMT" and len(text) <= 2)

                    # Check for large intra-line gap indicating embedded formula slot
                    has_gap_before = False
                    if i > 0:
                        prev_span = spans[i - 1]
                        if sr.x0 - prev_span["bbox"][2] > 12:
                            has_gap_before = True

                    if is_math_span or has_gap_before:
                        flush_current()

                    if is_math_span:
                        continue

                    if not text:
                        continue

                    current_chunk_words.append(text)
                    if current_rect is None:
                        current_rect = fitz.Rect(sr)
                    else:
                        current_rect |= sr

                flush_current()

        flush_current()

        # Deduplicate & construct ChunkItem instances
        items: List[ChunkItem] = []
        seen_texts = set()
        chunk_idx = 1

        for text, rect in raw_chunks:
            # Normalize whitespace
            norm_text = re.sub(r"\s+", " ", text).strip()
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)

            chunk_key = f"CHUNK_{chunk_idx}"
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            items.append(ChunkItem(id=chunk_idx, chunk_key=chunk_key, text=norm_text, bbox=bbox))
            chunk_idx += 1

        return items


def expand_chunks_in_markdown(text: str, chunks: Sequence[ChunkItem]) -> Tuple[str, int]:
    """
    Expands reference tokens [CHUNK_k] or {{CHUNK_k}} back into verified Unicode text.
    Returns (expanded_text, replacement_count).
    """
    chunk_map: Dict[str, str] = {c.chunk_key: c.text for c in chunks}
    # Also support numeric keys e.g. [CHUNK_1] -> "1"
    for c in chunks:
        chunk_map[str(c.id)] = c.text

    replacements = 0

    def replace_match(m: re.Match) -> str:
        nonlocal replacements
        key = m.group(1) or m.group(2)
        if key in chunk_map:
            replacements += 1
            return chunk_map[key]
        # Return original if unknown
        return m.group(0)

    pattern = r"\[CHUNK_([a-zA-Z0-9_]+)\]|\{\{C(?:HUNK_)?([a-zA-Z0-9_]+)\}\}"
    expanded = re.sub(pattern, replace_match, text)
    return expanded, replacements
