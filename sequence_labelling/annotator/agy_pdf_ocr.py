"""
agy_pdf_ocr.py — Autonomous End-to-End PDF OCR Converter powered by Google Antigravity (agy).

Combines:
1. Digital text chunk extraction (DigitalChunkExtractor) for 100% Vietnamese text fidelity.
2. High-resolution page rendering (150 DPI) with OpenCV/PyMuPDF.
3. Direct figure detection via AGY Vision model (eliminating ONNX/RF-DETR dependencies).
4. Chunk-injection prompt protocol (halving output token decode time).
5. Deterministic chunk reference expansion back to clean Unicode markdown.
6. Tier 1 (Typographic Archaeologist) & Tier 2 (Style Blueprinting) prompt contracts
   for authentic layout diversity without sequence labelling XML tags.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pymupdf

from sequence_labelling.annotator.pdf_converter import (
    SYSTEM_PROMPT_LONG_CONTEXT,
    prune_code_blocks,
    prune_think_tags,
)
from sequence_labelling.llm.agy_adapter import chat_with_agy
from tools.experiments.chunk_extractor import (
    ChunkItem,
    DigitalChunkExtractor,
    expand_chunks_in_markdown,
)

logger = logging.getLogger(__name__)

# Specialized OCR System Prompt for AGY with Authentic Layout Diversity & Direct Figure Detection
AGY_DIVERSE_RAW_OCR_SYSTEM_PROMPT = """You are an expert Vietnamese examination document OCR and Layout Transcription AI.
Transcribe the provided Vietnamese examination PDF page images into clean, faithful, and richly structured Markdown that mirrors the exact typographic design and spatial layout of each individual exam.

CRITICAL OCR & DIVERSITY CONTRACTS:
1. TIER 1: TYPOGRAPHIC ARCHAEOLOGIST (FAITHFUL LAYOUT MIRRORING):
   - Faithfully preserve the author's original question syntax, whether printed as "**Câu 1:**", "Câu 1.", "Câu 1 -", "Bài 1:", "Câu 1 (2,0 điểm):", "1.", "Question 1:", or other regional styles. NEVER standardize everything into "## Câu X".
   - Faithfully preserve the spatial arrangement of options as printed in the exam:
     * Inline horizontal choices (e.g. A. ...   B. ...   C. ...   D. ...) must remain inline.
     * 2x2 grid choices must remain in a 2x2 structure.
     * Vertical list choices must remain on individual lines.
   - Faithfully preserve document headings, school/exam headers, instructions, and sections in their natural Markdown hierarchy (#, ##, ###, bold, italic).

2. TIER 2: AUTHENTIC BLUEPRINT PALETTE (RECOGNIZE DIVERSE EXAM CONVENTIONS):
   - Recognize that Vietnamese exams follow varied educational formats: National Standard (THPTQG), Specialized Provincial (Chuyên), Competency Assessments (ĐGNL/ĐGTD), and Teacher Handouts.
   - Embrace the natural typographic diversity of each format without enforcing a uniform markdown mold.

3. SINGLE-COLUMN READING ORDER MERGING:
   - If a page has two columns, convert and merge them into a clean, single-column linear flow following logical reading order (Column 1 top-to-bottom, then Column 2 top-to-bottom).
   - Do NOT visually simulate columns with horizontal spacing or tabs.

4. STRICT DISALLOWANCE OF SEQUENCE LABELLING TAGS:
   - Do NOT emit <question_label>, <stem>, <option_label>, <option_text>, or <other> tags.
   - Transcribe all questions, stems, options, and metadata as authentic, un-tagged Markdown.

5. PAGE ENVELOPE & METADATA:
   - Wrap each individual page in <page> ... </page>.
   - Append page metadata at the end of each page:
     <page_metadata>
     {"p": <page_num>, "w": <page_width_pts>, "h": <page_height_pts>}
     </page_metadata>

6. STRICT HTML TABLES ONLY:
   - Convert all tables, matrices, and choice grids to standard HTML <table>...</table> elements (<table><tr><th>...</th></tr><tr><td>...</td></tr></table>). NEVER use Markdown pipe tables (| col | col |).

7. DIRECT FIGURE DETECTION:
   - Identify every geometric figure, coordinate plot, function graph, or diagram.
   - For each figure, emit:
     <figure id="fig-p{page}-{idx}" bbox="x1,y1,x2,y2" description="{concise description}" />
   - Coordinates (x1,y1,x2,y2) must be integer pixel bounding boxes on the rendered page image.

8. DIGITAL CHUNK ACCELERATION:
   - Verified digital text chunks are listed for each page as [CHUNK_k].
   - Whenever narrative text, options, or stems match a chunk, output [CHUNK_k] to guarantee 100% authentic Vietnamese text fidelity.
   - Convert all mathematical formulas and expressions into standard LaTeX ($...$ inline, $$...$$ block).
"""


class AgyPDFOCRConverter:
    """
    End-to-End autonomous PDF OCR converter using agy CLI.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-flash-low",
        thinking: Optional[str] = "low",
        dpi: int = 150,
        batch_size: int = 2,
        min_chars: int = 5,
        min_words: int = 2,
    ):
        self.model = model
        self.thinking = thinking
        self.dpi = dpi
        self.batch_size = max(1, batch_size)
        self.extractor = DigitalChunkExtractor(min_chars=min_chars, min_words=min_words)

    def _build_page_prompt(
        self,
        page_num: int,
        width_pts: float,
        height_pts: float,
        chunks: Sequence[ChunkItem],
    ) -> str:
        """Constructs prompt instructions and digital chunk inventory for a page."""
        lines = [
            f"=== DOCUMENT PAGE {page_num} (File: page_{page_num}.jpg, Dimensions: {int(width_pts)}x{int(height_pts)} pts) ===",
            "DIRECT FIGURE DETECTION INSTRUCTION:",
            "Detect any geometric figures, function graphs, coordinate planes, or tables.",
            f"For figures, emit: <figure id=\"fig-p{page_num}-<idx>\" bbox=\"x1,y1,x2,y2\" description=\"...\" />",
            "",
            f"VERIFIED DIGITAL TEXT CHUNKS FOR PAGE {page_num}:",
        ]
        if chunks:
            for c in chunks:
                lines.append(c.to_prompt_line())
        else:
            lines.append("(No extractable digital chunks found on this page)")

        lines.extend([
            "",
            "RULES FOR THIS PAGE:",
            "1. Transcribe the natural text and layout of the page. Do NOT use sequence labelling XML tags (<question_label>, <stem>, <option_label>, <other>).",
            "2. Whenever output text matches any verified chunk above, emit [CHUNK_id] to ensure 100% authentic Unicode text.",
            "3. Render inline math in $...$ and display math in $$...$$.",
            f"4. Wrap page content in <page> ... </page> and end with <page_metadata>{{\"p\": {page_num}, \"w\": {int(width_pts)}, \"h\": {int(height_pts)}}}</page_metadata>.",
        ])
        return "\n".join(lines).replace("\x00", "")

    def convert_pdf_page_batch(
        self,
        pdf_path: Union[str, Path],
        target_pages: List[int],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Converts a small batch of pages (e.g. 2 pages) in a single vision call.
        Returns (expanded_markdown, telemetry).
        """
        t0 = time.time()
        doc = pymupdf.open(str(pdf_path))

        page_images: List[Tuple[str, bytes]] = []
        page_chunks: Dict[int, List[ChunkItem]] = {}
        all_chunks: List[ChunkItem] = []
        page_prompts: List[str] = []

        for p_num in target_pages:
            if p_num < 1 or p_num > len(doc):
                continue
            page = doc[p_num - 1]
            rect = page.rect

            # Render image @ 150 DPI
            pix = page.get_pixmap(dpi=self.dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            _, img_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            page_images.append((f"page_{p_num}.jpg", img_bytes.tobytes()))

            # Extract digital chunks
            chunks = self.extractor.extract_page_chunks(page, p_num)
            page_chunks[p_num] = chunks
            all_chunks.extend(chunks)

            # Build prompt
            prompt_str = self._build_page_prompt(
                page_num=p_num,
                width_pts=rect.width,
                height_pts=rect.height,
                chunks=chunks,
            )
            page_prompts.append(prompt_str)

        doc.close()

        if not page_images:
            return "", {"error": "No valid pages in batch"}

        combined_prompt = "\n\n".join(page_prompts)

        # Call agy Vision with Tier 1 + Tier 2 Diverse Raw OCR System Prompt
        t_call_start = time.time()
        raw_response, conv_id, usage = chat_with_agy(
            prompt=combined_prompt,
            system=AGY_DIVERSE_RAW_OCR_SYSTEM_PROMPT,
            model=self.model,
            thinking=self.thinking,
            images=page_images,
            timeout=300,
            caller="agy_pdf_ocr",
            auto_log=True,
        )
        call_duration = time.time() - t_call_start

        cleaned_raw = prune_code_blocks(prune_think_tags(raw_response))

        # Expand chunks deterministically
        expanded_md, replacements = expand_chunks_in_markdown(cleaned_raw, all_chunks)
        total_duration = time.time() - t0

        telemetry = {
            "target_pages": target_pages,
            "duration_sec": round(total_duration, 2),
            "call_duration_sec": round(call_duration, 2),
            "chunks_total": len(all_chunks),
            "chunks_expanded": replacements,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return expanded_md, telemetry

    def convert_document(
        self,
        pdf_path: Union[str, Path],
        max_pages: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Converts an entire PDF document by processing pages in batches.
        Returns (full_document_markdown, summary_telemetry).
        """
        t0 = time.time()
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found: {pdf_path}")

        doc = pymupdf.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()

        if max_pages and max_pages > 0:
            total_pages = min(total_pages, max_pages)

        page_list = list(range(1, total_pages + 1))
        batches = [page_list[i : i + self.batch_size] for i in range(0, len(page_list), self.batch_size)]

        document_markdown_parts: List[str] = []
        batch_telemetries: List[Dict[str, Any]] = []

        total_input_tokens = 0
        total_output_tokens = 0

        for b_idx, batch_pages in enumerate(batches, start=1):
            if progress_callback:
                progress_callback(b_idx, len(batches), batch_pages)

            batch_md, telem = self.convert_pdf_page_batch(pdf_path, batch_pages)
            document_markdown_parts.append(batch_md)
            batch_telemetries.append(telem)

            total_input_tokens += telem.get("input_tokens", 0)
            total_output_tokens += telem.get("output_tokens", 0)

        full_markdown = "\n\n".join(document_markdown_parts).strip()
        total_duration = time.time() - t0

        summary = {
            "pdf_path": str(pdf_path),
            "filename": pdf_path.name,
            "total_pages": total_pages,
            "batches_count": len(batches),
            "total_duration_sec": round(total_duration, 2),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "batch_details": batch_telemetries,
        }

        return full_markdown, summary
