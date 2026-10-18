"""
Digital Chunk-Injection OCR Pipeline.

Extracts verified digital text chunks from digital exam PDFs, injects them into
the vision prompt, allows the vision LLM to reference chunks directly via [CHUNK_k]
instead of slowly rewriting narrative text, and deterministically expands the references
back into authentic Unicode Vietnamese text.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import fitz  # PyMuPDF
import numpy as np

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


class DigitalChunkOCR:
    """
    Accelerated Digital OCR via Chunk Injection.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        thinking: Optional[str] = "low",
        dpi: int = 150,
        min_chars: int = 5,
        min_words: int = 2,
    ):
        self.model = model
        self.thinking = thinking
        self.dpi = dpi
        self.extractor = DigitalChunkExtractor(min_chars=min_chars, min_words=min_words)

    def build_page_chunk_prompt(
        self,
        page_num: int,
        chunks: Sequence[ChunkItem],
        figure_info: str = "RF-DETR figure inventory: none.",
    ) -> str:
        """
        Builds the prompt section for a single page with its chunk inventory.
        """
        lines = [
            f"--- Document Page {page_num} (File: page_{page_num}.jpg) ---",
            figure_info,
            "",
            f"Verified Digital Text Chunks for Page {page_num}:",
        ]

        if chunks:
            for c in chunks:
                lines.append(c.to_prompt_line())
        else:
            lines.append("(No digital chunks extracted for this page)")

        lines.extend([
            "",
            "FAST OCR ACCELERATION RULES:",
            "1. Whenever you output narrative Vietnamese text, section headers, or question prompts matching any verified chunk, emit [CHUNK_id] instead of retyping the text.",
            "2. You may freely wrap [CHUNK_id] in Markdown styling (e.g. # [CHUNK_1], **[CHUNK_5]**).",
            "3. Convert all mathematical formulas, variables, and equations into standard LaTeX ($...$ inline, $$...$$ block).",
            "4. Convert all tables strictly into standard HTML <table>...</table> elements.",
            "5. You may emit regular text directly whenever convenient or for short connective words.",
            "6. Wrap the page content in <page> ... </page> and end with <page_metadata> ... </page_metadata>.",
        ])

        return "\n".join(lines)

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Processes a digital PDF using chunk-injection OCR.
        Returns (expanded_markdown, telemetry).
        """
        t0 = time.time()
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_doc_pages = len(doc)
        target_pages = [p for p in (pages or list(range(1, total_doc_pages + 1))) if 1 <= p <= total_doc_pages]

        t_extract_start = time.time()
        page_images: List[Tuple[str, bytes]] = []
        page_chunks: Dict[int, List[ChunkItem]] = {}
        all_chunks: List[ChunkItem] = []

        for p_num in target_pages:
            page = doc[p_num - 1]

            # Render image
            pix = page.get_pixmap(dpi=self.dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            _, img_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            page_images.append((f"page_{p_num}.jpg", img_bytes.tobytes()))

            # Extract chunks
            chunks = self.extractor.extract_page_chunks(page, p_num)
            page_chunks[p_num] = chunks
            all_chunks.extend(chunks)

        doc.close()
        extract_duration = time.time() - t_extract_start

        # Build combined prompt
        page_prompts = []
        for p_num in target_pages:
            p_prompt = self.build_page_chunk_prompt(
                page_num=p_num,
                chunks=page_chunks[p_num],
                figure_info="RF-DETR figure inventory: none.",
            )
            page_prompts.append(p_prompt)

        batch_user_prompt = "\n\n".join(page_prompts)

        # Call Vision LLM
        t_vision_start = time.time()
        raw_llm_response, _, usage = chat_with_agy(
            prompt=batch_user_prompt,
            system=SYSTEM_PROMPT_LONG_CONTEXT,
            model=self.model,
            thinking=self.thinking,
            images=page_images,
            timeout=300,
            caller="digital_chunk_ocr",
        )
        vision_duration = time.time() - t_vision_start

        # Clean raw response
        cleaned_raw = prune_code_blocks(prune_think_tags(raw_llm_response))

        # Deterministically expand [CHUNK_id] tokens
        t_expand_start = time.time()
        expanded_markdown, replacements_made = expand_chunks_in_markdown(cleaned_raw, all_chunks)
        expand_duration = time.time() - t_expand_start

        total_duration = time.time() - t0

        telemetry = {
            "total_duration_sec": total_duration,
            "extract_duration_sec": extract_duration,
            "vision_duration_sec": vision_duration,
            "expand_duration_sec": expand_duration,
            "chunks_available": len(all_chunks),
            "chunks_replaced": replacements_made,
            "chunk_usage_rate": (replacements_made / max(1, len(all_chunks))),
            "raw_char_count": len(cleaned_raw),
            "expanded_char_count": len(expanded_markdown),
            "token_compression_ratio": max(0.0, 1.0 - (len(cleaned_raw) / max(1, len(expanded_markdown)))),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        return expanded_markdown, telemetry
