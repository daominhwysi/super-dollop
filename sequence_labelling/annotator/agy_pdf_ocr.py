"""
agy_pdf_ocr.py — Autonomous End-to-End PDF OCR Converter powered by Google Antigravity (agy).

Combines:
1. High-resolution page rendering (150 DPI) with OpenCV/PyMuPDF.
2. Direct figure detection via AGY Vision model (eliminating ONNX/RF-DETR dependencies).
3. Tier 1 (Typographic Archaeologist) & Tier 2 (Style Blueprinting) prompt contracts
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

logger = logging.getLogger(__name__)

from dataclasses import asdict, dataclass
import random

# Specialized OCR System Prompt for AGY with Authentic Layout Diversity & Direct Figure Detection
AGY_DIVERSE_RAW_OCR_BASE_PROMPT = """You are an expert Vietnamese examination document OCR and Layout Transcription AI.
Transcribe the provided Vietnamese examination PDF page images into clean, faithful, and richly structured Markdown that mirrors the exact typographic design and spatial layout of each individual exam.

CRITICAL INVARIANTS:
1. STRICT RETENTION OF CONTENT & VERBATIM FIDELITY:
   - 100% of educational text, numbers, calculations, question choices, and math expressions ($...$, $$...$$) must be faithfully transcribed.
   - Never skip questions, choices, stems, figures, or reading passages.
   - Transcribe all questions, stems, options, and metadata as authentic document.
3. SINGLE-COLUMN READING ORDER:
   - If a page has two columns, convert and merge them into a single-column linear flow following logical reading order (Column 1 top-to-bottom, then Column 2 top-to-bottom).
4. PAGE ENVELOPE & METADATA:
   - Wrap each individual page in <page> ... </page>.
   - Append page metadata at the end of each page:
     <page_metadata>
     {"p": <page_num>, "w": <page_width_pts>, "h": <page_height_pts>}
     </page_metadata>
5. DIRECT FIGURE DETECTION:
   - Identify every geometric figure, coordinate plot, function graph, or diagram.
   - For each figure, emit:
     <figure id="fig-p{page}-{idx}" bbox="x1,y1,x2,y2" description="{concise description}" />
   - Coordinates (x1,y1,x2,y2) must be integer pixel bounding boxes on the rendered page image.
6. MATHEMATICAL FORMULAS:
   - Convert all mathematical formulas and expressions into standard LaTeX ($...$ inline, $$...$$ block).
"""

# Backward compatibility alias
AGY_DIVERSE_RAW_OCR_SYSTEM_PROMPT = AGY_DIVERSE_RAW_OCR_BASE_PROMPT


@dataclass
class DynamicStyleProfile:
    """
    20-Axis Combinatorial Style Profile for Randomized OCR Generation.
    Every axis controls typographic styling, delimiters, or spatial structure
    while keeping educational content 100% identical.
    """
    question_prefix_format: str = "hash_câu"          # hash_câu, bold_câu, dot_câu, colon_câu, bài, question
    stem_prefix_style: str = "plain"                  # plain, blockquote, italic, bold_lead
    stem_heading_level: str = "none"                  # none, h4, h3
    option_layout: str = "inline_horizontal"          # inline_horizontal, two_column_2x2, vertical_list, table_grid
    choice_letter_format: str = "letter_dot"          # letter_dot (A.), bold_letter_dot (**A.**), parens ((A)), brackets ([A]), slash (A/)
    choice_spacing: str = "double_space"              # double_space, tab_spaced, newline_separated
    table_format: str = "html_table"                  # html_table, markdown_pipe
    spatial_density: str = "standard"                 # standard, ultra_compact, relaxed
    stimulus_container: str = "blockquote"            # blockquote, h3_section, plain_passage, boxed
    subquestion_format: str = "letter_paren"          # letter_paren (a)), number_dot (1.1), roman ((i))
    score_point_tagging: str = "inline_parens"        # inline_parens ((1,0 điểm)), brackets ([1.0đ]), none
    figure_caption_position: str = "below"            # below, above, inline
    exam_header_style: str = "compact_meta"           # compact_meta, h1_banner, blockquote_header, none
    explanation_style: str = "bold_prefix"            # bold_prefix (**Lời giải:**), blockquote, plain
    section_delimiter: str = "h2_heading"             # h2_heading (## PHẦN I), bold_rule, horizontal_bar
    math_display_mode: str = "standard_block"         # standard_block ($$...$$), inline_everywhere ($...$)
    bullet_list_symbol: str = "dash"                  # dash (-), asterisk (*), bullet (•)
    true_false_choice_marker: str = "paren_tf"        # paren_tf (a) Đúng b) Sai), tf_table, tf_inline
    blank_fill_syntax: str = "underscore"             # underscore (______), bracket_dots ([...]), blank_box
    zero_newline_mode: bool = False                   # True = all choices on a single continuous line

    @classmethod
    def sample_random(cls, seed: Optional[int] = None) -> DynamicStyleProfile:
        """Sample a completely random style profile from the 20-axis combinatorial space."""
        rng = random.Random(seed)

        question_prefixes = ["hash_câu", "bold_câu", "dot_câu", "colon_câu", "bài", "question"]
        stem_prefixes = ["plain", "blockquote", "italic", "bold_lead"]
        stem_headings = ["none", "none", "h4"]
        option_layouts = ["inline_horizontal", "two_column_2x2", "vertical_list", "table_grid"]
        choice_letters = ["letter_dot", "bold_letter_dot", "parens", "brackets", "slash"]
        choice_spacings = ["double_space", "tab_spaced", "newline_separated"]
        table_formats = ["html_table", "markdown_pipe"]
        spatial_densities = ["standard", "relaxed", "ultra_compact"]
        stimulus_containers = ["blockquote", "h3_section", "plain_passage", "boxed"]
        subquestion_formats = ["letter_paren", "number_dot", "roman"]
        score_point_taggings = ["inline_parens", "brackets", "none"]
        figure_caption_positions = ["below", "above", "inline"]
        exam_header_styles = ["compact_meta", "h1_banner", "blockquote_header", "none"]
        explanation_styles = ["bold_prefix", "blockquote", "plain"]
        section_delimiters = ["h2_heading", "bold_rule", "horizontal_bar"]
        math_display_modes = ["standard_block", "inline_everywhere"]
        bullet_list_symbols = ["dash", "asterisk", "bullet"]
        true_false_choice_markers = ["paren_tf", "tf_table", "tf_inline"]
        blank_fill_syntaxes = ["underscore", "bracket_dots", "blank_box"]
        zero_newline = rng.random() < 0.20  # 20% probability of continuous non-newline layout

        return cls(
            question_prefix_format=rng.choice(question_prefixes),
            stem_prefix_style=rng.choice(stem_prefixes),
            stem_heading_level=rng.choice(stem_headings),
            option_layout=rng.choice(option_layouts),
            choice_letter_format=rng.choice(choice_letters),
            choice_spacing=rng.choice(choice_spacings),
            table_format=rng.choice(table_formats),
            spatial_density=rng.choice(spatial_densities),
            stimulus_container=rng.choice(stimulus_containers),
            subquestion_format=rng.choice(subquestion_formats),
            score_point_tagging=rng.choice(score_point_taggings),
            figure_caption_position=rng.choice(figure_caption_positions),
            exam_header_style=rng.choice(exam_header_styles),
            explanation_style=rng.choice(explanation_styles),
            section_delimiter=rng.choice(section_delimiters),
            math_display_mode=rng.choice(math_display_modes),
            bullet_list_symbol=rng.choice(bullet_list_symbols),
            true_false_choice_marker=rng.choice(true_false_choice_markers),
            blank_fill_syntax=rng.choice(blank_fill_syntaxes),
            zero_newline_mode=zero_newline,
        )

    def to_prompt_directives(self) -> List[str]:
        """Converts style choices into explicit, actionable prompt directives."""
        directives: List[str] = []

        # Axis 1: Question Prefix
        prefix_map = {
            "hash_câu": 'Render question labels as Markdown H2 headings: "## Câu {n}" or "## Câu {n}."',
            "bold_câu": 'Render question labels in bold: "**Câu {n}:**" or "**Câu {n}.**"',
            "dot_câu": 'Render question labels cleanly with dots: "Câu {n}."',
            "colon_câu": 'Render question labels with colon: "Câu {n}:"',
            "bài": 'Render question labels as Bài: "Bài {n}:" or "## Bài {n}."',
            "question": 'Render question labels in English style: "Question {n}:" or "Question {n}."',
        }
        directives.append(prefix_map.get(self.question_prefix_format, prefix_map["hash_câu"]))

        # Axis 2 & 3: Stem Prefix & Heading
        if self.stem_heading_level == "h4":
            directives.append('Render question stem text under an H4 header: "#### {stem_text}"')
        elif self.stem_prefix_style == "blockquote":
            directives.append('Format question stems as Markdown blockquotes: "> {stem_text}"')
        elif self.stem_prefix_style == "italic":
            directives.append('Format question stems in italics: "*{stem_text}*"')
        elif self.stem_prefix_style == "bold_lead":
            directives.append('Format the first phrase of question stems in bold.')

        # Axis 4 & 20: Option Layout & Zero Newline Mode
        if self.zero_newline_mode:
            directives.append('Render choices immediately contiguous without extra line breaks: e.g. "A. opt1 B. opt2 C. opt3 D. opt4"')
        else:
            layout_map = {
                "inline_horizontal": 'Arrange 4 choices horizontally on the same line: "A. ...   B. ...   C. ...   D. ..."',
                "two_column_2x2": 'Arrange choices in a 2x2 grid (two choices on line 1, two on line 2).',
                "vertical_list": 'Arrange each choice on its own individual line (vertical list).',
                "table_grid": 'Place options inside a structured choice grid or table.',
            }
            directives.append(layout_map.get(self.option_layout, layout_map["inline_horizontal"]))

        # Axis 5: Choice Letter Format
        choice_letter_map = {
            "letter_dot": 'Format choice markers as "A.", "B.", "C.", "D."',
            "bold_letter_dot": 'Format choice markers in bold: "**A.**", "**B.**", "**C.**", "**D.**"',
            "parens": 'Format choice markers in parentheses: "(A)", "(B)", "(C)", "(D)"',
            "brackets": 'Format choice markers in square brackets: "[A]", "[B]", "[C]", "[D]"',
            "slash": 'Format choice markers with slashes: "A/", "B/", "C/", "D/"',
        }
        directives.append(choice_letter_map.get(self.choice_letter_format, choice_letter_map["letter_dot"]))

        # Axis 7: Table Format
        if self.table_format == "html_table":
            directives.append('Render all tables, matrices, and choice grids as HTML tables: <table><tr><td>...</td></tr></table>')
        else:
            directives.append('Render data tables as standard Markdown pipe tables: | Col1 | Col2 |')

        # Axis 8: Spatial Density
        if self.spatial_density == "ultra_compact":
            directives.append('Use ultra-compact spacing (minimal blank lines between sections, tight text packing).')
        elif self.spatial_density == "relaxed":
            directives.append('Use relaxed vertical spacing with clear double-blank lines between questions.')

        # Axis 9: Stimulus Container
        stimulus_map = {
            "blockquote": 'Wrap reading passages / stimulus texts in Markdown blockquotes: "> ..."',
            "h3_section": 'Mark reading passages with an H3 heading: "### Đoạn văn đọc hiểu"',
            "plain_passage": 'Render reading passages directly as standard narrative text without special boxes.',
            "boxed": 'Enclose reading passages with border lines: "***\\nPassage text\\n***"',
        }
        directives.append(stimulus_map.get(self.stimulus_container, stimulus_map["blockquote"]))

        # Axis 11: Score Point Tagging
        if self.score_point_tagging == "inline_parens":
            directives.append('If point values exist on questions, keep them in parentheses: "(1,0 điểm)".')
        elif self.score_point_tagging == "brackets":
            directives.append('If point values exist, format them in brackets: "[1.0 điểm]".')

        return directives

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RandomizedPromptBuilder:
    """
    Builds Vision OCR system prompts with dynamic style profiles,
    guaranteeing 100% content/LaTeX preservation while randomizing layout.
    """

    @staticmethod
    def build_prompt_directives(profile: Optional[DynamicStyleProfile] = None) -> str:
        if profile is None:
            profile = DynamicStyleProfile.sample_random()

        directives = profile.to_prompt_directives()
        formatted_directives = "\n".join(f"   - {d}" for d in directives)

        return f"""DYNAMIC STYLE DIRECTIVES FOR THIS DOCUMENT (AUTHENTIC DIVERSITY):
Follow these specific typographic conventions for this run while keeping all content 100% faithful:
{formatted_directives}

CORE PRESERVATION INVARIANT:
- Keep 100% of educational text, numbers, calculations, and math expressions ($...$) completely faithful to the page image.
- STRICT DISALLOWANCE OF SEQUENCE LABELLING TAGS: Do NOT output sequence labelling XML tags.
- ONLY mutate typographic layout, spacing, and styling delimiters according to the style instructions above."""

    @staticmethod
    def build_system_prompt(profile: Optional[DynamicStyleProfile] = None) -> str:
        directives_block = RandomizedPromptBuilder.build_prompt_directives(profile)
        return AGY_DIVERSE_RAW_OCR_BASE_PROMPT + "\n\n" + directives_block


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
        min_chars: Optional[int] = None,
        min_words: Optional[int] = None,
        randomize_style: bool = True,
        style_seed: Optional[int] = None,
        account: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.thinking = thinking
        self.dpi = dpi
        self.batch_size = max(1, batch_size)
        self.randomize_style = randomize_style
        self.style_seed = style_seed
        self.account = account

    def _build_page_prompt(
        self,
        page_num: int,
        width_pts: float,
        height_pts: float,
    ) -> str:
        """Constructs prompt instructions for transcribing a page image."""
        lines = [
            f"=== DOCUMENT PAGE {page_num} (File: page_{page_num}.jpg, Dimensions: {int(width_pts)}x{int(height_pts)} pts) ===",
            "DIRECT FIGURE DETECTION INSTRUCTION:",
            "Detect any geometric figures, function graphs, coordinate planes, or tables.",
            f'For figures, emit: <figure id="fig-p{page_num}-<idx>" bbox="x1,y1,x2,y2" description="..." />',
            "",
            "RULES FOR THIS PAGE:",
            "1. Transcribe the natural text, questions, choices, and layout faithfully from the page image. Do NOT use sequence labelling XML tags (<question_label>, <stem>, <option_label>, <other>).",
            "2. Render inline math in $...$ and display math in $$...$$.",
            f'3. Wrap page content in <page> ... </page> and end with <page_metadata>{{"p": {page_num}, \"w\": {int(width_pts)}, \"h\": {int(height_pts)}}}</page_metadata>.',
        ]
        return "\n".join(lines).replace("\x00", "")

    def convert_pdf_page_batch(
        self,
        pdf_path: Union[str, Path],
        target_pages: List[int],
        style_profile: Optional[DynamicStyleProfile] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Converts a small batch of pages (e.g. 2 pages) in a single vision call.
        Returns (clean_markdown, telemetry).
        """
        t0 = time.time()
        doc = pymupdf.open(str(pdf_path))

        page_images: List[Tuple[str, bytes]] = []
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

            # Build prompt
            prompt_str = self._build_page_prompt(
                page_num=p_num,
                width_pts=rect.width,
                height_pts=rect.height,
            )
            page_prompts.append(prompt_str)

        doc.close()

        if not page_images:
            return "", {"error": "No valid pages in batch"}

        combined_prompt = "\n\n".join(page_prompts)

        # Prepare dynamic style directives at the TAIL of the user prompt to maximize KV-cache prefix hits
        profile_dict = None
        style_directives = ""
        if style_profile is not None:
            profile_dict = style_profile.to_dict()
            style_directives = RandomizedPromptBuilder.build_prompt_directives(style_profile)
        elif self.randomize_style:
            style_profile = DynamicStyleProfile.sample_random(seed=self.style_seed)
            profile_dict = style_profile.to_dict()
            style_directives = RandomizedPromptBuilder.build_prompt_directives(style_profile)

        if style_directives:
            combined_prompt = f"{combined_prompt}\n\n{style_directives}"

        # 100% static system prompt across all calls ensuring optimal KV prefix caching
        system_prompt = AGY_DIVERSE_RAW_OCR_BASE_PROMPT

        # Call agy Vision with Tier 1 + Tier 2 Diverse Raw OCR System Prompt
        t_call_start = time.time()
        raw_response, conv_id, usage = chat_with_agy(
            prompt=combined_prompt,
            system=system_prompt,
            model=self.model,
            thinking=self.thinking,
            images=page_images,
            timeout=300,
            caller="agy_pdf_ocr",
            auto_log=True,
            account=self.account,
        )
        call_duration = time.time() - t_call_start

        cleaned_md = prune_code_blocks(prune_think_tags(raw_response))
        total_duration = time.time() - t0

        telemetry = {
            "target_pages": target_pages,
            "duration_sec": round(total_duration, 2),
            "call_duration_sec": round(call_duration, 2),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "style_profile": profile_dict,
        }

        return cleaned_md, telemetry

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

        # Sample document-level style profile once per document for typographic consistency and maximum cache reuse
        doc_style_profile: Optional[DynamicStyleProfile] = None
        if self.randomize_style:
            doc_style_profile = DynamicStyleProfile.sample_random(seed=self.style_seed)

        for b_idx, batch_pages in enumerate(batches, start=1):
            if progress_callback:
                progress_callback(b_idx, len(batches), batch_pages)

            batch_attempts = 2
            batch_md = ""
            telem = {}
            for b_attempt in range(batch_attempts):
                try:
                    batch_md, telem = self.convert_pdf_page_batch(pdf_path, batch_pages, style_profile=doc_style_profile)
                    break
                except Exception as exc:
                    if "Agy5HourCapError" in type(exc).__name__:
                        raise exc
                    if b_attempt < batch_attempts - 1:
                        logger.warning(f"Page batch {b_idx} failed on attempt {b_attempt + 1}: {exc}. Retrying batch...")
                        time.sleep(10)
                        continue
                    raise exc

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
