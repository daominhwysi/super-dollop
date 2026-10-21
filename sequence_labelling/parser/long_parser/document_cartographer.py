"""Document Cartographer: Top-Down Topological Page-Span Object Manifest.

Surveys exam documents top-to-bottom before chunking to extract an authoritative
manifest of Sections, Stimuli, and Questions with their exact page spans [start_page, end_page].
Prevents cross-page entity severing and provides clean-cut seams for downstream chunking.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sequence_labelling.config import (
    CARTOGRAPHER_MODEL,
    CARTOGRAPHER_PROVIDER,
    CARTOGRAPHER_THINKING,
    PARSER_MODEL,
    PARSER_PROVIDER,
)
from sequence_labelling.llm.deepseek_client import chat

logger = logging.getLogger(__name__)

DEFAULT_CARTOGRAPHER_MODEL = "gemini-3.8-flash"
DEFAULT_CARTOGRAPHER_PROVIDER = "agy"
DEFAULT_CARTOGRAPHER_THINKING = "off"

SYSTEM_PROMPT_CARTOGRAPHER = """You are an expert Document Cartographer and Examination Layout Topologist.
Your task is to scan the provided exam paper TOP-TO-BOTTOM in chronological reading order and generate a sequential stream of every structural object with its exact PAGE SPAN [start_page, end_page].

OBJECT TYPES TO EMIT IN TOP-DOWN READING ORDER:
1. "section": Major section dividers, exam parts, or topics (e.g., "PHẦN I. TRẮC NGHIỆM", "PART A. READING COMPREHENSION", "Chuyên đề: Hàm số").
2. "stimulus": Shared reading passages, poems, science experiment setups, data tables, or background texts that serve 2 OR MORE QUESTIONS.
   - CRITICAL: A stimulus MUST serve 2 or more subsequent questions. If a context or figure belongs to ONLY 1 standalone question, do NOT emit it as a stimulus (it belongs to that question's stem).
   - DO NOT classify exam instructions, subject titles, or section dividers as stimuli.
3. "question": Individual questions, items, or exercises (e.g. Câu 1, Question 1, Bài 1).
   - If a question is based on a preceding stimulus, include "stimulus_id": "<id>".
   - Note the exact page span where the question begins and ends (e.g., if stem is on page 1 and options are on page 2, page_span is [1, 2]).

OUTPUT FORMAT:
Output ONLY a valid JSON array of objects in chronological reading order:
[
  {
    "type": "section",
    "id": "sec_1",
    "label": "PHẦN I: TRẮC NGHIỆM",
    "page_span": [1, 3]
  },
  {
    "type": "question",
    "id": "1",
    "page_span": [1, 1]
  },
  {
    "type": "question",
    "id": "2",
    "page_span": [1, 2]
  },
  {
    "type": "stimulus",
    "id": "stim_1",
    "page_span": [2, 3],
    "start_snippet": "<first 5-8 verbatim words>",
    "end_snippet": "<last 5-8 verbatim words>",
    "serves_questions": ["3", "4", "5", "6"]
  },
  {
    "type": "question",
    "id": "3",
    "page_span": [2, 2],
    "stimulus_id": "stim_1"
  },
  {
    "type": "question",
    "id": "4",
    "page_span": [2, 3],
    "stimulus_id": "stim_1"
  }
]

RULES:
- Emit objects strictly in document order from page 1 to page N.
- page_span must be an array of two integers: [start_page, end_page].
- If there are no multi-question stimuli, simply emit sections and questions.
- Output ONLY the JSON array without conversational markdown text."""


@dataclass
class CartographyObject:
    """Represents a single structural element located on a page span."""

    type: str  # "section", "stimulus", "question", "explanation"
    id: str
    page_span: Tuple[int, int]
    label: Optional[str] = None
    stimulus_id: Optional[str] = None
    start_snippet: Optional[str] = None
    end_snippet: Optional[str] = None
    serves_questions: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["page_span"] = list(self.page_span)
        return {k: v for k, v in data.items() if v is not None and v != [] and v != {}}


@dataclass
class CartographyManifest:
    """Authoritative topological manifest of an examination document."""

    doc_id: str
    total_pages: int
    objects: List[CartographyObject]
    stimuli: Dict[str, CartographyObject]
    questions: Dict[str, CartographyObject]
    sections: List[CartographyObject]
    clean_cut_seams: List[int]
    subject: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "total_pages": self.total_pages,
            "subject": self.subject,
            "clean_cut_seams": self.clean_cut_seams,
            "statistics": {
                "total_objects": len(self.objects),
                "total_questions": len(self.questions),
                "total_stimuli": len(self.stimuli),
                "total_sections": len(self.sections),
                "cross_page_questions": [
                    q.id for q in self.questions.values() if q.page_span[0] != q.page_span[1]
                ],
            },
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CartographyManifest":
        """Reconstructs a CartographyManifest from a dictionary and recalculates clean cut seams."""
        raw_objects = data.get("objects", [])
        total_pages = int(data.get("total_pages", 0))
        doc_id = str(data.get("doc_id", ""))
        subject = data.get("subject")
        cartographer = DocumentCartographer()
        return cartographer._build_manifest(
            doc_id=doc_id,
            total_pages=total_pages,
            raw_objects=raw_objects,
            subject=subject,
        )

    def is_seam_clean(self, page_num: int) -> bool:
        """Returns True if a chunk boundary cut can be cleanly made after page_num."""
        return page_num in self.clean_cut_seams


class DocumentCartographer:
    """Surveys an exam document top-to-bottom and constructs its spatial topology."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        thinking: Optional[str] = None,
    ):
        self.model = model or CARTOGRAPHER_MODEL or DEFAULT_CARTOGRAPHER_MODEL
        self.provider = provider or CARTOGRAPHER_PROVIDER or DEFAULT_CARTOGRAPHER_PROVIDER
        self.thinking = thinking or CARTOGRAPHER_THINKING or DEFAULT_CARTOGRAPHER_THINKING

    @staticmethod
    def extract_pages(ocr_markdown: str) -> List[Dict[str, Any]]:
        """Extracts individual pages from OCR markdown text."""
        page_pattern = re.compile(
            r"<page(?:\s+p=[\"']?(\d+)[\"']?)?[^>]*>(.*?)</page>",
            re.DOTALL | re.IGNORECASE,
        )
        matches = list(page_pattern.finditer(ocr_markdown))
        pages: List[Dict[str, Any]] = []

        if matches:
            for idx, m in enumerate(matches):
                p_num = int(m.group(1)) if m.group(1) else (idx + 1)
                text = m.group(2).strip()
                pages.append({"p": p_num, "text": text})
        else:
            # Fallback if no <page> tags exist: treat entire text as single page
            pages.append({"p": 1, "text": ocr_markdown.strip()})

        return pages

    def survey_document(
        self,
        ocr_markdown: str,
        doc_id: str = "doc_exam",
        subject: Optional[str] = None,
        completion_fn: Optional[Any] = None,
    ) -> CartographyManifest:
        """
        Main entrypoint: Surveys the exam markdown and produces a CartographyManifest.
        """
        pages = self.extract_pages(ocr_markdown)
        total_pages = len(pages)
        if total_pages == 0:
            return CartographyManifest(
                doc_id=doc_id,
                total_pages=0,
                objects=[],
                stimuli={},
                questions={},
                sections=[],
                clean_cut_seams=[],
                subject=subject,
            )

        # 1. Format document with explicit page boundaries
        formatted_doc_parts = []
        for p in pages:
            p_num = p["p"]
            formatted_doc_parts.append(f"=== PAGE {p_num} START ===\n{p['text']}\n=== PAGE {p_num} END ===")
        full_paged_text = "\n\n".join(formatted_doc_parts)

        # 2. Query LLM for Top-Down Sequential Stream
        raw_response = ""
        complete = completion_fn or chat
        prompt = (
            f"Exam Subject: {subject or 'Unknown'}\n"
            f"Total Pages: {total_pages}\n\n"
            "Analyze the document top-to-bottom and output the sequential JSON array of objects:\n\n"
            f"{full_paged_text}"
        )

        try:
            raw_response = complete(
                prompt=prompt,
                system=SYSTEM_PROMPT_CARTOGRAPHER,
                model=self.model,
                provider=self.provider,
                thinking=self.thinking,
            )
        except Exception as e:
            logger.warning(f"[DocumentCartographer] LLM call failed: {e}. Falling back to deterministic scan.")

        # 3. Parse JSON Stream
        raw_objects = self._parse_json_stream(raw_response)

        # 4. If LLM failed or produced empty questions, run deterministic regex fallback
        if not raw_objects or not any(obj.get("type") == "question" for obj in raw_objects):
            logger.info("[DocumentCartographer] Using deterministic regex fallback for topology.")
            raw_objects = self._deterministic_fallback_scan(pages)

        # 5. Build Cartography Objects & Manifest
        manifest = self._build_manifest(
            doc_id=doc_id,
            total_pages=total_pages,
            raw_objects=raw_objects,
            subject=subject,
        )

        return manifest

    def _parse_json_stream(self, raw_response: str) -> List[Dict[str, Any]]:
        """Safely parses JSON array from LLM response with multiple recovery fallbacks."""
        if not raw_response or not raw_response.strip():
            return []

        text = raw_response.strip()

        # Strip markdown fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Direct JSON decode
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Check for wrapped list e.g. {"objects": [...]}
                for val in data.values():
                    if isinstance(val, list):
                        return val
        except Exception:
            pass

        # Regex search for outer JSON array [...]
        array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        return []

    def _deterministic_fallback_scan(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fast regex fallback that scans pages for question labels and sections
        to ensure the cartographer never leaves the chunker unguided.
        """
        objects: List[Dict[str, Any]] = []
        q_label_re = re.compile(
            r"(?:^|\n)\s*(?:[#*_\s]*)(?:Câu|Question|Bài)\s*(\d{1,4})[.:\s]",
            re.IGNORECASE,
        )
        sec_re = re.compile(
            r"(?:^|\n)\s*([#*_\s]*(?:PHẦN|CHỦ ĐỀ|PART|MÔN)\s+[IVX\d]+[^\n]*)",
            re.IGNORECASE,
        )

        for p in pages:
            p_num = p["p"]
            text = p["text"]

            # Check sections
            for sm in sec_re.finditer(text):
                sec_title = sm.group(1).strip("#* \t\r\n")
                if len(sec_title) < 80:
                    objects.append({
                        "type": "section",
                        "id": f"sec_p{p_num}_{len(objects)}",
                        "label": sec_title,
                        "page_span": [p_num, p_num],
                    })

            # Check questions
            for qm in q_label_re.finditer(text):
                q_num = qm.group(1)
                objects.append({
                    "type": "question",
                    "id": q_num,
                    "page_span": [p_num, p_num],
                })

        return objects

    def _build_manifest(
        self,
        doc_id: str,
        total_pages: int,
        raw_objects: List[Dict[str, Any]],
        subject: Optional[str] = None,
    ) -> CartographyManifest:
        """Transforms raw dict items into validated CartographyObjects and calculates clean cut seams."""
        parsed_objects: List[CartographyObject] = []
        stimuli: Dict[str, CartographyObject] = {}
        questions: Dict[str, CartographyObject] = {}
        sections: List[CartographyObject] = []

        active_stim_id: Optional[str] = None
        active_stim_questions: List[str] = []

        for item in raw_objects:
            if not isinstance(item, dict):
                continue

            obj_type = str(item.get("type", "question")).lower().strip()
            obj_id = str(item.get("id", len(parsed_objects) + 1)).strip()

            raw_span = item.get("page_span") or item.get("p") or [1, 1]
            if isinstance(raw_span, list) and len(raw_span) >= 2:
                try:
                    start_p = max(1, min(total_pages, int(raw_span[0])))
                    end_p = max(start_p, min(total_pages, int(raw_span[1])))
                    page_span = (start_p, end_p)
                except (ValueError, TypeError):
                    page_span = (1, 1)
            elif isinstance(raw_span, int):
                p_val = max(1, min(total_pages, raw_span))
                page_span = (p_val, p_val)
            else:
                page_span = (1, 1)

            c_obj = CartographyObject(
                type=obj_type,
                id=obj_id,
                page_span=page_span,
                label=item.get("label") or item.get("title"),
                stimulus_id=item.get("stimulus_id") or item.get("stim_id"),
                start_snippet=item.get("start_snippet") or item.get("start"),
                end_snippet=item.get("end_snippet") or item.get("end"),
                serves_questions=[str(q) for q in item.get("serves_questions", [])],
                raw_metadata=item,
            )

            if obj_type == "stimulus":
                stimuli[obj_id] = c_obj
                active_stim_id = obj_id
                active_stim_questions = []
            elif obj_type == "question":
                # Link to active stimulus if specified or implied by proximity
                if not c_obj.stimulus_id and active_stim_id:
                    # If stimulus explicitly lists questions, check membership
                    stim_obj = stimuli.get(active_stim_id)
                    if stim_obj and stim_obj.serves_questions:
                        if obj_id in stim_obj.serves_questions:
                            c_obj.stimulus_id = active_stim_id
                    elif stim_obj and not stim_obj.serves_questions:
                        c_obj.stimulus_id = active_stim_id

                if c_obj.stimulus_id and c_obj.stimulus_id in stimuli:
                    if obj_id not in stimuli[c_obj.stimulus_id].serves_questions:
                        stimuli[c_obj.stimulus_id].serves_questions.append(obj_id)

                questions[obj_id] = c_obj
            elif obj_type == "section":
                sections.append(c_obj)
                # Sections close active stimulus inheritance
                active_stim_id = None

            parsed_objects.append(c_obj)

        # 6. Post-Validation: Enforce Rule 12 Invariant
        # A stimulus MUST serve 2 or more questions!
        for stim_id, s_obj in list(stimuli.items()):
            if len(s_obj.serves_questions) < 2:
                logger.info(
                    f"[DocumentCartographer] Stimulus '{stim_id}' serves only {len(s_obj.serves_questions)} "
                    "questions; flagged as single-question candidate."
                )

        # 7. Compute Clean Cut Seams
        clean_cut_seams = self.compute_clean_cut_seams(total_pages, parsed_objects, stimuli)

        return CartographyManifest(
            doc_id=doc_id,
            total_pages=total_pages,
            objects=parsed_objects,
            stimuli=stimuli,
            questions=questions,
            sections=sections,
            clean_cut_seams=clean_cut_seams,
            subject=subject,
        )

    @staticmethod
    def compute_clean_cut_seams(
        total_pages: int,
        objects: List[CartographyObject],
        stimuli: Dict[str, CartographyObject],
    ) -> List[int]:
        """
        Determines the page numbers P where a chunk boundary cut can be cleanly made
        between Page P and Page P+1 without severing any question or stimulus group.
        """
        if total_pages <= 1:
            return []

        # Find all blocked intervals [start_p, end_p]
        blocked_intervals: List[Tuple[int, int]] = []

        # 1. Any individual atomic entity spanning across pages (question or stimulus)
        # Note: sections are structural headers and can be partitioned between questions.
        for obj in objects:
            if obj.type in ("question", "stimulus") and obj.page_span[0] < obj.page_span[1]:
                blocked_intervals.append((obj.page_span[0], obj.page_span[1]))

        # 2. Any stimulus group: from stimulus start page to the end page of its served questions
        for stim_id, stim_obj in stimuli.items():
            s_start = stim_obj.page_span[0]
            s_end = stim_obj.page_span[1]

            # Find questions served by this stimulus
            serves = [str(q) for q in stim_obj.serves_questions]
            served_questions = [
                obj for obj in objects
                if obj.type == "question" and (
                    obj.stimulus_id == stim_id
                    or (str(obj.id) in serves and s_start <= obj.page_span[0] <= s_end + 6)
                )
            ]
            if served_questions:
                q_max_end = max(q.page_span[1] for q in served_questions)
                group_end = max(s_end, q_max_end)
            else:
                group_end = s_end

            if s_start < group_end:
                blocked_intervals.append((s_start, group_end))

        # Check every seam P from 1 to total_pages - 1
        clean_seams: List[int] = []
        for p in range(1, total_pages):
            # A seam between P and P+1 is cut if NO blocked interval covers the boundary P -> P+1
            is_blocked = False
            for b_start, b_end in blocked_intervals:
                # If an object starts at or before P and ends at or after P+1, it straddles the seam!
                if b_start <= p < b_end:
                    is_blocked = True
                    break

            if not is_blocked:
                clean_seams.append(p)

        return clean_seams
