import re
import time
import uuid
from typing import Dict, Any, List, Optional

from sequence_labelling.config import PARSER_MODEL, PARSER_PROVIDER, PARSER_THINKING
from sequence_labelling.annotator.annotate_ocr import OCRAnnotator
from sequence_labelling.annotator.xml_checker import XMLChecker
from sequence_labelling.parser.parser import parse_spans_into_structured_questions
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import AnchoredXMLLLMExamParser


_ALLOWED_XML_TAGS = {
    "section",
    "stimulus",
    "question",
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "explanation",
}


class ParserAgentWorker:
    """
    Parser Agent Worker utilizing XML Sequence Annotation with Compact Anchors.
    Extracts structured questions and stimuli from raw chunk text.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        thinking: Optional[str] = None,
        annotator_ready: bool = False,
        max_attempts: int = 1,
        **kwargs,
    ):
        self.model = model or PARSER_MODEL
        self.provider = provider or PARSER_PROVIDER
        self.thinking = thinking or PARSER_THINKING
        self.max_attempts = max_attempts
        self.anchored_parser = AnchoredXMLLLMExamParser(
            model=self.model,
            provider=self.provider,
            thinking=self.thinking,
        )
        self.annotator = None
        self.annotator_ready = False

    @staticmethod
    def _normalize_text_for_comparison(text: str) -> str:
        return re.sub(r"[\s\-\*\#]+", "", text or "")

    @staticmethod
    def _is_valid_annotation_xml(raw_ocr_text: str, raw_xml: str) -> bool:
        if not raw_xml:
            return False
        return XMLChecker.is_valid_xml(raw_xml)

    def process_chunk(
        self,
        raw_chunk_text: str,
        chunk_index: int = 0,
        page_ids: Optional[List[int]] = None,
        overlap_page_ids: Optional[List[int]] = None,
        page_offset_ranges: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document chunk into structured questions and stimuli blocks.
        """
        if not raw_chunk_text.strip():
            return {
                "chunk_index": chunk_index,
                "raw_chunk_text": raw_chunk_text,
                "raw_xml": "",
                "spans": [],
                "page_ids": page_ids or [],
                "overlap_page_ids": overlap_page_ids or [],
                "page_offset_ranges": page_offset_ranges or [],
                "parse_status": "empty",
                "parse_diagnostics": {"attempts": 0, "validation_errors": []},
                "questions": [],
                "stimuli": {},
                "method": "empty",
            }

        structured_questions: List[Dict[str, Any]] = []
        stimuli: Dict[str, str] = {}
        spans: List[Dict[str, Any]] = []
        tagged_text = ""
        chunk_request_id = uuid.uuid4().hex[:10]
        validation_errors: List[str] = []
        attempts_used = 1
        method = "llm_xml_anchored"

        # Compute chunk-level metrics for LLM prompt awareness
        chunk_metrics = {
            "estimated_tokens": max(50, len(raw_chunk_text.split())),
            "words": len(raw_chunk_text.split()),
            "page_ids": page_ids or [],
        }

        try:
            anchored_res = self.anchored_parser.parse_exam_chunk(
                raw_chunk_text,
                chunk_metrics=chunk_metrics,
            )
            structured_questions = anchored_res.get("questions") or []
            stimuli = anchored_res.get("stimuli") or {}
            tagged_text = anchored_res.get("raw_xml") or ""
            spans = anchored_res.get("spans") or []
            method = anchored_res.get("method", "llm_xml_anchored")
        except Exception as e:
            validation_errors.append(f"Parsing failed: {e}")
            print(f"[Parser Worker Warning] Parsing error: {e}")
            structured_questions = []

        # Ensure chunk-local IDs are globally unique across chunks.
        scoped_stimuli: Dict[str, str] = {}
        stimulus_id_map: Dict[str, str] = {}

        for old_id, text in stimuli.items():
            if old_id.startswith(f"chunk_{chunk_index}_"):
                new_id = old_id
            else:
                new_id = f"chunk_{chunk_index}_{old_id}"
            scoped_stimuli[new_id] = text
            if old_id != new_id:
                stimulus_id_map[old_id] = new_id

        for idx, q in enumerate(structured_questions):
            base_id = q.get("id") or f"q_{idx + 1}"
            if not str(base_id).startswith(f"chunk_{chunk_index}_"):
                base_id = f"chunk_{chunk_index}_{base_id}"
            q["id"] = base_id

            stim_id = q.get("stimulus_id")
            if stim_id:
                mapped_id = stimulus_id_map.get(stim_id, stim_id)
                if not str(mapped_id).startswith(f"chunk_{chunk_index}_"):
                    mapped_id = f"chunk_{chunk_index}_{mapped_id}"
                q["stimulus_id"] = mapped_id
                q["stimulus_text"] = scoped_stimuli.get(mapped_id, q.get("stimulus_text", ""))

            q["chunk_index"] = chunk_index

        # Recover untagged questions via SequenceValidatorAgent if any questions were found
        recovered_count = 0
        repaired_questions = structured_questions
        if structured_questions:
            from sequence_labelling.parser.long_parser.validator_agent import SequenceValidatorAgent
            validator = SequenceValidatorAgent()

            active_stim_id = list(scoped_stimuli.keys())[-1] if scoped_stimuli else None
            repaired_questions, recovered_count = validator.repair_chunk_questions(
                raw_chunk_text=raw_chunk_text,
                parsed_questions=structured_questions,
                chunk_index=chunk_index,
                active_stimulus_id=active_stim_id
            )

            if recovered_count > 0:
                method = f"{method}+validator_repaired({recovered_count})"

        return {
            "chunk_index": chunk_index,
            "raw_chunk_text": raw_chunk_text,
            "spans": spans,
            "page_ids": page_ids or [],
            "overlap_page_ids": overlap_page_ids or [],
            "page_offset_ranges": page_offset_ranges or [],
            "parse_status": "ok" if not validation_errors else "error",
            "parse_diagnostics": {
                "attempts": attempts_used,
                "validation_errors": validation_errors,
                "request_id": chunk_request_id,
            },
            "questions": repaired_questions,
            "stimuli": scoped_stimuli,
            "spans_count": len(spans),
            "recovered_count": recovered_count,
            "method": method,
            "raw_xml": tagged_text
        }
