"""
XML Sequence Annotation Engine with Compact Stimulus Anchors.

Adapts proven XML sequence labeling system prompt from annotate_ocr.py:
1. Annotates raw OCR text with inline XML sequence tags and compact stimulus anchors.
2. Preserves conversation prefix for prompt caching.
3. Resolves stimulus anchor boundaries against source text on the application side.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sequence_labelling.config import PARSER_MODEL, PARSER_PROVIDER, PARSER_THINKING
from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.annotator.annotate_ocr import (
    clean_llm_response,
    load_few_shot_examples_xml,
)
from sequence_labelling.parser.parser import parse_spans_into_structured_questions


STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE = """# [System Config]
Role: You are an expert NLP sequence annotator for educational exam papers (TOEIC, SAT, High School National Exams).
Your model architecture supports up to 128,000 output tokens. You are capable of annotating large document chunks completely in a single output stream without truncation, skipping, or conversational refusal disclaimers.

Annotate the raw OCR text with inline XML sequence tags and compact stimulus anchors across the full output window until <|END|>.

## 🏷️ Complete Tag Dictionary:

1. <section>...</section>: Wrap major section/part titles, headers, directions, and subject block titles (e.g. "<section>PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn...</section>", "<section>## PART 5</section>", "<section>## Chủ đề Lịch sử có 17 câu hỏi từ 1 đến 17</section>"). Output full paired tags <section>...</section> containing verbatim text. Do NOT use anchor tags for section titles.
2. <stimulus id="stim_1" start_anchor="..." end_anchor="..." />: For shared reading passages, emails, articles, tables, figures, multi-passage sets, and explicit context prompts (e.g. "Dựa vào thông tin sau đây để giải quyết bài 4, 5...", "Dựa vào thông tin dưới đây để trả lời các câu từ 515-517...").
   CRITICAL DEFINITION & MULTI-QUESTION RULE: A stimulus ONLY applies if it is intimately related to 2 OR MORE QUESTIONS (shared reading passage, dataset, table, or multi-question context). If a context or text block is only related to 1 single standalone question, do NOT tag it as a stimulus—include it inside that question's <stem> instead!
   CRITICAL ARCHITECTURAL CONSTRAINT: <stimulus> MUST NEVER wrap or enclose other system tags (<stem>, <question_label>, <option_label>, <option_text>, <explanation>). Output self-closing anchor tag with id, start_anchor (first 3-10 verbatim words), and end_anchor (last 3-10 verbatim words).
3. <question_label>...</question_label>: Wrap question prefix indicators ONLY (e.g. "**101.**", "**131.**", "101.", "Câu 1.", "Câu 1:"). CRITICAL RULE: <question_label> must ONLY contain the short question number/label itself. NEVER wrap reading passage text, shared contexts, or multi-question instructions inside <question_label>!
4. <stem>...</stem>: Wrap the main text body of a question following the question label.
5. <option_label>...</option_label>: Wrap choice letters/prefixes (A., B., (A)) AND sub-item/sub-question indicators (a), b), c), d)) in essay, constructed-response, or True/False questions.
6. <option_text>...</option_text>: Wrap the textual content of choices or sub-question items following an <option_label>, or statement cells in tabular True/False questions.
7. <explanation>...</explanation>: Wrap reference explanations, answers explanation texts, and solutions for questions.

---

## ⛔ Strict Ground-Truth Annotation Rules:

1. VERBATIM QUESTION & SECTION TEXT (100% RETENTION): Do NOT alter, correct, spell-check, or omit any character, typo, LaTeX expression ($...$, $$...$$), standard math abbreviation ('VT', 'VP', 'đpcm'), or page marker inside question elements (<question_label>, <stem>, <option_label>, <option_text>, <explanation>) or section elements (<section>).
2. ANCHOR TAG EXCEPTION (COMPACT STIMULUS SHORTCUT): Self-closing <stimulus id="..." start_anchor="..." end_anchor="..." /> tags are intentionally compact references ONLY for reading passages, shared texts, tables, and context prompts. For stimuli, output ONLY the self-closing anchor tag with start_anchor (first 3-10 verbatim words) and end_anchor (last 3-10 verbatim words). Do NOT wrap questions or other system tags inside stimulus tags, and do NOT absorb reading passage text into <question_label>!
3. FULL ANNOTATION COVERAGE: All question labels, stems, option labels, option texts, explanations, and section headers in the provided chunk MUST be annotated completely from start to finish. You have a 128k token output window—do NOT truncate or emit conversational refusal text!
4. NO MARKDOWN CODEBLOCKS: Output ONLY the annotated text directly. Do not wrap the output in ```xml codeblocks.
5. END DELIMITER: Append <|END|> at the very end of your output to indicate the annotation is complete.
6. STRICT TARGET BOUNDARY RULE: Annotate ONLY the raw text provided inside the boundary delimiters <<<TARGET_TEXT_START>>> and <<<TARGET_TEXT_END>>>.
7. STIMULUS DISCRIMINATION & MULTI-QUESTION RULE: A <stimulus> tag MUST ONLY be created if the passage/context/data block intimately relates to 2 OR MORE QUESTIONS. If text or a table is associated with only 1 single question, include it directly inside that question's <stem>...</stem>.
8. TABULAR & UNLABELED TRUE/FALSE SUB-QUESTIONS: When sub-questions or True/False statements are presented inside HTML tables (<table>...</table>), Markdown tables, or lists without explicit option labels (such as a), b) or A.), each statement cell or item text to be evaluated MUST still be tagged as <option_text>...</option_text> (e.g., <td><option_text>Statement text...</option_text></td>). Table formatting tags (<table>, <tr>, <th>, <td>), header titles ("Phát biểu", "Đúng", "Sai"), and choice indicators (○, ✓, [ ]) remain un-tagged structure.
9. NO QUESTION_LABEL EXPANSION: <question_label> must start and close strictly around the question indicator (e.g. `<question_label>**Câu 1.**</question_label>`). Never extend <question_label> to encompass reading passages or stems.
10. XML TAG MATCHING & INTEGRITY: Every opened tag (<section>, <question_label>, <stem>, <option_label>, <option_text>, <explanation>) MUST have its exact matching closing tag. NEVER produce mismatched closing tags (e.g. <stem>...</option_text>) or leave tags unclosed. Compact stimulus anchor tags <stimulus ... /> MUST be self-closing.
11. ACCEPTABLE TEXT OMISSION: If extraneous non-question lecture notes or book study guides are omitted while all exam questions, stems, choices, and solutions are fully annotated, this is valid and acceptable.
"""


def _normalize_text_for_matching(text: str) -> Tuple[str, List[int]]:
    norm_chars = []
    index_map = []
    in_space = False

    for orig_idx, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                norm_chars.append(" ")
                index_map.append(orig_idx)
                in_space = True
        else:
            norm_chars.append(ch)
            index_map.append(orig_idx)
            in_space = False

    norm_text = "".join(norm_chars)
    return norm_text, index_map


def find_anchor_position(
    raw_text: str, anchor: str, cursor: int = 0
) -> Tuple[int, int]:
    if not anchor or not raw_text:
        return -1, -1

    anchor_str = anchor.strip()
    if not anchor_str:
        return -1, -1

    pos = raw_text.find(anchor_str, max(0, cursor))
    if pos != -1:
        return pos, pos + len(anchor_str)

    pos = raw_text.find(anchor_str, 0)
    if pos != -1:
        return pos, pos + len(anchor_str)

    norm_text, index_map = _normalize_text_for_matching(raw_text)
    norm_anchor, _ = _normalize_text_for_matching(anchor_str)
    norm_anchor = norm_anchor.strip()

    if not norm_anchor:
        return -1, -1

    norm_pos = norm_text.find(norm_anchor)
    if norm_pos != -1:
        orig_start = index_map[norm_pos]
        norm_end_idx = norm_pos + len(norm_anchor) - 1
        orig_end = index_map[min(norm_end_idx, len(index_map) - 1)] + 1
        return orig_start, orig_end

    if len(anchor_str) > 12:
        short_anchor = anchor_str[:12]
        pos = raw_text.find(short_anchor, max(0, cursor))
        if pos != -1:
            return pos, pos + len(short_anchor)
        pos = raw_text.find(short_anchor, 0)
        if pos != -1:
            return pos, pos + len(short_anchor)

    return -1, -1


def recover_text_from_anchors(
    raw_text: str,
    start_anchor: Optional[str] = None,
    end_anchor: Optional[str] = None,
    cursor: int = 0,
) -> Tuple[str, int, int]:
    if not start_anchor:
        return "", cursor, cursor

    s_str = start_anchor.strip()
    start_pos, start_end = find_anchor_position(raw_text, s_str, cursor)
    if start_pos == -1:
        return s_str, cursor, cursor

    if end_anchor and isinstance(end_anchor, str) and end_anchor.strip():
        e_str = end_anchor.strip()
        end_pos, end_len_pos = find_anchor_position(raw_text, e_str, start_pos)
        if end_pos != -1 and end_len_pos > start_pos:
            return raw_text[start_pos:end_len_pos], start_pos, end_len_pos

    return raw_text[start_pos:start_end], start_pos, start_end


def parse_xml_with_anchors(
    raw_ocr_text: str, tagged_xml: str
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """
    Parses tagged XML string including self-closing attribute anchor tags
    (<stimulus id="..." start_anchor="..." end_anchor="..."/>).
    Returns (spans, stimuli, questions).
    """
    spans: List[Dict[str, Any]] = []
    stimuli: Dict[str, str] = {}
    current_text_pos = 0

    # Clean XML output
    cleaned_xml = clean_llm_response(tagged_xml).replace("<|END|>", "").strip()

    # Regex matching tags including self-closing tags with attributes
    tag_re = re.compile(
        r"<(/)?([a-zA-Z_0-9\-]+)(?:\s+([^/>]*))?(/\s*)?>", re.DOTALL
    )

    stack = []
    allowed_tags = {
        "section",
        "stimulus",
        "question_label",
        "stem",
        "option_label",
        "option_text",
        "explanation",
    }

    raw_chars = []
    pos = 0

    for match in tag_re.finditer(cleaned_xml):
        start, end = match.span()
        text_before = cleaned_xml[pos:start]
        raw_chars.append(text_before)

        is_closing = bool(match.group(1))
        tag_name = match.group(2).lower()
        attr_str = match.group(3) or ""
        is_self_closing = bool(match.group(4))

        if tag_name not in allowed_tags:
            raw_chars.append(match.group(0))
            pos = end
            continue

        # Parse attributes
        params = {}
        if attr_str:
            attr_matches = re.findall(
                r'([a-zA-Z_0-9\-]+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', attr_str
            )
            for k, v1, v2, v3 in attr_matches:
                params[k] = v1 or v2 or v3

        # Anchored stimulus tags only: <stimulus id="..." start_anchor="..." end_anchor="..." />
        if tag_name == "stimulus" and (is_self_closing or "start_anchor" in params):
            start_anchor = params.get("start_anchor")
            end_anchor = params.get("end_anchor")
            recovered_text, s_pos, e_pos = recover_text_from_anchors(
                raw_ocr_text, start_anchor, end_anchor, cursor=current_text_pos
            )
            current_text_pos = max(current_text_pos, e_pos)

            stim_id = params.get("id") or f"stim_{len(stimuli) + 1}"
            stimuli[stim_id] = recovered_text
            spans.append(
                {
                    "start": s_pos,
                    "end": e_pos,
                    "label": "stimulus",
                    "text": recovered_text,
                    "params": {"id": stim_id},
                }
            )
            pos = end
            continue

        # Non-stimulus tags must not be self-closing/anchored (anchored stimulus only)
        if is_self_closing:
            pos = end
            continue

        if not is_closing:
            stack.append(
                {
                    "tag_name": tag_name,
                    "tag_start_idx": len("".join(raw_chars)),
                    "params": params,
                }
            )
        else:
            match_idx = -1
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx]["tag_name"] == tag_name:
                    match_idx = idx
                    break

            if match_idx != -1:
                open_info = stack.pop(match_idx)
                tag_start_idx = open_info["tag_start_idx"]
                tag_end_idx = len("".join(raw_chars))
                span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]

                span_obj = {
                    "start": tag_start_idx,
                    "end": tag_end_idx,
                    "label": tag_name,
                    "text": span_text,
                }
                if open_info["params"]:
                    span_obj["params"] = open_info["params"]

                if tag_name == "stimulus":
                    stim_id = open_info["params"].get("id") or f"stim_{len(stimuli) + 1}"
                    stimuli[stim_id] = span_text
                    span_obj["params"] = {"id": stim_id}

                spans.append(span_obj)

        pos = end

    raw_chars.append(cleaned_xml[pos:])
    full_text = "".join(raw_chars)

    structured_questions, parsed_stimuli = parse_spans_into_structured_questions(
        raw_ocr_text, spans
    )
    stimuli.update(parsed_stimuli)

    return spans, stimuli, structured_questions





class AnchoredXMLLLMExamParser:
    """
    Full XML LLM exam parser using two-pass multi-role execution and anchor recovery.
    """

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None, thinking: Optional[Any] = None):
        self.model = model or PARSER_MODEL
        self.provider = provider or PARSER_PROVIDER
        self.thinking = thinking if thinking is not None else PARSER_THINKING
        script_dir = Path(__file__).resolve().parent
        examples_dir = script_dir.parent.parent / "annotator" / "examples" / "annotator"
        few_shot_xml = load_few_shot_examples_xml(examples_dir) if examples_dir.exists() else ""
        self.system_prompt = STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE.strip() + "\n" + few_shot_xml

    def parse_exam_chunk(
        self,
        raw_ocr_text: str,
        completion_fn: Optional[Any] = None,
        chunk_metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Runs anchor-aware XML annotation on raw OCR text.
        """
        if not raw_ocr_text or not raw_ocr_text.strip():
            return {
                "questions": [],
                "stimuli": {},
                "spans": [],
                "raw_xml": "",
                "method": "empty",
            }

        complete = completion_fn or chat

        metrics_header = ""
        if chunk_metrics:
            est_tokens = chunk_metrics.get("estimated_tokens") or max(50, len(raw_ocr_text.split()))
            words = chunk_metrics.get("words") or len(raw_ocr_text.split())
            page_ids = chunk_metrics.get("page_ids", [])
            page_str = f"Pages {page_ids[0]} to {page_ids[-1]} ({len(page_ids)} pages)" if page_ids else "Single Chunk"
            metrics_header = (
                f"[Chunk Budget & Metrics: ~{est_tokens} estimated tokens, {words} words, {page_str}. "
                "Annotate ALL questions in this chunk completely from start to end without truncation, skipping, or refusal text.]\n\n"
            )

        parser_prompt = (
            f"{metrics_header}"
            "Annotate ONLY the raw OCR text contained between the boundary delimiters below:\n\n"
            "<<<TARGET_TEXT_START>>>\n"
            f"{raw_ocr_text}\n"
            "<<<TARGET_TEXT_END>>>"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": parser_prompt},
        ]

        raw_xml = complete(
            messages=messages,
            model=self.model,
            provider=self.provider,
            thinking=self.thinking,
        )

        final_xml = raw_xml or ""
        spans, stimuli, questions = parse_xml_with_anchors(raw_ocr_text, final_xml)

        return {
            "questions": questions,
            "stimuli": stimuli,
            "spans": spans,
            "raw_xml": final_xml,
            "method": "llm_xml_anchored",
        }
