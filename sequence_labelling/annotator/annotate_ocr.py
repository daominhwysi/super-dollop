import os
import sys
import json
import re
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Set up path to import src modules
script_dir = Path(__file__).resolve().parent
workspace_dir = script_dir.parent.parent
sys.path.append(str(workspace_dir))

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_THINKING,
    PARSER_MAX_TOKENS,
    get_provider_base_url,
)
from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.annotator.xml_checker import XMLChecker

try:
    from src.token_tracker import log_response
except ImportError:
    # Fallback log_response if src.token_tracker is absent
    def log_response(response, model=""):
        pass

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(dotenv_path=workspace_dir / ".env")

# ── Sequence Labelling Tag Definitions ────────────────────────────────────────

BASE_TAGS = [
    "question",
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "stimulus",
    "section",
    "explanation",
]


def get_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generates BIO tag to ID mapping and reverse ID to tag mapping."""
    tag_to_id = {"O": 0}
    for tag in BASE_TAGS:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag


TAG_TO_ID, ID_TO_TAG = get_tag_mappings()

SYSTEM_PROMPT = """# [System Config]
Role: You are an expert NLP sequence annotator for educational exam papers (TOEIC, SAT, High School Exams).
Your task is to annotate raw OCR text of exam papers by wrapping specific components in precise inline XML tags for sequence labelling.

## 🏷️ Tag Dictionary:

1. <section>...</section>: Wrap major section/part titles, headers, directions, and subject block titles (e.g. "<section>PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn...</section>", "<section>## PART 5</section>", "<section>## Chủ đề Địa lí có 17 câu hỏi từ 501 đến 517</section>"). Output full paired tags <section>...</section> containing verbatim text. Do NOT use anchor tags for section titles.
2. <stimulus id="stim_1" start_anchor="..." end_anchor="..." />: For shared reading passages, emails, articles, tables, figures, multi-passage sets, and explicit context prompts (e.g. "Dựa vào thông tin sau đây để giải quyết bài 4, 5...", "Dựa vào thông tin dưới đây để trả lời các câu từ 515-517..."). CRITICAL DEFINITION & MULTI-QUESTION RULE: A stimulus ONLY applies if it is intimately related to 2 OR MORE QUESTIONS (shared reading passage, dataset, table, or multi-question context). If a context or text block is only related to 1 single standalone question, do NOT tag it as a stimulus—include it inside that question's <stem> instead! A stimulus is a CRUCIAL, essential shared content block without which those 2+ questions CANNOT be answered. Output self-closing anchor tag with id, start_anchor (first 3-10 verbatim words), and end_anchor (last 3-10 verbatim words).
3. <question_label>...</question_label>: Wrap question prefix indicators (e.g. "**101.**", "**131.**", "101.", "Câu 1:").
4. <stem>...</stem>: Wrap the main text body of a question following the question label.
5. <option_label>...</option_label>: Wrap choice letters/prefixes and sub-item/sub-question indicators (e.g. "(A)", "(B)", "(C)", "(D)", "A.", "B.", "a)", "b)", "c)", "d)", "a.", "b."). ESSAY & SUB-QUESTION RULE: In essay, long-answer, constructed-response, or multi-part questions where sub-items like "a)", "b)", "c)", "d)" appear without multiple-choice options to select, wrap the sub-item labels "a)", "b)" in <option_label>...</option_label> and their corresponding body text in <option_text>...</option_text>. Never absorb sub-item labels "a)", "b)" into <stem>!
6. <option_text>...</option_text>: Wrap the textual content of choices or sub-question items following an <option_label>.
7. <explanation>...</explanation>: Wrap reference explanations, answers explanation texts, and solutions for questions.
8. <figure id="fig_N" description="..." bbox="x1,y1,x2,y2" />: This is an immutable figure placeholder produced by the vision OCR stage and deterministic box projector. The bbox is in original rendered-page pixel coordinates using xyxy order. Preserve the complete self-closing tag, all three attributes, and its position exactly. It is source text, not a sequence-labelling tag.

---

## ⛔ Strict Rules:

1. **VERBATIM QUESTION & SECTION TEXT (NO TEXT MODIFICATION):** Do NOT alter, correct, spell-check, or omit any character, typo, LaTeX expression ($...$ or $$...$$), or page marker (`<|page|>Page X`) inside question elements (<question_label>, <stem>, <option_label>, <option_text>, <explanation>) or section elements (<section>). Preserve 100% verbatim input text layout.
2. **ANCHOR TAG EXCEPTION (COMPACT STIMULUS SHORTCUT):** Self-closing `<stimulus id="..." start_anchor="..." end_anchor="..." />` tags are intentionally compact references ONLY for reading passages, shared texts, tables, and context prompts. For stimuli, output ONLY the self-closing anchor tag with `start_anchor` (first 3-10 verbatim words) and `end_anchor` (last 3-10 verbatim words). Do NOT use anchor tags for `<section>`; section headers MUST always be wrapped in full paired `<section>...</section>` tags.
3. **FULL ANNOTATION COVERAGE:** Output the ENTIRE input text from start to end. ALL components in the input text MUST be annotated using their appropriate XML tags from the Tag Dictionary (<section>, <stimulus>, <question_label>, <stem>, <option_label>, <option_text>, <explanation>). You MUST annotate every question, stem, option label, and option text without exception—do NOT skip annotating questions or options when a stimulus is present, and do NOT leave text untagged!
4. **NO MARKDOWN CODEBLOCKS:** Output ONLY the annotated text directly. Do not wrap the output in ```xml codeblocks.
5. **CONCISE THINKING TRACE:** Use `<think>` block for concise reasoning (< 300 words) highlighting layout, edge cases, and verification.
6. **END DELIMITER:** Append `<|END|>` at the very end of your output to indicate the annotation is complete.
7. **STRICT TARGET BOUNDARY RULE:** Annotate ONLY the raw text provided inside the boundary delimiters `<<<TARGET_TEXT_START>>>` and `<<<TARGET_TEXT_END>>>` in the user prompt. DO NOT repeat, re-print, or output any text from previous turns or from the `<example>` blocks. Start outputting directly with the first token of the target text. Stop immediately and append `<|END|>` as soon as you reach the last token of the target text.
8. **SUB-QUESTION & CHOICE LABELS:** Sub-item markers (e.g. "a)", "b)", "c)", "d)") in essay, long-answer, true/false, or structured questions must ALWAYS be tagged as <option_label> (and their body text as <option_text>). Never absorb "a)", "b)" sub-item indicators into <stem>!
9. **FEW-SHOT EXAMPLES NOTICE:** The `<example>` blocks provided at the bottom of this system prompt are for reference and formatting demonstration ONLY. Never repeat, copy, or output text from any example blocks in your response.
10. **MULTI-TURN CONTINUATION RULE:** When requested to continue in a multi-turn conversation ("Continue from the very exact next token..."), resume outputting directly from the exact next token where your previous turn left off. Do NOT repeat or re-print any previously generated content, section headers, or tags from earlier turns. Continue annotating until the end of the input text and append `<|END|>`.
11. **FIGURE IMMUTABILITY:** Copy every `<figure ... />` placeholder character-for-character. Do not wrap only part of it, convert it to paired tags, rewrite its description, renumber its ID, or remove it. It may remain inside a surrounding `<stem>`, `<option_text>`, or `<stimulus>` span when context requires.
12. **STIMULUS DISCRIMINATION & MULTI-QUESTION RULE:** A `<stimulus>` tag MUST ONLY be created if the passage/context/data block intimately relates to 2 OR MORE QUESTIONS (e.g. reading passage for questions 6-10, dataset for questions 515-517, or prompt "Dựa vào thông tin sau đây để giải quyết bài 4, 5..."). If a piece of text or table is associated with only 1 single question, include it directly inside that question's <stem>...</stem> rather than tagging it as a <stimulus>. Never tag generic section headers, subject titles, exam metadata, or question range announcements (e.g. "## Chủ đề Địa lí có 17 câu hỏi từ 501 đến 517", "PHẦN I. TRẮC NGHIỆM", "Môn: Toán") as `<stimulus>`!
13. **TABULAR & UNLABELED TRUE/FALSE SUB-QUESTIONS:** When sub-questions or True/False statements are presented inside HTML tables (`<table>...</table>`), Markdown tables, or lists without explicit option labels (such as `a)`, `b)` or `A.`), each statement cell or item text to be evaluated MUST still be tagged as `<option_text>...</option_text>` (e.g., `<td><option_text>Statement text...</option_text></td>`). Table formatting tags (`<table>`, `<tr>`, `<th>`, `<td>`), header titles ("Phát biểu", "Đúng", "Sai"), and choice indicators (`○`, `✓`, `[ ]`) remain un-tagged structure.
14. **PAGE TAGS & METADATA PRUNING:** Prune and omit `<pages>`, `</pages>`, `<page>`, `</page>`, and `<page_metadata>...</page_metadata>` from the XML output. Do not retain page boundary tags or metadata blocks in the annotated XML. Maintain continuous elements (<stem>, <option_text>, <explanation>, <section>) seamlessly across page breaks without splitting them.
15. **XML TAG MATCHING & INTEGRITY (NO MISMATCHED/UNCLOSED TAGS):** Every opened tag (`<section>`, `<question_label>`, `<stem>`, `<option_label>`, `<option_text>`, `<explanation>`) MUST have its exact matching closing tag (`</section>`, `</question_label>`, `</stem>`, `</option_label>`, `</option_text>`, `</explanation>`). NEVER produce mismatched closing tags (e.g. `<stem>...</option_text>`) or leave opening tags unclosed. Compact stimulus anchor tags `<stimulus ... />` and vision figures `<figure ... />` MUST always be formatted as self-closing tags with `/>`.
"""



def clean_llm_response(text: str) -> str:
    """Removes think tags and markdown code block wrappers from LLM response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
    return text.strip()


def trim_unclosed_annotation_tag(raw_result: str, raw_ocr_text: str) -> str:
    """
    Safely trims unclosed LLM annotation tags (e.g. '<option_label', '</stem') at cutoff boundary.
    Distinguishes LLM-added annotation tags from literal original OCR text (e.g. '$x < y$')
    by checking against allowed schema tags and verifying if the trailing string exists in raw_ocr_text.
    """
    match = re.search(r"(</?([a-zA-Z_0-9]*))$", raw_result)
    if not match:
        return raw_result

    full_match = match.group(1)      # e.g. "<option_label" or "</stem" or "<"
    tag_prefix = match.group(2)      # e.g. "option_label" or "stem" or ""

    # Check if tag_prefix matches any allowed schema tag name or prefix of allowed tag
    is_schema_tag = False
    if tag_prefix == "":
        is_schema_tag = True  # Opening '<'
    else:
        for allowed in BASE_TAGS:
            if allowed.startswith(tag_prefix.lower()):
                is_schema_tag = True
                break

    if not is_schema_tag:
        # Not a prefix of any allowed XML tag -> must be original OCR text (e.g. '$x < y$')
        return raw_result

    # Strip XML tags from consumed output to compute position in raw_ocr_text
    raw_without_trailing = raw_result[:-len(full_match)]
    cleaned_consumed = clean_llm_response(raw_without_trailing)
    cleaned_consumed_no_tags = re.sub(r"</?[a-zA-Z_0-9]+>", "", cleaned_consumed)
    consumed_len = len(cleaned_consumed_no_tags.strip())

    # Look at remaining OCR text starting from current offset
    remaining_ocr_text = raw_ocr_text[consumed_len:].lstrip()

    # If remaining_ocr_text starts with full_match, then full_match is literal text in original OCR!
    if remaining_ocr_text.startswith(full_match):
        return raw_result

    # Otherwise, full_match is an unclosed LLM annotation tag that was cut off mid-generation. Trim it!
    return raw_without_trailing


def _to_int(value: Any) -> Optional[int]:
    """Convert token-like values to int when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _extract_usage_tokens(usage: Any) -> Dict[str, Optional[int]]:
    """Extract usage token fields from dict/object usage payloads."""
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    usage_dict = usage
    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump") and callable(usage.model_dump):
            try:
                dumped = usage.model_dump()
                if isinstance(dumped, dict):
                    usage_dict = dumped
            except Exception:
                usage_dict = usage
        elif hasattr(usage, "dict") and callable(usage.dict):
            try:
                dumped = usage.dict()
                if isinstance(dumped, dict):
                    usage_dict = dumped
            except Exception:
                usage_dict = usage

    if not isinstance(usage_dict, dict):
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": None,
            "total_tokens": None,
            "completion_tokens_details": getattr(usage, "completion_tokens_details", None),
        }

    return {
        "prompt_tokens": _to_int(usage_dict.get("prompt_tokens")),
        "completion_tokens": _to_int(usage_dict.get("completion_tokens")),
        "total_tokens": _to_int(usage_dict.get("total_tokens")),
    }


def parse_xml_annotations(tagged_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses tagged XML string to produce clean raw text and character offset spans.
    Supports XML tags with parameters/attributes (e.g. <question stimulus_id="stim_1">).
    Preserves raw text exact alignment.
    """
    allowed_tags = set(BASE_TAGS)
    raw_chars = []
    spans = []

    tag_pattern = re.compile(r"<(/)?([a-zA-Z_0-9\-]+)(?:\s+([^>]*))?>")
    pos = 0
    tag_stack = []

    for match in tag_pattern.finditer(tagged_text):
        start, end = match.span()
        text_before = tagged_text[pos:start]
        raw_chars.append(text_before)

        is_closing = bool(match.group(1))
        tag_name = match.group(2)
        attr_str = match.group(3) or ""

        if tag_name in allowed_tags:
            if not is_closing:
                params = {}
                if attr_str:
                    attr_matches = re.findall(
                        r'([a-zA-Z_0-9\-]+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', attr_str
                    )
                    for k, v1, v2, v3 in attr_matches:
                        params[k] = v1 or v2 or v3
                tag_stack.append(
                    {
                        "tag_name": tag_name,
                        "tag_start_idx": len("".join(raw_chars)),
                        "params": params,
                        "raw_end": end,
                    }
                )
            else:
                match_idx = -1
                for idx in range(len(tag_stack) - 1, -1, -1):
                    if tag_stack[idx]["tag_name"] == tag_name:
                        match_idx = idx
                        break

                if match_idx != -1:
                    open_info = tag_stack.pop(match_idx)
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

                    if tag_name in ["option_label", "question_label"]:
                        full_current_text = "".join(raw_chars)
                        if (
                            tag_start_idx >= 2
                            and full_current_text[tag_start_idx - 2 : tag_start_idx] == "**"
                            and tagged_text[end : end + 2] == "**"
                        ):
                            tag_start_idx -= 2
                            raw_chars.append("**")
                            tag_end_idx = len("".join(raw_chars))
                            span_text = "".join(raw_chars)[tag_start_idx:tag_end_idx]
                            pos = end + 2
                            span_obj["start"] = tag_start_idx
                            span_obj["end"] = tag_end_idx
                            span_obj["text"] = span_text
                            spans.append(span_obj)
                            continue

                    spans.append(span_obj)
        else:
            raw_chars.append(match.group(0))

        pos = end

    raw_chars.append(tagged_text[pos:])
    raw_text = "".join(raw_chars)

    return raw_text, spans


def tokenize_raw_text(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenizes raw text into words/tokens with exact character offsets.
    Handles Vietnamese unicode, numbers, punctuation, LaTeX expressions ($...$).
    """
    token_pattern = re.compile(
        r"(\$\$.*?\$\$|\$.*?\$|<\|page\|>Page \d+|\w+|[^\w\s])", re.DOTALL
    )

    tokens = []
    offsets = []

    for match in token_pattern.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))

    return tokens, offsets


def align_spans_to_bio_tags(
    raw_text: str,
    tokens: List[str],
    offsets: List[Tuple[int, int]],
    spans: List[Dict[str, Any]],
    tag_to_id: Dict[str, int] = TAG_TO_ID,
) -> Tuple[List[str], List[int]]:
    """
    Aligns character-level spans with token offsets to produce sequence labelling BIO tags & IDs.
    """
    # Clean spans to ignore leading/trailing whitespace inside span boundaries
    clean_spans = []
    for span in spans:
        span_text = span.get("text", "")
        start = span["start"]
        end = span["end"]
        if span_text:
            stripped = span_text.strip()
            if not stripped:
                continue
            leading = len(span_text) - len(span_text.lstrip())
            trailing = len(span_text) - len(span_text.rstrip())
            clean_start = start + leading
            clean_end = end - trailing
        else:
            clean_start = start
            clean_end = end

        clean_spans.append(
            {"start": clean_start, "end": clean_end, "label": span["label"]}
        )

    bio_tags = []
    label_ids = []

    for start_char, end_char in offsets:
        # Find non-whitespace anchor inside the token
        non_space_idx = start_char
        for idx in range(start_char, end_char):
            if idx < len(raw_text) and not raw_text[idx].isspace():
                non_space_idx = idx
                break

        matched_span = None
        for span in clean_spans:
            if span["start"] <= non_space_idx < span["end"]:
                matched_span = span
                break

        if matched_span is None:
            tag = "O"
        else:
            span_label = matched_span["label"]
            if non_space_idx == matched_span["start"]:
                tag = f"B-{span_label}"
            else:
                tag = f"I-{span_label}"

        bio_tags.append(tag)
        label_ids.append(tag_to_id.get(tag, 0))

    return bio_tags, label_ids


def validate_bio_sequence(tags: List[str]) -> bool:
    """
    Validates BIO sequence consistency (e.g. check that I-tag follows matching B-tag or I-tag).
    """
    valid = True
    prev_tag = "O"
    for idx, tag in enumerate(tags):
        if tag.startswith("I-"):
            entity = tag[2:]
            if prev_tag == "O" or (
                prev_tag[2:] != entity
                and not (prev_tag.startswith("B-") or prev_tag.startswith("I-"))
            ):
                print(
                    f"  [Warning] Inconsistent BIO transition at index {idx}: '{prev_tag}' -> '{tag}'"
                )
                valid = False
        prev_tag = tag
    return valid


def export_conll(tokens: List[str], tags: List[str]) -> str:
    """Exports tokens and BIO tags into standard CoNLL 2-column format."""
    lines = []
    for token, tag in zip(tokens, tags):
        lines.append(f"{token}\t{tag}")
    return "\n".join(lines)


def get_client_and_model(
    model_name: Optional[str] = None,
    deepseek_key: Optional[str] = None,
    llm_key: Optional[str] = None,
    xah_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Tuple[Optional[OpenAI], str]:
    """
    Initializes OpenAI client routed to the appropriate provider (Codex, Xah, NVIDIA, Vilao, DeepSeek) based on config.
    """
    target_model = model_name or PARSER_MODEL
    target_provider = (provider or PARSER_PROVIDER or "xah").lower()

    deepseek_key = deepseek_key or os.environ.get("DEEPSEEK_API_KEY")
    nvidia_key = os.environ.get("NVIDIA_API_KEY") or deepseek_key
    llm_key = llm_key or os.environ.get("LLM_API_KEY")
    xah_key = xah_key or os.environ.get("XAH_API_KEY") or os.environ.get("LLM_API_KEY")
    cmd_key = os.environ.get("CMD_API_KEY") or os.environ.get("COMMANDCODE_API_KEY") or deepseek_key

    if target_provider in ["codex", "openai_codex"]:
        print(f"Routing to OpenAI Codex with model: {target_model}")
        return None, target_model
    elif target_provider in ["agy", "antigravity"]:
        print(f"Routing to Antigravity CLI (agy) with model: {target_model}")
        return None, target_model
    elif target_provider == "nvidia":
        print(f"Routing to NVIDIA NIM with model: {target_model}")
        return OpenAI(
            api_key=nvidia_key, base_url="https://integrate.api.nvidia.com/v1"
        ), target_model
    elif target_provider == "xah":
        print(f"Routing to Xah.io API with model: {target_model}")
        return OpenAI(api_key=xah_key, base_url="https://api.xah.io/v1"), target_model
    elif target_provider == "vilao":
        print(f"Routing to Vilao.ai API with model: {target_model}")
        return OpenAI(api_key=llm_key, base_url="https://api.vilao.ai/v1"), target_model
    elif target_provider == "commandcode":
        cmd_base_url = get_provider_base_url("commandcode")
        print(f"Routing to CommandCode API with model: {target_model} at {cmd_base_url}")
        return OpenAI(api_key=cmd_key, base_url=cmd_base_url), target_model
    else:
        print(f"Routing to DeepSeek API with model: {target_model}")
        return OpenAI(
            api_key=deepseek_key, base_url="https://api.deepseek.com"
        ), target_model


def load_few_shot_examples_xml(example_dir: Path, max_pairs: int = 5) -> str:
    """Loads few-shot example pairs from in_X.md and out_X.md files and packs them into XML templates."""
    examples_xml = []
    i = 1
    loaded_count = 0

    while loaded_count < max_pairs:
        in_file = example_dir / f"in_{i}.md"
        out_file = example_dir / f"out_{i}.md"
        if not (in_file.exists() and out_file.exists()):
            break
        try:
            with open(in_file, "r", encoding="utf-8") as f_in, open(out_file, "r", encoding="utf-8") as f_out:
                in_txt = f_in.read().strip()
                out_txt = f_out.read().strip()
                examples_xml.append("<example>\n<input>\n" + in_txt + "\n</input>\n<output>\n" + out_txt + "\n</output>\n</example>")
                loaded_count += 1
        except Exception:
            pass
        i += 1

    continuation_example = (
        "<example>"
        "<!-- Demonstration of multi-turn continuation when output token limit is reached mid-sequence -->\n"
        "<input_turn_1>"
        "Annotate ONLY the raw OCR text:\n"
        "**101.** Question stem text.\n"
        "A. Option 1\n"
        "B. Option 2\n\n"
        "**102.** Second question stem.\n"
        "A. Choice A\n"
        "B. Choice B\n"
        "</input_turn_1>\n"
        "<output_turn_1>\n"
        "<question_label>**101.**</question_label> <stem>Question stem text.</stem>\n"
        "- <option_label>A.</option_label> <option_text>Option 1</option_text>\n"
        "- <option_label>B.</option_label> <option_text>Option 2</option_text>\n\n"
        "<question_label>**102.**</question_label> <"
        "</output_turn_1>\n"
        "<input_turn_2>"
        "Continue from the very exact next token where you left off. Start outputting directly from the next character without repeating any previously generated content or tags.\n"
        "</input_turn_2>"
        "<output_turn_2>"
        "stem>Second question stem.</stem>\n"
        "- <option_label>A.</option_label> <option_text>Choice A</option_text>\n"
        "- <option_label>B.</option_label> <option_text>Choice B</option_text>\n"
        "<|END|>\n"
        "</output_turn_2>"
        "</example>"
    )
    examples_xml.append(continuation_example)

    if examples_xml:
        print(f"  Loaded {len(examples_xml)} few-shot XML example(s) for system prompt.")
        return "\n\n## 💡 Demonstration Examples (FOR REFERENCE ONLY):\n\n" + "\n\n".join(examples_xml)
    return ""

def prune_hallucinated_xml_tail(raw_xml: str, raw_ocr_text: str) -> str:
    """
    Prunes hallucinated questions/passages emitted by the LLM beyond the end of the input OCR text string.
    """
    lines = raw_xml.splitlines()
    pruned_lines = []

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("<!--"):
            pruned_lines.append(line)
            continue

        clean_text = re.sub(r"<[^>]+>", "", line_s).strip()
        q_match = re.search(r"\*\*?(\d{1,4})\.\*\*?", clean_text)
        if q_match:
            q_num = q_match.group(1)
            if f"**{q_num}.**" not in raw_ocr_text and f"{q_num}." not in raw_ocr_text:
                break

        pruned_lines.append(line)

    return "\n".join(pruned_lines)


class OCRAnnotator:
    """
    Sequence Labelling & OCR Entity Annotator using LLM.
    Converts raw OCR text to inline XML, character spans, and BIO sequence-labelled tokens.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        llm_key: Optional[str] = None,
    ):
        self.deepseek_key = deepseek_key or os.environ.get("DEEPSEEK_API_KEY")
        self.llm_key = llm_key or os.environ.get("LLM_API_KEY")
        self.provider = provider or PARSER_PROVIDER or "xah"
        self.client, self.model_name = get_client_and_model(
            model, self.deepseek_key, self.llm_key, provider=provider
        )
        self.few_shot_xml = load_few_shot_examples_xml(script_dir / "examples" / "annotator")
        self.system_prompt = SYSTEM_PROMPT.strip() + "\n" + self.few_shot_xml
        self.thinking_disabled = PARSER_THINKING == "disabled"

    def _make_extra_body(self) -> Optional[Dict[str, Any]]:
        if self.thinking_disabled:
            return {"thinking": {"type": "disabled"}}
        return None

    def annotate_text(
        self,
        raw_ocr_text: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Annotates raw OCR text with LLM and aligns character spans to BIO sequence labels.
        Automatically handles output truncation by continuing annotation in multiple rounds
        using chat conversation history until the <|END|> delimiter is found.
        """
        if not raw_ocr_text.strip():
            raise ValueError("Input OCR text is empty.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Annotate ONLY the raw OCR text contained between the boundary delimiters below:\n\n"
                    "<<<TARGET_TEXT_START>>>\n"
                    f"{raw_ocr_text}\n"
                    "<<<TARGET_TEXT_END>>>"
                ),
            },
        ]

        all_raw_results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        max_iterations = 5

        if self.provider in ["codex", "openai_codex", "agy", "antigravity"]:
            for iteration in range(max_iterations):
                iteration_request_id = (
                    f"{request_id}_iter_{iteration + 1}" if request_id else None
                )
                raw_result = chat(
                    messages=messages,
                    model=self.model_name,
                    provider=self.provider,
                    thinking=PARSER_THINKING,
                    max_tokens=PARSER_MAX_TOKENS,
                )

                if "<|END|>" in raw_result:
                    all_raw_results.append(raw_result)
                    break

                raw_result_trimmed = trim_unclosed_annotation_tag(raw_result, raw_ocr_text)
                all_raw_results.append(raw_result_trimmed)

                messages.append({"role": "assistant", "content": raw_result_trimmed})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue from the very exact next token where you left off. "
                            "Start outputting directly from the next character without repeating any previously generated content or tags."
                        )
                    }
                )
        else:
            for iteration in range(max_iterations):
                iteration_request_id = (
                    f"{request_id}_iter_{iteration + 1}" if request_id else None
                )
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    frequency_penalty=0.2,
                )
                extra_body = self._make_extra_body()
                if extra_body:
                    kwargs["extra_body"] = extra_body

                import time
                start_time = time.time()
                response = self.client.chat.completions.create(**kwargs)
                duration_sec = time.time() - start_time

                if hasattr(response, "usage") and response.usage:
                    usage_counts = _extract_usage_tokens(response.usage)
                    total_prompt_tokens += usage_counts["prompt_tokens"] or 0
                    total_completion_tokens += usage_counts["completion_tokens"] or 0

                try:
                    from sequence_labelling.llm.llm_logger import log_llm_call
                    log_llm_call(
                        messages=messages,
                        response=response,
                        model=self.model_name,
                        provider=self.provider,
                        request_id=iteration_request_id,
                        duration_sec=duration_sec,
                    )
                except Exception as e:
                    print(f"[LLM Logger Warning] Failed to log LLM request: {e}")

                raw_result = response.choices[0].message.content or ""

                if "<|END|>" in raw_result:
                    all_raw_results.append(raw_result)
                    break

                # Safely trim unclosed trailing XML tag if present, preserving literal OCR text
                raw_result_trimmed = trim_unclosed_annotation_tag(raw_result, raw_ocr_text)
                all_raw_results.append(raw_result_trimmed)

                messages.append({"role": "assistant", "content": raw_result_trimmed})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue from the very exact next token where you left off. "
                            "Start outputting directly from the next character without repeating any previously generated content or tags."
                        )
                    }
                )

        full_raw_response = "".join(all_raw_results)
        cleaned_tagged_text = clean_llm_response(full_raw_response)
        cleaned_tagged_text = cleaned_tagged_text.replace("<|END|>", "")

        raw_xml = prune_hallucinated_xml_tail(cleaned_tagged_text, raw_ocr_text)
        parsed_raw_text, spans = parse_xml_annotations(cleaned_tagged_text)

        tokens, offsets = tokenize_raw_text(raw_ocr_text)
        bio_tags, label_ids = align_spans_to_bio_tags(
            raw_ocr_text, tokens, offsets, spans, TAG_TO_ID
        )

        validate_bio_sequence(bio_tags)
        xml_check_res = XMLChecker.check(raw_xml)
        if not xml_check_res.is_valid:
            print(f"  [XML Checker Warning] Issues detected in annotated XML:\n{xml_check_res.summary()}")

        return {
            "raw_text": raw_ocr_text,
            "raw_xml": raw_xml,
            "spans": spans,
            "tokens": tokens,
            "offsets": offsets,
            "tags": bio_tags,
            "labels": label_ids,
            "label_mapping": TAG_TO_ID,
            "annotated": True,
            "xml_is_valid": xml_check_res.is_valid,
            "xml_issues": [str(i) for i in xml_check_res.issues],
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

    def annotate_text_stream(
        self,
        raw_ocr_text: str,
        callback=None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Annotates raw OCR text with LLM using token streaming,
        invoking callback on each stream chunk.
        Automatically handles output truncation by continuing annotation in
        multiple rounds using chat conversation history until <|END|> is found.
        """
        if not raw_ocr_text.strip():
            raise ValueError("Input OCR text is empty.")

        words = raw_ocr_text.split()
        original_input_tokens = max(30, int(len(words) * 1.3))
        estimated_total_tokens = int(original_input_tokens * 1.8)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Annotate ONLY the raw OCR text contained between the boundary delimiters below:\n\n"
                    "<<<TARGET_TEXT_START>>>\n"
                    f"{raw_ocr_text}\n"
                    "<<<TARGET_TEXT_END>>>"
                ),
            },
        ]

        all_raw_results = []
        streamed_token_count = 0
        total_prompt_tokens = 0
        max_iterations = 5

        if self.provider in ["codex", "openai_codex", "agy", "antigravity"]:
            for iteration in range(max_iterations):
                iteration_request_id = (
                    f"{request_id}_iter_{iteration + 1}" if request_id else None
                )
                raw_result = chat(
                    messages=messages,
                    model=self.model_name,
                    provider=self.provider,
                    thinking=PARSER_THINKING,
                    max_tokens=PARSER_MAX_TOKENS,
                )
                chunk_tokens = max(1, len(raw_result.split()))
                streamed_token_count += chunk_tokens
                if callback:
                    callback(streamed_token_count, estimated_total_tokens, raw_result)

                if "<|END|>" in raw_result:
                    all_raw_results.append(raw_result)
                    break

                raw_result_trimmed = trim_unclosed_annotation_tag(raw_result, raw_ocr_text)
                all_raw_results.append(raw_result_trimmed)

                messages.append({"role": "assistant", "content": raw_result_trimmed})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue from the very exact next token where you left off. "
                            "Start outputting directly from the next character without repeating any previously generated content or tags."
                        )
                    }
                )
        else:
            from sequence_labelling.llm.llm_logger import StreamingLLMLogger
            import time

            max_retries = 3
            retry_delay = 2.0

            for iteration in range(max_iterations):
                iteration_request_id = (
                    f"{request_id}_iter_{iteration + 1}" if request_id else None
                )

                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    frequency_penalty=0.2,
                    stream=True,
                )
                extra_body = self._make_extra_body()
                if extra_body:
                    kwargs["extra_body"] = extra_body

                full_chunks = []
                iteration_base_tokens = streamed_token_count

                for attempt in range(max_retries):
                    attempt_request_id = (
                        f"{iteration_request_id}_attempt_{attempt + 1}" if iteration_request_id else None
                    )
                    stream_logger = StreamingLLMLogger(
                        messages=messages,
                        model=self.model_name,
                        provider=self.provider,
                        request_id=attempt_request_id,
                        flush_interval_sec=5.0,
                    )

                    try:
                        response = self.client.chat.completions.create(**kwargs)
                        full_chunks = []
                        current_attempt_tokens = iteration_base_tokens
                        saw_any_chunk = False
                        finish_reason = None
                        for chunk in response:
                            if chunk.choices and len(chunk.choices) > 0:
                                saw_any_chunk = True
                                choice = chunk.choices[0]
                                delta = choice.delta
                                chunk_finish = getattr(choice, "finish_reason", None)
                                if chunk_finish:
                                    finish_reason = chunk_finish
                                content = getattr(delta, "content", None) or ""
                                reasoning = getattr(delta, "reasoning_content", None) or ""
                                usage = getattr(chunk, "usage", None)

                                if usage:
                                    usage_counts = _extract_usage_tokens(usage)
                                    if usage_counts["prompt_tokens"] is not None:
                                        total_prompt_tokens = max(
                                            total_prompt_tokens,
                                            usage_counts["prompt_tokens"],
                                        )

                                if content or reasoning or usage:
                                    stream_logger.append_chunk(
                                        content=content,
                                        reasoning=reasoning,
                                        usage=usage,
                                    )

                                if content:
                                    full_chunks.append(content)
                                    chunk_tokens = max(1, len(content.split()))
                                    current_attempt_tokens += chunk_tokens
                                    if callback:
                                        callback(current_attempt_tokens, estimated_total_tokens, content)

                        if finish_reason in ("length", "max_tokens"):
                            print(f"  [Notice] Stream reached max output token limit (finish_reason='{finish_reason}') on iteration {iteration + 1}. Preserving {len(full_chunks)} content chunks for continuation.")

                        if not full_chunks and saw_any_chunk:
                            raise RuntimeError("LLM streaming response had no content chunks.")
                        if not full_chunks:
                            raise RuntimeError("LLM streaming response was empty (no chunks received).")

                        streamed_token_count = current_attempt_tokens
                        break
                    except Exception as e:
                        # If content chunks were already accumulated before the stream error/limit drop,
                        # preserve them so the continuation loop can resume generation seamlessly.
                        if full_chunks:
                            print(f"  [Notice] Stream interrupted mid-output on attempt {attempt + 1} ({len(full_chunks)} content chunks collected). Preserving output for continuation. Cause: {e}")
                            streamed_token_count = current_attempt_tokens
                            break

                        streamed_token_count = iteration_base_tokens
                        print(f"  [Warning] OCR annotation stream error on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(retry_delay)
                    finally:
                        stream_logger.finalize()

                raw_result = "".join(full_chunks)

                if "<|END|>" in raw_result:
                    all_raw_results.append(raw_result)
                    break

                # Safely trim unclosed trailing XML tag if present, preserving literal OCR text
                raw_result_trimmed = trim_unclosed_annotation_tag(raw_result, raw_ocr_text)
                all_raw_results.append(raw_result_trimmed)

                messages.append({"role": "assistant", "content": raw_result_trimmed})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue from the very exact next token where you left off. "
                            "Start outputting directly from the next character without repeating any previously generated content or tags."
                        )
                    }
                )

        full_raw_response = "".join(all_raw_results)
        cleaned_tagged_text = clean_llm_response(full_raw_response)
        cleaned_tagged_text = cleaned_tagged_text.replace("<|END|>", "")

        raw_xml = prune_hallucinated_xml_tail(cleaned_tagged_text, raw_ocr_text)
        parsed_raw_text, spans = parse_xml_annotations(cleaned_tagged_text)

        tokens, offsets = tokenize_raw_text(raw_ocr_text)
        bio_tags, label_ids = align_spans_to_bio_tags(
            raw_ocr_text, tokens, offsets, spans, TAG_TO_ID
        )

        validate_bio_sequence(bio_tags)
        xml_check_res = XMLChecker.check(raw_xml)
        if not xml_check_res.is_valid:
            print(f"  [XML Checker Warning] Issues detected in streamed XML:\n{xml_check_res.summary()}")

        return {
            "raw_text": raw_ocr_text,
            "raw_xml": raw_xml,
            "spans": spans,
            "tokens": tokens,
            "offsets": offsets,
            "tags": bio_tags,
            "labels": label_ids,
            "label_mapping": TAG_TO_ID,
            "annotated": True,
            "xml_is_valid": xml_check_res.is_valid,
            "xml_issues": [str(i) for i in xml_check_res.issues],
            "total_tokens": total_prompt_tokens + streamed_token_count,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": streamed_token_count,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Annotate raw OCR Vietnamese exams & generate sequence labelling datasets using LLMs"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(script_dir / "out"),
        help="Input file or directory containing markdown files from OCR",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(workspace_dir / "output" / "real-exams"),
        help="Directory to save annotated JSON, XML, and CoNLL files",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit number of files to process"
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-annotation even if output exists",
    )
    parser.add_argument(
        "--export-conll", action="store_true", help="Also export CoNLL format files"
    )
    parser.add_argument("--model", default=None, help="Model identifier")
    parser.add_argument(
        "--provider", choices=["deepseek", "nvidia", "vilao", "xah", "commandcode", "codex"], default=None
    )

    args = parser.parse_args()

    annotator = OCRAnnotator(model=args.model, provider=args.provider)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    md_files = []
    if input_path.is_file():
        md_files = [input_path]
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(".md"):
                    md_files.append(Path(root) / file)

    if not md_files:
        print(f"No .md files found in '{args.input}'.")
        sys.exit(0)

    if args.limit is not None:
        md_files = md_files[: args.limit]

    print(f"Found {len(md_files)} file(s) to process.\n")

    for idx, file_path in enumerate(tqdm(md_files, desc="[Annotating Files]", unit="file")):
        print(f"\n[{idx + 1}/{len(md_files)}] Processing: {file_path.name}")

        rel_sig = file_path.name
        path_hash = hashlib.md5(rel_sig.encode("utf-8")).hexdigest()[:8]

        json_out = output_path / f"real_exam_{path_hash}.json"
        xml_out = output_path / f"real_exam_{path_hash}.xml"
        conll_out = output_path / f"real_exam_{path_hash}.conll"

        if json_out.exists() and not args.force:
            print(f"  Skipping existing output: {json_out.name}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_ocr_text = f.read()

            result = annotator.annotate_text(raw_ocr_text)
            result["exam_id"] = f"real_{path_hash}"
            result["created_at"] = datetime.now().isoformat()
            result["is_real"] = True

            # Write JSON output
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Write XML output
            with open(xml_out, "w", encoding="utf-8") as f:
                f.write(result["raw_xml"])

            # Write CoNLL output if requested
            if args.export_conll:
                conll_text = export_conll(result["tokens"], result["tags"])
                with open(conll_out, "w", encoding="utf-8") as f:
                    f.write(conll_text)

            print(
                f"  -> Saved JSON ({len(result['spans'])} spans, {len(result['tokens'])} tokens) to: {json_out.name}"
            )

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    print("\nAnnotation completed!")


if __name__ == "__main__":
    main()
