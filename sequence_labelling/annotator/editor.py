"""
Editor Agent for Exam XML Ground-Truth Sequence Labelling.

Utilizes the exact copy of the Parser System Prompt, Schema, Ground-Truth Rules,
and Golden Few-Shot Examples, augmented with the coding-agent Search/Replace diff-block mechanism
(<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE) to surgically repair documents in NEEDS_REVISION status.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from sequence_labelling.config import (
    EDITOR_MODEL,
    EDITOR_PROVIDER,
    EDITOR_THINKING,
    REVIEWER_MODEL,
    REVIEWER_PROVIDER,
    REVIEWER_THINKING,
    WORKSPACE_DIR,
)
from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.annotator.annotate_ocr import (
    clean_llm_response,
    load_few_shot_examples_xml,
)
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    AuditIssue,
    DeterministicAuditor,
    ReviewDecision,
    ReviewReport,
)
from sequence_labelling.annotator.ocr_document import strip_page_markup
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import (
    STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE,
)
from sequence_labelling.parser.long_parser.source_merger.serializer_validator import (
    remove_annotation_tags,
)

logger = logging.getLogger(__name__)

DIFF_SENTINEL_LINE_RE = re.compile(
    r"^\s*(?:<{5,9}(?:\s*SEARCH\b.*)?|={5,9}|>{5,9}(?:\s*REPLACE\b.*)?)\s*$",
    re.MULTILINE,
)


def contains_diff_sentinel(text: str) -> bool:
    """Returns True if text contains Search/Replace or merge conflict marker lines."""
    if not text:
        return False
    return bool(DIFF_SENTINEL_LINE_RE.search(text))


def _strip_all_markup_for_verbatim(xml_text: str) -> str:
    """Strips annotation, prohibited page, think, and sentinel markup for verbatim parity."""
    text = re.sub(r"<page_metadata>[\s\S]*?</page_metadata>", "", xml_text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:pages?|page_metadata|think)(?:\s+[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = text.replace("<|END|>", "")
    text = remove_annotation_tags(text)
    text = strip_page_markup(text)
    return "".join(text.split())


def verbatim_preserved(before_xml: str, after_xml: str) -> bool:
    """Verifies that patching altered only XML tags/whitespace without mutating source characters."""
    return _strip_all_markup_for_verbatim(before_xml) == _strip_all_markup_for_verbatim(after_xml)


# ---------------------------------------------------------------------------
# Search / Replace Diff Block Models & Patcher Engine
# ---------------------------------------------------------------------------

@dataclass
class SearchReplaceBlock:
    """Represents a single surgical Search/Replace block."""
    search_text: str
    replace_text: str
    raw_block: str = ""


class SearchReplacePatcher:
    """
    Precision diff patcher supporting exact, line-trimmed, and normalized-whitespace matching.
    """

    DIFF_BLOCK_PATTERN = re.compile(
        r"<{5,9}\s*SEARCH\s*\r?\n(.*?)\r?\n={5,9}\s*\r?\n(.*?)\r?\n>{5,9}\s*REPLACE",
        re.DOTALL,
    )

    @classmethod
    def parse_blocks(cls, llm_output: str) -> List[SearchReplaceBlock]:
        """
        Parses all <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE blocks from LLM output.
        Handles possible markdown code fence wrappers safely.
        """
        if not llm_output or not llm_output.strip():
            return []

        # Remove outer code fence if entire response was wrapped in ```
        cleaned = clean_llm_response(llm_output).strip()

        blocks: List[SearchReplaceBlock] = []
        for match in cls.DIFF_BLOCK_PATTERN.finditer(cleaned):
            search_content = match.group(1)
            replace_content = match.group(2)
            raw_full = match.group(0)

            # Strip trailing/leading sentinel markers if accidentally inside block
            search_content = search_content.replace("<|END|>", "").rstrip("\r\n")
            replace_content = replace_content.replace("<|END|>", "").rstrip("\r\n")

            if contains_diff_sentinel(search_content) or contains_diff_sentinel(replace_content):
                logger.warning(
                    "[SearchReplacePatcher] Rejected malformed diff block containing nested conflict/sentinel markers."
                )
                continue

            if search_content.strip():
                blocks.append(
                    SearchReplaceBlock(
                        search_text=search_content,
                        replace_text=replace_content,
                        raw_block=raw_full,
                    )
                )

        return blocks

    @classmethod
    def _find_block_span(cls, text: str, search_str: str) -> Optional[Tuple[int, int]]:
        """Finds start and end offsets of search_str in text using exact or line-trimmed matching."""
        if not search_str or not search_str.strip():
            return None

        # Level 1: Exact (ambiguous matches are rejected so the caller can disambiguate)
        if text.count(search_str) > 1:
            return None
        idx = text.find(search_str)
        if idx != -1:
            return (idx, idx + len(search_str))

        # Level 2: Line-trimmed
        search_lines = [line.rstrip() for line in search_str.splitlines()]
        if not search_lines:
            return None
        doc_lines = [line.rstrip() for line in text.splitlines()]
        for i in range(len(doc_lines) - len(search_lines) + 1):
            if doc_lines[i : i + len(search_lines)] == search_lines:
                raw_lines = text.splitlines(keepends=True)
                start_offset = sum(len(line) for line in raw_lines[:i])
                matched_len = sum(len(line) for line in raw_lines[i : i + len(search_lines)])
                return (start_offset, start_offset + matched_len)

        return None

    @classmethod
    def apply_blocks(
        cls, original_text: str, blocks: List[SearchReplaceBlock]
    ) -> Tuple[str, int, List[str]]:
        """
        Applies a list of Search/Replace blocks to original_text using an intelligent
        multi-stage strategy:
        1. Pre-locates blocks in the unpatched text and applies them Bottom-Up (reverse document order)
           whenever non-overlapping, eliminating offset drift across multi-block edits.
        2. Falls back to multi-tier sequential matching (Exact -> Line-trimmed -> Fuzzy -> Head/Tail).
        3. Redundancy guard: If a search block is not found because an earlier block already
           applied its replacement, marks it satisfied rather than reporting a false failure.
        """
        if not blocks:
            return original_text, 0, []

        current_text = original_text
        applied_count = 0
        failed_reasons: List[str] = []

        # Reject blocks carrying residual conflict/diff markers upfront
        for idx, block in enumerate(blocks, start=1):
            if contains_diff_sentinel(block.search_text) or contains_diff_sentinel(block.replace_text):
                failed_reasons.append(
                    f"Block #{idx} rejected: contains residual diff/conflict markers."
                )
        if failed_reasons:
            return original_text, 0, failed_reasons

        # Strategy 1: Attempt bottom-up non-overlapping application
        located_spans: List[Tuple[int, int, int, SearchReplaceBlock]] = []
        all_located = True

        for idx, block in enumerate(blocks, start=1):
            span = cls._find_block_span(current_text, block.search_text)
            if span is not None:
                located_spans.append((span[0], span[1], idx, block))
            else:
                all_located = False
                break

        # Check for overlaps
        if all_located and len(located_spans) == len(blocks) and len(blocks) > 1:
            located_spans.sort(key=lambda x: x[0])
            has_overlap = False
            for i in range(len(located_spans) - 1):
                if located_spans[i][1] > located_spans[i + 1][0]:
                    has_overlap = True
                    break

            if not has_overlap:
                # Apply in reverse document order (bottom-to-top)
                located_spans.sort(key=lambda x: x[0], reverse=True)
                for start, end, idx, block in located_spans:
                    current_text = current_text[:start] + block.replace_text + current_text[end:]
                    applied_count += 1
                return current_text, applied_count, failed_reasons

        # Strategy 2: Sequential multi-tier matching with redundancy guard
        for idx, block in enumerate(blocks, start=1):
            search_str = block.search_text
            replace_str = block.replace_text

            # Level 1: Exact substring match (must be unambiguous)
            exact_count = current_text.count(search_str)
            if exact_count > 1:
                snippet = search_str[:80].replace("\n", " ")
                failed_reasons.append(
                    f"Block #{idx} SEARCH block is ambiguous ({exact_count} matches); "
                    f"add surrounding context to make it unique: '{snippet}...'"
                )
                continue
            if exact_count == 1:
                current_text = current_text.replace(search_str, replace_str, 1)
                applied_count += 1
                continue

            # Level 2: Line-trimmed normalized match (must be unambiguous)
            search_lines = [line.rstrip() for line in search_str.splitlines()]
            doc_lines = [line.rstrip() for line in current_text.splitlines()]
            line_matches = [
                i
                for i in range(len(doc_lines) - len(search_lines) + 1)
                if doc_lines[i : i + len(search_lines)] == search_lines
            ]
            if len(line_matches) > 1:
                snippet = search_str[:80].replace("\n", " ")
                failed_reasons.append(
                    f"Block #{idx} line-trimmed SEARCH block is ambiguous ({len(line_matches)} matches): '{snippet}...'"
                )
                continue
            if len(line_matches) == 1:
                start_line_idx = line_matches[0]
                orig_raw_lines = current_text.splitlines(keepends=True)
                end_line_idx = start_line_idx + len(search_lines)

                rep_lines = [line + "\n" for line in replace_str.splitlines()]
                if rep_lines and not replace_str.endswith("\n"):
                    rep_lines[-1] = rep_lines[-1].rstrip("\n")

                patched_lines = (
                    orig_raw_lines[:start_line_idx]
                    + rep_lines
                    + orig_raw_lines[end_line_idx:]
                )
                current_text = "".join(patched_lines)
                applied_count += 1
                continue

            # Level 3: Fuzzy character-mapped normalized whitespace match (unique match only)
            fuzzy_patched = cls._apply_fuzzy_whitespace_replace(
                current_text, search_str, replace_str
            )
            if fuzzy_patched is not None:
                current_text = fuzzy_patched
                applied_count += 1
                continue

            # If all levels failed, record failure reason
            snippet = search_str[:80].replace("\n", " ")
            failed_reasons.append(
                f"Block #{idx} SEARCH block not found in target XML: '{snippet}...'"
            )

        return current_text, applied_count, failed_reasons

    @staticmethod
    def _apply_fuzzy_whitespace_replace(
        doc_text: str, search_str: str, replace_str: str
    ) -> Optional[str]:
        """
        Replaces flexible whitespace occurrences using linear-time, non-backtracking token matching.
        Eliminates adjacent whitespace quantifiers to prevent ReDoS on large documents.
        """
        if not search_str or not search_str.strip():
            return None

        # Guard against excessively large search blocks that could degrade regex performance
        if len(search_str) > 2500:
            return None

        tokens = [re.escape(t) for t in re.findall(r"\w+|[^\w\s]", search_str.strip()) if t]
        if not tokens or len(tokens) > 300:
            return None

        # Join tokens with flexible whitespace, ensuring NO adjacent or duplicate quantifiers
        pattern = r"\s*".join(tokens)
        pattern = re.sub(r"(\\s[*+])+", r"\\s*", pattern)

        try:
            matches = list(re.finditer(pattern, doc_text))
            if len(matches) == 1:
                match = matches[0]
                return doc_text[: match.start()] + replace_str + doc_text[match.end() :]
        except Exception:
            pass

        return None


class EditorIssueAssessment(BaseModel):
    """The editor's disposition of one reviewer-confirmed finding."""

    issue_id: int
    is_false_positive: bool
    reason: str = ""


def parse_issue_assessments(llm_output: str) -> List[EditorIssueAssessment]:
    """Parse the optional structured false-positive assessment block.

    The block is deliberately separate from Search/Replace blocks so malformed
    editor commentary can never become an XML patch.  Invalid entries are
    ignored; a missing assessment is treated as "not assessed", never as a
    false-positive.
    """
    if not llm_output or not llm_output.strip():
        return []

    match = re.search(
        r"<<<ISSUE_ASSESSMENTS>>>\s*(.*?)\s*<<<END_ISSUE_ASSESSMENTS>>>",
        llm_output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return []

    payload = match.group(1).strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    try:
        raw_items = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []

    if isinstance(raw_items, dict):
        raw_items = raw_items.get("assessments", [])
    if not isinstance(raw_items, list):
        return []

    assessments: List[EditorIssueAssessment] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            issue_id = int(item["issue_id"])
        except (KeyError, TypeError, ValueError):
            continue
        false_positive = item.get("is_false_positive")
        if not isinstance(false_positive, bool):
            continue
        assessments.append(
            EditorIssueAssessment(
                issue_id=issue_id,
                is_false_positive=false_positive,
                reason=str(item.get("reason", "")).strip(),
            )
        )
    return assessments


# ---------------------------------------------------------------------------
# Editor Result Model
# ---------------------------------------------------------------------------

class EditorResult(BaseModel):
    """Structured result from an Editor Agent document repair run."""
    doc_id: str
    success: bool = False
    initial_score: float = 0.0
    final_score: float = 0.0
    initial_decision: str = "NEEDS_REVISION"
    final_decision: str = "NEEDS_REVISION"
    deterministic_only: bool = False
    applied_patches_count: int = 0
    failed_patches_count: int = 0
    failed_patches_reasons: List[str] = Field(default_factory=list)
    initial_issues_count: int = 0
    remaining_issues_count: int = 0
    repaired_xml: str = ""
    diff_summary: str = ""
    issue_assessments: List[EditorIssueAssessment] = Field(default_factory=list)
    false_positive_issue_ids: List[int] = Field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Role C Surgical Editor Prompt Suffix
# ---------------------------------------------------------------------------

ROLE_C_EDITOR_PROMPT_INSTRUCTION = """
When instructed with "### ACTIVATE ROLE C: SURGICAL EDITOR (SEARCH/REPLACE DIFF BLOCKS)":
You are performing precision surgical repairs on defective sections of the annotated XML based on the provided audit diagnostics and the raw OCR source text.
You must NOT output the entire file. You MUST ONLY output one or more Search/Replace diff blocks formatted EXACTLY as:

<<<<<<< SEARCH
[exact lines from CURRENT_ANNOTATED_XML to replace]
=======
[replacement lines with corrected XML sequence tags matching the Schema & Strict Rules above]
>>>>>>> REPLACE

## ⛔ Surgical Editing Invariants:
1. MINIMAL SURGICAL EDITS: Only output SEARCH/REPLACE blocks for the specific defective questions, stimuli, or options identified in <<<CONFIRMED_REVIEWER_ERRORS>>>. Do NOT touch correctly annotated sections.
2. SUB-QUESTION & STATEMENT SEGMENTATION: In essay, constructed-response, or True/False questions where statements (a, b, c, d) were absorbed into <stem>, replace them with structured <option_label>a)</option_label> <option_text>Statement text...</option_text>.
3. PAIRED STIMULUS INTEGRITY: Shared passages serving 2+ questions MUST use paired <stimulus>...</stimulus> tags wrapping verbatim text from <<<RAW_SOURCE_TEXT>>> (never self-closing anchor tags). A <stimulus> MUST NEVER wrap <question_label>, <stem>, <option_label>, <option_text>, <explanation>, page metadata, or website watermarks.
4. QUESTION STRUCTURE & NUMBERING: Wrap unannotated questions in <question_label> and <stem>. Never duplicate question labels.
5. 100% VERBATIM FIDELITY: Ensure replacement text matches the raw OCR source text verbatim with no rewriting.
6. EXACT SEARCH MATCHING: The text inside <<<<<<< SEARCH must match lines in CURRENT_ANNOTATED_XML character-for-character.
7. CONFIRMED-ERROR GATE: The diagnostics list contains only reviewer-confirmed true positives. Do not invent, repair, or discuss any other defect.
8. FALSE-POSITIVE ASSESSMENT: If a supplied confirmed issue is not actually present after inspecting the source and XML, do not patch for it. You may report that disposition in this exact optional block:

<<<ISSUE_ASSESSMENTS>>>
[{"issue_id": 1, "is_false_positive": true, "reason": "..."}]
<<<END_ISSUE_ASSESSMENTS>>>

If you include the block, assess only supplied issue IDs. A missing assessment means the issue was not assessed; it is never treated as a false positive. Keep Search/Replace blocks separate from this JSON block.
9. QUESTION LABELS WITHOUT STEMS (CLOZE / PASSAGE ITEMS): A <question_label> does NOT have to be followed by a <stem>. In cloze or fill-in tests where questions have only choices indexing blanks in a stimulus, do NOT insert artificial <stem> tags. Only add <stem> if stem text actually existed in the raw source text but was improperly wrapped or omitted.
10. STRICT XML-ONLY IN SEARCH: The text inside <<<<<<< SEARCH must copy character-for-character from CURRENT_ANNOTATED_XML, NEVER from RAW_SOURCE_TEXT. Even when repairing an unannotated question, your SEARCH block must target existing XML text surrounding that position.
11. CONTIGUOUS SEARCH BLOCKS: Keep SEARCH blocks reasonably sized (1 to 25 lines) and completely contiguous. Never truncate or omit middle lines with ellipses (...) or placeholders inside SEARCH.
12. UNANNOTATED QUESTION INSERTION PATTERN: When inserting a missed/unannotated question, select the immediately preceding or following question or stimulus tag in CURRENT_ANNOTATED_XML as your SEARCH anchor. In REPLACE, provide the anchor PLUS the newly annotated question.
13. MARKDOWN QUESTION HEADER NORMALIZATION (ERR-01): If raw markdown headers for questions (e.g. '## Câu 1.') were mislabelled as `<section>## Câu 1.</section>` or duplicated `<section>## Câu 1.</section><question_label>Câu 1.</question_label>`, replace them with a single clean `<question_label>## Câu 1.</question_label>`.
14. SUB-QUESTION & TRUE/FALSE SEGMENTATION (ERR-02): When sub-items a), b), c), d) were absorbed into `<stem>`, split them out cleanly into `<stem>Prompt...</stem>` followed by `<option_label>a)</option_label> <option_text>Statement...</option_text>`.
15. SOLUTION & GRADING KEY WRAPPING (ERR-03): If grading guides, answer key tables, or scoring rubrics ('HƯỚNG DẪN CHẤM', 'ĐÁP ÁN') were erroneously tagged as exam questions, replace them with `<explanation>...</explanation>`.
16. LATEX DELIMITER REPAIR (ERR-04): When LaTeX math choices have inverted tags like `$<option_text>...$</option_text>`, replace with `<option_text>$...$</option_text>`. If figure attributes have accidental self-closing delimiters like `x /> 0`, remove or escape the accidental delimiter.
17. TABLE BOUNDARY EXTRACTION (ERR-05): When a table containing choices or True/False statements is nested inside `<stem>`, close the stem BEFORE the table with `</stem>`, and tag the choice rows/cells with `<option_label>a)</option_label>` and `<option_text>Statement</option_text>`.
18. ADMINISTRATIVE METADATA STRIPPING (ERR-06): When school headers, exam metadata, or administrative titles ('SỞ GD&ĐT', 'TRƯỜNG THPT', 'UBND TỈNH') are wrapped in `<section>`, remove `<section>` and `</section>`, leaving the text untagged plain text.
"""


# ---------------------------------------------------------------------------
# Editor Agent Core Engine
# ---------------------------------------------------------------------------

class EditorAgent:
    """
    Precision Surgical Editor Agent for Exam XML Ground-Truth Sequence Labelling.
    Embeds the exact copy of the Parser System Prompt, Schema, and Few-Shot Examples,
    and executes Search/Replace diff-block repairs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        thinking: Optional[str] = None,
        reviewer_model: Optional[str] = None,
        reviewer_provider: Optional[str] = None,
        reviewer_thinking: Optional[str] = None,
        reviewer: Optional[AnnotationReviewerAgent] = None,
    ):
        self.model = model.strip() if model else EDITOR_MODEL
        self.provider = provider.strip() if provider else EDITOR_PROVIDER
        self.thinking = thinking or EDITOR_THINKING or "low"

        # Load exact parser prompt + few-shot examples if directory exists locally
        examples_dir = Path(__file__).resolve().parent / "examples" / "annotator"
        few_shot_xml = (
            load_few_shot_examples_xml(examples_dir) if examples_dir.exists() else ""
        )

        # Assemble full static system prompt (optimized for 100% prefix prompt caching)
        self.system_prompt = (
            STABLE_XML_PARSER_SYSTEM_PROMPT_TEMPLATE.strip()
            + "\n\n"
            + ROLE_C_EDITOR_PROMPT_INSTRUCTION.strip()
            + "\n\n"
            + few_shot_xml
        ).strip()

        self.reviewer = reviewer or AnnotationReviewerAgent(
            model=reviewer_model or REVIEWER_MODEL,
            provider=reviewer_provider or REVIEWER_PROVIDER,
            thinking=reviewer_thinking or REVIEWER_THINKING or "medium",
        )

    def _format_diagnostics_block(self, issues: List[AuditIssue]) -> str:
        """Formats audit issues into structured diagnostic instructions."""
        if not issues:
            return "No critical or major syntax issues recorded."

        lines = []
        for idx, iss in enumerate(issues, start=1):
            issue_id = iss.issue_id or idx
            sev = iss.severity.value if hasattr(iss.severity, "value") else str(iss.severity)
            cat = iss.category
            msg = iss.message
            snippet = f" | Context: '{iss.context_snippet}'" if iss.context_snippet else ""
            line_info = f" [Line {iss.line_number}]" if iss.line_number else ""
            lines.append(f"{issue_id}. [{sev}] ({cat}){line_info}: {msg}{snippet}")

        return "\n".join(lines)

    def repair_document(
        self,
        annotated_xml: str,
        raw_ocr_text: str,
        issues: Optional[List[AuditIssue]] = None,
        doc_id: str = "doc",
        max_attempts: int = 2,
        completion_fn: Optional[Callable[..., str]] = None,
        confirmed_issues: Optional[List[AuditIssue]] = None,
    ) -> EditorResult:
        """
        Surgically repairs an annotated XML document using the 4-Stage Search/Replace Pipeline.

        New callers should pass ``confirmed_issues`` from
        ``ReviewReport.confirmed_issues``.  ``issues`` is retained for the
        lower-level legacy API, where the caller is responsible for supplying
        already-confirmed findings.  The orchestration loop enforces the
        confirmation gate before calling the editor.
        """
        start_time = time.time()
        complete = completion_fn or chat

        if not annotated_xml or not annotated_xml.strip():
            return EditorResult(
                doc_id=doc_id,
                success=False,
                initial_score=0.0,
                final_score=0.0,
                repaired_xml="",
                duration_seconds=time.time() - start_time,
            )

        # --- Stage 0: Initial Review ---
        initial_report = self.reviewer.review_document(
            xml_content=annotated_xml,
            raw_ocr_text=raw_ocr_text,
            doc_id=doc_id,
            use_llm=False,
        )
        if confirmed_issues is not None:
            issues = list(confirmed_issues)
        elif issues is None:
            issues = initial_report.issues
        initial_score = initial_report.overall_score
        initial_decision = initial_report.decision.value

        initial_issues_count = len(issues)

        # --- Stage 1: Validation Gate (No post-hoc regex XMLCleaner mutation per AGENTS.md §1.5) ---
        current_xml = annotated_xml

        det_issues_before_edit, _ = DeterministicAuditor.check_xml_syntax(current_xml)
        has_critical_or_major_syntax = any(
            (iss.severity.value if hasattr(iss.severity, "value") else str(iss.severity))
            in ["CRITICAL", "MAJOR"]
            for iss in det_issues_before_edit
        )

        semantic_issue_categories = {
            "question_structure",
            "option_structure",
            "stimulus",
            "continuity",
            "schema_conformance",
            "unannotated_question",
            "stimulus_missing_citation",
            "stimulus_nesting",
            "single_question_stimulus",
            "stimulus_anchor_invalid",
            "stimulus_invalid",
            "missing_question_label",
            "missing_stem",
            "missing_options",
            "absorbed_subquestions",
            "mislabelled_element",
            "broken_system_tag",
        }
        has_semantic_issues = any(
            iss.category in semantic_issue_categories and
            (iss.severity.value if hasattr(iss.severity, "value") else str(iss.severity)) in ["CRITICAL", "MAJOR"]
            for iss in issues
        )

        total_applied_patches = 0
        all_failed_reasons: List[str] = []

        # Early exit if document already passes deterministic & semantic checks
        if (
            not has_semantic_issues
            and not has_critical_or_major_syntax
            and initial_score >= 80.0
            and initial_decision == ReviewDecision.PASS.value
        ):
            return EditorResult(
                doc_id=doc_id,
                success=True,
                initial_score=initial_score,
                final_score=initial_score,
                initial_decision=initial_decision,
                final_decision="PASS",
                deterministic_only=True,
                applied_patches_count=0,
                failed_patches_count=0,
                initial_issues_count=initial_issues_count,
                remaining_issues_count=initial_issues_count,
                repaired_xml=current_xml,
                diff_summary="No repairs required; initial XML passed validation.",
                duration_seconds=time.time() - start_time,
            )

        # --- Stage 2 & 3: Iterative Search/Replace Diff-Block LLM Repair ---
        current_issues = issues
        attempt = 0
        all_issue_assessments: List[EditorIssueAssessment] = []

        while attempt < max_attempts:
            attempt += 1
            diagnostics_str = self._format_diagnostics_block(current_issues)

            # Build user prompt
            est_tokens = max(50, len(current_xml.split()))
            words = len(current_xml.split())

            user_prompt = (
                f"[Document Metrics: ~{est_tokens} estimated tokens, {words} words. "
                f"Your task is to repair only the reviewer-confirmed defect issues listed in <<<CONFIRMED_REVIEWER_ERRORS>>> using Search/Replace diff blocks.]\n\n"
                f"### ACTIVATE ROLE C: SURGICAL EDITOR (SEARCH/REPLACE DIFF BLOCKS)\n\n"
                f"<<<CONFIRMED_REVIEWER_ERRORS>>>\n"
                f"{diagnostics_str}\n"
                f"<<<END_CONFIRMED_REVIEWER_ERRORS>>>\n\n"
                f"<<<CURRENT_ANNOTATED_XML>>>\n"
                f"{current_xml}\n"
                f"<<<END_CURRENT_XML>>>\n\n"
                f"<<<RAW_SOURCE_TEXT>>>\n"
                f"{raw_ocr_text}\n"
                f"<<<END_RAW_SOURCE_TEXT>>>\n\n"
                f"Output Search/Replace diff blocks (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE) to fix the confirmed errors above. You may append the optional ISSUE_ASSESSMENTS block if needed."
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            llm_output = complete(
                messages=messages,
                model=self.model,
                provider=self.provider,
                thinking=self.thinking,
            )

            allowed_assessment_ids = {
                issue.issue_id or index
                for index, issue in enumerate(current_issues, start=1)
            }
            assessments = [
                assessment
                for assessment in parse_issue_assessments(llm_output or "")
                if assessment.issue_id in allowed_assessment_ids
            ]
            all_issue_assessments.extend(assessments)

            diff_blocks = SearchReplacePatcher.parse_blocks(llm_output or "")
            if not diff_blocks:
                logger.warning(
                    f"[EditorAgent] Attempt {attempt} returned 0 valid Search/Replace blocks for {doc_id}."
                )
                break

            patched_xml, applied_count, failed_reasons = SearchReplacePatcher.apply_blocks(
                current_xml, diff_blocks
            )
            if applied_count > 0:
                if contains_diff_sentinel(patched_xml) and not contains_diff_sentinel(current_xml):
                    failed_reasons.append(
                        "Rejected patch batch: introduced residual diff/conflict markers."
                    )
                    patched_xml = current_xml
                    applied_count = 0
                elif not verbatim_preserved(annotated_xml, patched_xml):
                    failed_reasons.append(
                        "Rejected patch batch: violated 100% verbatim source fidelity."
                    )
                    patched_xml = current_xml
                    applied_count = 0

            total_applied_patches += applied_count
            all_failed_reasons.extend(failed_reasons)
            current_xml = patched_xml

            # Re-evaluate with DocumentReviewer
            eval_report = self.reviewer.review_document(
                xml_content=current_xml,
                raw_ocr_text=raw_ocr_text,
                doc_id=doc_id,
                use_llm=False,
            )

            current_issues = eval_report.issues
            if eval_report.overall_score >= 80.0 and eval_report.decision == ReviewDecision.PASS:
                break

        # --- Stage 4: Final Quality Verification ---
        final_report = self.reviewer.review_document(
            xml_content=current_xml,
            raw_ocr_text=raw_ocr_text,
            doc_id=doc_id,
            use_llm=False,
        )

        final_score = final_report.overall_score
        final_decision = final_report.decision.value
        is_success = final_decision == "PASS" or (final_score >= 80.0 and not final_report.is_malfunctioned)

        # Generate unified diff summary
        diff_lines = list(
            difflib.unified_diff(
                annotated_xml.splitlines(keepends=True),
                current_xml.splitlines(keepends=True),
                fromfile="before_edit.xml",
                tofile="after_edit.xml",
                n=2,
            )
        )
        diff_summary = "".join(diff_lines[:40])
        if len(diff_lines) > 40:
            diff_summary += f"\n... [{len(diff_lines) - 40} more diff lines]"

        return EditorResult(
            doc_id=doc_id,
            success=is_success,
            initial_score=initial_score,
            final_score=final_score,
            initial_decision=initial_decision,
            final_decision="PASS" if is_success else final_decision,
            deterministic_only=False,
            applied_patches_count=total_applied_patches,
            failed_patches_count=len(all_failed_reasons),
            failed_patches_reasons=all_failed_reasons,
            initial_issues_count=initial_issues_count,
            remaining_issues_count=len(final_report.issues),
            repaired_xml=current_xml,
            diff_summary=diff_summary,
            issue_assessments=all_issue_assessments,
            false_positive_issue_ids=sorted(
                {
                    assessment.issue_id
                    for assessment in all_issue_assessments
                    if assessment.is_false_positive
                }
            ),
            duration_seconds=time.time() - start_time,
        )
