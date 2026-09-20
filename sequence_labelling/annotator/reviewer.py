import os
import re
import json
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from pydantic import BaseModel, Field
from difflib import SequenceMatcher

from sequence_labelling.config import (
    REVIEWER_MODEL,
    REVIEWER_PROVIDER,
    REVIEWER_THINKING,
    REVIEWER_MIN_SCORE,
    PARSER_MODEL,
    PARSER_PROVIDER,
    get_provider_api_key,
)
from sequence_labelling.llm.deepseek_client import chat
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic

# Allowed tag schema for sequence labelling & embedded markdown/HTML structure
HTML_VOID_TAGS = {"br", "hr", "img", "col", "wbr", "input", "meta", "link"}
ALLOWED_FORMATTING_TAGS = {
    "table",
    "tr",
    "td",
    "th",
    "tbody",
    "thead",
    "tfoot",
    "b",
    "i",
    "u",
    "strong",
    "em",
    "sub",
    "sup",
    "span",
    "div",
    "p",
    "ol",
    "ul",
    "li",
    "code",
    "pre",
    "a",
    "s",
    "strike",
    "del",
    "center",
    "font",
    "mark",
    "small",
    "caption",
    "colgroup",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
}

# System tags that define sequence labeling structure and require strict closure/pairing
SYSTEM_TAGS = {
    "section",
    "stimulus",
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "explanation",
    "question",
    "figure",
}

ALLOWED_PAIRED_TAGS = {
    "section",
    "stimulus",
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "explanation",
    "question",
} | ALLOWED_FORMATTING_TAGS

ALLOWED_SELF_CLOSING_TAGS = {"stimulus", "figure"} | HTML_VOID_TAGS
PROHIBITED_TAGS = {"pages", "page", "page_metadata", "think"}


class ReviewDecision(str, Enum):
    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"
    DISCARD = "DISCARD"


class IssueSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class AuditIssue(BaseModel):
    issue_id: Optional[int] = None
    category: str
    severity: IssueSeverity
    message: str
    line_number: Optional[int] = None
    context_snippet: Optional[str] = None


class RubricScores(BaseModel):
    xml_well_formedness: float = Field(default=100.0, ge=0.0, le=100.0)
    schema_conformance: float = Field(default=100.0, ge=0.0, le=100.0)
    det_question_coverage: float = Field(default=100.0, ge=0.0, le=100.0)
    verbatim_fidelity: float = Field(default=100.0, ge=0.0, le=100.0)
    question_option_completeness: float = Field(default=100.0, ge=0.0, le=100.0)
    stimulus_accuracy: float = Field(default=100.0, ge=0.0, le=100.0)
    sequence_continuity: float = Field(default=100.0, ge=0.0, le=100.0)


class ReviewReport(BaseModel):
    doc_id: str
    file_path: Optional[str] = None
    raw_file_path: Optional[str] = None
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    deterministic_score: float = Field(default=0.0, ge=0.0, le=100.0)
    llm_score: Optional[float] = None
    grade: str = "F"  # A, B, C, D, F
    decision: ReviewDecision = ReviewDecision.DISCARD
    is_malfunctioned: bool = False
    discard_reasons: List[str] = Field(default_factory=list)
    rubric_scores: RubricScores = Field(default_factory=RubricScores)
    issues: List[AuditIssue] = Field(default_factory=list)
    llm_confirmations: List[Dict[str, Any]] = Field(default_factory=list)
    # Only these issues are safe to send to the editor.  ``issues`` remains the
    # complete audit trail, including deterministic findings that the semantic
    # reviewer rejected as false positives.
    confirmed_issues: List[AuditIssue] = Field(default_factory=list)
    confirmation_status: str = "not_run"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    parser_info: Optional[Dict[str, Any]] = None
    summary: str = ""
    reviewed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    reviewer_model: Optional[str] = None
    reviewer_provider: Optional[str] = None


class BatchReviewSummary(BaseModel):
    total_documents: int = 0
    passed_count: int = 0
    needs_revision_count: int = 0
    discarded_count: int = 0
    discarded_paths: List[str] = Field(default_factory=list)
    average_score: float = 0.0
    duration_sec: float = 0.0
    reports: List[ReviewReport] = Field(default_factory=list)
    failure_reasons_distribution: Dict[str, int] = Field(default_factory=dict)


def compute_grade(score: float) -> str:
    if score >= 90.0:
        return "A"
    elif score >= 80.0:
        return "B"
    elif score >= 70.0:
        return "C"
    elif score >= 60.0:
        return "D"
    return "F"


class DeterministicAuditor:
    """
    High-speed deterministic static analyzer and integrity auditor for sequence-labelled XML.
    """

    @staticmethod
    def strip_xml_tags(xml_text: str) -> str:
        """
        Strips valid XML/HTML tags and comments to reconstruct pure text content,
        preserving mathematical inequality comparisons (<, >) inside math text.
        """
        clean = re.sub(
            r"<!--.*?-->|</?[a-zA-Z][a-zA-Z0-9_\-:]*(?:\s+[^>]*)?/?>",
            "",
            xml_text,
            flags=re.DOTALL,
        )
        return clean

    @staticmethod
    def check_xml_syntax(xml_content: str) -> Tuple[List[AuditIssue], float]:
        """
        Validates tag pairing, closure, unclosed opening brackets, and structural XML syntax.
        """
        issues: List[AuditIssue] = []
        deductions = 0.0

        if not xml_content or not xml_content.strip():
            issues.append(
                AuditIssue(
                    category="xml_syntax",
                    severity=IssueSeverity.CRITICAL,
                    message="Document content is completely empty.",
                )
            )
            return issues, 0.0

        # Check for unclosed tag at the very end of generation
        trailing_match = re.search(r"</?([a-zA-Z_0-9\-]*)$", xml_content.strip())
        if trailing_match and trailing_match.group(1):
            tag_fragment = trailing_match.group(0)
            if tag_fragment.startswith("<"):
                issues.append(
                    AuditIssue(
                        category="xml_syntax",
                        severity=IssueSeverity.CRITICAL,
                        message=f"Output truncated mid-tag at file termination: '{tag_fragment}'",
                        context_snippet=xml_content[-80:],
                    )
                )
                deductions += 40.0

        # Lex tags with line numbers (stripping designated terminal sentinels <|END|>)
        clean_xml = re.sub(r"<\s*\|\s*END\s*\|\s*>", "", xml_content)
        clean_xml = re.sub(r"<\s*\|\s*endoftext\s*\|\s*>", "", clean_xml)
        tag_pattern = re.compile(r"<(/)?([a-zA-Z_][a-zA-Z0-9_\-]*)(?:\s+([^>]*))?(/)?>")
        lines = clean_xml.splitlines()

        tag_stack: List[Tuple[str, int, str]] = []  # (tag_name, line_num, full_tag)
        all_tags_found = 0

        current_line_num = 1
        pos = 0

        for match in tag_pattern.finditer(clean_xml):
            all_tags_found += 1
            start_pos = match.start()
            # Calculate line number
            current_line_num = xml_content.count("\n", 0, start_pos) + 1

            is_closing = bool(match.group(1))
            tag_name = match.group(2).lower()
            is_self_closing = bool(match.group(4)) or tag_name in ALLOWED_SELF_CLOSING_TAGS

            full_match = match.group(0)

            # Check self closing
            if is_self_closing or full_match.endswith("/>"):
                continue

            if not is_closing:
                # Opening tag: Only push system tags onto tag_stack to enforce strict closure and pairing.
                # Completely ignore non-system tags, tables, styling, and weird custom markup.
                if tag_name in SYSTEM_TAGS:
                    tag_stack.append((tag_name, current_line_num, full_match))
            else:
                # Closing tag - only system tags are enforced against tag_stack
                if tag_name in SYSTEM_TAGS:
                    if not tag_stack:
                        issues.append(
                            AuditIssue(
                                category="xml_syntax",
                                severity=IssueSeverity.MAJOR,
                                message=f"Unexpected closing tag '</{tag_name}>' with no matching open tag.",
                                line_number=current_line_num,
                                context_snippet=full_match,
                            )
                        )
                        deductions += 10.0
                    else:
                        last_open, last_line, last_tag = tag_stack.pop()
                        if last_open != tag_name:
                            issues.append(
                                AuditIssue(
                                    category="xml_syntax",
                                    severity=IssueSeverity.MAJOR,
                                    message=f"Mismatched closing tag '</{tag_name}>' at line {current_line_num}, expected '</{last_open}>' (opened at line {last_line}).",
                                    line_number=current_line_num,
                                    context_snippet=f"{last_tag} ... {full_match}",
                                )
                            )
                            deductions += 15.0

        # Any unclosed system tags remaining in stack? (unclosed non-system tags are ignored)
        for unclosed_name, unclosed_line, unclosed_tag in tag_stack:
            if unclosed_name in SYSTEM_TAGS:
                issues.append(
                    AuditIssue(
                        category="xml_syntax",
                        severity=IssueSeverity.MAJOR,
                        message=f"Unclosed tag '<{unclosed_name}>' opened at line {unclosed_line} was never closed.",
                        line_number=unclosed_line,
                        context_snippet=unclosed_tag,
                    )
                )
                deductions += 10.0

        if all_tags_found == 0:
            issues.append(
                AuditIssue(
                    category="xml_syntax",
                    severity=IssueSeverity.CRITICAL,
                    message="No XML sequence tags found in the entire document.",
                )
            )
            deductions += 100.0

        score = max(0.0, 100.0 - deductions)
        return issues, score

    @staticmethod
    def check_prohibited_tags(xml_content: str) -> Tuple[List[AuditIssue], float]:
        """
        Audits presence of prohibited tags like <pages>, <page>, <page_metadata>, <think>.
        """
        issues: List[AuditIssue] = []
        deductions = 0.0

        for prohibited in PROHIBITED_TAGS:
            pattern = re.compile(rf"</?{prohibited}(?:\s+[^>]*)?>", re.IGNORECASE)
            matches = list(pattern.finditer(xml_content))
            if matches:
                count = len(matches)
                first_match = matches[0]
                line_num = xml_content.count("\n", 0, first_match.start()) + 1
                issues.append(
                    AuditIssue(
                        category="prohibited_tags",
                        severity=IssueSeverity.CRITICAL if prohibited in ["pages", "page"] else IssueSeverity.MAJOR,
                        message=f"Found {count} instance(s) of prohibited tag '<{prohibited}>'. Page boundaries and metadata must be pruned.",
                        line_number=line_num,
                        context_snippet=first_match.group(0),
                    )
                )
                deductions += 20.0 * min(count, 3)

        score = max(0.0, 100.0 - deductions)
        return issues, score

    @staticmethod
    def check_stimulus_wrapping_system_tags(xml_content: str) -> Tuple[List[AuditIssue], float]:
        """
        Audits whether any <stimulus> tag illegally wraps or encloses system question elements
        (<stem>, <question_label>, <option_label>, <option_text>, <explanation>).
        Violations are fatal architectural malfunctions and trigger immediate auto-reject (CRITICAL).
        """
        issues: List[AuditIssue] = []
        deductions = 0.0

        if not xml_content:
            return issues, 100.0

        # 1. Check paired <stimulus ...>...</stimulus> (ensuring opening tag is not self-closing)
        for m in re.finditer(r"<stimulus\b(?![^>]*/>)([^>]*)>(.*?)</stimulus>", xml_content, re.DOTALL | re.IGNORECASE):
            inner_content = m.group(2)
            start_pos = m.start()
            line_num = xml_content.count("\n", 0, start_pos) + 1

            nested_tags = re.findall(
                r"<(stem|question_label|option_label|option_text|explanation)\b",
                inner_content,
                re.IGNORECASE,
            )
            if nested_tags:
                unique_nested = sorted(list(set(nested_tags)))
                issues.append(
                    AuditIssue(
                        category="stimulus_nesting",
                        severity=IssueSeverity.CRITICAL,
                        message=(
                            f"Stimulus tag illegally wraps system tag(s): {', '.join('<' + t + '>' for t in unique_nested)}. "
                            f"Stimulus must be a self-closing anchor tag or standalone passage, NEVER enclosing question elements."
                        ),
                        line_number=line_num,
                        context_snippet=m.group(0)[:120],
                    )
                )
                deductions += 100.0

        # 2. Check for unclosed paired <stimulus ...> tags that precede system tags without closing
        stim_open_pattern = re.compile(r"<stimulus\b(?![^>]*/>)([^>]*)>", re.IGNORECASE)
        for m in stim_open_pattern.finditer(xml_content):
            start_pos = m.end()
            # check if followed by closing </stimulus> before EOF
            close_match = xml_content.find("</stimulus>", start_pos)
            if close_match == -1:
                # Unclosed stimulus: check if system tags appear after this
                rem_text = xml_content[start_pos:]
                nested_tags = re.findall(
                    r"<(stem|question_label|option_label|option_text|explanation)\b",
                    rem_text,
                    re.IGNORECASE,
                )
                if nested_tags:
                    line_num = xml_content.count("\n", 0, m.start()) + 1
                    unique_nested = sorted(list(set(nested_tags)))
                    issues.append(
                        AuditIssue(
                            category="stimulus_nesting",
                            severity=IssueSeverity.CRITICAL,
                            message=(
                                f"Unclosed stimulus tag encloses system tag(s): {', '.join('<' + t + '>' for t in unique_nested)}. "
                                f"Stimulus must be a self-closing anchor tag, NEVER wrapping question elements."
                            ),
                            line_number=line_num,
                            context_snippet=m.group(0),
                        )
                    )
                    deductions += 100.0

        score = max(0.0, 100.0 - deductions)
        return issues, score

    @staticmethod
    def check_question_and_option_structure(
        xml_content: str,
        raw_ocr_text: Optional[str] = None,
    ) -> Tuple[List[AuditIssue], float, Dict[str, Any]]:
        """
        Audits questions, stems, choices, and sub-questions completeness and integrity.
        Considers error rate relative to the size of the entire document (e.g. 1 error in 30 questions is minor).
        Note: Figures (<figure ... />) are out of evaluation scope for now and are not penalized.
        """
        issues: List[AuditIssue] = []
        deductions = 0.0

        # Extract elements
        q_labels = re.findall(r"<question_label>(.*?)</question_label>", xml_content, re.DOTALL)
        stems = re.findall(r"<stem>(.*?)</stem>", xml_content, re.DOTALL)
        opt_labels = re.findall(r"<option_label>(.*?)</option_label>", xml_content, re.DOTALL)
        opt_texts = re.findall(r"<option_text>(.*?)</option_text>", xml_content, re.DOTALL)
        stimuli = re.findall(r"<stimulus\b([^>]*)/?>", xml_content, re.DOTALL)
        sections = re.findall(r"<section>(.*?)</section>", xml_content, re.DOTALL)
        figures = re.findall(r"<figure\b([^>]*)/?>", xml_content, re.DOTALL)

        total_questions = len(q_labels)

        metrics = {
            "questions_count": total_questions,
            "stems_count": len(stems),
            "option_labels_count": len(opt_labels),
            "option_texts_count": len(opt_texts),
            "stimuli_count": len(stimuli),
            "sections_count": len(sections),
            "figures_count": len(figures),
        }

        if total_questions == 0:
            # Check if raw source text also has 0 questions via DET parser (alien format: reference table, spec matrix, lecture notes)
            if raw_ocr_text and raw_ocr_text.strip():
                clean_raw = DeterministicAuditor.clean_raw_ocr_text(raw_ocr_text)
                try:
                    det_res = parse_chunk_deterministic(clean_raw)
                    det_q_count = len([s for s in det_res.spans if s.get("label") == "question_label"])
                except Exception:
                    det_q_count = 0

                if det_q_count == 0:
                    issues.append(
                        AuditIssue(
                            category="question_structure",
                            severity=IssueSeverity.INFO,
                            message="Zero questions detected, consistent with raw source document (reference table / specification matrix / non-exam document).",
                        )
                    )
                    return issues, 100.0, metrics

            issues.append(
                AuditIssue(
                    category="question_structure",
                    severity=IssueSeverity.CRITICAL,
                    message="Zero questions (<question_label>) detected in document.",
                )
            )
            return issues, 0.0, metrics

        # Empty stems check - scale severity and deduction by total questions
        empty_stems = [s for s in stems if not s.strip()]
        if empty_stems:
            stem_err_rate = len(empty_stems) / max(1, total_questions)
            is_major = stem_err_rate >= 0.15 or len(empty_stems) >= 4
            issues.append(
                AuditIssue(
                    category="question_structure",
                    severity=IssueSeverity.MAJOR if is_major else IssueSeverity.MINOR,
                    message=f"Detected {len(empty_stems)}/{total_questions} ({stem_err_rate:.1%}) empty <stem> tags.",
                )
            )
            deductions += min(35.0, stem_err_rate * 40.0 + (3.0 if not is_major else 15.0))

        # Empty option text check - scale severity and deduction
        empty_opts = [o for o in opt_texts if not o.strip()]
        if empty_opts:
            total_opts = max(1, len(opt_texts))
            opt_err_rate = len(empty_opts) / total_opts
            is_major = opt_err_rate >= 0.15 or len(empty_opts) >= 4
            issues.append(
                AuditIssue(
                    category="option_structure",
                    severity=IssueSeverity.MAJOR if is_major else IssueSeverity.MINOR,
                    message=f"Detected {len(empty_opts)}/{total_opts} ({opt_err_rate:.1%}) empty <option_text> tags.",
                )
            )
            deductions += min(25.0, opt_err_rate * 35.0 + (2.0 if not is_major else 10.0))

        # Ratio of option_labels to option_texts
        # Note: Tabular True/False questions may have option_text inside <td> without option_label (standard table HTML)
        if len(opt_labels) > 0 and len(opt_texts) == 0:
            issues.append(
                AuditIssue(
                    category="option_structure",
                    severity=IssueSeverity.CRITICAL,
                    message=f"Found {len(opt_labels)} <option_label> tags but ZERO <option_text> tags (orphaned choice labels).",
                )
            )
            deductions += 40.0
        elif len(opt_labels) > 0 and len(opt_texts) > 0:
            diff = abs(len(opt_labels) - len(opt_texts))
            max_opts = max(len(opt_labels), len(opt_texts))
            diff_ratio = diff / max_opts
            # Only flag if substantial mismatch and not standard table variance
            if diff_ratio >= 0.30 and diff > 3:
                is_major = diff_ratio >= 0.50
                issues.append(
                    AuditIssue(
                        category="option_structure",
                        severity=IssueSeverity.MAJOR if is_major else IssueSeverity.MINOR,
                        message=f"Imbalance between option labels ({len(opt_labels)}) and option texts ({len(opt_texts)}) ({diff_ratio:.1%} mismatch).",
                    )
                )
                deductions += min(15.0, diff_ratio * 20.0)

        # Sub-question check: check if stems contain un-tagged sub-item patterns (e.g. "\n- a)" or "\n- b)")
        subitem_in_stem_count = 0
        for s in stems:
            if re.search(r"(?:^|\n)\s*[-*•]?\s*[a-d]\)\s+[A-ZÀ-Ỹ0-9]", s):
                subitem_in_stem_count += 1

        if subitem_in_stem_count > 0:
            subitem_ratio = subitem_in_stem_count / max(1, total_questions)
            # MAJOR severity means systemic/repetitive error across a large portion (>= 15%) of the document
            is_major = subitem_ratio >= 0.15 or subitem_in_stem_count >= 4
            issues.append(
                AuditIssue(
                    category="question_structure",
                    severity=IssueSeverity.MAJOR if is_major else IssueSeverity.MINOR,
                    message=(
                        f"Found {subitem_in_stem_count}/{total_questions} ({subitem_ratio:.1%}) question stems containing "
                        f"un-tagged sub-questions ('a)', 'b)'). These must be tagged in <option_label> + <option_text>."
                    ),
                )
            )
            deductions += min(30.0, subitem_ratio * 40.0 + (3.0 if not is_major else 12.0))

        score = max(0.0, 100.0 - deductions)
        return issues, score, metrics

    @staticmethod
    def check_sequence_continuity(xml_content: str) -> Tuple[List[AuditIssue], float]:
        """
        Deprecated: Monotonic sequence continuity check has been removed in favor of
        format-invariant DET question coverage. Retained for backwards compatibility.
        """
        return [], 100.0

    @staticmethod
    def check_question_coverage_via_det(
        xml_content: str, raw_ocr_text: Optional[str]
    ) -> Tuple[List[AuditIssue], float, Dict[str, Any]]:
        """
        Audits parser question recall and coverage using the deterministic (DET) parser.
        Detects any question present in the raw source text that the parser left unannotated.
        Format-invariant: documents with 0 questions in both raw text and XML (e.g. reference tables,
        specification matrices, lecture notes) are awarded 100.0 without penalty.
        """
        issues: List[AuditIssue] = []
        metrics: Dict[str, Any] = {
            "det_total_questions": 0,
            "xml_total_questions": 0,
            "matched_questions_count": 0,
            "unannotated_questions_count": 0,
            "question_coverage_ratio": 1.0,
        }

        if not raw_ocr_text or not raw_ocr_text.strip():
            return issues, 100.0, metrics

        clean_raw = DeterministicAuditor.clean_raw_ocr_text(raw_ocr_text)
        if not clean_raw.strip():
            return issues, 100.0, metrics

        try:
            det_result = parse_chunk_deterministic(clean_raw)
        except Exception:
            return issues, 100.0, metrics

        # 1. Extract candidate questions from DET parse result
        det_questions: List[Dict[str, Any]] = []
        for i, span in enumerate(det_result.spans):
            if span.get("label") == "question_label":
                # Find matching stem
                stem_text = ""
                for j in range(i + 1, min(i + 5, len(det_result.spans))):
                    if det_result.spans[j].get("label") == "stem":
                        stem_text = det_result.spans[j].get("text", "")
                        break
                    elif det_result.spans[j].get("label") == "question_label":
                        break
                det_questions.append({
                    "label": span.get("text", "").strip(),
                    "stem": stem_text.strip(),
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                })

        # 2. Extract annotated questions from XML
        xml_matches = list(re.finditer(r"<question_label>(.*?)</question_label>", xml_content, re.DOTALL))
        xml_questions: List[Dict[str, Any]] = []
        for idx, qm in enumerate(xml_matches):
            ql_text = qm.group(1).strip()
            next_start = xml_matches[idx + 1].start() if idx + 1 < len(xml_matches) else len(xml_content)
            chunk = xml_content[qm.end():next_start]
            stem_m = re.search(r"<stem>(.*?)</stem>", chunk, re.DOTALL)
            stem_text = stem_m.group(1).strip() if stem_m else ""
            xml_questions.append({
                "label": ql_text,
                "stem": stem_text,
            })

        metrics["det_total_questions"] = len(det_questions)
        metrics["xml_total_questions"] = len(xml_questions)

        # If DET finds 0 questions in the raw document:
        # Non-exam / reference / specification text has no questions to annotate -> 100% valid
        if len(det_questions) == 0:
            return issues, 100.0, metrics

        # 3. Align DET questions against XML questions
        matched_det_indices = set()
        unannotated_questions = []

        for d_idx, q_det in enumerate(det_questions):
            d_label = q_det["label"]
            d_nums = re.findall(r"\d+", d_label)
            d_clean_label = re.sub(r"[^\w\s]", "", d_label).strip().lower()
            d_stem_norm = re.sub(r"\s+", " ", q_det["stem"]).strip().lower()
            d_stem_prefix = d_stem_norm[:50]

            is_matched = False
            for x_idx, q_xml in enumerate(xml_questions):
                x_label = q_xml["label"]
                x_nums = re.findall(r"\d+", x_label)
                x_clean_label = re.sub(r"[^\w\s]", "", x_label).strip().lower()
                x_stem_norm = re.sub(r"\s+", " ", q_xml["stem"]).strip().lower()

                # Strategy A: Question numbers match and labels align
                if d_nums and x_nums and d_nums == x_nums:
                    # If numbers match, verify stem isn't completely contradictory if both stems exist
                    if len(d_stem_prefix) >= 15 and len(x_stem_norm) >= 15:
                        sim = SequenceMatcher(None, d_stem_prefix, x_stem_norm[:60]).ratio()
                        if sim >= 0.40 or d_stem_prefix[:25] in x_stem_norm or x_stem_norm[:25] in d_stem_norm:
                            is_matched = True
                            break
                    else:
                        is_matched = True
                        break

                # Strategy B: Clean labels match exactly
                if d_clean_label and d_clean_label == x_clean_label:
                    is_matched = True
                    break

                # Strategy C: High stem overlap (even if label formatting differed)
                if len(d_stem_prefix) >= 20 and len(x_stem_norm) >= 20:
                    if d_stem_prefix[:35] in x_stem_norm or x_stem_norm[:35] in d_stem_norm:
                        is_matched = True
                        break
                    sim = SequenceMatcher(None, d_stem_prefix, x_stem_norm[:60]).ratio()
                    if sim >= 0.75:
                        is_matched = True
                        break

            if is_matched:
                matched_det_indices.add(d_idx)
            else:
                unannotated_questions.append(q_det)

        matched_count = len(matched_det_indices)
        total_det = len(det_questions)
        coverage_ratio = matched_count / total_det if total_det > 0 else 1.0

        metrics["matched_questions_count"] = matched_count
        metrics["unannotated_questions_count"] = len(unannotated_questions)
        metrics["question_coverage_ratio"] = round(coverage_ratio, 3)

        coverage_score = round(coverage_ratio * 100.0, 1)

        # 4. Generate issues for unannotated questions
        if unannotated_questions:
            unannot_ratio = len(unannotated_questions) / total_det
            if unannot_ratio >= 0.50 and len(unannotated_questions) >= 4:
                severity = IssueSeverity.CRITICAL
            elif unannot_ratio >= 0.20 or len(unannotated_questions) >= 3:
                severity = IssueSeverity.MAJOR
            else:
                severity = IssueSeverity.MINOR

            for u_q in unannotated_questions[:10]:
                snippet = f"{u_q['label']} {u_q['stem'][:70]}".strip()
                issues.append(
                    AuditIssue(
                        category="unannotated_question",
                        severity=severity,
                        message=f"Question '{u_q['label']}' present in raw source text was left unannotated by the parser.",
                        context_snippet=snippet,
                    )
                )
            if len(unannotated_questions) > 10:
                issues.append(
                    AuditIssue(
                        category="unannotated_question",
                        severity=severity,
                        message=f"... and {len(unannotated_questions) - 10} additional unannotated questions omitted by parser.",
                    )
                )

        return issues, coverage_score, metrics

    @staticmethod
    def check_stimulus_anchors(
        xml_content: str, pure_text: str, raw_ocr_text: Optional[str] = None
    ) -> Tuple[List[AuditIssue], float]:
        """
        Audits stimulus anchor tags (start_anchor, end_anchor) and multi-question rule.
        """
        issues: List[AuditIssue] = []
        deductions = 0.0

        stimulus_tags = re.findall(r"<stimulus\b([^>]*)/?>", xml_content)
        if not stimulus_tags:
            return issues, 100.0

        for stim_idx, stim_attr in enumerate(stimulus_tags):
            start_m = re.search(r'start_anchor="([^"]*)"', stim_attr)
            end_m = re.search(r'end_anchor="([^"]*)"', stim_attr)

            if not start_m or not end_m:
                issues.append(
                    AuditIssue(
                        category="stimulus",
                        severity=IssueSeverity.MAJOR,
                        message=f"Stimulus #{stim_idx+1} missing required 'start_anchor' or 'end_anchor' attribute.",
                        context_snippet=f"<stimulus {stim_attr} />",
                    )
                )
                deductions += 15.0
                continue

            start_anchor = start_m.group(1).strip()
            end_anchor = end_m.group(1).strip()

            if len(start_anchor) < 3 or len(end_anchor) < 3:
                issues.append(
                    AuditIssue(
                        category="stimulus",
                        severity=IssueSeverity.MINOR,
                        message=f"Stimulus #{stim_idx+1} has very short anchors ('{start_anchor}', '{end_anchor}'). Recommended 3-10 words.",
                    )
                )
                deductions += 5.0

            # If raw OCR text is provided, verify anchors exist in source document
            if raw_ocr_text:
                clean_raw = DeterministicAuditor.clean_raw_ocr_text(raw_ocr_text)
                norm_raw = " ".join(clean_raw.split())
                norm_start = " ".join(start_anchor.split())
                norm_end = " ".join(end_anchor.split())

                if norm_start and norm_start not in norm_raw:
                    issues.append(
                        AuditIssue(
                            category="stimulus",
                            severity=IssueSeverity.MAJOR,
                            message=f"Stimulus #{stim_idx+1} start_anchor '{start_anchor}' not found in raw source document text.",
                        )
                    )
                    deductions += 15.0

                if norm_end and norm_end not in norm_raw:
                    issues.append(
                        AuditIssue(
                            category="stimulus",
                            severity=IssueSeverity.MAJOR,
                            message=f"Stimulus #{stim_idx+1} end_anchor '{end_anchor}' not found in raw source document text.",
                        )
                    )
                    deductions += 15.0

        score = max(0.0, 100.0 - deductions)
        return issues, score

    @staticmethod
    def clean_raw_ocr_text(raw_ocr_text: Optional[str]) -> str:
        """
        Strips wrapper metadata (like <page_metadata> JSON blocks, <page>, <pages>)
        to extract pure source document text for verbatim comparison.
        """
        if not raw_ocr_text:
            return ""
        # Strip <page_metadata> blocks whether closed with </page_metadata> or terminated at </page>
        cleaned = re.sub(
            r"<page_metadata>.*?(?:</page_metadata>|</page>)",
            "</page>",
            raw_ocr_text,
            flags=re.DOTALL,
        )
        # Strip <pages>, </pages>, <page>, </page>
        cleaned = re.sub(r"</?pages?>", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def check_verbatim_alignment(
        xml_content: str, raw_ocr_text: Optional[str]
    ) -> Tuple[List[AuditIssue], float, Dict[str, Any]]:
        """
        Audits verbatim content fidelity against raw OCR text if provided.
        Precision-oriented: verifies that all tagged text in XML is 100% faithful to the source
        with zero hallucination or paraphrasing, via span-level substring containment & local LCS alignment.
        Decoupled from crude document-length ratio: omitting non-exam lecture/guide text is NOT penalized.
        """
        issues: List[AuditIssue] = []
        pure_text = DeterministicAuditor.strip_xml_tags(xml_content)
        annotated_chars = len(pure_text)
        annotated_words = len(pure_text.split())

        clean_raw = DeterministicAuditor.clean_raw_ocr_text(raw_ocr_text) if raw_ocr_text else ""
        raw_chars = len(clean_raw) if clean_raw else 0
        raw_words = len(clean_raw.split()) if clean_raw else 0

        metrics: Dict[str, Any] = {
            "annotated_char_count": annotated_chars,
            "annotated_word_count": annotated_words,
            "raw_char_count": raw_chars if raw_ocr_text else None,
            "retention_ratio": None,
            "verbatim_span_fidelity": 100.0,
        }

        if not raw_ocr_text or raw_chars == 0:
            return issues, 100.0, metrics

        metrics["raw_char_count"] = raw_chars
        metrics["raw_word_count"] = raw_words

        retention_ratio = annotated_chars / raw_chars
        metrics["retention_ratio"] = round(retention_ratio, 3)

        # 1. Hallucination / runaway repetition checks
        if retention_ratio > 1.35:
            issues.append(
                AuditIssue(
                    category="verbatim_fidelity",
                    severity=IssueSeverity.CRITICAL,
                    message=f"Severe hallucination / repetition: Annotated output is {retention_ratio*100:.1f}% the size of raw OCR text ({annotated_chars}/{raw_chars} chars).",
                )
            )
        elif retention_ratio > 1.15:
            issues.append(
                AuditIssue(
                    category="verbatim_fidelity",
                    severity=IssueSeverity.MINOR,
                    message=f"Possible text repetition/hallucination: Output size is {retention_ratio*100:.1f}% of input.",
                )
            )

        # Non-exam lecture/guide text omission is recorded as INFO, zero penalty
        if retention_ratio < 0.25:
            issues.append(
                AuditIssue(
                    category="verbatim_fidelity",
                    severity=IssueSeverity.INFO,
                    message=f"Annotated text represents {retention_ratio*100:.1f}% of raw document. Extraneous non-exam sections (lectures, guides) pruned.",
                )
            )

        norm_raw = " ".join(clean_raw.split())
        norm_raw_stripped = " ".join(DeterministicAuditor.strip_xml_tags(clean_raw).split())

        # 2. Extract content spans from XML (<stem>, <option_text>, <question_label>, <explanation>, <section>)
        content_spans: List[Tuple[str, str]] = []
        span_pattern = re.compile(
            r"<(stem|option_text|question_label|explanation|section)\b[^>]*>(.*?)</\1>",
            re.DOTALL | re.IGNORECASE,
        )
        for m in span_pattern.finditer(xml_content):
            tag_name = m.group(1).lower()
            inner = m.group(2)
            clean_span = DeterministicAuditor.strip_xml_tags(inner).strip()
            if len(clean_span) >= 3:
                content_spans.append((tag_name, clean_span))

        if not content_spans:
            return issues, 100.0, metrics

        total_weight = 0
        weighted_score = 0.0
        hallucinated_spans: List[Tuple[str, str, float]] = []

        for tag, text in content_spans:
            norm_span = " ".join(text.split())
            if not norm_span:
                continue
            w = len(norm_span)
            total_weight += w

            # A. Exact substring match in normalized raw text (with or without stripped XML/figure markup)
            if norm_span in norm_raw or norm_span in norm_raw_stripped:
                weighted_score += w * 1.0
                continue

            # B. Local LCS / SequenceMatcher alignment (test against stripped raw text first to avoid figure/table tag drag)
            prefix_len = min(30, len(norm_span))
            prefix = norm_span[:prefix_len]
            pos = norm_raw_stripped.find(prefix)
            target_raw = norm_raw_stripped if pos != -1 else norm_raw
            if pos == -1:
                pos = norm_raw.find(prefix)

            best_ratio = 0.0
            if pos != -1:
                window = target_raw[pos : pos + len(norm_span) + 30]
                best_ratio = SequenceMatcher(None, norm_span, window).ratio()
            else:
                words = norm_span.split()
                if len(words) >= 4:
                    anchor = " ".join(words[:4])
                    pos = norm_raw_stripped.find(anchor)
                    target_raw = norm_raw_stripped if pos != -1 else norm_raw
                    if pos == -1:
                        pos = norm_raw.find(anchor)
                    if pos != -1:
                        window = target_raw[pos : pos + len(norm_span) + 30]
                        best_ratio = SequenceMatcher(None, norm_span, window).ratio()
                    else:
                        anchor_tail = " ".join(words[-4:])
                        pos = norm_raw_stripped.find(anchor_tail)
                        target_raw = norm_raw_stripped if pos != -1 else norm_raw
                        if pos == -1:
                            pos = norm_raw.find(anchor_tail)
                        if pos != -1:
                            window = target_raw[max(0, pos - len(norm_span) - 20) : pos + len(anchor_tail)]
                            best_ratio = SequenceMatcher(None, norm_span, window).ratio()

            if best_ratio >= 0.90:
                # Minor whitespace or LaTeX formatting variance ($ vs $$) -> full credit
                weighted_score += w * 1.0
            elif best_ratio >= 0.75:
                weighted_score += w * best_ratio
            else:
                weighted_score += w * max(0.0, best_ratio)
                hallucinated_spans.append((tag, norm_span, best_ratio))

        verbatim_fidelity_score = (
            round((weighted_score / total_weight) * 100.0, 1) if total_weight > 0 else 100.0
        )

        if retention_ratio > 1.35:
            verbatim_fidelity_score = max(0.0, verbatim_fidelity_score - 40.0)
        elif retention_ratio > 1.15:
            verbatim_fidelity_score = max(0.0, verbatim_fidelity_score - 10.0)

        metrics["verbatim_span_fidelity"] = verbatim_fidelity_score

        if hallucinated_spans:
            hallucinated_weight = sum(len(s[1]) for s in hallucinated_spans)
            hallucinated_ratio = hallucinated_weight / total_weight if total_weight > 0 else 0.0
            is_major = hallucinated_ratio >= 0.15 or len(hallucinated_spans) >= 4

            for h_tag, h_span, h_ratio in hallucinated_spans[:5]:
                issues.append(
                    AuditIssue(
                        category="verbatim_fidelity",
                        severity=IssueSeverity.MAJOR if is_major else IssueSeverity.MINOR,
                        message=f"Annotated text in <{h_tag}> appears modified or hallucinated ({h_ratio*100:.1f}% similarity).",
                        context_snippet=h_span[:80],
                    )
                )

        return issues, verbatim_fidelity_score, metrics



PARSER_ERROR_PENALTIES: Dict[str, Dict[IssueSeverity, float]] = {
    # Unannotated question present in raw text
    "unannotated_question": {
        IssueSeverity.CRITICAL: 50.0,
        IssueSeverity.MAJOR: 20.0,
        IssueSeverity.MINOR: 10.0,
        IssueSeverity.INFO: 0.0,
    },

    # Stimulus anchor omitted citation / source attribution
    "stimulus_missing_citation": {
        IssueSeverity.CRITICAL: 25.0,
        IssueSeverity.MAJOR: 12.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Stimulus illegally wrapping system tags (Fatal architectural flaw)
    "stimulus_nesting": {
        IssueSeverity.CRITICAL: 100.0,
        IssueSeverity.MAJOR: 50.0,
        IssueSeverity.MINOR: 20.0,
        IssueSeverity.INFO: 0.0,
    },
    # Stimulus used for single question instead of 2+ questions
    "single_question_stimulus": {
        IssueSeverity.CRITICAL: 20.0,
        IssueSeverity.MAJOR: 10.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Stimulus anchor not matching raw text or invalid
    "stimulus_anchor_invalid": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Broken, unclosed, or mismatched SYSTEM tags
    "broken_system_tag": {
        IssueSeverity.CRITICAL: 40.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Missing question label
    "missing_question_label": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Missing stem
    "missing_stem": {
        IssueSeverity.CRITICAL: 35.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Missing choices / options
    "missing_options": {
        IssueSeverity.CRITICAL: 35.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Absorbed subquestions a), b) inside stem
    "absorbed_subquestions": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },

    # Mislabelled element (e.g. stem tagged as section)
    "mislabelled_element": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Prohibited tags retained (<page>, <page_metadata>)
    "prohibited_tags": {
        IssueSeverity.CRITICAL: 40.0,
        IssueSeverity.MAJOR: 20.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    # Truncated output at EOF
    "truncated_output": {
        IssueSeverity.CRITICAL: 50.0,
        IssueSeverity.MAJOR: 25.0,
        IssueSeverity.MINOR: 10.0,
        IssueSeverity.INFO: 0.0,
    },
    # Standard fallback categories
    "xml_syntax": {
        IssueSeverity.CRITICAL: 40.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "schema_conformance": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 10.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "stimulus": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "question_structure": {
        IssueSeverity.CRITICAL: 40.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "option_structure": {
        IssueSeverity.CRITICAL: 35.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "verbatim_fidelity": {
        IssueSeverity.CRITICAL: 40.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
    "llm_semantic": {
        IssueSeverity.CRITICAL: 30.0,
        IssueSeverity.MAJOR: 15.0,
        IssueSeverity.MINOR: 5.0,
        IssueSeverity.INFO: 0.0,
    },
}


def is_spurious_end_sentinel_issue(category: str, message: str) -> bool:
    """
    Detects false-positive issues complaining about the absence of the <|END|> sentinel.
    The <|END|> token is an internal LLM generation stop delimiter that is automatically
    stripped upon saving, so its absence in annotated XML documents is completely expected
    and valid.
    """
    msg_clean = message.lower()
    cat_clean = category.lower().strip()

    # Must specifically mention <|end|> or end sentinel
    mentions_end_sentinel = (
        "<|end|>" in msg_clean
        or "< | end | >" in msg_clean
        or "end sentinel" in msg_clean
        or ("sentinel" in msg_clean and "end" in msg_clean)
        or ("sentinel" in msg_clean and cat_clean == "truncated_output")
    )
    if not mentions_end_sentinel:
        return False

    # Check for words indicating complaint about absence / missing / requiring it
    absence_cues = [
        "without",
        "missing",
        "require",
        "lack",
        "not found",
        "omitted",
        "premature",
        "no <|end|>",
        "ends without",
        "does not end",
        "doesn't end",
    ]
    return any(cue in msg_clean for cue in absence_cues)


def _audit_issue_payload(issues: List[AuditIssue]) -> List[Dict[str, Any]]:
    """Serialize deterministic findings with stable IDs for LLM confirmation."""
    return [
        {
            "issue_id": issue.issue_id or index,
            "category": issue.category,
            "severity": issue.severity.value,
            "message": issue.message,
            "line_number": issue.line_number,
            "context_snippet": issue.context_snippet,
        }
        for index, issue in enumerate(issues, start=1)
    ]


def _assign_issue_ids(issues: List[AuditIssue]) -> None:
    """Assign deterministic, per-review IDs used by the confirmation protocol."""
    for index, issue in enumerate(issues, start=1):
        if issue.issue_id is None:
            issue.issue_id = index


def _confirmed_issue_keys(confirmations: List[Dict[str, Any]]) -> Set[int]:
    """Return IDs explicitly confirmed as true positives by the reviewer LLM."""
    return {
        int(item["issue_id"])
        for item in confirmations
        if item.get("is_true_positive") is True and item.get("issue_id") is not None
    }


def _normalise_issue_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _is_duplicate_llm_issue(
    llm_issue: Dict[str, Any], deterministic_issues: List[Dict[str, Any]]
) -> Optional[int]:
    """Return the matching deterministic issue ID when an LLM issue repeats it."""
    explicit_id = llm_issue.get("deterministic_issue_id", llm_issue.get("duplicate_of"))
    if explicit_id is not None:
        try:
            explicit_id = int(explicit_id)
        except (TypeError, ValueError):
            explicit_id = None
        if explicit_id and any(item.get("issue_id") == explicit_id for item in deterministic_issues):
            return explicit_id

    llm_key = (
        _normalise_issue_text(llm_issue.get("category")),
        _normalise_issue_text(llm_issue.get("severity")),
        _normalise_issue_text(llm_issue.get("message")),
    )
    for item in deterministic_issues:
        det_key = (
            _normalise_issue_text(item.get("category")),
            _normalise_issue_text(item.get("severity")),
            _normalise_issue_text(item.get("message")),
        )
        if llm_key == det_key:
            return item.get("issue_id")
    return None


def _normalise_llm_confirmations(
    raw_confirmations: Any, deterministic_issues: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Normalize LLM true-positive confirmations and attach source context."""
    if not isinstance(raw_confirmations, list):
        return []

    by_id = {item.get("issue_id"): item for item in deterministic_issues}
    confirmations: List[Dict[str, Any]] = []
    for item in raw_confirmations:
        if not isinstance(item, dict):
            continue

        issue_id = item.get("issue_id", item.get("deterministic_issue_id"))
        try:
            issue_id = int(issue_id) if issue_id is not None else None
        except (TypeError, ValueError):
            issue_id = None

        matched = by_id.get(issue_id)
        if matched is None:
            candidate_category = _normalise_issue_text(item.get("category"))
            candidate_message = _normalise_issue_text(item.get("message"))
            for candidate in deterministic_issues:
                if (
                    candidate_category == _normalise_issue_text(candidate.get("category"))
                    and candidate_message == _normalise_issue_text(candidate.get("message"))
                ):
                    matched = candidate
                    issue_id = candidate.get("issue_id")
                    break

        confirmed = item.get("is_true_positive", item.get("true_positive", item.get("confirmed")))
        if isinstance(confirmed, str):
            normalised_confirmed = confirmed.strip().lower()
            if normalised_confirmed in {"true", "yes", "1", "confirmed"}:
                confirmed = True
            elif normalised_confirmed in {"false", "no", "0", "rejected", "false_positive"}:
                confirmed = False
            else:
                # An ambiguous response must not be treated as a false
                # positive or allowed through to the editor.
                confirmed = None
        elif not isinstance(confirmed, bool):
            confirmed = None

        confirmation = {
            "issue_id": issue_id,
            "is_true_positive": confirmed,
            "reason": str(item.get("reason", item.get("explanation", ""))).strip(),
        }
        if matched is not None:
            confirmation.update(
                {
                    "category": matched.get("category"),
                    "severity": matched.get("severity"),
                    "message": matched.get("message"),
                }
            )
        else:
            confirmation.update(
                {
                    "category": item.get("category"),
                    "severity": item.get("severity"),
                    "message": item.get("message"),
                }
            )
        confirmations.append(confirmation)
    return confirmations


def compute_llm_penalties(
    llm_issues: List[AuditIssue],
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Computes algorithmic penalties for issues detected by the reviewer LLM based on their label / category and severity.
    Returns:
        (total_penalty, discard_reasons, rubric_deductions)
    """
    total_penalty = 0.0
    discard_reasons: List[str] = []
    rubric_deductions: Dict[str, float] = {
        "xml_well_formedness": 0.0,
        "schema_conformance": 0.0,
        "det_question_coverage": 0.0,
        "verbatim_fidelity": 0.0,
        "question_option_completeness": 0.0,
        "stimulus_accuracy": 0.0,
    }

    for iss in llm_issues:
        cat = iss.category.lower().strip()
        sev = iss.severity
        if is_spurious_end_sentinel_issue(cat, iss.message):
            continue
        cat_map = PARSER_ERROR_PENALTIES.get(cat, PARSER_ERROR_PENALTIES["llm_semantic"])
        penalty = cat_map.get(sev, 5.0)
        total_penalty += penalty

        # Fatal architectural flaws trigger immediate discard
        if sev == IssueSeverity.CRITICAL:
            discard_reasons.append(f"[{cat.upper()}] {iss.message}")
        elif cat == "stimulus_nesting":
            discard_reasons.append(f"[{cat.upper()}] {iss.message}")

        # Map penalty to rubric dimensions
        if cat in ["broken_system_tag", "xml_syntax", "truncated_output"]:
            rubric_deductions["xml_well_formedness"] += penalty
        elif cat in ["prohibited_tags", "schema_conformance", "mislabelled_element"]:
            rubric_deductions["schema_conformance"] += penalty
        elif cat in ["unannotated_question"]:
            rubric_deductions["det_question_coverage"] += penalty
        elif cat in ["verbatim_mutation", "verbatim_fidelity"]:
            rubric_deductions["verbatim_fidelity"] += penalty
        elif cat in [
            "missing_question_label",
            "missing_stem",
            "missing_options",
            "absorbed_subquestions",
            "question_structure",
            "option_structure",
        ]:
            rubric_deductions["question_option_completeness"] += penalty
        elif cat in [
            "stimulus_missing_citation",
            "stimulus_nesting",
            "single_question_stimulus",
            "stimulus_anchor_invalid",
            "stimulus",
        ]:
            rubric_deductions["stimulus_accuracy"] += penalty
        else:
            rubric_deductions["schema_conformance"] += penalty * 0.5
            rubric_deductions["question_option_completeness"] += penalty * 0.5

    return total_penalty, discard_reasons, rubric_deductions


class DeepSeekReviewer:
    """
    LLM-powered Semantic Reviewer using DeepSeek Client for in-depth quality analysis.
    Informed by the complete original parser ground-truth prompt rules, constraints, and raw OCR source text.
    """

    SYSTEM_PROMPT = """# [System Config]
Role: You are an expert AI Quality Assurance & Sequence Labelling Auditor for Exam Documents (TOEIC, SAT, High School National Exams).
Your task is to critically inspect an annotated XML exam document produced by an automated sequence-labelling parser against its original raw OCR source text and strict ground-truth specifications.

CRITICAL INSTRUCTION ON SCORING:
You do NOT calculate or provide any numerical scores or rubric scores.
Your sole responsibility is to validate the deterministic parser findings and identify any additional errors, defects, and discrepancies caused by the parser.
The audit algorithm will automatically compute document scores and assign penalties based on the error labels and severity you report.

CRITICAL INSTRUCTION ON DUPLICATES:
The user prompt may contain numbered findings from a deterministic parser pre-check.
You MUST review every numbered finding and return one confirmation for each finding when possible,
stating whether it is a true positive. Confirmed deterministic findings are already present in the
final audit and MUST NOT be repeated in your `issues` array. The `issues` array must contain only
new parser errors that are not represented by the deterministic findings. If a deterministic finding
is a false positive, report that in `parser_error_confirmations`; do not copy it into `issues`.

---

## 🛠️ How the Sequence Labelling Parser Works (Architecture & Principles):
The parser is an autoregressive sequence annotator that consumes raw OCR text of educational exam papers and annotates questions, options, sections, and stimuli using inline XML tags.

### Complete Real-World Parser Example:
#### Raw OCR Input Text:
```text
SỞ GD&ĐT HÀ NỘI
TRƯỜNG THPT CHUYÊN HÀ NỘI - AMSTERDAM
ĐỀ THI THỬ TỐT NGHIỆP THPT NĂM 2025
Môn: KHOA HỌC TỰ NHIÊN - VẬT LÍ

PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.

Dựa vào thông tin sau đây để trả lời các câu 1 và 2:
Sóng điện từ là điện từ trường lan truyền trong không gian. Trong quá trình lan truyền sóng điện từ, tại một điểm bất kỳ trong không gian, vectơ cường độ điện trường $\\vec{E}$ và vectơ cảm ứng từ $\\vec{B}$ luôn dao động cùng pha, vuông phương với nhau và vuông góc với phương truyền sóng. Tốc độ lan truyền của sóng điện từ trong chân không bằng $c = 3 \\cdot 10^8\\text{ m/s}$.
(Trích Sách giáo khoa Vật lí 12 - Bộ Kết nối tri thức với cuộc sống, NXB Giáo dục Việt Nam)

Câu 1: Sóng điện từ
A. là sóng dọc lan truyền trong môi trường vật chất.
B. có thành phần điện trường và từ trường dao động cùng pha.
C. không truyền được trong môi trường chân không.
D. có vectơ $\\vec{E}$ và $\\vec{B}$ dao động cùng phương.

Câu 2: Trong chân không, một sóng điện từ có tần số $f = 100\\text{ MHz}$. Bước sóng của sóng này là
A. $3\\text{ m}$.
B. $0{,}3\\text{ m}$.
C. $30\\text{ m}$.
D. $300\\text{ m}$.

PHẦN II. Câu trắc nghiệm đúng sai. Thí sinh trả lời câu 3.
Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.

Câu 3: Cho con lắc lò xo dao động điều hòa theo phương ngang với phương trình $x = 4\\cos(10t)\\text{ cm}$.
a) Biên độ dao động của con lắc là $A = 4\\text{ cm}$.
b) Tần số góc dao động là $\\omega = 10\\text{ rad/s}$.
c) Gia tốc cực đại của vật là $a_{\\max} = 400\\text{ cm/s}^2$.
d) Tại thời điểm $t = 0$, vật đang đi qua vị trí cân bằng theo chiều dương.

HƯỚNG DẪN GIẢI CHI TIẾT:
Câu 1: Chọn B vì vectơ E và B dao động cùng pha.
Câu 2: Bước sóng $\\lambda = c / f = 3 \\cdot 10^8 / 10^8 = 3\\text{ m}$. Chọn A.
```

#### Expected Parser Output XML:
```xml
<section>SỞ GD&ĐT HÀ NỘI
TRƯỜNG THPT CHUYÊN HÀ NỘI - AMSTERDAM
ĐỀ THI THỬ TỐT NGHIỆP THPT NĂM 2025
Môn: KHOA HỌC TỰ NHIÊN - VẬT LÍ

PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.</section>

<stimulus id="stim_1" start_anchor="Dựa vào thông tin sau đây" end_anchor="NXB Giáo dục Việt Nam)" />

<question_label>Câu 1:</question_label> <stem>Sóng điện từ</stem>
- <option_label>A.</option_label> <option_text>là sóng dọc lan truyền trong môi trường vật chất.</option_text>
- <option_label>B.</option_label> <option_text>có thành phần điện trường và từ trường dao động cùng pha.</option_text>
- <option_label>C.</option_label> <option_text>không truyền được trong môi trường chân không.</option_text>
- <option_label>D.</option_label> <option_text>có vectơ $\\vec{E}$ và $\\vec{B}$ dao động cùng phương.</option_text>

<question_label>Câu 2:</question_label> <stem>Trong chân không, một sóng điện từ có tần số $f = 100\\text{ MHz}$. Bước sóng của sóng này là</stem>
- <option_label>A.</option_label> <option_text>$3\\text{ m}$.</option_text>
- <option_label>B.</option_label> <option_text>$0{,}3\\text{ m}$.</option_text>
- <option_label>C.</option_label> <option_text>$30\\text{ m}$.</option_text>
- <option_label>D.</option_label> <option_text>$300\\text{ m}$.</option_text>

<section>PHẦN II. Câu trắc nghiệm đúng sai. Thí sinh trả lời câu 3.
Trong mỗi ý a), b), c), d) ở mỗi câu, thí sinh chọn đúng hoặc sai.</section>

<question_label>Câu 3:</question_label> <stem>Cho con lắc lò xo dao động điều hòa theo phương ngang với phương trình $x = 4\\cos(10t)\\text{ cm}$.</stem>
- <option_label>a)</option_label> <option_text>Biên độ dao động của con lắc là $A = 4\\text{ cm}$.</option_text>
- <option_label>b)</option_label> <option_text>Tần số góc dao động là $\\omega = 10\\text{ rad/s}$.</option_text>
- <option_label>c)</option_label> <option_text>Gia tốc cực đại của vật là $a_{\\max} = 400\\text{ cm/s}^2$.</option_text>
- <option_label>d)</option_label> <option_text>Tại thời điểm $t = 0$, vật đang đi qua vị trí cân bằng theo chiều dương.</option_text>

<explanation>HƯỚNG DẪN GIẢI CHI TIẾT:
Câu 1: Chọn B vì vectơ E và B dao động cùng pha.
Câu 2: Bước sóng $\\lambda = c / f = 3 \\cdot 10^8 / 10^8 = 3\\text{ m}$. Chọn A.</explanation>
```

---

## 🏷️ System Tags Dictionary & Auditing Scope:
### SYSTEM XML TAGS (The ONLY tags you should audit for syntax and closure):
1. `<section>...</section>`: Major section/part titles, exam directions, headers. Must be full paired tags containing verbatim text. Never use anchor tags for sections.
2. `<stimulus id="stim_N" start_anchor="..." end_anchor="..." />`: Compact self-closing anchor tag for shared reading passages, tables, or datasets serving 2 OR MORE QUESTIONS.
   - Must be self-closing (`/>`).
   - `start_anchor`: First 3-10 verbatim words.
   - `end_anchor`: Last 3-10 verbatim words. MUST include the citation/source attribution at the end if present!
   - FATAL CONSTRAINT: Must NEVER wrap or enclose question elements (`<stem>`, `<question_label>`, `<option_label>`, `<option_text>`, `<explanation>`).
   - MULTI-QUESTION CONSTRAINT: A stimulus ONLY applies if shared across 2+ questions. Single-question context belongs in `<stem>`.
3. `<question_label>...</question_label>`: Question prefix indicator ONLY (e.g. "**101.**", "Câu 1:"). Must NEVER wrap stems or reading passages.
4. `<stem>...</stem>`: Main text body of the question.
5. `<option_label>...</option_label>`: Choice letters/prefixes (A., B.) AND sub-question markers (a), b), c), d)) in True/False or structured questions.
6. `<option_text>...</option_text>`: Content of choices or sub-question items, or statement cells in tabular True/False questions.
7. `<explanation>...</explanation>`: Solutions, explanations, and answer keys.
8. `<question>...</question>`: Legacy wrapper (rarely used, paired tag).
9. `<figure id="..." description="..." bbox="..." />`: Vision OCR figure placeholder (immutable source text).

### NON-SYSTEM XML TAGS & FORMATTING (COMPLETELY IGNORE IF BROKEN OR UNCLOSED):
- HTML formatting tags: `<table>`, `<tr>`, `<td>`, `<th>`, `<b>`, `<i>`, `<u>`, `<strong>`, `<em>`, `<span>`, `<div>`, `<p>`, `<ul>`, `<ol>`, `<li>`, `<br>`, etc.
- LaTeX mathematical comparisons: `$x < y$`, `$a > b$`, `$\\alpha < \\beta$`.
- RULE: You MUST ONLY care about SYSTEM tags if they are missing closing tags, broken, or mismatched. You MUST COMPLETELY IGNORE broken, unclosed, or mismatched NON-SYSTEM tags! Never report unclosed table or text styling tags as syntax errors.
- VERBATIM SOURCE FIDELITY: The parser pipeline uses a source-grounded compiler that automatically enforces 100% exact verbatim preservation of original OCR characters by projecting tag offsets onto the source text. You do NOT need to check for word-level paraphrasing or text alterations; focus 100% of your audit on system XML tag placement, structure, boundaries, and coverage.

---

## 📋 Comprehensive Taxonomy of Possible Parser Errors:
When reporting issues, you MUST categorize each error using one of the following exact labels:

1. `unannotated_question`:
   - A question present in the raw OCR source text was completely omitted or left unannotated by the parser.
   - Example: "Question 'Câu 5' present in raw OCR text was omitted/unannotated by the parser."

2. `stimulus_missing_citation`:
   - A `<stimulus>` anchor tag omitted the citation, source attribution, reading passage title, or author info at the end (or beginning) of the passage.
   - Example: "Stimulus 'stim_1' end_anchor cut off at 'trong chân không.' omitting the source citation '(Trích SGK Vật lí 12, NXB Giáo dục)'."

3. `stimulus_nesting`:
   - [CRITICAL FATAL] A `<stimulus>` tag illegally wraps or encloses other system tags (`<stem>`, `<question_label>`, `<option_label>`, `<option_text>`, `<explanation>`).
   - Example: "Stimulus tag <stimulus ...>...</stimulus> illegally encloses <question_label> and <stem>."

4. `single_question_stimulus`:
   - A `<stimulus>` tag was created for context, passage, or table that only serves 1 single question instead of 2 or more questions.
   - Example: "Stimulus 'stim_2' only relates to Câu 4; single-question context must be included directly inside <stem>."

5. `stimulus_anchor_invalid`:
   - The `start_anchor` or `end_anchor` text in a `<stimulus>` tag does not match the raw source text verbatim, or anchors are too short (< 3 words) or ambiguous.
   - Example: "Stimulus 'stim_1' start_anchor 'Đọc đoạn' is too short and ambiguous."

6. `broken_system_tag`:
   - A SYSTEM tag is missing a closing tag, has a mismatched closing tag, or has broken syntax (e.g. truncated mid-tag).
   - Example: "Unclosed system tag <stem> at question Câu 3 was never closed before <option_label>."
   - Note: Do NOT flag non-system tags (e.g. unclosed <td>, <b>, <tr>) under this or any category!

7. `missing_question_label`:
   - A question stem or options appear without a `<question_label>` tag.
   - Example: "Question stem 'Cho hình chóp...' has no preceding <question_label> tag."

8. `missing_stem`:
   - A question has question statement or prompt text in the raw source document, but the parser omitted the `<stem>` tag or left an empty `<stem></stem>`.
   - IMPORTANT INVARIANT: A `<question_label>` does NOT have to be followed by a `<stem>`. In cloze tests, fill-in-the-blank passages, or reading comprehension items where blanks are numbered within a reading passage or stimulus, questions legitimately consist of only `<question_label>` followed immediately by `<option_label>`. These are 100% VALID sequence labeling outputs. NEVER flag `<question_label>` followed by `<option_label>` as `missing_stem`.
   - Example: "Raw text has question statement 'Tính thể tích khối lăng trụ...', but parser omitted the <stem> tag and jumped directly to choices."

9. `missing_options`:
   - A multiple-choice or True/False question stem has options in the raw text, but the parser omitted all `<option_label>` and `<option_text>` tags.
   - Example: "Question 'Câu 7' stem is tagged, but choices A, B, C, D present in raw text were omitted."

10. `absorbed_subquestions`:
    - Sub-questions (a), b), c), d)) in essay, constructed-response, or True/False questions were absorbed into `<stem>` instead of being tagged with `<option_label>` and `<option_text>`.
    - Example: "Sub-items 'a)', 'b)' in Câu 3 were absorbed into <stem> instead of tagged with <option_label> + <option_text>."

11. `mislabelled_element`:
    - An element was tagged with the wrong system tag (e.g., question stem tagged as <section>, explanation tagged as <stem>, or question label encompassing the stem text).
    - Example: "Question stem was tagged as <section> instead of <stem>."

12. `prohibited_tags`:
    - Prohibited page tags (`<page>`, `<pages>`, `<page_metadata>`) or `<think>` tags were retained in the output instead of being pruned.
    - Example: "Document retains unpruned <page> and <page_metadata> tags."

13. `truncated_output`:
    - Output generation was cut off mid-tag, mid-word, or mid-sentence before the end of the document (e.g. trailing unclosed tag fragment like `<opt` or abruptly terminated sentence at file end, or questions missing at document end relative to raw source text).
    - NEVER report missing `<|END|>` under this category: annotated XML documents are stored WITHOUT `<|END|>` (it is automatically stripped upon saving); its absence is completely normal and valid.
    - Example: "Document was truncated mid-tag '<opt' at file end."

---

## ⛔ Severity Levels:
- **CRITICAL**: Fatal architectural violations that render the document unusable or poison downstream models (e.g. `stimulus_nesting`, truncated output mid-tag at EOF, unpruned `<page_metadata>`, massive unannotated questions).
- **MAJOR**: Systemic or repairable defects (e.g., `broken_system_tag`, systemic `absorbed_subquestions` across the entire document, multiple dropped questions).
- **MINOR**: Isolated, low-frequency anomalies or one-off glitches (e.g. 1 isolated question out of 30 with an issue, minor anchor omission like missing citation part in stimulus).
- **INFO**: Informational observations with zero penalty (e.g. omitted extraneous non-exam lecture text, solved examples, valid table structures).

## 💡 Evaluation Rules & Tolerances:
1. **Acceptable Text Omission**: If extraneous non-exam lecture notes, study guides, or teacher info blurbs were omitted BUT all questions, stems, choices, and solutions were fully annotated, this omission is ACCEPTABLE. Do NOT report as an error.
2. **Figures Out of Scope**: Do NOT evaluate or report issues on `<figure ... />` tags.
3. **Question Labels without Stems (Cloze / Fill-in Exams)**: A `<question_label>` does NOT need to be followed by a `<stem>`. In cloze tests, fill-in-the-blank items, or passage items where the blank/prompt is embedded in a reading passage or stimulus, questions legitimately contain only `<question_label>` followed immediately by `<option_label>`. These are completely VALID. NEVER report `missing_stem` for cloze items or questions with valid choices.
4. **Format Invariance**: Non-consecutive question numbering, essay/short-answer questions without choices, and non-exam reference documents with 0 questions in both raw text and XML are completely VALID.
5. **Terminal Sentinel (<|END|>)**: Annotated XML documents are stored WITHOUT `<|END|>` (it is automatically stripped after generation). Normal documents end with standard XML closing tags (such as `</explanation>`, `</option_text>`, or `</stem>`). The absence of `<|END|>` is completely EXPECTED and NORMAL. You MUST NEVER report missing `<|END|>` as an error, truncation, or defect. (If `<|END|>` happens to be present, it is also harmless and must not be flagged as a syntax error).

---

## Output Format:
Respond ONLY with a valid JSON object with NO markdown codeblocks or extra text:
{
  "parser_error_confirmations": [
    {
      "issue_id": 1,
      "is_true_positive": true,
      "reason": "Brief evidence-based explanation"
    }
  ],
  "issues": [
    {
      "category": "<exact category label from taxonomy above>",
      "severity": "CRITICAL" | "MAJOR" | "MINOR" | "INFO",
      "message": "<concise description of the specific error caused by the parser>",
      "context_snippet": "<verbatim snippet from XML where the error occurred>"
    }
  ],
  "summary": "<concise overall review summary>"
}
"""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        thinking: Optional[str] = None,
    ):
        self.model = model.strip() if model else (REVIEWER_MODEL or PARSER_MODEL)
        self.provider = provider.strip() if provider else (REVIEWER_PROVIDER or PARSER_PROVIDER)
        self.thinking = thinking or REVIEWER_THINKING or "medium"

    @staticmethod
    def _sample_xml_safely(xml_content: str, max_chars: int) -> str:
        """
        Safely samples XML without splitting mid-tag or creating broken XML fragments.
        """
        if len(xml_content) <= max_chars:
            return xml_content

        target_head = int(max_chars * 0.7)
        target_tail = int(max_chars * 0.3)

        # Find clean boundary for head (e.g., </explanation>\n\n, </section>\n\n, </stem>\n\n, \n\n)
        head_cut = target_head
        for sep in ["</explanation>\n\n", "</section>\n\n", "</stem>\n\n", "</stimulus>\n\n", "\n\n"]:
            idx = xml_content.rfind(sep, 0, target_head + 500)
            if idx != -1 and idx > target_head - 2500:
                head_cut = idx + len(sep)
                break

        # Find clean boundary for tail (e.g., \n\n<question_label>, \n\n<section>, \n\n<stimulus>, \n\n)
        tail_start_target = len(xml_content) - target_tail
        tail_cut = tail_start_target
        for sep in ["\n\n<question_label>", "\n\n<section>", "\n\n<stimulus>", "\n\n"]:
            idx = xml_content.find(sep, max(head_cut, tail_start_target - 1500), tail_start_target + 1500)
            if idx != -1:
                tail_cut = idx + (2 if sep.startswith("\n\n") else 0)
                break

        head = xml_content[:head_cut].rstrip()
        tail = xml_content[tail_cut:].lstrip()
        elided_len = max(0, tail_cut - head_cut)

        return (
            f"{head}\n\n"
            f"<!-- ... [AUDITOR_SAMPLING_WINDOW: {elided_len} characters elided by audit tool for token budget. "
            f"This is an auditor sampling window, NOT a document flaw] ... -->\n\n"
            f"{tail}"
        )

    def review_semantic(
        self,
        xml_content: str,
        deterministic_metrics: Dict[str, Any],
        raw_ocr_text: Optional[str] = None,
        deterministic_issues: Optional[List[Dict[str, Any]]] = None,
        max_xml_chars: int = 80000,
    ) -> Optional[Dict[str, Any]]:
        """
        Calls DeepSeek / LLM to semantically rate annotation quality and detect subtle malfunctions.
        Supplies raw OCR text for omission verification and pedagogical assessment.
        """
        xml_sample = self._sample_xml_safely(xml_content, max_chars=max_xml_chars)

        raw_section = ""
        if raw_ocr_text and raw_ocr_text.strip():
            clean_raw = DeterministicAuditor.clean_raw_ocr_text(raw_ocr_text)
            raw_sample = self._sample_xml_safely(clean_raw, max_chars=40000)
            raw_section = f"\n\n### Original Raw OCR Source Text (for omission verification):\n```text\n{raw_sample}\n```\n"

        deterministic_issues = deterministic_issues or []
        deterministic_findings_section = ""
        if deterministic_issues:
            deterministic_findings_section = (
                "\n\n### Deterministic Parser Findings Requiring Confirmation\n"
                "These findings are already included in the audit's parser issue list. "
                "Confirm each one as true or false, but do not repeat any of them in `issues`.\n"
                "Return confirmations using the exact `issue_id` values below.\n"
                f"```json\n{json.dumps(deterministic_issues, ensure_ascii=False, indent=2)}\n```\n"
            )

        user_prompt = (
            f"Review the following annotated XML exam document against its raw OCR source:\n\n"
            f"### Document Metrics from Deterministic Pre-check:\n"
            f"- Total Questions: {deterministic_metrics.get('questions_count', 'N/A')}\n"
            f"- Total Option Labels: {deterministic_metrics.get('option_labels_count', 'N/A')}\n"
            f"- Total Option Texts: {deterministic_metrics.get('option_texts_count', 'N/A')}\n"
            f"- Total Stimuli: {deterministic_metrics.get('stimuli_count', 'N/A')}\n"
            f"- Retention Ratio: {deterministic_metrics.get('retention_ratio', 'N/A')}\n"
            f"{deterministic_findings_section}"
            f"{raw_section}\n"
            f"### Target Annotated XML:\n"
            f"```xml\n{xml_sample}\n```\n\n"
            f"### Evaluation Guidelines & Context:\n"
            f"1. No Scoring: Do NOT provide numerical scores or rubric scores. The audit algorithm assigns penalties based on the error categories you report.\n"
            f"2. System XML Tags Only: ONLY check system tags (<section>, <stimulus>, <question_label>, <stem>, <option_label>, <option_text>, <explanation>, <figure>) for missing closing tags or broken syntax. Completely IGNORE broken, unclosed, or mismatched non-system tags (<table>, <tr>, <td>, <th>, <b>, <i>, <span>, <p>, etc.) and math comparison operators (<, >).\n"
            f"3. Parser Error Taxonomy: Categorize all issues using the standard categories: 'unannotated_question', 'stimulus_missing_citation', 'stimulus_nesting', 'single_question_stimulus', 'stimulus_anchor_invalid', 'broken_system_tag', 'missing_question_label', 'missing_stem', 'missing_options', 'absorbed_subquestions', 'mislabelled_element', 'prohibited_tags', 'truncated_output'.\n"
            f"4. Acceptable Text Omission: If the annotated XML omitted long lecture notes, textbook theory, or info blurbs but annotated ALL exam questions and choices fully and accurately without affecting the end exam result, this lost text is completely ACCEPTABLE. Do NOT report as an error.\n"
            f"5. Error Rate vs Document Size: Consider error frequency relative to total questions. If a document has 30 questions and only 1 isolated mislabelled stem or minor glitch (>96% accuracy), classify it as MINOR.\n"
            f"6. Figures Out of Scope: Do NOT evaluate or penalize figure tags (<figure ... />) or figure mentions.\n"
            f"7. Auto-Reject Stimulus Nesting: If <stimulus> wraps <stem>, <option_label>, <question_label>, <option_text>, or <explanation>, flag as CRITICAL under 'stimulus_nesting'.\n"
            f"8. Mathematical '<' or '>' comparison symbols inside math formulas/LaTeX are valid content, not broken XML tags.\n"
            f"9. Vietnamese mathematical abbreviations ('VT', 'VP', 'đpcm') are valid standard notation.\n"
            f"10. Documents may legitimately mix solved examples (<stem> + <explanation>) and unsolved exercises (<stem> only).\n"
            f"11. If an [AUDITOR_SAMPLING_WINDOW] comment is present, it was injected by the test auditor tool for large files; do not treat the sampling marker or jump across it as an omission or error.\n"
            f"12. Terminal Sentinel (<|END|>): Do NOT require or expect '<|END|>'. In annotated XML documents, '<|END|>' is stripped upon saving; its absence is completely normal and valid. NEVER report missing '<|END|>' as an error or truncation.\n"
            f"13. Duplicate prevention: validate every deterministic finding in `parser_error_confirmations`; return only additional errors in `issues`.\n"
            f"14. Question Labels without Stems: In cloze tests, fill-in-the-blank items, or reading passage questions with embedded blanks, questions legitimately consist of only <question_label> followed immediately by <option_label>. NEVER flag these as 'missing_stem' or as an error.\n\n"
            f"Inspect the document, confirm the supplied deterministic findings, identify only additional parser errors, and return your JSON report."
        )

        try:
            raw_response = chat(
                prompt=user_prompt,
                system=self.SYSTEM_PROMPT,
                model=self.model,
                provider=self.provider,
                thinking=self.thinking,
            )

            cleaned = raw_response.strip()
            # Strip think tags
            cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
            # Strip code blocks
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if len(lines) >= 2:
                    if lines[-1].strip() == "```":
                        cleaned = "\n".join(lines[1:-1])
                    else:
                        cleaned = "\n".join(lines[1:])
            cleaned = cleaned.strip()

            # Find json block
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            target_json = json_match.group(0) if json_match else cleaned

            # Sanitize LLM JSON quirks (trailing commas, LaTeX backslashes like \alpha, \frac, \underline)
            sanitized = re.sub(r",\s*([\]}])", r"\1", target_json)
            escape_pattern = r'(\\+)([^"\\/bfnrtu]|u(?![\da-fA-F]{4}))'
            def _fix_slashes(m):
                slashes = m.group(1)
                follow = m.group(2)
                if len(slashes) % 2 == 1:
                    return slashes + "\\" + follow
                return m.group(0)
            sanitized = re.sub(escape_pattern, _fix_slashes, sanitized)
            try:
                return json.loads(sanitized, strict=False)
            except Exception:
                try:
                    return json.loads(target_json, strict=False)
                except Exception as parse_err:
                    print(f"[Reviewer Warning] DeepSeek LLM JSON parse failed: {parse_err}")
                    return None
        except Exception as e:
            print(f"[Reviewer Warning] DeepSeek LLM evaluation call failed: {e}")
            return None


class AnnotationReviewerAgent:
    """
    Main Reviewer Agent for Azozo Sequence Labelling and Exam Document Pipeline.
    Combines deterministic static checks with DeepSeek semantic auditing to rate quality
    and automatically quarantine/discard malfunctioned documents.
    """

    def __init__(
        self,
        min_score: int = REVIEWER_MIN_SCORE,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        thinking: Optional[str] = None,
    ):
        self.min_score = min_score
        self.model = model.strip() if model else REVIEWER_MODEL
        self.provider = provider.strip() if provider else REVIEWER_PROVIDER
        self.llm_reviewer = DeepSeekReviewer(
            model=self.model, provider=self.provider, thinking=thinking
        )

    def review_document(
        self,
        xml_content: str,
        raw_ocr_text: Optional[str] = None,
        doc_id: str = "doc_001",
        file_path: Optional[str] = None,
        raw_file_path: Optional[str] = None,
        use_llm: bool = True,
        cached_llm_result: Optional[Dict[str, Any]] = None,
    ) -> ReviewReport:
        """
        Runs the full hybrid review on an annotated XML document.
        Combines deterministic rule checks with DeepSeek LLM semantic review.
        """
        issues: List[AuditIssue] = []
        discard_reasons: List[str] = []

        # 1. XML Syntax & Well-Formedness
        syntax_issues, syntax_score = DeterministicAuditor.check_xml_syntax(xml_content)
        issues.extend(syntax_issues)

        # 2. Prohibited Tags & Schema Conformance
        prohibited_issues, prohibited_score = DeterministicAuditor.check_prohibited_tags(
            xml_content
        )
        issues.extend(prohibited_issues)
        s_syntax = round(syntax_score * 0.70 + prohibited_score * 0.30, 1)

        # 3. Stimulus Nesting & Anchors (Architecture & Grounding)
        stim_nest_issues, stim_nest_score = (
            DeterministicAuditor.check_stimulus_wrapping_system_tags(xml_content)
        )
        issues.extend(stim_nest_issues)

        stim_anchor_issues, stim_anchor_score = DeterministicAuditor.check_stimulus_anchors(
            xml_content=xml_content,
            pure_text=DeterministicAuditor.strip_xml_tags(xml_content),
            raw_ocr_text=raw_ocr_text,
        )
        issues.extend(stim_anchor_issues)
        stim_score = min(stim_nest_score, stim_anchor_score)

        # 4. Question and Option Structure Integrity
        q_issues, q_score, metrics = (
            DeterministicAuditor.check_question_and_option_structure(
                xml_content, raw_ocr_text=raw_ocr_text
            )
        )
        issues.extend(q_issues)
        s_structure = round(q_score * 0.70 + stim_score * 0.30, 1)

        # 5. DET Question Coverage (Recall)
        det_cov_issues, det_cov_score, det_cov_metrics = (
            DeterministicAuditor.check_question_coverage_via_det(
                xml_content=xml_content, raw_ocr_text=raw_ocr_text
            )
        )
        issues.extend(det_cov_issues)
        metrics.update(det_cov_metrics)

        # 6. Verbatim Content Fidelity (Precision)
        verbatim_issues, verbatim_score, verbatim_metrics = (
            DeterministicAuditor.check_verbatim_alignment(
                xml_content=xml_content, raw_ocr_text=raw_ocr_text
            )
        )
        issues.extend(verbatim_issues)

        # IDs are assigned only after all deterministic checks have run.  The
        # same IDs are shown to the LLM and later carried into the editor
        # prompt, so the editor never has to infer which finding it is fixing.
        _assign_issue_ids(issues)
        metrics.update(verbatim_metrics)

        # Calculate generalized composite deterministic score
        if raw_ocr_text and raw_ocr_text.strip():
            det_score = (
                s_syntax * 0.25
                + det_cov_score * 0.35
                + verbatim_score * 0.25
                + s_structure * 0.15
            )
        else:
            det_score = s_syntax * 0.60 + s_structure * 0.40

        det_score = max(0.0, min(100.0, round(det_score, 1)))

        rubric = RubricScores(
            xml_well_formedness=round(syntax_score, 1),
            schema_conformance=round(prohibited_score, 1),
            det_question_coverage=round(det_cov_score, 1),
            verbatim_fidelity=round(verbatim_score, 1),
            question_option_completeness=round(q_score, 1),
            stimulus_accuracy=round(stim_score, 1),
            sequence_continuity=100.0,
        )

        # Check for hard critical malfunctions that require immediate discard
        critical_issues = [iss for iss in issues if iss.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            for c_iss in critical_issues:
                discard_reasons.append(f"[{c_iss.category.upper()}] {c_iss.message}")

        overall_score = det_score
        llm_score_val = None
        llm_confirmations: List[Dict[str, Any]] = []
        confirmed_issues: List[AuditIssue] = []
        confirmation_status = "not_run"
        summary_text = f"Deterministic audit: {len(issues)} issue(s), {len(critical_issues)} critical."

        # 7. LLM Semantic Review (or reuse cached LLM review if present)
        llm_result = None
        if use_llm and syntax_score > 20.0:
            confirmation_status = "unavailable"
            deterministic_issue_payload = _audit_issue_payload(issues)
            if cached_llm_result:
                llm_result = cached_llm_result
            else:
                llm_result = self.llm_reviewer.review_semantic(
                    xml_content=xml_content,
                    deterministic_metrics=metrics,
                    raw_ocr_text=raw_ocr_text,
                    deterministic_issues=deterministic_issue_payload,
                )

            if llm_result:
                confirmation_status = "complete"
                llm_confirmations = _normalise_llm_confirmations(
                    llm_result.get("parser_error_confirmations", []),
                    deterministic_issue_payload,
                )
                raw_llm_issues = llm_result.get("issues", [])
                llm_issues: List[AuditIssue] = []
                for iss in raw_llm_issues:
                    if isinstance(iss, dict):
                        cat_val = str(iss.get("category", "llm_semantic")).strip()
                        msg_val = str(iss.get("message", "Semantic issue detected by LLM")).strip()

                        # Deterministic findings are already in `issues`. The LLM
                        # should confirm them separately, but filter duplicates as
                        # a defensive guard against prompt non-compliance.
                        duplicate_id = _is_duplicate_llm_issue(iss, deterministic_issue_payload)
                        if duplicate_id is not None:
                            continue

                        # Filter out spurious false-positive issues complaining about missing <|END|> sentinel
                        if is_spurious_end_sentinel_issue(cat_val, msg_val):
                            continue

                        try:
                            sev = IssueSeverity(iss.get("severity", "MINOR").upper())
                        except Exception:
                            sev = IssueSeverity.MINOR
                        llm_issues.append(
                            AuditIssue(
                                category=cat_val,
                                severity=sev,
                                message=msg_val,
                                context_snippet=iss.get("context_snippet"),
                            )
                        )
                issues.extend(llm_issues)

                # LLM-reported additional issues are already semantic findings
                # and therefore enter the editor handoff as confirmed errors.
                # Deterministic findings enter only when the LLM explicitly
                # confirms their stable issue_id as a true positive.
                _assign_issue_ids(issues)
                confirmed_ids = _confirmed_issue_keys(llm_confirmations)
                confirmed_issues = [
                    issue
                    for issue in issues
                    if issue.issue_id in confirmed_ids
                ]
                for issue in llm_issues:
                    confirmed_issues.append(issue)

                # Algorithmic penalty assignment on all confirmed issues (confirmed deterministic + new LLM issues)
                total_penalties, llm_discard_reasons, rubric_deductions = compute_llm_penalties(confirmed_issues)

                if "score" in llm_result and not raw_llm_issues and not confirmed_issues:
                    # Backward compatibility for cached/mock results with score and originally zero issues
                    llm_score_val = float(llm_result["score"])
                else:
                    llm_score_val = max(0.0, 100.0 - total_penalties)

                # Update rubric scores by subtracting algorithmic deductions
                rubric.xml_well_formedness = max(
                    0.0, round(rubric.xml_well_formedness - rubric_deductions["xml_well_formedness"], 1)
                )
                rubric.schema_conformance = max(
                    0.0, round(rubric.schema_conformance - rubric_deductions["schema_conformance"], 1)
                )
                rubric.det_question_coverage = max(
                    0.0, round(rubric.det_question_coverage - rubric_deductions["det_question_coverage"], 1)
                )
                rubric.verbatim_fidelity = max(
                    0.0, round(rubric.verbatim_fidelity - rubric_deductions["verbatim_fidelity"], 1)
                )
                rubric.question_option_completeness = max(
                    0.0,
                    round(rubric.question_option_completeness - rubric_deductions["question_option_completeness"], 1),
                )
                rubric.stimulus_accuracy = max(
                    0.0, round(rubric.stimulus_accuracy - rubric_deductions["stimulus_accuracy"], 1)
                )

                # Fallback: if cached/mock had rubric_scores and no raw issues were present, blend
                if "rubric_scores" in llm_result and not raw_llm_issues:
                    llm_rubric = llm_result.get("rubric_scores", {})
                    if isinstance(llm_rubric, dict):
                        rubric.xml_well_formedness = round(
                            (rubric.xml_well_formedness + float(llm_rubric.get("xml_well_formedness", rubric.xml_well_formedness))) / 2, 1
                        )
                        rubric.schema_conformance = round(
                            (rubric.schema_conformance + float(llm_rubric.get("schema_conformance", rubric.schema_conformance))) / 2, 1
                        )
                        if "det_question_coverage" in llm_rubric:
                            rubric.det_question_coverage = round(
                                (rubric.det_question_coverage + float(llm_rubric.get("det_question_coverage", rubric.det_question_coverage))) / 2, 1
                            )
                        rubric.verbatim_fidelity = round(
                            (rubric.verbatim_fidelity + float(llm_rubric.get("verbatim_fidelity", rubric.verbatim_fidelity))) / 2, 1
                        )
                        rubric.question_option_completeness = round(
                            (rubric.question_option_completeness + float(llm_rubric.get("question_option_completeness", rubric.question_option_completeness))) / 2, 1
                        )
                        rubric.stimulus_accuracy = round(
                            (rubric.stimulus_accuracy + float(llm_rubric.get("stimulus_accuracy", rubric.stimulus_accuracy))) / 2, 1
                        )
                        if "sequence_continuity" in llm_rubric:
                            rubric.sequence_continuity = round(
                                (rubric.sequence_continuity + float(llm_rubric.get("sequence_continuity", rubric.sequence_continuity))) / 2, 1
                            )

                # Discard reasons in complete confirmation mode derive from confirmed issues and LLM findings
                discard_reasons = list(llm_discard_reasons)

                if llm_result.get("is_malfunctioned"):
                    for d_reason in llm_result.get("discard_reasons", []):
                        if is_spurious_end_sentinel_issue("", str(d_reason)):
                            continue
                        formatted_d = f"[LLM_AUDIT] {d_reason}" if "[LLM_AUDIT]" not in d_reason else d_reason
                        if formatted_d not in discard_reasons:
                            discard_reasons.append(formatted_d)

                if llm_result.get("summary"):
                    raw_summary = str(llm_result["summary"]).strip()
                    if is_spurious_end_sentinel_issue("truncated_output", raw_summary) and not llm_issues:
                        summary_text = "Review completed: false-positive end sentinel truncation ignored; document well-formed."
                    else:
                        summary_text = raw_summary

                # Weighted score: 40% deterministic rule adherence, 60% LLM semantic judgment
                overall_score = round(det_score * 0.40 + llm_score_val * 0.60, 1)

        overall_score = max(0.0, min(100.0, round(overall_score, 1)))
        grade = compute_grade(overall_score)

        # Decision logic: evaluate active issues (confirmed issues if LLM ran, else all deterministic issues)
        active_issues = (
            confirmed_issues
            if use_llm and confirmation_status == "complete"
            else issues
        )
        has_critical = any(iss.severity == IssueSeverity.CRITICAL for iss in active_issues)
        has_major = any(iss.severity == IssueSeverity.MAJOR for iss in active_issues)

        is_malfunctioned = len(discard_reasons) > 0 or overall_score < self.min_score or has_critical
        if is_malfunctioned:
            decision = ReviewDecision.DISCARD
            if not discard_reasons:
                if has_critical:
                    critical_msgs = [f"[{iss.category.upper()}] {iss.message}" for iss in active_issues if iss.severity == IssueSeverity.CRITICAL]
                    discard_reasons.extend(critical_msgs)
                else:
                    discard_reasons.append(
                        f"Overall score {overall_score:.1f}/100 is below minimum threshold {self.min_score}."
                    )
        elif has_major:
            # Any unaddressed MAJOR issue (syntax, unannotated questions, missing options, etc.) requires revision
            decision = ReviewDecision.NEEDS_REVISION
        elif overall_score >= 80.0:
            decision = ReviewDecision.PASS
        else:
            decision = ReviewDecision.NEEDS_REVISION

        return ReviewReport(
            doc_id=doc_id,
            file_path=str(file_path) if file_path else None,
            raw_file_path=str(raw_file_path) if raw_file_path else None,
            overall_score=overall_score,
            deterministic_score=round(det_score, 1),
            llm_score=round(llm_score_val, 1) if llm_score_val is not None else None,
            grade=grade,
            decision=decision,
            is_malfunctioned=is_malfunctioned,
            discard_reasons=discard_reasons,
            rubric_scores=rubric,
            issues=issues,
            llm_confirmations=llm_confirmations,
            metrics=metrics,
            summary=summary_text,
            reviewer_model=self.model,
            reviewer_provider=self.provider,
            confirmed_issues=confirmed_issues,
            confirmation_status=confirmation_status,
        )

    def review_file(
        self,
        xml_path: Union[str, Path],
        raw_path: Optional[Union[str, Path]] = None,
        use_llm: bool = True,
        save_audit_json: bool = True,
        overwrite: bool = False,
    ) -> ReviewReport:
        """
        Reviews a single XML file from disk against its raw markdown source if available.
        """
        xml_p = Path(xml_path)
        if not xml_p.exists():
            raise FileNotFoundError(f"Target XML file does not exist: {xml_path}")

        with open(xml_p, "r", encoding="utf-8", errors="replace") as f:
            xml_content = f.read()

        raw_content = None
        raw_p = Path(raw_path) if raw_path else None

        # Auto-discover matching raw .md file if not provided
        if not raw_p or not raw_p.exists():
            # Check parallel folder under sequence_labelling_input_data
            str_path = str(xml_p)
            if "sequence_labelling_annotated" in str_path:
                candidate_raw_str = str_path.replace(
                    "sequence_labelling_annotated", "sequence_labelling_input_data"
                )
                candidate_p = Path(candidate_raw_str)
                # If path was .../exam_123/merged.xml, candidate is .../exam_123.md
                if candidate_p.name in ["merged.xml", "merged.json"]:
                    candidate_md = candidate_p.parent.with_suffix(".md")
                    if candidate_md.exists():
                        raw_p = candidate_md

        if raw_p and raw_p.exists():
            with open(raw_p, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

        # Extract parser metadata from merged.json if available
        parser_info = None
        if xml_p.parent.is_dir():
            merged_json = xml_p.parent / "merged.json"
            if merged_json.exists():
                try:
                    with open(merged_json, "r", encoding="utf-8", errors="replace") as f_json:
                        j_data = json.load(f_json)
                        parser_info = {
                            "merge_status": j_data.get("merge_status"),
                            "total_chunks": j_data.get("total_chunks"),
                            "duration_seconds": j_data.get("duration_seconds"),
                            "questions_count": j_data.get("questions_count"),
                            "linker_skipped": j_data.get("linker_skipped"),
                        }
                except Exception:
                    pass

        doc_id = xml_p.parent.name if xml_p.name in ["merged.xml", "merged.json"] else xml_p.stem

        # Check if existing audit report has cached LLM result to preserve
        cached_llm = None
        audit_file_target = (
            (xml_p.parent / "audit_report.json")
            if xml_p.name in ["merged.xml", "merged.json"]
            else xml_p.with_suffix(".audit.json")
        )
        if not overwrite and not use_llm and audit_file_target.exists():
            try:
                with open(audit_file_target, "r", encoding="utf-8") as f_prev:
                    prev_data = json.load(f_prev)
                    if prev_data.get("llm_score") is not None:
                        cached_llm = {
                            "score": prev_data.get("llm_score"),
                            "rubric_scores": prev_data.get("rubric_scores"),
                            "issues": [
                                iss for iss in prev_data.get("issues", [])
                                if (
                                    iss.get("source") == "llm"
                                    or (
                                        "source" not in iss
                                        and (
                                            iss.get("category") in PARSER_ERROR_PENALTIES
                                            or iss.get("category") == "llm_semantic"
                                        )
                                    )
                                )
                            ],
                            "parser_error_confirmations": prev_data.get("llm_confirmations", []),
                            "summary": prev_data.get("summary", ""),
                            "is_malfunctioned": prev_data.get("is_malfunctioned", False),
                            "discard_reasons": [
                                r.replace("[LLM_AUDIT] ", "")
                                for r in prev_data.get("discard_reasons", [])
                                if "[LLM_AUDIT]" in r
                            ],
                        }
            except Exception:
                pass

        report = self.review_document(
            xml_content=xml_content,
            raw_ocr_text=raw_content,
            doc_id=doc_id,
            file_path=str(xml_p),
            raw_file_path=str(raw_p) if raw_p else None,
            use_llm=use_llm,
            cached_llm_result=cached_llm,
        )

        if parser_info:
            report.parser_info = parser_info

        if save_audit_json:
            with open(audit_file_target, "w", encoding="utf-8") as f_out:
                json.dump(report.model_dump(), f_out, indent=2, ensure_ascii=False)

        return report

    def discard_document(
        self,
        xml_path: Union[str, Path],
        report: ReviewReport,
        discard_dir: Union[str, Path],
        move_raw: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Quarantines / discards a malfunctioned document by moving its directory to discard_dir
        and saving an accompanying audit report JSON.
        """
        xml_p = Path(xml_path).resolve()
        discard_base = Path(discard_dir).resolve()

        if not xml_p.exists():
            return {"success": False, "error": f"File not found: {xml_path}"}

        # Determine target move root
        if xml_p.name in ["merged.xml", "merged.json"]:
            source_folder = xml_p.parent
        else:
            source_folder = xml_p

        # Find relative path structure if inside a known base dir
        rel_path = None
        for known_parent in ["sequence_labelling_annotated", "annotator/out", "data"]:
            if known_parent in str(source_folder):
                parts = str(source_folder).split(known_parent)
                rel_path = parts[-1].lstrip(os.sep)
                break

        if not rel_path:
            rel_path = source_folder.name

        dest_target = discard_base / rel_path
        audit_file_path = (dest_target / "audit_report.json") if dest_target.is_dir() or source_folder.is_dir() else dest_target.with_suffix(".audit.json")

        result = {
            "success": True,
            "dry_run": dry_run,
            "source_path": str(source_folder),
            "destination_path": str(dest_target),
            "audit_file": str(audit_file_path),
            "discard_reasons": report.discard_reasons,
            "overall_score": report.overall_score,
        }

        if dry_run:
            return result

        dest_target.parent.mkdir(parents=True, exist_ok=True)

        # Move source folder / file
        if dest_target.exists():
            if dest_target.is_dir():
                shutil.rmtree(dest_target)
            else:
                dest_target.unlink()

        shutil.move(str(source_folder), str(dest_target))

        # Write audit report JSON
        if dest_target.is_dir():
            audit_file = dest_target / "audit_report.json"
        else:
            audit_file = dest_target.with_suffix(".audit.json")

        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

        # Optionally move raw input file if requested
        if move_raw and report.raw_file_path:
            raw_src = Path(report.raw_file_path)
            if raw_src.exists():
                raw_dest = dest_target.parent / raw_src.name if dest_target.is_dir() else dest_target.with_suffix(raw_src.suffix)
                try:
                    shutil.move(str(raw_src), str(raw_dest))
                    result["raw_moved_to"] = str(raw_dest)
                except Exception as e:
                    result["raw_move_warning"] = str(e)

        return result

    @staticmethod
    def discover_review_targets(
        annotated_dir: Union[str, Path],
        max_merged_tokens: int = 500_000,
    ) -> List[Path]:
        """
        Discovers XML files for review following the threshold rule:
        - If a document is <= max_merged_tokens (~500k tokens), review the merged.xml version.
        - If a document exceeds max_merged_tokens, fallback to chunk level (chunk_*.xml).
        - Never review both merged and chunk versions for the same document.
        - Standalone XML files are reviewed as-is.
        """
        base_dir = Path(annotated_dir)
        if base_dir.is_file():
            return [base_dir]

        targets: List[Path] = []

        # 1. Find all directories containing merged.xml
        merged_files = sorted(base_dir.rglob("merged.xml"))
        exam_dirs_with_merged = set()

        for m_file in merged_files:
            exam_dir = m_file.parent
            exam_dirs_with_merged.add(exam_dir)

            try:
                content = m_file.read_text(encoding="utf-8", errors="replace")
                # Estimate tokens: approx 3.5 chars per token
                est_tokens = len(content) / 3.5
            except Exception:
                est_tokens = 0

            chunks_dir = exam_dir / "chunks"
            chunk_files = sorted(chunks_dir.glob("chunk_*.xml")) if chunks_dir.exists() else []

            if est_tokens > max_merged_tokens and chunk_files:
                # Fallback to chunk level if document exceeds 500k token threshold
                targets.extend(chunk_files)
            else:
                # Process merged level
                targets.append(m_file)

        # 2. Find any chunk files in folders that DO NOT have merged.xml
        all_chunks = sorted(base_dir.rglob("chunk_*.xml"))
        for c_file in all_chunks:
            exam_dir = c_file.parent.parent if c_file.parent.name == "chunks" else c_file.parent
            if exam_dir not in exam_dirs_with_merged:
                targets.append(c_file)

        # 3. Find any standalone .xml files (not named merged.xml and not chunk_*.xml)
        all_xml = sorted(base_dir.rglob("*.xml"))
        for x_file in all_xml:
            if x_file.name == "merged.xml" or x_file.name.startswith("chunk_"):
                continue
            targets.append(x_file)

        targets.sort()
        return targets

    def batch_review(
        self,
        annotated_dir: Union[str, Path],
        raw_dir: Optional[Union[str, Path]] = None,
        discard_dir: Optional[Union[str, Path]] = None,
        auto_discard: bool = False,
        save_audit_json: bool = True,
        use_llm: bool = True,
        concurrency: int = 4,
        max_merged_tokens: int = 500_000,
        output_report_path: Optional[Union[str, Path]] = None,
        progress_callback=None,
        overwrite: bool = False,
        filter_decision: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BatchReviewSummary:
        """
        Performs batch review across all XML documents in a directory.
        Processes merged.xml by default if <= 500k tokens, falling back to chunk level if > 500k tokens.
        Never processes both chunk and merged versions for the same document.
        Scans and resumes from existing audit_report.json unless overwrite is True.
        Saves progress, Markdown report, and JSON summary on the fly as documents finish.
        """
        start_time = time.time()
        base_dir = Path(annotated_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Annotated directory not found: {annotated_dir}")

        # Gather target xml files following 500k token merged-vs-chunk rule
        all_targets = self.discover_review_targets(base_dir, max_merged_tokens=max_merged_tokens)
        all_targets.sort()

        if limit is not None and limit > 0:
            all_targets = all_targets[:limit]

        total_docs = len(all_targets)
        reports: List[ReviewReport] = []
        discarded_paths: List[str] = []
        failure_reasons_distribution: Dict[str, int] = {}

        passed = 0
        needs_revision = 0
        discarded = 0
        total_score_sum = 0.0

        # Scan targets for existing audits
        to_process_files: List[Path] = []
        cached_reports_map: Dict[Path, ReviewReport] = {}

        for xml_p in all_targets:
            audit_file = (
                (xml_p.parent / "audit_report.json")
                if xml_p.name in ["merged.xml", "merged.json"]
                else xml_p.with_suffix(".audit.json")
            )
            is_done = False
            cached_rep = None

            if not overwrite and audit_file.exists():
                try:
                    with open(audit_file, "r", encoding="utf-8") as f_a:
                        audit_data = json.load(f_a)
                        has_required_llm = (not use_llm) or (audit_data.get("llm_score") is not None)
                        if has_required_llm and "overall_score" in audit_data and "decision" in audit_data:
                            cached_rep = ReviewReport.model_validate(audit_data)
                            is_done = True
                except Exception:
                    is_done = False

            if filter_decision and is_done and cached_rep:
                if isinstance(filter_decision, (list, tuple, set)):
                    allowed_filters = {str(d).upper().strip() for d in filter_decision}
                else:
                    allowed_filters = {d.upper().strip() for d in str(filter_decision).split(",") if d.strip()}

                if "FAILED" in allowed_filters or "DISCARD_AND_REVISION" in allowed_filters:
                    allowed_filters.update({"DISCARD", "NEEDS_REVISION"})
                if "DISCARDS" in allowed_filters:
                    allowed_filters.add("DISCARD")

                if (
                    cached_rep.decision.value.upper() in allowed_filters
                    or cached_rep.decision.name.upper() in allowed_filters
                ):
                    to_process_files.append(xml_p)
                else:
                    cached_reports_map[xml_p] = cached_rep
            elif is_done and cached_rep:
                cached_reports_map[xml_p] = cached_rep
            else:
                to_process_files.append(xml_p)

        # Seed reports with loaded cached audits
        for rep in cached_reports_map.values():
            reports.append(rep)
            total_score_sum += rep.overall_score
            if rep.decision == ReviewDecision.PASS:
                passed += 1
            elif rep.decision == ReviewDecision.NEEDS_REVISION:
                needs_revision += 1
            else:
                discarded += 1
                if rep.file_path:
                    discarded_paths.append(rep.file_path)
                for r in rep.discard_reasons:
                    cat = r.split("]")[0].lstrip("[") if "]" in r else "OTHER"
                    failure_reasons_distribution[cat] = failure_reasons_distribution.get(cat, 0) + 1

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        report_lock = threading.Lock()

        def process_single(xml_p: Path) -> Tuple[ReviewReport, Optional[Dict[str, Any]]]:
            # Locate raw path
            raw_file = None
            if raw_dir:
                try:
                    rel = xml_p.relative_to(base_dir)
                    if rel.name in ["merged.xml", "merged.json"]:
                        if rel.parent == Path("."):
                            candidate = Path(raw_dir) / f"{base_dir.name}.md"
                        else:
                            candidate = Path(raw_dir) / rel.parent.with_suffix(".md")
                    else:
                        candidate = Path(raw_dir) / rel.with_suffix(".md")
                    if candidate.exists():
                        raw_file = candidate
                except Exception:
                    pass

            report = self.review_file(
                xml_p,
                raw_path=raw_file,
                use_llm=use_llm,
                save_audit_json=save_audit_json,
                overwrite=overwrite,
            )

            discard_res = None
            if auto_discard and report.decision == ReviewDecision.DISCARD and discard_dir:
                discard_res = self.discard_document(
                    xml_path=xml_p,
                    report=report,
                    discard_dir=discard_dir,
                    move_raw=False,
                    dry_run=False,
                )

            return report, discard_res

        completed = len(cached_reports_map)

        if to_process_files:
            max_workers = max(1, min(concurrency, len(to_process_files)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(process_single, f): f for f in to_process_files}

                for future in as_completed(future_to_file):
                    f = future_to_file[future]
                    completed += 1
                    try:
                        rep, disc_info = future.result()
                        with report_lock:
                            reports.append(rep)
                            total_score_sum += rep.overall_score

                            if rep.decision == ReviewDecision.PASS:
                                passed += 1
                            elif rep.decision == ReviewDecision.NEEDS_REVISION:
                                needs_revision += 1
                            else:
                                discarded += 1
                                discarded_paths.append(str(f))
                                seen_cats_for_doc = set()
                                for r in rep.discard_reasons:
                                    cat = r.split("]")[0].lstrip("[") if "]" in r else "OTHER"
                                    if cat not in seen_cats_for_doc:
                                        failure_reasons_distribution[cat] = (
                                            failure_reasons_distribution.get(cat, 0) + 1
                                        )
                                        seen_cats_for_doc.add(cat)

                            curr_avg = round(total_score_sum / max(1, len(reports)), 1)
                            curr_duration = round(time.time() - start_time, 2)

                            # On-the-fly report and progress export
                            if output_report_path:
                                partial_summary = BatchReviewSummary(
                                    total_documents=total_docs,
                                    passed_count=passed,
                                    needs_revision_count=needs_revision,
                                    discarded_count=discarded,
                                    discarded_paths=discarded_paths,
                                    average_score=curr_avg,
                                    duration_sec=curr_duration,
                                    reports=reports,
                                    failure_reasons_distribution=failure_reasons_distribution,
                                )
                                self.export_markdown_report(partial_summary, output_report_path)
                                out_p = Path(output_report_path)
                                json_report_path = out_p.with_suffix(".json")
                                with open(json_report_path, "w", encoding="utf-8") as f_j:
                                    json.dump(partial_summary.model_dump(), f_j, indent=2, ensure_ascii=False)

                                progress_path = out_p.parent / "review_progress.json"
                                pct = round((completed / max(1, total_docs)) * 100, 1)
                                eta_sec = (
                                    round((curr_duration / completed) * (total_docs - completed), 1)
                                    if completed > 0
                                    else 0.0
                                )
                                prog_payload = {
                                    "status": "IN_PROGRESS" if completed < total_docs else "COMPLETED",
                                    "total_documents": total_docs,
                                    "completed_documents": completed,
                                    "progress_percent": pct,
                                    "passed_count": passed,
                                    "needs_revision_count": needs_revision,
                                    "discarded_count": discarded,
                                    "average_score": curr_avg,
                                    "elapsed_seconds": curr_duration,
                                    "eta_seconds": eta_sec,
                                    "last_completed_document": rep.doc_id,
                                    "last_decision": rep.decision.value,
                                    "last_overall_score": rep.overall_score,
                                    "updated_at": datetime.now().isoformat(),
                                }
                                with open(progress_path, "w", encoding="utf-8") as f_pr:
                                    json.dump(prog_payload, f_pr, indent=2, ensure_ascii=False)

                        if progress_callback:
                            progress_callback(completed, total_docs, rep)
                    except Exception as e:
                        pass
        else:
            # All files were cached / done
            curr_avg = round(total_score_sum / max(1, len(reports)), 1)
            curr_duration = round(time.time() - start_time, 2)
            if output_report_path:
                summary = BatchReviewSummary(
                    total_documents=total_docs,
                    passed_count=passed,
                    needs_revision_count=needs_revision,
                    discarded_count=discarded,
                    discarded_paths=discarded_paths,
                    average_score=curr_avg,
                    duration_sec=curr_duration,
                    reports=reports,
                    failure_reasons_distribution=failure_reasons_distribution,
                )
                self.export_markdown_report(summary, output_report_path)
                out_p = Path(output_report_path)
                json_report_path = out_p.with_suffix(".json")
                with open(json_report_path, "w", encoding="utf-8") as f_j:
                    json.dump(summary.model_dump(), f_j, indent=2, ensure_ascii=False)

                progress_path = out_p.parent / "review_progress.json"
                prog_payload = {
                    "status": "COMPLETED",
                    "total_documents": total_docs,
                    "completed_documents": completed,
                    "progress_percent": 100.0,
                    "passed_count": passed,
                    "needs_revision_count": needs_revision,
                    "discarded_count": discarded,
                    "average_score": curr_avg,
                    "elapsed_seconds": curr_duration,
                    "eta_seconds": 0.0,
                    "updated_at": datetime.now().isoformat(),
                }
                with open(progress_path, "w", encoding="utf-8") as f_pr:
                    json.dump(prog_payload, f_pr, indent=2, ensure_ascii=False)

        avg_score = round(total_score_sum / max(1, len(reports)), 1)
        duration = round(time.time() - start_time, 2)

        final_summary = BatchReviewSummary(
            total_documents=total_docs,
            passed_count=passed,
            needs_revision_count=needs_revision,
            discarded_count=discarded,
            discarded_paths=discarded_paths,
            average_score=avg_score,
            duration_sec=duration,
            reports=reports,
            failure_reasons_distribution=failure_reasons_distribution,
        )

        if output_report_path:
            self.export_markdown_report(final_summary, output_report_path)
            json_report_path = Path(output_report_path).with_suffix(".json")
            with open(json_report_path, "w", encoding="utf-8") as f_j:
                json.dump(final_summary.model_dump(), f_j, indent=2, ensure_ascii=False)

        return final_summary

    @staticmethod
    def export_markdown_report(summary: BatchReviewSummary, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Renders a clean, formatted Markdown audit summary report with live progress status.
        """
        is_in_progress = len(summary.reports) < summary.total_documents and summary.total_documents > 0
        pct = (len(summary.reports) / max(1, summary.total_documents)) * 100

        status_badge = (
            f"⏳ **IN PROGRESS** (`{len(summary.reports)}/{summary.total_documents}` processed — `{pct:.1f}%`)"
            if is_in_progress
            else f"✅ **COMPLETED** (`{summary.total_documents}` documents)"
        )

        lines = [
            "# 📋 Annotation Quality Audit & Document Review Report",
            "",
            f"- **Status**: {status_badge}",
            f"- **Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Documents Reviewed**: `{len(summary.reports)}` / `{summary.total_documents}`",
            f"- **Average Quality Score**: **{summary.average_score:.1f} / 100**",
            f"- **Passed**: `{summary.passed_count}` ({summary.passed_count/max(1, len(summary.reports))*100:.1f}%)",
            f"- **Needs Revision**: `{summary.needs_revision_count}` ({summary.needs_revision_count/max(1, len(summary.reports))*100:.1f}%)",
            f"- **Discarded / Malfunctioned**: `{summary.discarded_count}` ({summary.discarded_count/max(1, len(summary.reports))*100:.1f}%)",
            f"- **Duration / Elapsed**: {summary.duration_sec:.1f}s",
            "",
            "## 📊 Failure Reasons Distribution",
            "",
        ]

        if summary.failure_reasons_distribution:
            lines.append("| Failure Category | Malfunction Count |")
            lines.append("| :--- | :--- |")
            for cat, cnt in sorted(summary.failure_reasons_distribution.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| `{cat}` | {cnt} |")
        else:
            lines.append("*(No malfunctions detected so far)*")

        lines.extend([
            "",
            "## 📑 Detailed Document Audit Table",
            "",
            "| Document ID | Overall | Det. Score | LLM Score | Grade | Decision | Questions | Verbatim Ret. | Summary / Discard Reason |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for r in summary.reports:
            symbol_badge = (
                "🟢 PASS"
                if r.decision == ReviewDecision.PASS
                else ("🟡 REVISION" if r.decision == ReviewDecision.NEEDS_REVISION else "🔴 DISCARD")
            )
            q_cnt = r.metrics.get("questions_count", "N/A")
            llm_sc = f"{r.llm_score:.1f}" if r.llm_score is not None else "N/A"
            ret_str = f"{r.metrics.get('retention_ratio'):.1%}" if r.metrics.get("retention_ratio") is not None else "N/A"

            reason_snip = "; ".join(r.discard_reasons[:2]) if r.discard_reasons else (r.summary[:60] + "..." if len(r.summary) > 60 else r.summary)
            reason_snip = reason_snip.replace("\n", " ").replace("|", "\\|")

            lines.append(
                f"| `{r.doc_id}` | **{r.overall_score:.1f}** | {r.deterministic_score:.1f} | {llm_sc} | `{r.grade}` | {symbol_badge} | {q_cnt} | {ret_str} | {reason_snip} |"
            )

        markdown_content = "\n".join(lines) + "\n"

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(markdown_content)

        return markdown_content


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Reviewer for Sequence Labelling XML Documents"
    )
    parser.add_argument(
        "--annotated-dir",
        "-i",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Directory containing annotated documents (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--raw-dir",
        "-r",
        type=str,
        default="data/sequence_labelling_input_data",
        help="Directory containing raw OCR source documents",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Reviewer LLM model name (e.g. gpt-5.6-luna)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        help="LLM provider name (e.g. codex, agy, xah)",
    )
    parser.add_argument(
        "--thinking",
        "-t",
        type=str,
        default=None,
        help="Reasoning/thinking effort (high, medium, low)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=REVIEWER_MIN_SCORE,
        help=f"Minimum score threshold to pass (default: {REVIEWER_MIN_SCORE})",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)",
    )
    parser.add_argument(
        "--output-report",
        "-o",
        type=str,
        default="data/review_report.md",
        help="Path to export Markdown audit report (default: data/review_report.md)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate all files even if audit_report.json exists",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run fast deterministic static checks only, skipping LLM evaluation",
    )
    parser.add_argument(
        "--filter-decision",
        type=str,
        default=None,
        help="Only re-evaluate documents matching specific decisions (e.g. DISCARD, NEEDS_REVISION)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of documents to evaluate",
    )
    parser.add_argument(
        "--auto-discard",
        action="store_true",
        help="Automatically quarantine discarded documents into discard directory",
    )
    parser.add_argument(
        "--discard-dir",
        type=str,
        default="data/sequence_labelling_discarded",
        help="Quarantine folder for discarded documents",
    )

    args = parser.parse_args()

    agent = AnnotationReviewerAgent(
        min_score=args.min_score,
        model=args.model,
        provider=args.provider,
        thinking=args.thinking,
    )

    print("=" * 65)
    print("=== Azozo Sequence Labelling Quality Reviewer ===")
    print(f"  Annotated Directory : {args.annotated_dir}")
    print(f"  Raw OCR Directory   : {args.raw_dir}")
    print(f"  Reviewer Provider   : {agent.provider}")
    print(f"  Reviewer Model      : {agent.model}")
    print(f"  Thinking Effort     : {agent.llm_reviewer.thinking}")
    print(f"  Concurrency         : {args.concurrency} worker thread(s)")
    print(f"  LLM Semantic Audit  : {'Disabled (Deterministic Only)' if args.no_llm else 'Enabled'}")
    print(f"  Overwrite Mode      : {'Enabled' if args.overwrite else 'Disabled (Resume from cache)'}")
    if args.filter_decision:
        print(f"  Filter Decision     : {args.filter_decision}")
    if args.limit:
        print(f"  Document Limit      : {args.limit}")
    print(f"  Output Report       : {args.output_report}")
    print("=" * 65)

    summary = agent.batch_review(
        annotated_dir=args.annotated_dir,
        raw_dir=args.raw_dir,
        discard_dir=args.discard_dir if args.auto_discard else None,
        auto_discard=args.auto_discard,
        use_llm=not args.no_llm,
        concurrency=args.concurrency,
        output_report_path=args.output_report,
        overwrite=args.overwrite,
        filter_decision=args.filter_decision,
        limit=args.limit,
    )

    print("\n" + "=" * 65)
    print("=== Batch Review Completed ===")
    print(f"  Total Processed : {len(summary.reports)} / {summary.total_documents}")
    print(f"  Average Score   : {summary.average_score:.1f} / 100")
    print(f"  Passed          : {summary.passed_count}")
    print(f"  Needs Revision  : {summary.needs_revision_count}")
    print(f"  Discarded       : {summary.discarded_count}")
    print(f"  Duration        : {summary.duration_sec:.1f}s")
    print(f"  Report exported : {args.output_report}")
    print("=" * 65)


if __name__ == "__main__":
    main()
