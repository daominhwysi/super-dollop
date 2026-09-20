"""
XML Syntax & Tag Integrity Checker for OCR Sequence Labeling Outputs.

Validates:
1. Mismatched opening and closing tags (e.g. <stem>...</option_text>).
2. Unclosed tags (e.g. <stem> opened but never closed).
3. Unexpected closing tags without matching opening tags.
4. Truncated tags at EOF (e.g. <opt or </ste).
5. Prohibited tags (<pages>, <page>, <page_metadata>, <think>).
6. Self-closing tags (<stimulus ... />, <figure ... />) and embedded HTML elements.
7. Tag counts and structural sequence metrics.
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union

# Ensure workspace root is in sys.path
_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# Standard allowed sequence labeling tags
SEQUENCE_PAIRED_TAGS: Set[str] = {
    "section",
    "stimulus",
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "explanation",
    "question",  # Legacy / fallback wrapper
}

# System tags that define sequence labeling structure and require strict closure/pairing
SYSTEM_TAGS: Set[str] = SEQUENCE_PAIRED_TAGS | {"figure"}

# HTML formatting tags commonly found inside stems/tables
HTML_FORMATTING_TAGS: Set[str] = {
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

ALLOWED_PAIRED_TAGS: Set[str] = SEQUENCE_PAIRED_TAGS | HTML_FORMATTING_TAGS

# HTML void tags and sequence self-closing tags
HTML_VOID_TAGS: Set[str] = {"br", "hr", "img", "col", "wbr", "input", "meta", "link"}
ALLOWED_SELF_CLOSING_TAGS: Set[str] = {"stimulus", "figure"} | HTML_VOID_TAGS

# Prohibited tags that must be pruned from clean OCR annotations
PROHIBITED_TAGS: Set[str] = {"pages", "page", "page_metadata", "think"}


@dataclass
class XMLTagIssue:
    issue_type: str  # "mismatched_closing_tag", "unclosed_tag", "unexpected_closing_tag", "truncated_tag", "prohibited_tag", "unknown_tag", "empty_document"
    severity: str    # "CRITICAL", "MAJOR", "MINOR", "WARNING"
    message: str
    line_number: int
    column_number: Optional[int] = None
    tag_name: Optional[str] = None
    expected_tag: Optional[str] = None
    opened_at_line: Optional[int] = None
    context_snippet: Optional[str] = None

    def __str__(self) -> str:
        loc = f"Line {self.line_number}"
        if self.column_number is not None:
            loc += f":{self.column_number}"
        return f"[{self.severity}] [{self.issue_type}] {loc}: {self.message}"


@dataclass
class XMLValidationResult:
    is_valid: bool
    issues: List[XMLTagIssue] = field(default_factory=list)
    tag_counts: Dict[str, int] = field(default_factory=dict)
    total_tags: int = 0
    has_mismatched_tags: bool = False
    has_unclosed_tags: bool = False
    has_unexpected_closing_tags: bool = False
    has_truncated_tag: bool = False
    has_prohibited_tags: bool = False
    has_unknown_tags: bool = False
    file_path: Optional[str] = None

    @property
    def critical_issues(self) -> List[XMLTagIssue]:
        return [i for i in self.issues if i.severity == "CRITICAL"]

    @property
    def major_issues(self) -> List[XMLTagIssue]:
        return [i for i in self.issues if i.severity == "MAJOR"]

    @property
    def error_messages(self) -> List[str]:
        return [str(i) for i in self.issues]

    def summary(self) -> str:
        if self.is_valid and not self.issues:
            return "XML Validation PASSED: No tag mismatches or syntax errors found."
        
        lines = [
            f"XML Validation {'PASSED (with warnings)' if self.is_valid else 'FAILED'}:",
            f"  - Total tags analyzed: {self.total_tags}",
            f"  - Total issues found : {len(self.issues)}",
        ]
        if self.has_mismatched_tags:
            lines.append("  - ❌ Contains Mismatched Closing Tag(s)")
        if self.has_unclosed_tags:
            lines.append("  - ❌ Contains Unclosed Opening Tag(s)")
        if self.has_unexpected_closing_tags:
            lines.append("  - ❌ Contains Unexpected Closing Tag(s)")
        if self.has_truncated_tag:
            lines.append("  - ❌ Output Truncated Mid-Tag at EOF")
        if self.has_prohibited_tags:
            lines.append("  - ⚠️ Contains Prohibited Tags (page boundaries / think)")
        if self.has_unknown_tags:
            lines.append("  - ⚠️ Contains Unknown / Non-Standard XML Tags")

        lines.append("\nDetailed Issues:")
        for issue in self.issues:
            lines.append(f"  • {issue}")
            if issue.context_snippet:
                lines.append(f"    Snippet: {issue.context_snippet.strip()}")

        return "\n".join(lines)


class XMLChecker:
    """
    High-accuracy, tolerant static XML tag integrity auditor.
    Detects mismatched closing tags, unclosed tags, premature stream cutoffs,
    and prohibited metadata tags in OCR agent outputs.
    """

    @classmethod
    def check(
        cls,
        xml_content: str,
        file_path: Optional[str] = None,
        allow_unknown_tags: bool = False,
    ) -> XMLValidationResult:
        """
        Performs comprehensive XML tag structure validation.
        """
        issues: List[XMLTagIssue] = []
        tag_counts: Dict[str, int] = {}
        total_tags = 0

        has_mismatched_tags = False
        has_unclosed_tags = False
        has_unexpected_closing_tags = False
        has_truncated_tag = False
        has_prohibited_tags = False
        has_unknown_tags = False

        if not xml_content or not xml_content.strip():
            issues.append(
                XMLTagIssue(
                    issue_type="empty_document",
                    severity="CRITICAL",
                    message="Document content is completely empty.",
                    line_number=1,
                )
            )
            return XMLValidationResult(
                is_valid=False,
                issues=issues,
                tag_counts=tag_counts,
                total_tags=0,
                file_path=file_path,
            )

        # 1. Check for unclosed tag / cutoff fragment at the very end of file
        trailing_match = re.search(r"</?([a-zA-Z_0-9\-]*)$", xml_content.strip())
        if trailing_match and trailing_match.group(1):
            tag_fragment = trailing_match.group(0)
            if tag_fragment.startswith("<"):
                last_line = xml_content.count("\n") + 1
                issues.append(
                    XMLTagIssue(
                        issue_type="truncated_tag",
                        severity="CRITICAL",
                        message=f"Output truncated mid-tag at end of generation: '{tag_fragment}'",
                        line_number=last_line,
                        context_snippet=xml_content[-80:],
                    )
                )
                has_truncated_tag = True

        # 2. Check for prohibited tags (<pages>, <page>, <page_metadata>, <think>)
        for prohibited in PROHIBITED_TAGS:
            pattern = re.compile(rf"</?{prohibited}(?:\s+[^>]*)?>", re.IGNORECASE)
            for m in pattern.finditer(xml_content):
                has_prohibited_tags = True
                line_num = xml_content.count("\n", 0, m.start()) + 1
                col_num = m.start() - xml_content.rfind("\n", 0, m.start())
                issues.append(
                    XMLTagIssue(
                        issue_type="prohibited_tag",
                        severity="CRITICAL" if prohibited in ["pages", "page"] else "MAJOR",
                        message=f"Found prohibited tag '<{prohibited}>'. Page boundaries and metadata must be pruned.",
                        line_number=line_num,
                        column_number=col_num,
                        tag_name=prohibited,
                        context_snippet=m.group(0),
                    )
                )

        # 3. Clean harmless end-of-text delimiters and comments before tag stack check
        clean_xml = re.sub(r"<!--.*?-->", "", xml_content, flags=re.DOTALL)
        clean_xml = re.sub(r"<\s*\|\s*END\s*\|\s*>", "", clean_xml)
        clean_xml = re.sub(r"<\s*\|\s*endoftext\s*\|\s*>", "", clean_xml)

        # Regex matching tags with attribute-quote awareness (e.g. end_anchor="...</td>...")
        tag_pattern = re.compile(r'<(/)?([a-zA-Z_0-9\-]+)(?:\s+((?:[^"\'>]|"[^"]*"|\'[^\']*\')*))?(/)?>')

        tag_stack: List[Tuple[str, int, int, str]] = []  # (tag_name, line_num, col_num, full_tag)

        for match in tag_pattern.finditer(clean_xml):
            total_tags += 1
            start_pos = match.start()
            line_num = xml_content.count("\n", 0, start_pos) + 1
            col_num = start_pos - xml_content.rfind("\n", 0, start_pos)

            is_closing = bool(match.group(1))
            tag_name = match.group(2).lower()
            attrs_str = match.group(3) or ""
            is_self_closing_slash = bool(match.group(4)) or attrs_str.strip().endswith("/")
            full_match = match.group(0)

            # Check if tag is self-closing
            is_self_closing = is_self_closing_slash or tag_name in ALLOWED_SELF_CLOSING_TAGS

            # Record tag occurrence count (once per element: on open or self-closing)
            if not is_closing:
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

            if is_self_closing or full_match.endswith("/>"):
                continue

            if not is_closing:
                # Opening tag
                if not allow_unknown_tags and tag_name not in ALLOWED_PAIRED_TAGS and tag_name not in PROHIBITED_TAGS:
                    has_unknown_tags = True
                    issues.append(
                        XMLTagIssue(
                            issue_type="unknown_tag",
                            severity="MAJOR",
                            message=f"Unknown or non-standard XML tag '<{tag_name}>'",
                            line_number=line_num,
                            column_number=col_num,
                            tag_name=tag_name,
                            context_snippet=full_match,
                        )
                    )
                # Only push system tags onto tag_stack to enforce strict closure and pairing,
                # ignoring unclosed non-system tags (e.g. <td>, <tr>, <th>, <table>, <b>, <i>, <p>, etc.)
                if tag_name in SYSTEM_TAGS:
                    tag_stack.append((tag_name, line_num, col_num, full_match))
            else:
                # Closing tag - only system tags are enforced against tag_stack
                if tag_name in SYSTEM_TAGS:
                    if not tag_stack:
                        has_unexpected_closing_tags = True
                        issues.append(
                            XMLTagIssue(
                                issue_type="unexpected_closing_tag",
                                severity="MAJOR",
                                message=f"Unexpected closing tag '</{tag_name}>' with no matching open tag.",
                                line_number=line_num,
                                column_number=col_num,
                                tag_name=tag_name,
                                context_snippet=full_match,
                            )
                        )
                    else:
                        last_open, last_line, last_col, last_tag = tag_stack.pop()
                        if last_open != tag_name:
                            has_mismatched_tags = True
                            issues.append(
                                XMLTagIssue(
                                    issue_type="mismatched_closing_tag",
                                    severity="MAJOR",
                                    message=(
                                        f"Mismatched closing tag '</{tag_name}>' at line {line_num}:{col_num}, "
                                        f"expected '</{last_open}>' (opened at line {last_line}:{last_col})."
                                    ),
                                    line_number=line_num,
                                    column_number=col_num,
                                    tag_name=tag_name,
                                    expected_tag=last_open,
                                    opened_at_line=last_line,
                                    context_snippet=f"{last_tag} ... {full_match}",
                                )
                            )

        # 4. Check for unclosed system tags remaining in stack (unclosed non-system tags are ignored)
        while tag_stack:
            unclosed_name, unclosed_line, unclosed_col, unclosed_tag = tag_stack.pop()
            if unclosed_name in SYSTEM_TAGS:
                has_unclosed_tags = True
                issues.append(
                    XMLTagIssue(
                        issue_type="unclosed_tag",
                        severity="MAJOR",
                        message=f"Unclosed tag '<{unclosed_name}>' opened at line {unclosed_line}:{unclosed_col} was never closed.",
                        line_number=unclosed_line,
                        column_number=unclosed_col,
                        tag_name=unclosed_name,
                        context_snippet=unclosed_tag,
                    )
                )

        if total_tags == 0:
            issues.append(
                XMLTagIssue(
                    issue_type="empty_document",
                    severity="CRITICAL",
                    message="No XML sequence tags found in the document.",
                    line_number=1,
                )
            )

        # Document is invalid if it has mismatched tags, unclosed system tags, unexpected closing tags, truncated tags, unknown tags, or fatal prohibited tags
        is_valid = not (
            has_mismatched_tags
            or has_unclosed_tags
            or has_unexpected_closing_tags
            or has_truncated_tag
            or (not allow_unknown_tags and has_unknown_tags)
            or any(i.severity == "CRITICAL" for i in issues)
        )

        return XMLValidationResult(
            is_valid=is_valid,
            issues=issues,
            tag_counts=tag_counts,
            total_tags=total_tags,
            has_mismatched_tags=has_mismatched_tags,
            has_unclosed_tags=has_unclosed_tags,
            has_unexpected_closing_tags=has_unexpected_closing_tags,
            has_truncated_tag=has_truncated_tag,
            has_prohibited_tags=has_prohibited_tags,
            has_unknown_tags=has_unknown_tags,
            file_path=file_path,
        )

    @classmethod
    def validate_file(cls, file_path: Union[str, Path]) -> XMLValidationResult:
        """
        Validates an XML file from path.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text(encoding="utf-8")
        return cls.check(content, file_path=str(path))

    @classmethod
    def is_valid_xml(cls, xml_content: str) -> bool:
        """
        Fast boolean check helper.
        """
        return cls.check(xml_content).is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Azozo XML Tag & Syntax Checker for OCR Agent Outputs"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to an XML file or a directory containing XML files to audit",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Scan directories recursively for .xml files (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output errors and failed files",
    )
    args = parser.parse_args()

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"❌ Error: Target path '{args.target}' does not exist.")
        sys.exit(1)

    if target_path.is_file():
        result = XMLChecker.validate_file(target_path)
        print(result.summary())
        sys.exit(0 if result.is_valid else 1)
    else:
        # Directory scan
        xml_files = list(target_path.glob("**/*.xml")) if args.recursive else list(target_path.glob("*.xml"))
        print(f"🔍 Scanning {len(xml_files)} XML file(s) in '{target_path}'...")

        passed = 0
        failed = 0
        failed_files: List[Tuple[Path, XMLValidationResult]] = []

        for xml_p in xml_files:
            res = XMLChecker.validate_file(xml_p)
            if res.is_valid:
                passed += 1
                if not args.quiet:
                    print(f"  🟢 [PASS] {xml_p.relative_to(target_path)}")
            else:
                failed += 1
                failed_files.append((xml_p, res))
                print(f"  🔴 [FAIL] {xml_p.relative_to(target_path)}")
                for err in res.error_messages:
                    print(f"       {err}")

        print("\n" + "=" * 60)
        print(f"Total Scanned: {len(xml_files)} | Passed: {passed} | Failed: {failed}")
        print("=" * 60)

        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
