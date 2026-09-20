#!/usr/bin/env python3
"""Separate deterministic-parser findings from LLM findings in an audit report.

The audit report's ``issues`` entries do not contain provenance.  This script
re-runs the deterministic checks against each XML/raw-text pair and matches the
reported findings against those results:

* ``parser``: exact category/severity/message match from the deterministic audit
* ``ambiguous``: same category was emitted by the deterministic audit, but the
  message did not match exactly (the script refuses to guess)
* ``llm``: no deterministic finding matched and the category was not emitted by
  the deterministic audit for that document
* ``unknown``: the source XML was unavailable, so provenance could not be checked

Usage:
    python scripts/separate_audit_issues.py
    python scripts/separate_audit_issues.py --input data/review_report.json \
        --output data/review_report_separated.json
    python scripts/separate_audit_issues.py --exam-root data/sequence_labelling_annotated
    python scripts/separate_audit_issues.py --exam-root data/sequence_labelling_annotated --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_labelling.annotator.reviewer import DeterministicAuditor  # noqa: E402


IssueKey = Tuple[str, str, str, Optional[int], Optional[str]]
StrictIssueKey = Tuple[str, str, str]


def _value(value: Any) -> str:
    """Return a stable string for enum/string values."""
    return getattr(value, "value", str(value))


def _issue_key(issue: Dict[str, Any]) -> IssueKey:
    return (
        str(issue.get("category", "")),
        _value(issue.get("severity", "")),
        str(issue.get("message", "")),
        issue.get("line_number"),
        issue.get("context_snippet"),
    )


def _strict_issue_key(issue: Dict[str, Any]) -> StrictIssueKey:
    return (
        str(issue.get("category", "")),
        _value(issue.get("severity", "")),
        str(issue.get("message", "")),
    )


def _issue_dict(issue: Any) -> Dict[str, Any]:
    """Convert an AuditIssue (Pydantic v1/v2) to a plain dictionary."""
    if hasattr(issue, "model_dump"):
        result = issue.model_dump()
        result["severity"] = _value(result.get("severity", ""))
        return result
    if hasattr(issue, "dict"):
        result = issue.dict()
        result["severity"] = _value(result.get("severity", ""))
        return result
    return {
        "category": str(getattr(issue, "category", "")),
        "severity": _value(getattr(issue, "severity", "")),
        "message": str(getattr(issue, "message", "")),
        "line_number": getattr(issue, "line_number", None),
        "context_snippet": getattr(issue, "context_snippet", None),
    }


def _read_optional(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def deterministic_issues(xml_content: str, raw_ocr_text: Optional[str]) -> List[Dict[str, Any]]:
    """Run the same deterministic checks used by the hybrid reviewer."""
    issues: List[Any] = []

    syntax_issues, _ = DeterministicAuditor.check_xml_syntax(xml_content)
    issues.extend(syntax_issues)

    prohibited_issues, _ = DeterministicAuditor.check_prohibited_tags(xml_content)
    issues.extend(prohibited_issues)

    nesting_issues, _ = DeterministicAuditor.check_stimulus_wrapping_system_tags(xml_content)
    issues.extend(nesting_issues)

    anchor_issues, _ = DeterministicAuditor.check_stimulus_anchors(
        xml_content=xml_content,
        pure_text=DeterministicAuditor.strip_xml_tags(xml_content),
        raw_ocr_text=raw_ocr_text,
    )
    issues.extend(anchor_issues)

    structure_issues, _, _ = DeterministicAuditor.check_question_and_option_structure(
        xml_content, raw_ocr_text=raw_ocr_text
    )
    issues.extend(structure_issues)

    coverage_issues, _, _ = DeterministicAuditor.check_question_coverage_via_det(
        xml_content=xml_content, raw_ocr_text=raw_ocr_text
    )
    issues.extend(coverage_issues)

    fidelity_issues, _, _ = DeterministicAuditor.check_verbatim_alignment(
        xml_content=xml_content, raw_ocr_text=raw_ocr_text
    )
    issues.extend(fidelity_issues)

    return [_issue_dict(issue) for issue in issues]


def classify_report_issues(report: Dict[str, Any]) -> Dict[str, Any]:
    xml_content = _read_optional(report.get("file_path"))
    raw_ocr_text = _read_optional(report.get("raw_file_path"))
    reported_issues = list(report.get("issues") or [])

    if xml_content is None:
        enriched = [dict(issue, source="unknown", match_type="source_xml_unavailable") for issue in reported_issues]
        return {
            "doc_id": report.get("doc_id"),
            "deterministic_issue_count": None,
            "issues": enriched,
            "issues_by_source": {
                "parser": [],
                "llm": [],
                "ambiguous": [],
                "unknown": enriched,
            },
        }

    parser_issues = deterministic_issues(xml_content, raw_ocr_text)
    remaining_exact = Counter(_issue_key(issue) for issue in parser_issues)
    remaining_strict = Counter(_strict_issue_key(issue) for issue in parser_issues)
    parser_categories = {str(issue.get("category", "")) for issue in parser_issues}

    enriched: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for original_issue in reported_issues:
        issue = dict(original_issue)
        exact_key = _issue_key(issue)
        strict_key = _strict_issue_key(issue)

        if remaining_exact[exact_key] > 0:
            source, match_type = "parser", "exact"
            remaining_exact[exact_key] -= 1
            remaining_strict[strict_key] -= 1
        elif remaining_strict[strict_key] > 0:
            source, match_type = "parser", "category_severity_message"
            remaining_strict[strict_key] -= 1
        elif str(issue.get("category", "")) in parser_categories:
            source, match_type = "ambiguous", "category_only"
        else:
            source, match_type = "llm", "no_deterministic_match"

        issue["source"] = source
        issue["match_type"] = match_type
        enriched.append(issue)
        grouped[source].append(issue)

    return {
        "doc_id": report.get("doc_id"),
        "deterministic_issue_count": len(parser_issues),
        "issues": enriched,
        "issues_by_source": {
            "parser": grouped.get("parser", []),
            "llm": grouped.get("llm", []),
            "ambiguous": grouped.get("ambiguous", []),
            "unknown": grouped.get("unknown", []),
        },
    }


def annotate_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return one per-exam audit report with provenance fields added."""
    classified = classify_report_issues(report)
    annotated = dict(report)
    annotated["issues"] = classified["issues"]
    annotated["issues_by_source"] = classified["issues_by_source"]
    annotated["deterministic_issue_count"] = classified["deterministic_issue_count"]
    annotated["provenance_method"] = (
        "Deterministic checks were re-run and matched by category/severity/message; "
        "category-only matches are ambiguous."
    )
    annotated["provenance_summary"] = {
        source: len(issues)
        for source, issues in classified["issues_by_source"].items()
    }
    return annotated


def separate_report(input_path: Path) -> Dict[str, Any]:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    classified_reports: List[Dict[str, Any]] = []
    totals = Counter()

    for original_report in report.get("reports", []):
        classified = classify_report_issues(original_report)
        classified_reports.append(classified)
        for issue in classified["issues"]:
            totals[issue["source"]] += 1

    return {
        "source_report": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification_method": (
            "Re-run deterministic checks and match category/severity/message; "
            "category-only matches remain ambiguous."
        ),
        "summary": {
            "documents": len(classified_reports),
            "total_issues": sum(totals.values()),
            "parser_issues": totals["parser"],
            "llm_issues": totals["llm"],
            "ambiguous_issues": totals["ambiguous"],
            "unknown_issues": totals["unknown"],
        },
        "reports": classified_reports,
    }


def separate_exam_reports(exam_root: Path, in_place: bool) -> Dict[str, Any]:
    """Annotate each exam's audit_report.json, in place or as a sidecar file."""
    audit_paths = sorted(exam_root.glob("*/audit_report.json"))
    totals = Counter()
    processed = 0
    output_paths: List[str] = []

    for audit_path in audit_paths:
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        annotated = annotate_report(report)
        target = audit_path if in_place else audit_path.with_name("audit_report_separated.json")
        target.write_text(json.dumps(annotated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        processed += 1
        output_paths.append(str(target))
        for source, count in annotated["provenance_summary"].items():
            totals[source] += count

    return {
        "exam_reports": processed,
        "parser_issues": totals["parser"],
        "llm_issues": totals["llm"],
        "ambiguous_issues": totals["ambiguous"],
        "unknown_issues": totals["unknown"],
        "in_place": in_place,
        "output_examples": output_paths[:3],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/review_report.json",
        help="Existing aggregate audit report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/review_report_separated.json",
        help="Output JSON containing issue provenance labels.",
    )
    parser.add_argument(
        "--exam-root",
        type=Path,
        help="Process each */audit_report.json under this directory instead of an aggregate report.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="When used with --exam-root, update each audit_report.json directly. Without it, write audit_report_separated.json sidecars.",
    )
    args = parser.parse_args()

    if args.exam_root:
        result = separate_exam_reports(args.exam_root.resolve(), in_place=args.in_place)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    result = separate_report(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
