#!/usr/bin/env python3
"""
strip_annotation_tags.py — Deterministic Tag Stripper for Vision OCR Outputs.

Removes sequence labelling XML tags (<question_label>, <stem>, <option_label>,
<option_text>, <other>) while strictly preserving:
- Original transcribed text verbatim (no synthetic alteration).
- Page wrappers (<pages>, <page>).
- Page metadata (<page_metadata>...</page_metadata>).
- HTML tables (<table>...</table>).
- Direct figure tags (<figure ... />).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("strip_annotation_tags")

# Tags that must be removed completely (opening and closing)
SEQUENCE_TAG_PATTERN = re.compile(
    r"</?(?:question_label|stem|option_label|option_text|other)\b[^>]*>",
    re.IGNORECASE,
)

# Residual whitespace pattern to clean up 3+ blank lines created by tag removal
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def strip_tags_from_text(content: str) -> str:
    """
    Strips sequence labelling XML tags from OCR text while keeping content intact.
    Preserves page boundaries, page metadata, tables, and figure tags.
    """
    if not content:
        return ""

    # 1. Strip sequence tags
    cleaned = SEQUENCE_TAG_PATTERN.sub("", content)

    # 2. Normalize excessive blank lines (max 2 consecutive newlines)
    cleaned = EXCESS_NEWLINES_PATTERN.sub("\n\n", cleaned)

    return cleaned.strip() + "\n"


def verify_no_sequence_tags(content: str) -> List[str]:
    """
    Returns any forbidden sequence labelling tags found in the content.
    """
    matches = SEQUENCE_TAG_PATTERN.findall(content)
    return matches


def process_directory(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Processes all .md files in input_dir and saves clean raw text to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list(input_dir.glob("*.md")))
    total_files = len(input_files)
    logger.info(f"Found {total_files} documents in {input_dir} to strip.")

    success_count = 0
    failed_count = 0
    total_tags_stripped = 0
    verification_errors = []

    for idx, in_file in enumerate(input_files, start=1):
        rel_name = in_file.name
        out_file = output_dir / rel_name

        if out_file.exists() and not overwrite:
            logger.debug(f"Skipping existing: {rel_name}")
            continue

        try:
            raw_content = in_file.read_text(encoding="utf-8")
            tag_matches = SEQUENCE_TAG_PATTERN.findall(raw_content)
            tag_count = len(tag_matches)
            total_tags_stripped += tag_count

            cleaned_content = strip_tags_from_text(raw_content)

            # Verification: zero forbidden tags remaining
            remaining_tags = verify_no_sequence_tags(cleaned_content)
            if remaining_tags:
                err_msg = f"Residual tags found in {rel_name}: {remaining_tags[:5]}"
                logger.error(err_msg)
                verification_errors.append(err_msg)
                failed_count += 1
                continue

            out_file.write_text(cleaned_content, encoding="utf-8")
            success_count += 1

            if idx % 50 == 0 or idx == total_files:
                logger.info(f"Progress: {idx}/{total_files} documents processed.")

        except Exception as e:
            logger.error(f"Failed to process {rel_name}: {e}")
            failed_count += 1

    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "total_source_files": total_files,
        "successfully_stripped": success_count,
        "failed_files": failed_count,
        "total_tags_removed": total_tags_stripped,
        "verification_errors": verification_errors,
        "all_passed": (failed_count == 0 and len(verification_errors) == 0),
    }

    logger.info(f"Tag stripping complete: {success_count}/{total_files} successfully stripped.")
    logger.info(f"Total sequence labelling XML tags removed: {total_tags_stripped:,}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Strip sequence labelling tags from OCR markdown outputs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/batch_500_diverse_output",
        help="Input directory containing annotated OCR markdown files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/batch_500_diverse_raw",
        help="Output directory to write clean raw text markdown files",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/stripping_report.json",
        help="Path to save execution report",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files in output directory",
    )
    args = parser.parse_args()

    summary = process_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        overwrite=not args.no_overwrite,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
