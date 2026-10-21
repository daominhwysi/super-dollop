#!/usr/bin/env python3
"""
generate_dataset_metadata.py — Batch Document Metadata Generator using GPT-6 Luna.

Iterates over all documents in data/sequence_labelling_input_data/, extracts
structured metadata (Subject, Examination, Grade, Academic Year, Curriculum, Authority, Title),
saves individual JSONs in data/sequence_labelling_metadata/, and compiles a master catalog.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.metadata.extractor import (
    extract_and_save_metadata,
    extract_document_metadata,
)
from sequence_labelling.metadata.schema import ExamMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_metadata")


def compile_catalog_and_report(
    metadata_dir: Path,
    catalog_path: Path,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compiles all individual *.metadata.json files in metadata_dir into a single
    consolidated catalog and generates a comprehensive Markdown report.
    """
    json_files = sorted(list(metadata_dir.glob("*.metadata.json")))
    catalog: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(json_files),
        "documents": {},
    }

    subjects = Counter()
    grades = Counter()
    exams = Counter()
    curriculums = Counter()
    years = Counter()
    methods = Counter()

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            doc_id = data.get("document_id") or jf.stem.replace(".metadata", "")
            catalog["documents"][doc_id] = data

            # Track distributions
            subj = data.get("subject") or "Unknown / Null"
            subjects[subj] += 1

            grd = data.get("grade") or "Unknown / Null"
            grades[grd] += 1

            exm = data.get("examination") or "Unknown / Null"
            exams[exm] += 1

            curr = data.get("curriculum") or "Unknown / Null"
            curriculums[curr] += 1

            yr = data.get("academic_year") or "Unknown / Null"
            years[yr] += 1

            method = data.get("metadata_extractor", {}).get("method") or "unknown"
            methods[method] += 1
        except Exception as e:
            logger.warning(f"Error reading metadata file '{jf.name}': {e}")

    catalog["statistics"] = {
        "by_subject": dict(subjects.most_common()),
        "by_grade": dict(grades.most_common()),
        "by_examination": dict(exams.most_common(15)),
        "by_curriculum": dict(curriculums.most_common()),
        "by_academic_year": dict(years.most_common(15)),
        "by_extractor_method": dict(methods.most_common()),
    }

    # Save catalog JSON
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # Generate Markdown Report if requested
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            "# Dataset Metadata Catalog & Stratification Analysis",
            "",
            f"- **Generated At:** `{catalog['generated_at']}`",
            f"- **Total Documents Cataloged:** `{catalog['total_documents']}`",
            "",
            "## 1. Distribution by Subject (Môn học)",
            "| Subject | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]
        tot = max(1, catalog["total_documents"])
        for s, c in subjects.most_common():
            md_lines.append(f"| {s} | {c} | {c / tot * 100:.1f}% |")

        md_lines.extend([
            "",
            "## 2. Distribution by Grade (Khối lớp)",
            "| Grade | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for g, c in grades.most_common():
            md_lines.append(f"| {g} | {c} | {c / tot * 100:.1f}% |")

        md_lines.extend([
            "",
            "## 3. Distribution by Curriculum Standard",
            "| Curriculum | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for cur, c in curriculums.most_common():
            md_lines.append(f"| {cur} | {c} | {c / tot * 100:.1f}% |")

        md_lines.extend([
            "",
            "## 4. Top Examinations",
            "| Examination Type | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for ex, c in exams.most_common(15):
            md_lines.append(f"| {ex} | {c} | {c / tot * 100:.1f}% |")

        md_lines.extend([
            "",
            "## 5. Extraction Method Provenance",
            "| Method | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        for m, c in methods.most_common():
            md_lines.append(f"| `{m}` | {c} | {c / tot * 100:.1f}% |")

        report_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info(f"Summary report written to: {report_path}")

    return catalog


def process_single_document(
    md_file: Path,
    output_dir: Path,
    model: str,
    provider: str,
    overwrite: bool,
    existing_originals: Dict[str, str],
) -> Tuple[str, bool, Optional[str], Optional[str], Optional[str]]:
    """
    Process a single document file.
    Returns: (doc_id, success, subject, grade, examination)
    """
    doc_id = md_file.stem
    target_meta_file = output_dir / f"{doc_id}.metadata.json"

    original_filename = existing_originals.get(doc_id)

    # Check if already processed
    if target_meta_file.exists() and not overwrite:
        try:
            data = json.loads(target_meta_file.read_text(encoding="utf-8"))
            if data.get("schema_version") == 2 and data.get("metadata_extractor", {}).get("method") == "llm":
                return (doc_id, True, data.get("subject"), data.get("grade"), data.get("examination"))
            if not original_filename and data.get("original_filename"):
                original_filename = data.get("original_filename")
        except Exception:
            pass

    try:
        ocr_text = md_file.read_text(encoding="utf-8")
        meta = extract_document_metadata(
            ocr_markdown=ocr_text,
            filename=md_file.name,
            doc_id=doc_id,
            original_filename=original_filename,
            model=model,
            provider=provider,
        )

        # Write metadata atomically
        temp_file = target_meta_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
        temp_file.replace(target_meta_file)

        return (doc_id, True, meta.subject, meta.grade, meta.examination)
    except Exception as e:
        logger.error(f"Failed to process '{doc_id}': {e}")
        return (doc_id, False, None, None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate document metadata using GPT-6 Luna via aixingialaire."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"),
        help="Input directory containing OCR markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_metadata"),
        help="Output directory for individual metadata JSON files.",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_metadata_catalog.json"),
        help="Path for consolidated catalog JSON.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(WORKSPACE_DIR / "artifacts" / "reports" / "dataset_metadata_catalog.md"),
        help="Path for markdown summary report.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent workers (default: 2).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-6-luna",
        help="LLM model identifier (default: gpt-6-luna).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="aixingialaire",
        help="LLM provider (default: aixingialaire).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (e.g. for testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force reprocessing even if valid v2 metadata exists.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    catalog_path = Path(args.catalog_path)
    report_path = Path(args.report_path)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all markdown files
    all_files = sorted(list(input_dir.glob("*.md")))
    total_docs = len(all_files)

    # Pre-index existing original filenames from legacy metadata if available
    existing_originals: Dict[str, str] = {}
    for jf in output_dir.glob("*.metadata.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("original_filename"):
                existing_originals[jf.stem.replace(".metadata", "")] = d["original_filename"]
        except Exception:
            pass

    # Filter files needing processing if not overwrite
    if not args.overwrite:
        pending_files = []
        for f in all_files:
            meta_f = output_dir / f"{f.stem}.metadata.json"
            if not meta_f.exists():
                pending_files.append(f)
            else:
                try:
                    data = json.loads(meta_f.read_text(encoding="utf-8"))
                    if data.get("schema_version") != 2 or data.get("metadata_extractor", {}).get("method") != "llm":
                        pending_files.append(f)
                except Exception:
                    pending_files.append(f)
    else:
        pending_files = list(all_files)

    if args.limit:
        pending_files = pending_files[:args.limit]

    logger.info("==================================================")
    logger.info("Batch Exam Metadata Generation with GPT-6 Luna")
    logger.info(f"  Input Directory   : {input_dir}")
    logger.info(f"  Output Directory  : {output_dir}")
    logger.info(f"  Total Markdown Docs: {total_docs}")
    logger.info(f"  Already Processed : {total_docs - len(pending_files)}")
    logger.info(f"  To Process        : {len(pending_files)}")
    logger.info(f"  Concurrency       : {args.concurrency}")
    logger.info(f"  Model / Provider  : {args.model} via {args.provider}")
    logger.info(f"  Overwrite         : {args.overwrite}")
    logger.info("==================================================")

    if not pending_files:
        logger.info("All documents are already processed! Re-compiling catalog...")
        compile_catalog_and_report(output_dir, catalog_path, report_path)
        return

    start_time = time.time()
    completed_count = 0
    success_count = 0
    total_to_process = len(pending_files)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_file = {
            executor.submit(
                process_single_document,
                md_file,
                output_dir,
                args.model,
                args.provider,
                args.overwrite,
                existing_originals,
            ): md_file
            for md_file in pending_files
        }

        for future in as_completed(future_to_file):
            completed_count += 1
            doc_id, success, subj, grd, exm = future.result()
            if success:
                success_count += 1
                logger.info(
                    f"[{completed_count}/{total_to_process}] ({(completed_count/total_to_process)*100:.1f}%) "
                    f"✔ {doc_id} -> Subject: {subj or 'None'} | Grade: {grd or 'None'} | Exam: {exm or 'None'}"
                )
            else:
                logger.warning(
                    f"[{completed_count}/{total_to_process}] ({(completed_count/total_to_process)*100:.1f}%) "
                    f"✘ {doc_id} -> Failed"
                )

            # Periodically compile catalog every 50 completed items
            if completed_count % 50 == 0:
                compile_catalog_and_report(output_dir, catalog_path, report_path)

    # Final catalog compilation
    elapsed = time.time() - start_time
    logger.info("==================================================")
    logger.info(f"Batch Metadata Generation Finished in {elapsed:.1f}s!")
    logger.info(f"  Successfully Processed: {success_count}/{total_to_process}")
    logger.info("==================================================")

    compile_catalog_and_report(output_dir, catalog_path, report_path)


if __name__ == "__main__":
    main()
