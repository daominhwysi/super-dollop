#!/usr/bin/env python3
"""
Batch Script to Re-Merge All Annotated Exam Folders in data/sequence_labelling_annotated.
Applies the fixed Source Merger with self-closing stimulus anchor tag support,
strip terminal sentinels, and accurate structured question/stimuli mapping.
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.parser.long_parser.sequence_reconciler import (
    reconcile_parser_chunk_results,
    merge_chunk_xmls,
)
from sequence_labelling.parser.long_parser.source_merger.serializer_validator import remove_annotation_tags


def _chunk_sort_key(p: Path) -> int:
    m = re.search(r"chunk_(\d+)", p.name)
    return int(m.group(1)) if m else 999


def sanitize_prohibited_tags(xml_content: str) -> str:
    cleaned = re.sub(r"</?page_metadata>\s*\{[\s\S]*?\}\s*</?page_metadata>", "", xml_content)
    cleaned = re.sub(r"<page_metadata>[\s\S]*?</page_metadata>", "", cleaned)
    cleaned = re.sub(r"</?pages?>", "", cleaned)
    cleaned = re.sub(r"</?page_metadata>", "", cleaned)
    cleaned = cleaned.replace("<|END|>", "").replace("<|endoftext|>", "").rstrip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def remerge_exam_dir(exam_dir: Path, raw_base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Remerges an individual exam directory.
    """
    chunks_dir = exam_dir / "chunks"
    chunk_json_files = []
    chunk_xml_files = []

    if chunks_dir.exists():
        chunk_json_files = sorted(
            [p for p in chunks_dir.glob("chunk_*.json") if not p.name.endswith(".audit.json") and not p.name.startswith(".")],
            key=_chunk_sort_key
        )
        chunk_xml_files = sorted(
            [p for p in chunks_dir.glob("chunk_*.xml") if not p.name.endswith(".audit.xml") and not p.name.startswith(".")],
            key=_chunk_sort_key
        )

    if not chunk_json_files and not chunk_xml_files:
        # Single-file exam without chunks
        merged_xml_path = exam_dir / "merged.xml"
        if merged_xml_path.exists():
            content = merged_xml_path.read_text(encoding="utf-8")
            cleaned = sanitize_prohibited_tags(content)
            if cleaned != content.rstrip():
                merged_xml_path.write_text(cleaned + "\n", encoding="utf-8")
                return {"doc_id": exam_dir.name, "status": "cleaned_sentinel", "chunks_count": 0}
        return {"doc_id": exam_dir.name, "status": "skipped_no_chunks", "chunks_count": 0}

    chunk_results = []
    if chunk_json_files:
        for c_json in chunk_json_files:
            try:
                data = json.loads(c_json.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "chunk_index" in data and "raw_chunk_text" in data:
                    chunk_results.append(data)
            except Exception:
                pass

    # Prefer chunk_*.xml on disk as the canonical XML source (repaired chunks edit chunk_*.xml,
    # while chunk_*.json may retain pre-repair raw_xml / raw_chunk_text).
    xml_by_index: Dict[int, str] = {}
    if chunk_xml_files:
        for idx, c_xml_path in enumerate(chunk_xml_files):
            m = re.search(r"chunk_(\d+)", c_xml_path.name)
            c_idx = int(m.group(1)) if m else idx
            xml_by_index[c_idx] = sanitize_prohibited_tags(c_xml_path.read_text(encoding="utf-8"))

    def _persist_merge_result(merge_result: Dict[str, Any], status_name: str, n_chunks: int) -> Dict[str, Any]:
        merged_xml = sanitize_prohibited_tags(merge_result["merged_xml"])
        questions = merge_result["structured_questions"]
        stimuli = merge_result["structured_stimuli"]
        diag = merge_result["diagnostics"]

        (exam_dir / "merged.xml").write_text(merged_xml, encoding="utf-8")

        merged_json_path = exam_dir / "merged.json"
        merged_json_data = {}
        if merged_json_path.exists():
            try:
                merged_json_data = json.loads(merged_json_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        merged_json_data.update({
            "remerge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "merge_status": "success",
            "questions_count": len(questions),
            "stimuli_count": len(stimuli),
            "questions": questions,
            "stimuli": stimuli,
            "diagnostics": diag,
            "merge_diagnostics": diag,
            "source_fidelity_ok": diag.get("validation", {}).get("source_fidelity_ok", True),
        })
        merged_json_path.write_text(
            json.dumps(merged_json_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return {
            "doc_id": exam_dir.name,
            "status": status_name,
            "chunks_count": n_chunks,
            "questions_count": len(questions),
            "stimuli_count": len(stimuli),
            "source_fidelity_ok": diag.get("validation", {}).get("source_fidelity_ok", True),
        }

    if len(chunk_xml_files) == 1:
        only_xml = next(iter(xml_by_index.values()))
        detagged_only = remove_annotation_tags(only_xml)
        raw_matches = (
            len(chunk_results) == 1
            and detagged_only.strip() == chunk_results[0].get("raw_chunk_text", "").strip()
        )
        if not chunk_json_files or not raw_matches:
            try:
                res = merge_chunk_xmls([only_xml], raw_chunk_inputs=[detagged_only])
                return _persist_merge_result(res, "remerged_single_chunk_xml", 1)
            except Exception as err:
                return {
                    "doc_id": exam_dir.name,
                    "status": "error",
                    "error": str(err),
                    "chunks_count": 1,
                }

    if chunk_results and (not chunk_xml_files or len(chunk_results) == len(chunk_xml_files)):
        try:
            synced_results = []
            text_drifted = False
            for item in chunk_results:
                c_idx = int(item["chunk_index"])
                item_copy = dict(item)
                if c_idx in xml_by_index:
                    disk_xml = xml_by_index[c_idx]
                    item_copy["raw_xml"] = disk_xml
                    if remove_annotation_tags(disk_xml).strip() != item.get("raw_chunk_text", "").strip():
                        text_drifted = True
                synced_results.append(item_copy)

            merge_result = reconcile_parser_chunk_results(synced_results)

            # When repaired chunk XML text diverges from cached raw_chunk_text (e.g. HTML table/bold edits),
            # also evaluate source-grounded parser-text merging and keep whichever drops fewer spans.
            if text_drifted and chunk_xml_files:
                try:
                    ordered_xmls = [
                        xml_by_index[k] for k in sorted(xml_by_index.keys())
                    ]
                    alt_result = merge_chunk_xmls(ordered_xmls, allow_parser_text_as_source=True)
                    orig_proj = sum(
                        p.get("spans_projected", 0)
                        for p in merge_result.get("diagnostics", {}).get("parses", [])
                    )
                    alt_proj = sum(
                        p.get("spans_projected", 0)
                        for p in alt_result.get("diagnostics", {}).get("parses", [])
                    )
                    orig_rej = sum(
                        p.get("spans_rejected", 0)
                        for p in merge_result.get("diagnostics", {}).get("parses", [])
                    )
                    alt_rej = sum(
                        p.get("spans_rejected", 0)
                        for p in alt_result.get("diagnostics", {}).get("parses", [])
                    )
                    if (
                        alt_result.get("diagnostics", {}).get("validation", {}).get("source_fidelity_ok", False)
                        and (alt_rej < orig_rej or (alt_rej == orig_rej and alt_proj > orig_proj))
                        and len(alt_result.get("annotations", [])) >= len(merge_result.get("annotations", []))
                    ):
                        merge_result = alt_result
                except Exception:
                    pass

            return _persist_merge_result(merge_result, "remerged_success", len(chunk_results))
        except Exception as err:
            return {
                "doc_id": exam_dir.name,
                "status": "error",
                "error": str(err),
                "chunks_count": len(chunk_results),
            }
    elif chunk_xml_files:
        try:
            xml_contents = [p.read_text(encoding="utf-8") for p in chunk_xml_files]
            res = merge_chunk_xmls(xml_contents, allow_parser_text_as_source=True)
            (exam_dir / "merged.xml").write_text(res["merged_xml"], encoding="utf-8")
            return {
                "doc_id": exam_dir.name,
                "status": "remerged_xml_fallback",
                "chunks_count": len(chunk_xml_files),
                "questions_count": res.get("total_questions", 0),
            }
        except Exception as err:
            return {
                "doc_id": exam_dir.name,
                "status": "error",
                "error": str(err),
                "chunks_count": len(chunk_xml_files),
            }

    return {"doc_id": exam_dir.name, "status": "unknown", "chunks_count": 0}


def main():
    parser = argparse.ArgumentParser(description="Batch re-merge all annotated exam folders")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Root path to annotated dataset directory (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=8,
        help="Number of parallel workers for re-merging (default: 8)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 70)
    print("🔄 AZOZO BATCH RE-MERGER")
    print("=" * 70)
    print(f"  Target Directory : {input_dir}")
    print(f"  Concurrency      : {args.concurrency}")
    print("=" * 70)

    # Find all exam directories
    merged_files = sorted(input_dir.rglob("merged.xml"))
    exam_dirs = [p.parent for p in merged_files]

    print(f"\nDiscovered {len(exam_dirs)} exam directories to process.")
    start_time = time.time()

    results = []
    success_count = 0
    cleaned_count = 0
    skipped_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(remerge_exam_dir, ed): ed for ed in exam_dirs}
        for completed, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            results.append(res)
            st = res.get("status")
            if "remerged" in st:
                success_count += 1
            elif st == "cleaned_sentinel":
                cleaned_count += 1
            elif st == "skipped_no_chunks":
                skipped_count += 1
            elif st == "error":
                error_count += 1
                print(f"  ❌ Error re-merging {res.get('doc_id')}: {res.get('error')}")
            print(f"[Re-merging exams] {completed}/{len(futures)} completed: {res.get('doc_id')}")

    duration = time.time() - start_time
    print("\n" + "=" * 70)
    print("📊 BATCH RE-MERGE COMPLETED")
    print("=" * 70)
    print(f"  Total Exam Dirs Processed : {len(exam_dirs)}")
    print(f"  Re-merged with Chunks     : {success_count}")
    print(f"  Cleaned Sentinels (Single): {cleaned_count}")
    print(f"  Skipped (Already Clean)   : {skipped_count}")
    print(f"  Errors                    : {error_count}")
    print(f"  Total Duration            : {duration:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
