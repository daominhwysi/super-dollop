import os
import sys
import glob
import json
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from sequence_labelling.parser.long_parser.sequence_reconciler import (
    merge_chunk_xmls,
    reconcile_parser_chunk_results,
)


def generate_merger_summary_report(
    input_file: str,
    chunk_xml_files: List[str],
    merge_result: Dict[str, Any]
) -> str:
    input_filename = os.path.basename(input_file)
    input_stem = os.path.splitext(input_filename)[0]

    md = []
    md.append(f"# 🔀 Sequence Merger Visualization Report: `{input_stem}`")
    md.append("")
    md.append(f"- **Input OCR Document:** `{input_file}`")
    md.append(f"- **Parsed Chunks Merged:** `{len(chunk_xml_files)}` chunks")
    md.append(f"- **Total Unique Questions Parsed:** `{merge_result.get('total_questions')}` questions")
    md.append(f"- **Question Range:** `{merge_result.get('question_range')}`")
    md.append(f"- **Boundary Items Deduplicated:** `{merge_result.get('deduplicated_count')}` duplicate entries")
    md.append(f"- **Merged Unified Output File:** 👉 [merged_full_document.xml](merged_full_document.xml)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 📦 Input Parsed Chunks")
    md.append("")
    md.append("| Chunk | Source File | Status |")
    md.append("| :--- | :--- | :--- |")

    for idx, xml_path in enumerate(chunk_xml_files):
        fname = os.path.basename(xml_path)
        md.append(f"| **Chunk {idx + 1}** | [{fname}](../../parser_visualize/results/{input_stem}/{fname}) | ✅ Merged |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🛠️ Verification & Reconciler Metrics")
    diagnostics = merge_result.get("diagnostics", {})
    validation = diagnostics.get("validation", {})
    md.append(f"- **Source Fidelity:** `{validation.get('source_fidelity_ok', False)}`")
    md.append(f"- **Balanced Tags:** `{validation.get('balanced_ok', False)}`")
    md.append(f"- **Source Authority:** `{merge_result.get('source_authority', 'original_chunk_text')}`")
    md.append(f"- **Conflicts Resolved:** `{len(diagnostics.get('conflicts', []))}`")

    return "\n".join(md)


def main():
    default_parser_results_dir = "/home/daominhwysi/project/azozo-experiment/tests/parser_visualize/results/ocr_input_20260722_222013"
    parser_dir = sys.argv[1] if len(sys.argv) > 1 else default_parser_results_dir

    input_stem = os.path.basename(parser_dir.rstrip("/"))
    base_code_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_code_dir, "results", input_stem)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Locate all chunk_*.xml files from parser_visualize output
    chunk_files = sorted(glob.glob(os.path.join(parser_dir, "chunk_*_parsed.xml")))
    if not chunk_files:
        print(f"Error: No chunk_*_parsed.xml files found in {parser_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} parsed chunk XML files in: {parser_dir}")
    chunk_contents = []
    for fpath in chunk_files:
        print(f" Loading: {os.path.basename(fpath)}")
        with open(fpath, "r", encoding="utf-8") as f:
            chunk_contents.append(f.read())

    manifest_path = os.path.join(parser_dir, "parser_chunks_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        chunk_results = []
        for item in manifest["chunks"]:
            with open(os.path.join(parser_dir, item["raw_chunk_text_file"]), "r", encoding="utf-8") as f:
                raw_text = f.read()
            with open(os.path.join(parser_dir, item["parsed_xml_file"]), "r", encoding="utf-8") as f:
                parsed_xml = f.read()
            chunk_results.append({
                **item,
                "raw_chunk_text": raw_text,
                "raw_xml": parsed_xml,
            })
        merge_result = reconcile_parser_chunk_results(chunk_results)
        merge_result["total_questions"] = len(merge_result["structured_questions"])
        merge_result["question_range"] = "source-scoped"
        merge_result["deduplicated_count"] = len(
            merge_result["diagnostics"].get("conflicts", [])
        )
        merge_result["source_authority"] = "original_chunk_text"
    else:
        print("Warning: parser manifest missing; parsing raw text via _strip_xml for source-grounded merging.")
        from sequence_labelling.parser.long_parser.source_merger.canonical_source import _strip_xml
        raw_inputs = [_strip_xml(c) for c in chunk_contents]
        merge_result = merge_chunk_xmls(
            chunk_contents,
            raw_chunk_inputs=raw_inputs,
        )

    # 3. Output merged_full_document.xml
    merged_xml_path = os.path.join(out_dir, "merged_full_document.xml")
    header_comment = (
        f"<!-- 🌟 MERGED FULL DOCUMENT XML: {input_stem} -->\n"
        f"<!-- Total Unique Questions: {merge_result.get('total_questions')} | Question Range: {merge_result.get('question_range')} -->\n"
        f"<!-- Deduplicated Boundary Items: {merge_result.get('deduplicated_count')} | Chunks Merged: {len(chunk_files)} -->\n\n"
    )
    full_merged_content = header_comment + merge_result.get("merged_xml", "")

    with open(merged_xml_path, "w", encoding="utf-8") as f:
        f.write(full_merged_content)
    print(f"  -> Generated unified merged XML document: {merged_xml_path}")

    # 4. Generate Merger Visualization Summary Report
    summary_md = generate_merger_summary_report(input_stem, chunk_files, merge_result)
    summary_path = os.path.join(out_dir, "merger_visualization.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Successfully generated merger visualization report at: {summary_path}")


if __name__ == "__main__":
    main()
