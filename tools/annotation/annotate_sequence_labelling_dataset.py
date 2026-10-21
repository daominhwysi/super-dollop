import os
import sys
import json
import time
import uuid
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    PARSER_MAX_TOKENS,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
)
from sequence_labelling.annotator.ocr_document import OCRDocument
from sequence_labelling.parser.long_parser.sequence_reconciler import (
    DocumentStateStack,
    extract_metadata_headers_from_markdown,
    reconcile_parser_chunk_results,
)
from sequence_labelling.parser.long_parser.greedy_chunker import (
    PAGE_SEPARATOR,
    build_chunk_plan,
    greedy_oversize_chunker,
)
from sequence_labelling.parser.long_parser.parser_agent_worker import ParserAgentWorker
from sequence_labelling.annotator.xml_checker import XMLChecker
from sequence_labelling.annotator.xml_cleaner import XMLCleaner
from sequence_labelling.llm.agy_rate_limiter import Agy5HourCapError


def extract_pages_from_markdown(full_markdown: str) -> List[str]:
    """
    Extracts page blocks from OCR/markdown text using OCRDocument.
    """
    if not full_markdown:
        return []
    doc = OCRDocument.from_markdown(full_markdown)
    return [p.raw_text for p in doc.pages]


def process_single_document(
    file_path: Path,
    rel_path: Path,
    out_dir: Path,
    parser_model: str,
    parser_provider: str,
    target_tokens: int = CHUNKER_TARGET_TOKENS,
    max_tokens: int = CHUNKER_MAX_TOKENS,
    overlap_pages: int = CHUNKER_OVERLAP_PAGES,
    concurrency: int = 4,
    parser_thinking: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Annotates a single document using the chunker and parser worker,
    saving both the merged version and per-chunk output files.
    """
    start_time = time.time()
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    ocr_doc = OCRDocument.from_file(file_path)
    metadata_headers = ocr_doc.page_metadata_list
    state_stack = DocumentStateStack(doc_id=doc_id)
    completed_groups = state_stack.process_metadata_stream(metadata_headers)
    page_list = ocr_doc.to_chunk_pages()

    # --- Step 1: Chunker ---
    chunks = greedy_oversize_chunker(
        page_list,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_pages=overlap_pages,
    )
    chunk_plans = [
        build_chunk_plan(chunk_pages, chunk_index)
        for chunk_index, chunk_pages in enumerate(chunks)
    ]

    doc_out_dir = out_dir / rel_path.parent / rel_path.stem
    existing_chunks_dir = doc_out_dir / "chunks"

    # --- Step 2: Parallel Parser Swarm ---
    worker = ParserAgentWorker(
        model=parser_model,
        provider=parser_provider,
        thinking=parser_thinking,
    )
    chunk_results = [None] * len(chunks)
    max_workers = max(1, min(concurrency, len(chunks)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for plan in chunk_plans:
            chunk_idx = plan["chunk_index"]
            c_json = existing_chunks_dir / f"chunk_{chunk_idx}.json"
            c_xml = existing_chunks_dir / f"chunk_{chunk_idx}.xml"
            if not overwrite and existing_chunks_dir.exists() and c_xml.exists() and c_xml.stat().st_size > 100 and c_json.exists():
                try:
                    cached = json.loads(c_json.read_text(encoding="utf-8"))
                    if cached.get("parse_status") not in ("failed", "error") and cached.get("raw_xml"):
                        print(f"  ⚡ [Cached Chunk] Document '{rel_path}' Chunk {chunk_idx}: loaded from disk ({len(cached.get('questions', []))} questions)", flush=True)
                        chunk_results[chunk_idx] = cached
                        continue
                except Exception:
                    pass

            future = executor.submit(
                worker.process_chunk,
                plan["raw_chunk_text"],
                chunk_index=chunk_idx,
                page_ids=plan["page_ids"],
                overlap_page_ids=plan["overlap_page_ids"],
                page_offset_ranges=plan["page_offset_ranges"],
            )
            futures[future] = plan

        for future in as_completed(futures):
            plan = futures[future]
            chunk_idx = plan["chunk_index"]
            try:
                res = future.result()
                print(f"  ✅ [Parsed Chunk] Document '{rel_path}' Chunk {chunk_idx}: finished ({len(res.get('questions', []))} questions)", flush=True)
            except Agy5HourCapError:
                raise
            except Exception as e:
                print(f"[Error] Chunk {chunk_idx} failed for {rel_path}: {e}", flush=True)
                res = {
                    **plan,
                    "raw_xml": "",
                    "spans": [],
                    "parse_status": "failed",
                    "parse_diagnostics": {"attempts": worker.max_attempts, "validation_errors": [str(e)]},
                    "questions": [],
                    "stimuli": {},
                }
            chunk_results[chunk_idx] = res

    # --- Step 3: Save Per-Chunk Outputs First ---
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = doc_out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_file_info = []
    valid_chunk_results = sorted([r for r in chunk_results if r is not None], key=lambda x: x.get("chunk_index", 0))

    for cres in valid_chunk_results:
        idx = cres.get("chunk_index", 0)
        chunk_json_path = chunks_dir / f"chunk_{idx}.json"
        chunk_xml_path = chunks_dir / f"chunk_{idx}.xml"

        chunk_json_path.write_text(
            json.dumps(cres, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        clean_chunk = XMLCleaner.clean(cres.get("raw_xml", ""))
        chunk_xml_path.write_text(
            clean_chunk.cleaned_xml,
            encoding="utf-8"
        )
        chunk_file_info.append({
            "chunk_index": idx,
            "page_ids": cres.get("page_ids", []),
            "overlap_page_ids": cres.get("overlap_page_ids", []),
            "questions_count": len(cres.get("questions", [])),
            "parse_status": cres.get("parse_status", "unknown"),
            "json_path": str(chunk_json_path.relative_to(out_dir)),
            "xml_path": str(chunk_xml_path.relative_to(out_dir)),
        })

        diag = cres.get("parse_diagnostics", {})
        attempts = diag.get("attempts", 1)
        v_errors = diag.get("validation_errors", [])
        
        # Check XML syntax & tag matching
        chunk_xml_check = XMLChecker.check(cres.get("raw_xml", ""))
        if not chunk_xml_check.is_valid:
            v_errors.extend(chunk_xml_check.error_messages)
            print(
                f"  ⚠️ [XML TAG MISMATCH / ISSUE] Document '{rel_path}' Chunk {idx}: "
                f"{len(chunk_xml_check.issues)} issue(s) detected. Diagnostics: {chunk_xml_check.error_messages[:3]}"
            )
        elif attempts > 1 or v_errors:
            print(
                f"  ⚠️ [RETRY / WARNING] Document '{rel_path}' Chunk {idx}: "
                f"{attempts} attempt(s) used. Diagnostics: {v_errors}"
            )

    # --- Step 4: Sequence Reconciliation (Skip Linker) ---
    merge_status = "success"
    merge_error_msg = None
    try:
        merge_result = reconcile_parser_chunk_results(valid_chunk_results)
    except Exception as merge_err:
        print(f"[Warning] Merge invariant reconciliation failed for '{rel_path}': {merge_err}. Using fallback merger.")
        merge_status = "fallback_concatenation"
        merge_error_msg = str(merge_err)

        # Fallback: concatenate questions, stimuli, and XML from valid chunks
        all_q = []
        all_s = {}
        xml_parts = []
        for cr in valid_chunk_results:
            all_q.extend(cr.get("questions", []))
            all_s.update(cr.get("stimuli", {}))
            if cr.get("raw_xml"):
                xml_parts.append(cr["raw_xml"])

        merge_result = {
            "structured_questions": all_q,
            "structured_stimuli": all_s,
            "merged_xml": "\n".join(xml_parts),
            "diagnostics": {
                "merge_status": "fallback",
                "error": merge_error_msg,
            },
        }

    questions_count = len(merge_result.get("structured_questions", []))
    if questions_count == 0:
        print(f"  ℹ️ Document '{rel_path}': 0 questions found (non-exam or reference document).")

    duration = time.time() - start_time

    # Clean and validate merged XML output
    raw_merged_xml = merge_result.get("merged_xml", "")
    clean_merged = XMLCleaner.clean(raw_merged_xml)
    merged_xml_content = clean_merged.cleaned_xml
    merged_xml_check = XMLChecker.check(merged_xml_content)
    if not merged_xml_check.is_valid:
        print(f"  ❌ [MERGED XML TAG ISSUE] '{rel_path}': {len(merged_xml_check.issues)} issue(s) found in merged.xml")

    # Save Merged Version
    merged_data = {
        "doc_id": doc_id,
        "input_file": str(rel_path),
        "total_chunks": len(chunks),
        "chunk_token_config": {"target_tokens": target_tokens, "max_tokens": max_tokens},
        "linker_skipped": True,
        "merge_status": merge_status,
        "merge_error": merge_error_msg,
        "questions_count": questions_count,
        "stimuli_count": len(merge_result["structured_stimuli"]),
        "duration_seconds": round(duration, 2),
        "diagnostics": merge_result.get("diagnostics", {}),
        "xml_validation": {
            "is_valid": merged_xml_check.is_valid,
            "issues_count": len(merged_xml_check.issues),
            "issues": [str(i) for i in merged_xml_check.issues],
            "has_mismatched_tags": merged_xml_check.has_mismatched_tags,
            "has_unclosed_tags": merged_xml_check.has_unclosed_tags,
            "fixes_applied": clean_merged.fixes_applied,
        },
        "questions": merge_result["structured_questions"],
        "stimuli": merge_result["structured_stimuli"],
        "merged_xml": merged_xml_content,
        "chunks_summary": chunk_file_info,
    }

    if not merged_xml_content.strip() and len(full_text.strip()) > 300:
        print(f"❌ [Error] Merged XML is completely empty for '{rel_path}' despite non-empty input. Refusing to write empty output.")
        return {
            "status": "failed",
            "file": str(rel_path),
            "chunks": len(chunks),
            "questions": 0,
            "duration": round(duration, 2),
            "output_dir": str(doc_out_dir),
        }

    merged_json_path = doc_out_dir / "merged.json"
    merged_xml_path = doc_out_dir / "merged.xml"

    merged_json_path.write_text(
        json.dumps(merged_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    merged_xml_path.write_text(
        merged_xml_content,
        encoding="utf-8"
    )

    return {
        "status": merge_status,
        "file": str(rel_path),
        "chunks": len(chunks),
        "questions": len(merge_result["structured_questions"]),
        "duration": round(duration, 2),
        "output_dir": str(doc_out_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Annotate sequence_labelling_input_data with 48-64k chunks, skipping Linker, saving merged and per-chunk results."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"),
        help="Path to input markdown documents directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_annotated"),
        help="Path to save annotated JSON & XML outputs",
    )
    parser.add_argument("--model", type=str, default=PARSER_MODEL, help=f"Parser model name (default: {PARSER_MODEL})")
    parser.add_argument(
        "--provider",
        type=str,
        default=PARSER_PROVIDER,
        choices=["codex", "xah", "deepseek", "nvidia", "vilao", "commandcode", "agy", "antigravity", "aixingialaire"],
        help=f"Parser provider (default: {PARSER_PROVIDER})",
    )
    parser.add_argument("--target_tokens", type=int, default=CHUNKER_TARGET_TOKENS, help="Target chunk size in tokens")
    parser.add_argument("--max_tokens", type=int, default=CHUNKER_MAX_TOKENS, help="Maximum chunk size in tokens")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent chunk worker threads")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index (0-based) for parallel multi-worker runs")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel multi-worker runs")
    parser.add_argument("--pattern", type=str, default="*.md", help="Glob pattern for markdown files to process (e.g. 'exam-00*.md')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    target_model = args.model or PARSER_MODEL
    target_provider = args.provider or PARSER_PROVIDER

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    all_md_files = sorted(list(input_path.rglob(args.pattern)))
    if args.limit:
        all_md_files = all_md_files[: args.limit]

    total_files = len(all_md_files)
    if not args.overwrite:
        md_files = []
        for f in all_md_files:
            doc_dir = output_path / f.relative_to(input_path).parent / f.stem
            merged_json = doc_dir / "merged.json"
            merged_xml = doc_dir / "merged.xml"
            chunks_dir = doc_dir / "chunks"

            if not merged_json.exists() or not merged_xml.exists() or merged_xml.stat().st_size < 100:
                md_files.append(f)
                continue

            has_broken_chunk = False
            if chunks_dir.exists():
                for c_xml in chunks_dir.glob("chunk_*.xml"):
                    if c_xml.stat().st_size == 0:
                        has_broken_chunk = True
                        break
                if not has_broken_chunk:
                    for c_json in chunks_dir.glob("chunk_*.json"):
                        if c_json.name.endswith(".audit.json"):
                            continue
                        try:
                            c_data = json.loads(c_json.read_text(encoding="utf-8"))
                            if c_data.get("parse_status") in ("failed", "error") or not c_data.get("raw_xml"):
                                has_broken_chunk = True
                                break
                        except Exception:
                            has_broken_chunk = True
                            break

            if has_broken_chunk:
                md_files.append(f)
                continue

            try:
                xml_content = merged_xml.read_text(encoding="utf-8", errors="ignore")
                if "<" in xml_content and ">" in xml_content:
                    continue
            except Exception:
                pass
            md_files.append(f)
        skipped_count = total_files - len(md_files)
    else:
        md_files = all_md_files
        skipped_count = 0

    if args.num_shards > 1:
        md_files = [f for idx, f in enumerate(md_files) if idx % args.num_shards == args.shard_id]

    from sequence_labelling.config import get_provider_base_url
    parser_base_url = get_provider_base_url(target_provider)

    print("==================================================================")
    print("=== Sequence Labeling Batch Annotation Pipeline (Linker Skipped) =")
    print(f"  Input Directory   : {input_path}")
    print(f"  Output Directory  : {output_path}")
    print(f"  Total Documents   : {total_files}")
    print(f"  Already Done      : {skipped_count} (skipped)")
    print(f"  Shard             : {args.shard_id + 1}/{args.num_shards}" if args.num_shards > 1 else "  Shard             : Full Set (1/1)")
    print(f"  To Process        : {len(md_files)}")
    print(f"  Parser Provider   : {target_provider}")
    print(f"  Parser Model      : {target_model}")
    print(f"  Base URL          : {parser_base_url}")
    print(f"  Concurrency       : {args.concurrency} worker thread(s)")
    print(f"  Chunker Budget    : {args.target_tokens} - {args.max_tokens} tokens")
    print(f"  Max Output Tokens : {PARSER_MAX_TOKENS}")
    print(f"  Linker Enabled    : False (Skipped)")
    print(f"  Overwrite Mode    : {'Enabled (Overwrite existing)' if args.overwrite else 'Disabled (Skip existing)'}")
    print("==================================================================")

    success_count = 0
    failed_count = 0

    for file_idx, file_path in enumerate(md_files, start=1):
        rel_path = file_path.relative_to(input_path)
        print(f"[Annotating Documents] {file_idx}/{len(md_files)}: {rel_path}")

        try:
            res = process_single_document(
                file_path=file_path,
                rel_path=rel_path,
                out_dir=output_path,
                parser_model=target_model,
                parser_provider=target_provider,
                target_tokens=args.target_tokens,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                overwrite=args.overwrite,
            )
            success_count += 1
        except Exception as e:
            failed_count += 1
            print(f"\n[Error] Failed processing '{rel_path}': {e}")

    print("\n==================================================================")
    print(f"Sequence Labeling Annotation Completed!")
    print(f"  Successfully Processed: {success_count}/{len(md_files)}")
    print(f"  Skipped (Existing)    : {skipped_count}")
    print(f"  Failed                : {failed_count}")
    print(f"  Output saved to       : {output_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
