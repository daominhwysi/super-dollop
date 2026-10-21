"""Batch generation tool for DocumentCartographer manifests using gpt-6-luna."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from sequence_labelling.parser.long_parser.document_cartographer import DocumentCartographer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cartography_generator")


def discover_target_documents(
    annotated_dir: Path,
    input_dir: Path,
    target_multi_chunk_only: bool = True,
    subject_filter: Optional[str] = None,
    catalog_path: Optional[Path] = None,
) -> List[Tuple[str, Path, int]]:
    """
    Discovers candidate documents to survey.
    Returns [(doc_id, input_md_path, chunk_count)]
    """
    catalog = {}
    if catalog_path and catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f).get("documents", {})
        except Exception:
            pass

    targets: List[Tuple[str, Path, int]] = []

    for d in sorted(annotated_dir.iterdir()):
        if not d.is_dir():
            continue
        doc_id = d.name
        chunks_dir = d / "chunks"
        chunk_count = len(list(chunks_dir.glob("*.xml"))) if chunks_dir.exists() else 0

        if target_multi_chunk_only and chunk_count <= 1:
            continue

        if subject_filter:
            doc_meta = catalog.get(doc_id, {})
            subj = str(doc_meta.get("subject", ""))
            if subject_filter.lower() not in subj.lower():
                continue

        # Find raw markdown file
        md_file = input_dir / f"{doc_id}.md"
        if not md_file.exists():
            candidates = list(input_dir.glob(f"*{doc_id}*.md"))
            if candidates:
                md_file = candidates[0]
            else:
                continue

        targets.append((doc_id, md_file, chunk_count))

    return targets


def process_single_document(
    doc_id: str,
    md_path: Path,
    out_dir: Path,
    cartographer: DocumentCartographer,
    subject: Optional[str],
    progress_lock: Lock,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates cartography manifest for a single document."""
    out_file = out_dir / f"{doc_id}.json"
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("objects"):
                with progress_lock:
                    stats["completed"] += 1
                    stats["cached"] += 1
                    logger.info(f"[{stats['completed']}/{stats['total']}] [CACHED] {doc_id} ({subject})")
                return {"status": "cached", "doc_id": doc_id}
        except Exception:
            pass

    try:
        raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
        start_time = time.time()

        manifest = cartographer.survey_document(
            ocr_markdown=raw_text,
            doc_id=doc_id,
            subject=subject,
        )

        elapsed = time.time() - start_time
        manifest_dict = manifest.to_dict()
        manifest_dict["generation_time_seconds"] = round(elapsed, 2)
        manifest_dict["generator_model"] = cartographer.model
        manifest_dict["generator_provider"] = cartographer.provider

        # Atomic write
        tmp_file = out_dir / f"{doc_id}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2)
        tmp_file.replace(out_file)

        with progress_lock:
            stats["completed"] += 1
            stats["generated"] += 1
            logger.info(
                f"[{stats['completed']}/{stats['total']}] [OK] {doc_id} ({subject}) "
                f"- {len(manifest.questions)} Qs, {len(manifest.stimuli)} Stims, "
                f"Clean Seams: {len(manifest.clean_cut_seams)} in {elapsed:.1f}s"
            )

        return {"status": "ok", "doc_id": doc_id, "manifest": manifest_dict}
    except Exception as e:
        with progress_lock:
            stats["completed"] += 1
            stats["failed"] += 1
            logger.error(f"[{stats['completed']}/{stats['total']}] [FAIL] {doc_id}: {e}")
        return {"status": "error", "doc_id": doc_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch generate DocumentCartographer manifests.")
    parser.add_argument("--model", type=str, default="gpt-6-luna", help="LLM model (default: gpt-6-luna)")
    parser.add_argument("--provider", type=str, default="aixingialaire", help="LLM provider (default: aixingialaire)")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel worker threads (default: 2)")
    parser.add_argument("--all", action="store_true", help="Process ALL documents instead of multi-chunk only")
    parser.add_argument("--subject", type=str, default=None, help="Filter by subject (e.g. 'Tiếng Anh')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sequence_labelling_cartography"))
    args = parser.parse_args()

    annotated_dir = Path("data/sequence_labelling_annotated")
    input_dir = Path("data/sequence_labelling_input_data")
    catalog_path = Path("data/sequence_labelling_metadata_catalog.json")

    catalog = {}
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f).get("documents", {})

    targets = discover_target_documents(
        annotated_dir=annotated_dir,
        input_dir=input_dir,
        target_multi_chunk_only=not args.all,
        subject_filter=args.subject,
        catalog_path=catalog_path,
    )

    if args.limit:
        targets = targets[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("🚀 DOCUMENT CARTOGRAPHER BATCH GENERATOR")
    logger.info(f"Target Documents: {len(targets)} (Multi-chunk only: {not args.all})")
    logger.info(f"Model: {args.model} | Provider: {args.provider} | Concurrency: {args.concurrency}")
    logger.info(f"Output Directory: {args.out_dir}")
    logger.info("=" * 60)

    cartographer = DocumentCartographer(
        model=args.model,
        provider=args.provider,
        thinking="off",
    )

    stats = {
        "total": len(targets),
        "completed": 0,
        "cached": 0,
        "generated": 0,
        "failed": 0,
    }
    progress_lock = Lock()

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for doc_id, md_path, _ in targets:
            subject = catalog.get(doc_id, {}).get("subject")
            f = executor.submit(
                process_single_document,
                doc_id=doc_id,
                md_path=md_path,
                out_dir=args.out_dir,
                cartographer=cartographer,
                subject=subject,
                progress_lock=progress_lock,
                stats=stats,
            )
            futures.append(f)

        for future in as_completed(futures):
            future.result()

    total_time = time.time() - start_all
    logger.info("=" * 60)
    logger.info(
        f"🏁 COMPLETED in {total_time:.1f}s: "
        f"Generated={stats['generated']}, Cached={stats['cached']}, Failed={stats['failed']}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
