"""
run_luna_batch_ocr.py — Autonomous Production Batch OCR Runner using GPT-6 Luna via aixingialaire.

Features:
- Processes selected PDF exams from data/batch_420_exams_manifest.json.
- Thread-safe concurrent processing (--concurrency 2).
- Automatic resume / idempotent skip for already converted documents.
- Atomic state tracking (data/luna_ocr_420_state.json).
- Real-time token usage and cost accounting (46 VND/1M in, 230 VND/1M out).
- Generates clean, standardized Page-Aware Markdown (--- PAGE X ---) in data/sequence_labelling_input_data.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from openai import OpenAI

from sequence_labelling.config import get_provider_base_url, get_provider_api_key
from sequence_labelling.annotator.agy_pdf_ocr import AGY_DIVERSE_RAW_OCR_BASE_PROMPT, AgyPDFOCRConverter
from sequence_labelling.annotator.pdf_converter import (
    normalize_batch_metadata,
    prune_code_blocks,
    prune_think_tags,
)

# Logging
LOG_FILE = Path("artifacts/luna_batch_ocr_420.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("luna_batch_ocr")

DEFAULT_MANIFEST = Path("data/batch_420_exams_manifest.json")
DEFAULT_RAW_DIR = Path("data/sequence_labelling_input_data_raw")
DEFAULT_OUT_DIR = Path("data/sequence_labelling_input_data")
DEFAULT_STATE_FILE = Path("data/luna_ocr_420_state.json")

# Rates
RATE_INPUT_PER_M = 46.0   # VND
RATE_OUTPUT_PER_M = 230.0 # VND


class BatchOCRState:
    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.data: Dict[str, Any] = {
            "total_target_files": 0,
            "completed_count": 0,
            "failed_count": 0,
            "total_pages_processed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_vnd": 0.0,
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            "completed_docs": {},
            "failed_docs": {},
        }
        self.load()

    def load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file {self.state_file}: {e}")

    def save(self) -> None:
        self.data["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            tmp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state file {self.state_file}: {e}")

    def mark_completed(
        self,
        doc_id: str,
        pages: int,
        input_tokens: int,
        output_tokens: int,
        duration_sec: float,
        out_path: str,
    ) -> None:
        cost = (input_tokens * RATE_INPUT_PER_M / 1_000_000) + (output_tokens * RATE_OUTPUT_PER_M / 1_000_000)
        self.data["completed_count"] += 1
        self.data["total_pages_processed"] += pages
        self.data["total_input_tokens"] += input_tokens
        self.data["total_output_tokens"] += output_tokens
        self.data["total_cost_vnd"] += cost
        self.data["completed_docs"][doc_id] = {
            "pages": pages,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_vnd": round(cost, 2),
            "duration_sec": round(duration_sec, 2),
            "out_path": out_path,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.save()

    def mark_failed(self, doc_id: str, error: str) -> None:
        self.data["failed_count"] += 1
        self.data["failed_docs"][doc_id] = {
            "error": error,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.save()


def clean_markdown_artifacts(text: str) -> str:
    """Strip raw HTML inline styles (span, div) that GPT-6 Luna occasionally emits for colored headers."""
    # Remove div/span style tags but preserve contents
    cleaned = re.sub(r'</?(?:div|span|em|font|b)(?:\s+[^>]*)?>', '', text, flags=re.IGNORECASE)
    # Remove multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def ocr_single_page(
    client: OpenAI,
    model: str,
    prompt: str,
    img_bytes: bytes,
    max_retries: int = 5,
) -> Tuple[str, int, int]:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": AGY_DIVERSE_RAW_OCR_BASE_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }
                ],
                max_tokens=8192
            )
            raw = resp.choices[0].message.content or ""
            in_tok = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
            out_tok = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
            return raw, in_tok, out_tok
        except Exception as e:
            if attempt == max_retries:
                raise e
            sleep_sec = min(30.0, 2.0 * (1.5 ** attempt))
            logger.warning(f"Vision API error (attempt {attempt}/{max_retries}): {e}. Sleeping {sleep_sec:.1f}s...")
            time.sleep(sleep_sec)
    return "", 0, 0


def process_single_document(
    pdf_path: Path,
    out_dir: Path,
    client: OpenAI,
    model: str,
    converter: AgyPDFOCRConverter,
) -> Tuple[bool, str, int, int, int, float]:
    t0 = time.time()
    doc_stem = pdf_path.stem
    target_out = out_dir / f"{doc_stem}.md"

    # Check if already completed
    if target_out.exists() and target_out.stat().st_size > 200:
        return True, str(target_out), 0, 0, 0, 0.0

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return False, f"Failed to open PDF: {e}", 0, 0, 0, 0.0

    total_pages = len(doc)
    page_texts = []
    total_in_tok = 0
    total_out_tok = 0

    for p_idx in range(total_pages):
        page = doc[p_idx]
        p_num = p_idx + 1
        rect = page.rect
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")

        prompt = converter._build_page_prompt(
            page_num=p_num,
            width_pts=rect.width,
            height_pts=rect.height
        )

        try:
            raw_text, in_tok, out_tok = ocr_single_page(client, model, prompt, img_bytes)
            total_in_tok += in_tok
            total_out_tok += out_tok
        except Exception as e:
            doc.close()
            return False, f"Failed OCR on page {p_num}: {e}", 0, 0, 0, 0.0

        clean_page = prune_code_blocks(prune_think_tags(raw_text))
        clean_page = clean_markdown_artifacts(clean_page)
        clean_page = normalize_batch_metadata(clean_page, p_num)
        page_texts.append(clean_page)

    doc.close()

    full_markdown = "\n\n".join(page_texts)

    # Atomic write
    tmp_out = target_out.with_suffix(".tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        f.write(full_markdown)
    tmp_out.replace(target_out)

    duration = time.time() - t0
    return True, str(target_out), total_pages, total_in_tok, total_out_tok, duration


def main():
    parser = argparse.ArgumentParser(description="Autonomous Batch OCR with GPT-6 Luna via TMUX")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Path to manifest JSON")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to raw PDFs")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Path to output markdown dir")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent document workers")
    parser.add_argument("--model", type=str, default="gpt-6-luna", help="Vision LLM model")
    parser.add_argument("--provider", type=str, default="aixingialaire", help="Provider name")
    parser.add_argument("--limit", type=int, default=None, help="Optional max files limit")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    logger.info(f"Loaded manifest with {len(entries)} target exams.")

    # Filter targets that have valid PDFs
    target_pdfs = []
    for e in entries:
        p_path = Path(e["pdf_path"])
        if p_path.exists() and p_path.stat().st_size > 0:
            target_pdfs.append(p_path)

    if args.limit:
        target_pdfs = target_pdfs[:args.limit]

    logger.info(f"Target PDFs ready for OCR: {len(target_pdfs)} files.")

    # Initialize client
    base_url = get_provider_base_url(args.provider)
    api_key = get_provider_api_key(args.provider)
    if not api_key:
        logger.error(f"API key for provider '{args.provider}' is not set!")
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    converter = AgyPDFOCRConverter(model=args.model, thinking="low")
    state = BatchOCRState()
    state.data["total_target_files"] = len(entries)
    state.save()

    logger.info(f"Target exams in manifest: {len(entries)}. Starting continuous processing loop...")

    t_start_all = time.time()

    while True:
        # Check current progress
        completed_entries = []
        pending_entries = []
        waiting_conversion = []

        for e in entries:
            doc_name = Path(e["filename"]).stem
            p_path = Path(e["pdf_path"])
            out_f = out_dir / f"{doc_name}.md"

            if out_f.exists() and out_f.stat().st_size > 200:
                completed_entries.append(doc_name)
                if doc_name not in state.data["completed_docs"]:
                    state.mark_completed(doc_name, e.get("estimated_pages", 5), 0, 0, 0.0, str(out_f))
            elif p_path.exists() and p_path.stat().st_size > 0:
                pending_entries.append(p_path)
            else:
                waiting_conversion.append(doc_name)

        total_done = len(completed_entries)
        logger.info(f"Progress Status: {total_done}/{len(entries)} completed ({total_done/len(entries)*100:.1f}%), {len(pending_entries)} ready to OCR, {len(waiting_conversion)} waiting for PDF conversion.")

        if total_done >= len(entries):
            logger.info("ALL 420 TARGET EXAMS HAVE BEEN OCR-PROCESSED!")
            break

        if not pending_entries:
            if waiting_conversion:
                logger.info(f"Waiting for {len(waiting_conversion)} files to finish PDF conversion... Sleeping 10s.")
                time.sleep(10)
                continue
            else:
                logger.info("No more pending PDFs found. Finishing.")
                break

        # Process available pending in batches with ThreadPoolExecutor
        batch_to_run = pending_entries[:20]
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_pdf = {
                executor.submit(
                    process_single_document,
                    pdf_path,
                    out_dir,
                    client,
                    args.model,
                    converter,
                ): pdf_path
                for pdf_path in batch_to_run
            }

            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    ok, res_str, pages, in_tok, out_tok, dur = future.result()
                    if ok:
                        state.mark_completed(pdf_path.stem, pages, in_tok, out_tok, dur, res_str)
                        cost = (in_tok * RATE_INPUT_PER_M / 1_000_000) + (out_tok * RATE_OUTPUT_PER_M / 1_000_000)
                        logger.info(
                            f"[OCR OK] {pdf_path.stem[:45]}... ({pages}p, {dur:.1f}s, {in_tok}+{out_tok} tok, {cost:.2f} VND)"
                        )
                    else:
                        state.mark_failed(pdf_path.stem, res_str)
                        logger.error(f"[OCR FAIL] {pdf_path.stem}: {res_str}")
                except Exception as e:
                    state.mark_failed(pdf_path.stem, str(e))
                    logger.error(f"[OCR EXCEPTION] {pdf_path.stem}: {e}")

    total_time = time.time() - t_start_all
    logger.info("=" * 60)
    logger.info("BATCH OCR RUN COMPLETE!")
    logger.info(f"Total Completed: {state.data['completed_count']} files.")
    logger.info(f"Total Pages Processed: {state.data['total_pages_processed']} pages.")
    logger.info(f"Total Cost: {state.data['total_cost_vnd']:.2f} VND / 20,000 VND budget.")
    logger.info(f"Total Run Time: {total_time/60:.1f} minutes.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
