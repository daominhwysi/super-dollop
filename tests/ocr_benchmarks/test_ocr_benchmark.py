import sys
import time
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

# Add project root to sys.path
workspace_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_dir))

import fitz
from sequence_labelling.annotator.pdf_converter import PDFOCRConverter
from sequence_labelling.config import OCR_MODEL, OCR_PROVIDER, OCR_BATCH_SIZE, OCR_CONCURRENCY, LOGS_DIR

def benchmark_pdf_ocr(pdf_path_str: str, results_dir: Optional[Path] = None):
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"Error: File {pdf_path} does not exist.")
        return

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Starting PDF OCR Benchmark ===")
    print(f"File: {pdf_path.name}")
    print(f"Total Pages: {total_pages}")
    print(f"Model: {OCR_MODEL} (Provider: {OCR_PROVIDER})")
    print(f"Batch Size: {OCR_BATCH_SIZE}")
    print(f"Concurrency: {OCR_CONCURRENCY}")
    print("-----------------------------------")

    start_time = time.time()
    
    converter = PDFOCRConverter(
        model=OCR_MODEL,
        provider=OCR_PROVIDER,
        batch_size=OCR_BATCH_SIZE,
        concurrency=OCR_CONCURRENCY
    )

    def progress_callback(completed, total, msg):
        elapsed = time.time() - start_time
        print(f"[{elapsed:6.2f}s] ({completed}/{total} batches) - {msg}")

    ocr_result = converter.convert_pdf(pdf_path, progress_callback=progress_callback)
    
    end_time = time.time()
    total_duration = end_time - start_time
    sec_per_page = total_duration / max(1, total_pages)
    char_count = len(ocr_result)
    pages_per_min = (total_pages / (total_duration / 60))

    summary = {
        "timestamp": timestamp_str,
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "model": OCR_MODEL,
        "provider": OCR_PROVIDER,
        "batch_size": OCR_BATCH_SIZE,
        "concurrency": OCR_CONCURRENCY,
        "total_duration_seconds": round(total_duration, 2),
        "seconds_per_page": round(sec_per_page, 2),
        "pages_per_minute": round(pages_per_min, 2),
        "character_count": char_count
    }

    # Save outputs to results directory
    out_dir = results_dir or (Path(__file__).resolve().parent / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    md_output_path = out_dir / f"ocr_{pdf_path.stem}_{timestamp_str}.md"
    json_output_path = out_dir / f"summary_{pdf_path.stem}_{timestamp_str}.json"

    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(ocr_result)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("===================================")
    print(f"Benchmark Results:")
    print(f"Total Duration : {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Average / Page : {sec_per_page:.2f} seconds/page")
    print(f"Total Output   : {char_count} characters")
    print(f"Pages/Minute   : {pages_per_min:.2f} pages/min")
    print(f"Saved Text     : {md_output_path}")
    print(f"Saved Summary  : {json_output_path}")
    print("===================================")

if __name__ == "__main__":
    pdf_target = "/home/daominhwysi/project/azozo-experiment/ocr_logs/req_20260718_153303_f053ab47/input.pdf"
    benchmark_pdf_ocr(pdf_target)
