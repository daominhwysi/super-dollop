#!/usr/bin/env python3
"""
Classifies all PDF documents in data/sequence_labelling_input_data_raw into:
1. DIGITAL: Pages contain substantial extractable text stream.
2. SCANNED: Pages contain raster images with negligible or no extractable text.
3. HYBRID / MIXED: Document contains both digital and scanned pages.

Outputs summary statistics and saves detailed JSON report.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pymupdf

RAW_DIR = Path("data/sequence_labelling_input_data_raw")
OUTPUT_REPORT = Path("data/pdf_classification_report.json")


def analyze_pdf(pdf_path: Path) -> Dict[str, Any]:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return {
            "file_name": pdf_path.name,
            "error": str(e),
            "type": "CORRUPTED",
            "page_count": 0,
        }

    page_count = len(doc)
    if page_count == 0:
        return {
            "file_name": pdf_path.name,
            "type": "EMPTY",
            "page_count": 0,
        }

    total_chars = 0
    digital_pages = 0
    scanned_pages = 0
    page_stats = []

    for pno, page in enumerate(doc):
        text = page.get_text().strip()
        char_count = len(text)
        total_chars += char_count
        images = page.get_images()
        image_count = len(images)

        # Classification heuristics per page:
        # A page is digital if it has >= 100 characters of extractable text,
        # or >= 40 characters and fewer than 2 full-page raster images.
        is_digital = char_count >= 100 or (char_count >= 40 and image_count <= 2)
        if is_digital:
            digital_pages += 1
        else:
            scanned_pages += 1

        page_stats.append({
            "page": pno + 1,
            "chars": char_count,
            "images": image_count,
            "is_digital": is_digital,
        })

    avg_chars_per_page = total_chars / page_count
    digital_ratio = digital_pages / page_count

    # PDF-level classification
    if digital_ratio >= 0.85:
        pdf_type = "DIGITAL"
    elif digital_ratio <= 0.15:
        pdf_type = "SCANNED"
    else:
        pdf_type = "HYBRID"

    return {
        "file_name": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": round(avg_chars_per_page, 1),
        "digital_pages": digital_pages,
        "scanned_pages": scanned_pages,
        "digital_ratio": round(digital_ratio, 3),
        "type": pdf_type,
    }


def main():
    if not RAW_DIR.exists():
        print(f"Error: Directory {RAW_DIR} does not exist.")
        sys.exit(1)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    total_files = len(pdf_files)
    print(f"Analyzing {total_files} PDFs in '{RAW_DIR}'...")

    start_time = time.time()
    results = []
    counts = {"DIGITAL": 0, "SCANNED": 0, "HYBRID": 0, "CORRUPTED": 0, "EMPTY": 0}
    total_pages_by_type = {"DIGITAL": 0, "SCANNED": 0, "HYBRID": 0}

    for idx, pdf in enumerate(pdf_files, 1):
        info = analyze_pdf(pdf)
        results.append(info)
        t = info.get("type", "CORRUPTED")
        counts[t] = counts.get(t, 0) + 1
        if t in total_pages_by_type:
            total_pages_by_type[t] += info.get("page_count", 0)

        if idx % 50 == 0 or idx == total_files:
            print(f"  Processed {idx}/{total_files} ({(idx/total_files)*100:.1f}%) | Digital: {counts['DIGITAL']}, Scanned: {counts['SCANNED']}, Hybrid: {counts['HYBRID']}")

    elapsed = time.time() - start_time

    summary = {
        "total_pdfs": total_files,
        "elapsed_seconds": round(elapsed, 2),
        "counts": counts,
        "percentages": {
            k: round((v / total_files) * 100, 2) for k, v in counts.items()
        },
        "total_pages_by_type": total_pages_by_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": results,
    }

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total PDFs analyzed : {total_files}")
    print(f"Elapsed Time        : {elapsed:.2f}s ({total_files / max(0.01, elapsed):.1f} files/s)")
    print(f"  - DIGITAL         : {counts['DIGITAL']} ({summary['percentages']['DIGITAL']}%) | {total_pages_by_type['DIGITAL']:,} pages")
    print(f"  - SCANNED         : {counts['SCANNED']} ({summary['percentages']['SCANNED']}%) | {total_pages_by_type['SCANNED']:,} pages")
    print(f"  - HYBRID          : {counts['HYBRID']} ({summary['percentages']['HYBRID']}%) | {total_pages_by_type['HYBRID']:,} pages")
    if counts.get("CORRUPTED", 0) > 0:
        print(f"  - CORRUPTED       : {counts['CORRUPTED']} ({summary['percentages']['CORRUPTED']}%)")
    print("=" * 60)
    print(f"Detailed report saved to: {OUTPUT_REPORT.resolve()}")


if __name__ == "__main__":
    main()
