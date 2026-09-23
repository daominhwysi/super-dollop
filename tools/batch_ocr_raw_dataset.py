import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.pdf_converter import PDFOCRConverter

def run_batch_ocr(
    raw_dir: Path = WORKSPACE_DIR / "data" / "sequence_labelling_input_data_raw",
    out_dir: Path = WORKSPACE_DIR / "data" / "sequence_labelling_input_data",
    overwrite: bool = False,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    concurrency: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    if not raw_dir.exists():
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        sys.exit(1)

    all_pdf_files = sorted(list(raw_dir.rglob("*.pdf")))
    total_files = len(all_pdf_files)

    if not overwrite:
        files_to_process = [
            pdf for pdf in all_pdf_files
            if not (out_dir / pdf.relative_to(raw_dir).with_suffix(".md")).exists()
        ]
        skipped_count = total_files - len(files_to_process)
    else:
        files_to_process = all_pdf_files
        skipped_count = 0

    converter = PDFOCRConverter(
        model=model,
        provider=provider,
        concurrency=concurrency,
        batch_size=batch_size,
    )

    print(f"=== Starting Batch OCR on Raw Dataset ===")
    print(f"  Source Folder  : {raw_dir}")
    print(f"  Output Folder  : {out_dir}")
    print(f"  Total Raw PDFs : {total_files}")
    print(f"  Already Done   : {skipped_count} (skipped)")
    print(f"  To Process     : {len(files_to_process)}")
    print(f"  OCR Provider   : {converter.provider}")
    print(f"  OCR Model      : {converter.model}")
    print(f"  Base URL       : {converter.base_url}")
    print(f"  Batch Size     : {converter.batch_size} image(s)/batch")
    print(f"  Concurrency    : {converter.concurrency} parallel worker(s)")
    print(f"  Figure Detect  : {'Enabled' if converter.figure_detector else 'Disabled'}")
    print(f"  Overwrite Mode : {'Enabled (Overwrite existing)' if overwrite else 'Disabled (Skip existing)'}")
    print("=========================================")

    success_count = 0
    failed_count = 0

    total_to_process = len(files_to_process)
    for file_idx, pdf_path in enumerate(files_to_process, start=1):
        print(f"[Batch OCR Progress] {file_idx}/{total_to_process}: {pdf_path.name}")
        rel_path = pdf_path.relative_to(raw_dir)
        target_md = out_dir / rel_path.with_suffix(".md")

        target_md.parent.mkdir(parents=True, exist_ok=True)

        def make_callback(f_idx, f_total, f_name):
            def callback(stage: str, current: int, total: int, msg: str):
                print(f"[Batch OCR Progress] PDF {f_idx}/{f_total} {f_name}: {stage} {current}/{total} - {msg}")
            return callback

        cb = make_callback(file_idx, total_to_process, pdf_path.stem)

        try:
            ocr_text = converter.convert_pdf(pdf_path, progress_callback=cb)
            target_md.write_text(ocr_text, encoding="utf-8")
            success_count += 1
        except Exception as e:
            failed_count += 1
            print(f"\n[Error] Failed to process '{rel_path}': {e}")

    print("\n=========================================")
    print(f"Batch OCR Completed!")
    print(f"  Successfully Processed: {success_count}/{len(files_to_process)}")
    print(f"  Skipped (Existing)    : {skipped_count}")
    print(f"  Failed                : {failed_count}")
    print("=========================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch OCR all PDF files in data/sequence_labelling_input_data_raw")
    parser.add_argument("--raw-dir", type=str, default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data_raw"), help="Raw input directory containing PDFs")
    parser.add_argument("--out-dir", type=str, default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"), help="Output directory for OCR markdown files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output .md files")
    parser.add_argument("--model", type=str, default=None, help="OCR Vision LLM model identifier")
    parser.add_argument("--provider", type=str, default=None, help="OCR Provider (e.g. xah, commandcode, nvidia, deepseek, vilao)")
    parser.add_argument("--concurrency", type=int, default=None, help="Number of parallel OCR batch requests")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of images per OCR batch request")
    args = parser.parse_args()
    
    run_batch_ocr(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        overwrite=args.overwrite,
        model=args.model,
        provider=args.provider,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
