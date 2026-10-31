import os
import glob
import subprocess
import time
from pathlib import Path

RAW_DIR = Path("data/sequence_labelling_input_data_raw")

def main():
    docx_files = sorted(glob.glob(str(RAW_DIR / "*.docx")))
    print(f"Total DOCX in {RAW_DIR}: {len(docx_files)}")

    to_convert = []
    for d in docx_files:
        pdf = RAW_DIR / f"{Path(d).stem}.pdf"
        if not pdf.exists() or pdf.stat().st_size == 0:
            to_convert.append(d)

    print(f"Remaining DOCX needing conversion: {len(to_convert)}")

    converted_ok = 0
    failed_or_timeout = 0

    for idx, docx_path in enumerate(to_convert):
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(RAW_DIR),
            docx_path
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            pdf = RAW_DIR / f"{Path(docx_path).stem}.pdf"
            if pdf.exists() and pdf.stat().st_size > 0:
                converted_ok += 1
            else:
                failed_or_timeout += 1
                print(f"Failed ({idx+1}/{len(to_convert)}): {Path(docx_path).name}")
        except subprocess.TimeoutExpired:
            failed_or_timeout += 1
            print(f"Timeout ({idx+1}/{len(to_convert)}): {Path(docx_path).name}")
        except Exception as e:
            failed_or_timeout += 1
            print(f"Error ({idx+1}/{len(to_convert)}): {Path(docx_path).name}: {e}")

        if (idx + 1) % 25 == 0 or (idx + 1) == len(to_convert):
            print(f"Progress: {idx+1}/{len(to_convert)} (OK: {converted_ok}, Failed/Timeout: {failed_or_timeout})")

    # Verify
    all_pdfs = [d for d in docx_files if (RAW_DIR / f"{Path(d).stem}.pdf").exists()]
    print(f"Total Converted PDFs in {RAW_DIR}: {len(all_pdfs)}/{len(docx_files)}")

if __name__ == "__main__":
    main()
