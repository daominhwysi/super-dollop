import os
import shutil
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST_PATH = Path("scratch/curated_batch_manifest.json")
SRC_DIR = Path("../scrape-thuvienhoclieu/downloads")
RAW_DIR = Path("data/sequence_labelling_input_data_raw")
TARGET_MANIFEST = Path("data/batch_420_exams_manifest.json")

def convert_single_docx(docx_path: Path, out_dir: Path, worker_id: int = 0):
    user_prof = f"/tmp/libreoffice_prof_{worker_id}"
    try:
        cmd = [
            "libreoffice",
            f"-env:UserInstallation=file://{user_prof}",
            "--headless", "--convert-to", "pdf",
            "--outdir", str(out_dir), str(docx_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        pdf_path = out_dir / f"{docx_path.stem}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return True, str(pdf_path)
        return False, f"PDF not created or empty: {res.stderr}"
    except Exception as e:
        return False, str(e)

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found!")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sampled = data.get("sampled_exams", {})
    all_files = []
    for cat, flist in sampled.items():
        for item in flist:
            all_files.append((item["filename"], cat, item["pages"]))

    print(f"Total files in manifest: {len(all_files)}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    to_convert = []
    manifest_entries = []

    for filename, cat, pages in all_files:
        src = SRC_DIR / filename
        dst_docx = RAW_DIR / filename
        dst_pdf = RAW_DIR / f"{Path(filename).stem}.pdf"

        if not src.exists():
            print(f"Warning: Source not found: {src}")
            continue

        # Copy docx if not exists
        if not dst_docx.exists():
            shutil.copy2(src, dst_docx)
            copied += 1

        # Check if pdf already exists
        if not dst_pdf.exists() or dst_pdf.stat().st_size == 0:
            to_convert.append(dst_docx)

        manifest_entries.append({
            "filename": filename,
            "category": cat,
            "estimated_pages": pages,
            "docx_path": str(dst_docx),
            "pdf_path": str(dst_pdf)
        })

    print(f"Copied {copied} docx files to {RAW_DIR}")
    print(f"Need to convert {len(to_convert)} files to PDF...")

    # Parallel conversion with LibreOffice (max 4 parallel workers)
    if to_convert:
        success_cnt = 0
        fail_cnt = 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(convert_single_docx, docx, RAW_DIR, idx % 4): docx
                for idx, docx in enumerate(to_convert)
            }
            done_cnt = 0
            for future in as_completed(future_to_file):
                done_cnt += 1
                ok, err = future.result()
                if ok:
                    success_cnt += 1
                else:
                    fail_cnt += 1
                    docx = future_to_file[future]
                    print(f"Failed {docx.name}: {err}")
                if done_cnt % 50 == 0 or done_cnt == len(to_convert):
                    print(f"Converted {done_cnt}/{len(to_convert)} (Success: {success_cnt}, Failed: {fail_cnt})...")

    # Verify how many PDFs exist
    valid_pdfs = [e for e in manifest_entries if Path(e["pdf_path"]).exists() and Path(e["pdf_path"]).stat().st_size > 0]
    print(f"\nFinal Valid PDFs in {RAW_DIR}: {len(valid_pdfs)}/{len(all_files)}")

    # Save target manifest
    with open(TARGET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest to {TARGET_MANIFEST}")

if __name__ == "__main__":
    main()
