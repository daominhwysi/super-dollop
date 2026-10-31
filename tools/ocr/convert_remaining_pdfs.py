import os
import glob
import subprocess
import signal
import time
from pathlib import Path

RAW_DIR = Path("data/sequence_labelling_input_data_raw")

def main():
    docx_files = sorted(glob.glob(str(RAW_DIR / "*.docx")))
    print(f"Total DOCX: {len(docx_files)}")

    to_convert = []
    for d in docx_files:
        pdf = RAW_DIR / f"{Path(d).stem}.pdf"
        if not pdf.exists() or pdf.stat().st_size == 0:
            to_convert.append(d)

    print(f"Remaining to convert: {len(to_convert)}")

    ok_cnt = 0
    skip_cnt = 0

    t_start = time.time()
    for idx, docx_path in enumerate(to_convert):
        pdf_path = RAW_DIR / f"{Path(docx_path).stem}.pdf"
        t0 = time.time()
        user_prof = f"/tmp/lo_convert_{os.getpid()}"

        cmd = [
            "libreoffice",
            f"-env:UserInstallation=file://{user_prof}",
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(RAW_DIR),
            docx_path
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        try:
            proc.communicate(timeout=10)
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                ok_cnt += 1
                dur = time.time() - t0
                if (ok_cnt % 10 == 0) or (idx + 1 == len(to_convert)):
                    print(f"[{idx+1}/{len(to_convert)}] OK ({dur:.1f}s): {Path(docx_path).name[:45]}...")
            else:
                skip_cnt += 1
                print(f"[{idx+1}/{len(to_convert)}] FAILED: {Path(docx_path).name}")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            skip_cnt += 1
            print(f"[{idx+1}/{len(to_convert)}] TIMEOUT (Killed after 10s): {Path(docx_path).name}")
        except Exception as e:
            skip_cnt += 1
            print(f"[{idx+1}/{len(to_convert)}] ERROR: {Path(docx_path).name}: {e}")

    total_time = time.time() - t_start
    all_pdfs = [d for d in docx_files if (RAW_DIR / f"{Path(d).stem}.pdf").exists() and (RAW_DIR / f"{Path(d).stem}.pdf").stat().st_size > 0]
    print("=" * 60)
    print(f"CONVERSION COMPLETE in {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"New OK: {ok_cnt}, Skipped/Failed: {skip_cnt}")
    print(f"Total Valid PDFs in {RAW_DIR}: {len(all_pdfs)}/{len(docx_files)} ({len(all_pdfs)/len(docx_files)*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
