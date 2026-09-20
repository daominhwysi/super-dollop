import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
)
from scripts.annotate_sequence_labelling_dataset import process_single_document


def find_table_samples(annotated_dir: Path, input_dir: Path):
    table_samples = []
    for root, dirs, files in os.walk(annotated_dir):
        if "merged.xml" in files:
            rel_dir = Path(os.path.relpath(root, annotated_dir))
            input_file = input_dir / f"{rel_dir}.md"
            xml_file = Path(root) / "merged.xml"

            has_table = False
            if xml_file.exists():
                txt = xml_file.read_text(encoding="utf-8", errors="ignore")
                if "<table>" in txt or "<table " in txt:
                    has_table = True

            if not has_table and input_file.exists():
                txt = input_file.read_text(encoding="utf-8", errors="ignore")
                if "<table>" in txt or "<tr>" in txt or "| ---" in txt or "|---" in txt:
                    has_table = True

            if has_table and input_file.exists():
                table_samples.append((input_file, rel_dir))

    return sorted(table_samples, key=lambda x: str(x[1]))


def main():
    parser = argparse.ArgumentParser(description="Re-annotate sequence labeling samples containing tables.")
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
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent chunk worker threads")
    parser.add_argument("--skip-validator", action="store_true", help="Skip Role B Validator pass")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    samples = find_table_samples(output_path, input_path)

    print("==================================================================")
    print("=== Re-annotating Sequence Labeling Samples Containing Tables ===")
    print(f"  Input Directory   : {input_path}")
    print(f"  Output Directory  : {output_path}")
    print(f"  Parser Model      : {PARSER_MODEL} ({PARSER_PROVIDER})")
    print(f"  Validator Mode    : {'Single-Pass' if args.skip_validator else 'Two-Pass (Role A + Role B)'}")
    print(f"  Target Samples    : {len(samples)}")
    print("==================================================================")

    success_count = 0
    failed_count = 0

    pbar = tqdm(samples, desc="Re-annotating Table Samples", unit="sample")
    for file_path, rel_path in pbar:
        pbar.set_postfix({"sample": rel_path.name[:25], "success": success_count, "failed": failed_count})
        try:
            res = process_single_document(
                file_path=file_path,
                rel_path=rel_path,
                out_dir=output_path,
                parser_model=PARSER_MODEL,
                parser_provider=PARSER_PROVIDER,
                concurrency=args.concurrency,
                enable_validator=not args.skip_validator,
            )
            success_count += 1
            tqdm.write(f"  ✅ [Done] {rel_path} -> {res['questions']} questions, {res['duration']}s")
        except Exception as e:
            failed_count += 1
            print(f"\n[Error] Failed processing '{rel_path}': {e}")

    print("\n==================================================================")
    print(f"Re-annotation of Table Samples Completed!")
    print(f"  Successfully Processed: {success_count}/{len(samples)}")
    print(f"  Failed                : {failed_count}")
    print(f"  Output saved to       : {output_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
