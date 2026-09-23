#!/usr/bin/env python3
"""
Batch Folder Inference CLI for Sequence Labelling.
Recursively finds exam markdown/text files in an input directory, runs sliding-window
inference, and writes predicted XML annotations into an output directory.
"""

import os
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import argparse
import torch
from transformers import AutoTokenizer

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping


def main():
    parser = argparse.ArgumentParser(description="Batch sequence labelling inference on a folder.")
    parser.add_argument("--model-dir", type=str, default="results/mmbert_seqlabel_v2", help="Model directory")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing markdown/text exams")
    parser.add_argument("--output-dir", type=str, default="output/predictions", help="Output directory for predicted XMLs")
    parser.add_argument("--max-length", type=int, default=1024, help="Sliding window token length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_path.rglob("*") if f.is_file() and f.suffix in [".md", ".txt"]])
    if not files:
        print(f"No .md or .txt files found in '{args.input_dir}'.")
        return

    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} files to process in '{args.input_dir}'.")
    print(f"Loading model from '{args.model_dir}' onto {args.device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tag_to_id, id_to_tag = load_label_mapping(args.model_dir)

    model = EnhancedBertForTokenClassification.from_pretrained(
        args.model_dir,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    ).to(args.device)

    for file_idx, f in enumerate(files, start=1):
        print(f"[{file_idx}/{len(files)}] Annotating: {f.name}")
        try:
            rel_path = f.relative_to(input_path)
            out_file = output_path / rel_path.with_suffix(".xml")
            out_file.parent.mkdir(parents=True, exist_ok=True)

            text = f.read_text(encoding="utf-8")
            res = predict_text(
                text,
                model,
                tokenizer,
                id_to_tag,
                device=args.device,
                max_length=args.max_length,
                stride=args.stride
            )
            out_file.write_text(res["xml_text"], encoding="utf-8")
        except Exception as e:
            print(f"Error processing '{f.name}': {e}")

    print(f"\nBatch processing finished! Output XMLs saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
