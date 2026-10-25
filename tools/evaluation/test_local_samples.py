#!/usr/bin/env python3
"""
Test sequence labelling model on local sample exams in artifacts/samples.
Generates XML predictions, checks entity statistics, and validates zero character mutation.
"""

import os
import sys
import time
import json
import re
import argparse
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch
from transformers import AutoTokenizer

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping


def strip_xml_tags(xml_str: str) -> str:
    """Removes all XML markup tags to verify 100% character preservation."""
    return re.sub(r"</?[a-zA-Z_0-9]+(?:\s+[^>]*)?>", "", xml_str)


def run_sample_eval(
    model_name_or_path: str = "daominhwysi/sealior-modernbert-vi-exam-seq-labeling",
    sample_dir: Path = WORKSPACE_DIR / "artifacts" / "samples",
    out_dir: Path = WORKSPACE_DIR / "artifacts" / "samples_predictions",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_viterbi: bool = True
):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(list(sample_dir.glob("*.txt")))
    if not files:
        print(f"No sample .txt files found in {sample_dir}")
        return

    print("=" * 80)
    print("  VIETNAMESE SEQUENCE LABELLING - SAMPLE INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"  Model Source : {model_name_or_path}")
    print(f"  Inference Dev: {device}")
    print(f"  Viterbi Opt  : {'ENABLED (Constrained)' if use_viterbi else 'DISABLED (Argmax)'}")
    print(f"  Samples Dir  : {sample_dir} ({len(files)} files)")
    print(f"  Output Dir   : {out_dir}")
    print("=" * 80 + "\n")

    print(f"Loading tokenizer & label mapping from '{model_name_or_path}'...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tag_to_id, id_to_tag = load_label_mapping(model_name_or_path)

    print(f"Loading Enhanced ModernBERT token classification model on {device}...")
    model = EnhancedBertForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print(f"Model and weights initialized in {time.time() - t0:.2f}s.\n")

    summary_records = []

    for idx, f in enumerate(files, 1):
        raw_text = f.read_text(encoding="utf-8")
        raw_normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        print(f"[{idx}/{len(files)}] Running: {f.name} ({len(raw_normalized)} chars)...")

        t_start = time.time()
        res = predict_text(
            raw_normalized,
            model=model,
            tokenizer=tokenizer,
            id_to_tag=id_to_tag,
            device=device,
            max_length=2048,
            stride=256,
            batch_size=4,
            use_viterbi=use_viterbi
        )
        t_elapsed = time.time() - t_start

        xml_output = res["xml_text"]
        out_file = out_dir / f"{f.stem}.predicted.xml"
        out_file.write_text(xml_output, encoding="utf-8")

        # Entity count
        label_counts = {}
        for sp in res["spans"]:
            lbl = sp["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Character preservation check
        stripped = strip_xml_tags(xml_output)
        zero_mutation = (stripped == raw_normalized)

        summary_records.append({
            "name": f.stem,
            "tokens": res.get("num_tokens", len(tokenizer.tokenize(raw_normalized))),
            "time_sec": t_elapsed,
            "zero_mutation": zero_mutation,
            "counts": label_counts,
            "xml_sample": xml_output
        })

        status_str = "PASSED (100% exact)" if zero_mutation else f"FAILED ({abs(len(stripped) - len(raw_normalized))} diff)"
        print(f"    Latency   : {t_elapsed:.2f}s")
        print(f"    Entities  : {label_counts}")
        print(f"    Fidelity  : {status_str}")
        print()

    print("\n" + "=" * 90)
    print(f"{'Sample Name':<32} | {'Tokens':<6} | {'Time(s)':<7} | {'Q_Lab':<5} | {'Stem':<5} | {'Opt':<5} | {'Stim':<5} | {'Fidelity':<6}")
    print("=" * 90)
    for r in summary_records:
        c = r["counts"]
        q_cnt = c.get("question_label", 0)
        stem_cnt = c.get("stem", 0)
        opt_cnt = c.get("option_label", 0)
        stim_cnt = c.get("stimulus", 0)
        fid_str = "OK" if r["zero_mutation"] else "ERR"
        print(f"{r['name']:<32} | {r['tokens']:<6} | {r['time_sec']:<7.2f} | {q_cnt:<5} | {stem_cnt:<5} | {opt_cnt:<5} | {stim_cnt:<5} | {fid_str:<6}")
    print("=" * 90)
    print(f"\nAll predicted XML files successfully saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test sequence labelling model on local samples in artifacts/samples.")
    parser.add_argument("--model", type=str, default="daominhwysi/sealior-modernbert-vi-exam-seq-labeling", help="Hugging Face repo or local path")
    parser.add_argument("--samples-dir", type=Path, default=WORKSPACE_DIR / "artifacts" / "samples")
    parser.add_argument("--out-dir", type=Path, default=WORKSPACE_DIR / "artifacts" / "samples_predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-viterbi", action="store_true", default=False)
    args = parser.parse_args()

    run_sample_eval(
        model_name_or_path=args.model,
        sample_dir=args.samples_dir,
        out_dir=args.out_dir,
        device=args.device,
        use_viterbi=not args.no_viterbi
    )


if __name__ == "__main__":
    main()
