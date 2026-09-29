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
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch
from transformers import AutoTokenizer

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping


def test_samples():
    sample_dir = WORKSPACE_DIR / "artifacts" / "samples"
    out_dir = WORKSPACE_DIR / "artifacts" / "samples_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = "daominhwysi/mmbert-small-vi-exam-seq-labeling"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Running Local Set Inference Test ===")
    print(f"Model Source : {model_dir}")
    print(f"Inference Dev: {device}")
    print(f"Samples Dir  : {sample_dir}")
    print(f"Output Dir   : {out_dir}\n")

    print("Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tag_to_id, id_to_tag = load_label_mapping(model_dir)
    model = EnhancedBertForTokenClassification.from_pretrained(
        model_dir,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    ).to(device)
    print(f"Model ready in {time.time() - t0:.2f}s.\n")

    files = sorted([f for f in sample_dir.glob("*.txt")])
    if not files:
        print(f"No .txt files found in {sample_dir}")
        return

    results_summary = []

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processing: {f.name} ({f.stat().st_size / 1024:.1f} KB)...")
        raw_text = f.read_text(encoding="utf-8")

        start_time = time.time()
        res = predict_text(
            raw_text,
            model=model,
            tokenizer=tokenizer,
            id_to_tag=id_to_tag,
            device=device,
            max_length=2048,
            stride=256
        )
        elapsed = time.time() - start_time

        xml_output = res["xml_text"]
        out_file = out_dir / f"{f.stem}.predicted.xml"
        out_file.write_text(xml_output, encoding="utf-8")

        # Entity count
        label_counts = {}
        for sp in res["spans"]:
            lbl = sp["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Check character fidelity: stripping xml tags should equal clean raw text
        cleaned_xml = re.sub(r"</?[a-zA-Z_]+>", "", xml_output)
        clean_raw = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        zero_mutation = (cleaned_xml == clean_raw)

        results_summary.append({
            "file": f.name,
            "size_kb": f.stat().st_size / 1024,
            "tokens": res["num_tokens"],
            "elapsed_sec": elapsed,
            "zero_mutation": zero_mutation,
            "counts": label_counts,
            "xml_sample": xml_output[:300]
        })

        print(f"    Done in {elapsed:.2f}s ({res['num_tokens']} tokens).")
        print(f"    Entities: {label_counts}")
        print(f"    Zero Character Mutation: {'PASSED (100% exact)' if zero_mutation else 'FAILED'}")
        print()

    print("\n" + "="*80)
    print(f"{'File Name':<32} | {'Tokens':<6} | {'Time(s)':<7} | {'Q_Lab':<5} | {'Stem':<5} | {'Opt':<5} | {'Stim':<5}")
    print("="*80)
    for r in results_summary:
        cnt = r["counts"]
        q_cnt = cnt.get("question_label", 0)
        stem_cnt = cnt.get("stem", 0)
        opt_cnt = cnt.get("option_label", 0)
        stim_cnt = cnt.get("stimulus", 0)
        print(f"{r['file']:<32} | {r['tokens']:<6} | {r['elapsed_sec']:<7.1f} | {q_cnt:<5} | {stem_cnt:<5} | {opt_cnt:<5} | {stim_cnt:<5}")
    print("="*80)
    print(f"\nAll predicted XML files saved to: {out_dir}")


if __name__ == "__main__":
    test_samples()
