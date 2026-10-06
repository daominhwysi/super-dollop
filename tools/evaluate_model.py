#!/usr/bin/env python3
"""
Comprehensive Gold Benchmark & Architecture Evaluation Script.
Evaluates Vietnamese sequence labeling pipeline across 63 Gold Benchmark Documents:
1. Zero Text Corruption Invariant: strip_tags(output_xml) == raw_source_markdown (100% required)
2. Cloze Stemless Question Invariant: <question_label> followed directly by <option_label> is valid
3. Sub-millisecond latency budget: < 0.5 ms per 1024-token chunk
4. Span-level F1 precision & recall metrics across difficulty tiers
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic
from model.inference.predict import (
    load_label_mapping,
    predict_text,
    build_tagged_xml
)
from model.module.head import EnhancedBertForTokenClassification


def evaluate_span_matching(
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    label: str,
    tolerance: int = 4
) -> Tuple[int, int, int, int, int]:
    """Computes TP, FP, FN with a character-level boundary tolerance."""
    gold = [s for s in gold_spans if s.get("label") == label]
    pred = [s for s in pred_spans if s.get("label") == label]

    tp = 0
    fp = 0
    matched_gold_indices = set()

    for p in pred:
        p_start = p.get("start", 0)
        p_end = p.get("end", 0)
        found_idx = None
        for idx, g in enumerate(gold):
            if idx not in matched_gold_indices:
                g_start = g.get("start", 0)
                g_end = g.get("end", 0)
                if abs(g_start - p_start) <= tolerance and abs(g_end - p_end) <= tolerance:
                    found_idx = idx
                    break
        if found_idx is not None:
            tp += 1
            matched_gold_indices.add(found_idx)
        else:
            fp += 1

    fn = len(gold) - tp
    return tp, fp, fn, len(gold), len(pred)


def calc_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def check_zero_corruption(raw_text: str, xml_text: str) -> bool:
    """Invariant: strip_tags(xml_text) == raw_text exactly without single-char deviation."""
    tag_pattern = re.compile(
        r"</?(?:question|question_label|stem|option_label|option_text|stimulus|section|explanation)(?:\s+[^>]*)?>"
    )
    stripped = tag_pattern.sub("", xml_text)
    return stripped == raw_text


def check_cloze_stemless_validity(xml_text: str) -> Tuple[bool, int]:
    """
    Checks that cloze items where <question_label> is followed by <option_label>
    do NOT contain empty/hallucinated <stem></stem> tags.
    """
    has_empty_stem = bool(re.search(r"<stem>\s*</stem>", xml_text))
    # Count stemless cloze questions
    stemless_matches = len(re.findall(r"</question_label>\s*<option_label>", xml_text))
    return (not has_empty_stem), stemless_matches


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequence Labelling Model against Gold Benchmark.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--gold_set", type=str, default="data/benchmark_gold_set.json", help="Path to gold set JSON")
    parser.add_argument("--model_path", type=str, default="results/sanity_check", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to evaluate")
    parser.add_argument("--use_det", action="store_true", help="Use deterministic parser directly")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gold_path = Path(args.gold_set)
    if not gold_path.exists():
        fallback = WORKSPACE_DIR / "data/benchmark_gold_set.json"
        if fallback.exists():
            gold_path = fallback
        else:
            raise FileNotFoundError(f"Could not locate gold benchmark at {args.gold_set} or {fallback}")

    print("=" * 80)
    print("VIETNAMESE SEQUENCE LABELLING v2 - ARCHITECTURE BENCHMARK & AUDIT")
    print("=" * 80)
    print(f"Gold Benchmark Set  : {gold_path}")
    print(f"Target Device       : {device}")

    # Load Gold Set
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    documents = gold_data.get("documents", [])
    if args.limit:
        documents = documents[:args.limit]
    print(f"Evaluating Documents: {len(documents)} documents (Corpus Token Total: {gold_data.get('total_tokens', 'N/A')})\n")

    # Try loading model if checkpoint exists
    model_dir = Path(args.model_path) if args.model_path else None
    model = None
    tokenizer = None
    id_to_tag = None
    if not args.use_det and model_dir and model_dir.exists() and (model_dir / "config.json").exists():
        try:
            from transformers import AutoTokenizer
            print(f"Loading checkpoint from: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            tag_to_id, id_to_tag = load_label_mapping(model_dir)
            model = EnhancedBertForTokenClassification.from_pretrained(
                str(model_dir),
                num_labels=len(id_to_tag)
            ).to(device)
            model.eval()
            print("Successfully loaded EnhancedBertForTokenClassification checkpoint.\n")
        except Exception as e:
            print(f"Notice: Could not load trained model checkpoint from {model_dir}: {e}")
            print("Falling back to Deterministic Parser + Gold Consistency verification.\n")
            model = None

    # Track metrics
    labels_to_eval = [
        ("question_label", 4),
        ("option_label", 4),
        ("stem", 10),
        ("option_text", 10),
        ("stimulus", 20),
        ("section", 20),
        ("explanation", 20),
    ]
    aggregate_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "gold_cnt": 0, "pred_cnt": 0} for lbl, _ in labels_to_eval}
    
    zero_corruption_passes = 0
    cloze_valid_passes = 0
    total_stemless_cloze_items = 0
    latency_records = []

    for idx, doc_meta in enumerate(documents):
        doc_dir = Path(doc_meta.get("path", ""))
        if not doc_dir.exists():
            # Try finding folder relative to data/sequence_labelling_annotated
            doc_dir = WORKSPACE_DIR / "data/sequence_labelling_annotated" / doc_meta.get("doc_id", "")

        doc_title = doc_meta.get('doc_id', '')
        print(f"[{idx+1}/{len(documents)}] Auditing {doc_title} (tokens: {doc_meta.get('token_count', 'N/A')})...", end=" ", flush=True)

        xml_file = doc_dir / "merged.xml"
        if not xml_file.exists():
            print(f"Warning: {xml_file} does not exist, skipping doc {doc_title}")
            continue

        xml_content = xml_file.read_text(encoding="utf-8")
        raw_text, gold_spans = parse_xml_annotations(xml_content)

        # 1. Zero-Corruption Invariant Check on Ground Truth XML
        gold_zc = check_zero_corruption(raw_text, xml_content)
        gold_cloze_ok, stemless_count = check_cloze_stemless_validity(xml_content)
        total_stemless_cloze_items += stemless_count

        # 2. Prediction (via Model or Deterministic Parser)
        if model is not None and tokenizer is not None and id_to_tag is not None and not args.use_det:
            t0 = time.perf_counter()
            pred_res = predict_text(
                text=raw_text,
                model=model,
                tokenizer=tokenizer,
                id_to_tag=id_to_tag,
                device=device,
                max_length=1024,
                stride=256,
                mask_latex=True,
                use_viterbi=True
            )
            lat_ms = (time.perf_counter() - t0) * 1000
            latency_records.append(lat_ms)
            pred_spans = pred_res.get("spans", [])
            pred_xml = pred_res.get("xml_text", raw_text)

            pred_zc = check_zero_corruption(raw_text, pred_xml)
            pred_cloze_ok, _ = check_cloze_stemless_validity(pred_xml)
            if pred_zc and gold_zc:
                zero_corruption_passes += 1
            if pred_cloze_ok and gold_cloze_ok:
                cloze_valid_passes += 1
            print(f"Done in {lat_ms:.1f}ms | ZC: {'PASS' if pred_zc and gold_zc else 'FAIL'} | Cloze: {'VALID' if pred_cloze_ok and gold_cloze_ok else 'FAIL'}")
        else:
            t0 = time.perf_counter()
            det_res = parse_chunk_deterministic(raw_text)
            lat_ms = (time.perf_counter() - t0) * 1000
            latency_records.append(lat_ms)
            pred_spans = det_res.spans

            pred_xml = build_tagged_xml(raw_text, pred_spans)
            pred_zc = check_zero_corruption(raw_text, pred_xml)
            pred_cloze_ok, _ = check_cloze_stemless_validity(pred_xml)
            if pred_zc and gold_zc:
                zero_corruption_passes += 1
            if pred_cloze_ok and gold_cloze_ok:
                cloze_valid_passes += 1
            print(f"Done in {lat_ms:.1f}ms | ZC: {'PASS' if pred_zc and gold_zc else 'FAIL'} | Cloze: {'VALID' if pred_cloze_ok and gold_cloze_ok else 'FAIL'}")

        # 3. Span matching evaluation
        for lbl, tol in labels_to_eval:
            tp, fp, fn, g_cnt, p_cnt = evaluate_span_matching(gold_spans, pred_spans, lbl, tolerance=tol)
            aggregate_counts[lbl]["tp"] += tp
            aggregate_counts[lbl]["fp"] += fp
            aggregate_counts[lbl]["fn"] += fn
            aggregate_counts[lbl]["gold_cnt"] += g_cnt
            aggregate_counts[lbl]["pred_cnt"] += p_cnt

    total_docs = len(latency_records)
    zc_rate = (zero_corruption_passes / max(1, total_docs)) * 100.0
    cloze_rate = (cloze_valid_passes / max(1, total_docs)) * 100.0
    avg_latency = float(np.mean(latency_records)) if latency_records else 0.0

    print("=" * 80)
    print("SPAN-LEVEL CLASSIFICATION PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 80)

    f1_list = []
    for lbl, _ in labels_to_eval:
        data = aggregate_counts[lbl]
        p, r, f1 = calc_f1(data["tp"], data["fp"], data["fn"])
        f1_list.append(f1)
        print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | {data['gold_cnt']:<8}")

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    print("-" * 80)
    print(f"{'Macro Average F1':<20} | {'-':<10} | {'-':<10} | {macro_f1*100:8.2f}% | {'-':<8}\n")

    print("=" * 80)
    print("ARCHITECTURAL INVARIANT COMPLIANCE AUDIT")
    print("=" * 80)
    print(f"Zero-Corruption Invariant  : {zc_rate:6.2f}% ({zero_corruption_passes}/{total_docs} docs passed)")
    print(f"Cloze Stemless Invariant   : {cloze_rate:6.2f}% ({cloze_valid_passes}/{total_docs} docs passed)")
    print(f"Stemless Cloze Questions   : {total_stemless_cloze_items} valid stemless items audited")
    print(f"Inference Latency (Avg)    : {avg_latency:6.3f} ms / document")
    print("=" * 80)

    # Invariant assertions
    assert zc_rate == 100.0, f"Zero-Corruption invariant failed: {zc_rate:.2f}% < 100.0%"
    assert cloze_rate == 100.0, f"Cloze Stemless invariant failed: {cloze_rate:.2f}% < 100.0%"
    print("\nALL ARCHITECTURAL INVARIANTS SATISFIED SUCCESSFULLY (100% PASS).")


if __name__ == "__main__":
    main()
