#!/usr/bin/env python3
"""
tools/evaluate_trained_model_500.py — Evaluation of Trained Hugging Face Model (daominhwysi/mmbert-small-vi-exam-seq-labeling)
on the Newly Created 500 Exam Documents.

Evaluates:
1. Zero Text Corruption Invariant: strip_tags(pred_xml) == raw_text
2. Cloze Stemless Question Invariant: <question_label> directly followed by <option_label> is valid
3. Span-Level Classification Metrics: Precision, Recall, F1 across all educational exam entities
4. Latency and Throughput on real exam text
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping, build_tagged_xml
from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations


def clean_span(span: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Trims boundary whitespace to ensure fair character alignment."""
    start = span["start"]
    end = span["end"]
    span_txt = text[start:end]
    lead = len(span_txt) - len(span_txt.lstrip())
    trail = len(span_txt) - len(span_txt.rstrip())
    return {
        "start": start + lead,
        "end": end - trail,
        "label": span["label"],
        "text": text[start + lead : end - trail]
    }


def evaluate_span_matching(
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    label: str,
    tolerance: int = 6
) -> Tuple[int, int, int, int, int]:
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
    tag_pattern = re.compile(
        r"</?(?:question|question_label|stem|option_label|option_text|stimulus|section|explanation)(?:\s+[^>]*)?>"
    )
    stripped = tag_pattern.sub("", xml_text)
    return stripped == raw_text


def check_cloze_stemless_validity(xml_text: str) -> Tuple[bool, int]:
    has_empty_stem = bool(re.search(r"<stem>\s*</stem>", xml_text))
    stemless_matches = len(re.findall(r"</question_label>\s*<option_label>", xml_text))
    return (not has_empty_stem), stemless_matches


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on 500-Document Batch")
    parser.add_argument("--model-id", type=str, default="daominhwysi/mmbert-small-vi-exam-seq-labeling")
    parser.add_argument("--state-file", type=str, default="data/annotation_batch_500_state.json")
    parser.add_argument("--num-docs", type=int, default=15, help="Number of diverse documents to evaluate")
    parser.add_argument("--slice-chars", type=int, default=4000, help="Character slice per document for CPU evaluation")
    parser.add_argument("--output-json", type=str, default="artifacts/evaluation_trained_hf_model_results.json")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    all_docs = list(state.get("completed_docs", {}).items())
    # Stratified selection across the 500 documents (spaced evenly)
    step = max(1, len(all_docs) // args.num_docs)
    selected_docs = [all_docs[i] for i in range(0, min(len(all_docs), step * args.num_docs), step)][:args.num_docs]

    print("=" * 80)
    print("EVALUATION OF TRAINED MODEL: " + args.model_id)
    print("=" * 80)
    print(f"Target Documents Selected: {len(selected_docs)} (stratified across {len(all_docs)} total)")
    print(f"Slice Length per Document : {args.slice_chars} chars (CPU inference)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device         : {device}")

    print(f"\nLoading trained model & tokenizer from Hugging Face ({args.model_id})...")
    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tag_to_id, id_to_tag = load_label_mapping(args.model_id)
    model = EnhancedBertForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    ).to(device)
    model.eval()
    print(f"Model loaded successfully in {time.perf_counter() - t0_load:.2f}s.\n")

    labels_to_eval = [
        ("option_label", 4),
        ("option_text", 6),
        ("question_label", 6),
        ("stem", 8),
        ("section", 12),
        ("explanation", 12),
    ]

    aggregate_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "gold_cnt": 0, "pred_cnt": 0} for lbl, _ in labels_to_eval}
    doc_results = []
    zero_corruption_passes = 0
    cloze_valid_passes = 0
    latencies = []

    for idx, (doc_name, meta) in enumerate(selected_docs):
        xml_path = Path(meta["output_dir"]) / "merged.xml"
        if not xml_path.exists():
            continue

        xml_content = xml_path.read_text(encoding="utf-8")
        raw_text, gold_spans = parse_xml_annotations(xml_content)

        test_text = raw_text[:args.slice_chars]
        gold_slice = [clean_span(s, test_text) for s in gold_spans if s["end"] <= len(test_text)]

        t_start = time.perf_counter()
        pred_res = predict_text(
            text=test_text,
            model=model,
            tokenizer=tokenizer,
            id_to_tag=id_to_tag,
            device=device,
            max_length=1024,
            stride=256,
            mask_latex=True,
            use_viterbi=True
        )
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat_ms)

        pred_spans = [clean_span(s, test_text) for s in pred_res.get("spans", [])]
        pred_xml = pred_res.get("xml_text", test_text)

        zc = check_zero_corruption(test_text, pred_xml)
        cloze_ok, _ = check_cloze_stemless_validity(pred_xml)
        if zc:
            zero_corruption_passes += 1
        if cloze_ok:
            cloze_valid_passes += 1

        doc_metrics = {}
        for lbl, tol in labels_to_eval:
            tp, fp, fn, g_cnt, p_cnt = evaluate_span_matching(gold_slice, pred_spans, lbl, tolerance=tol)
            aggregate_counts[lbl]["tp"] += tp
            aggregate_counts[lbl]["fp"] += fp
            aggregate_counts[lbl]["fn"] += fn
            aggregate_counts[lbl]["gold_cnt"] += g_cnt
            aggregate_counts[lbl]["pred_cnt"] += p_cnt
            p, r, f1 = calc_f1(tp, fp, fn)
            doc_metrics[lbl] = {"prec": round(p * 100, 2), "rec": round(r * 100, 2), "f1": round(f1 * 100, 2), "gold": g_cnt, "pred": p_cnt}

        opt_f1 = doc_metrics["option_label"]["f1"]
        q_f1 = doc_metrics["question_label"]["f1"]
        stem_f1 = doc_metrics["stem"]["f1"]

        print(f"[{idx+1}/{len(selected_docs)}] {doc_name[:38]:<38} | {lat_ms:6.1f}ms | Opt F1: {opt_f1:5.1f}% | Q F1: {q_f1:5.1f}% | Stem F1: {stem_f1:5.1f}% | ZC: {zc}")

        doc_results.append({
            "doc_name": doc_name,
            "chars_evaluated": len(test_text),
            "latency_ms": round(lat_ms, 2),
            "zero_corruption": zc,
            "cloze_stemless_valid": cloze_ok,
            "metrics": doc_metrics
        })

    # Summary calculations
    total_audited = len(doc_results)
    zc_rate = (zero_corruption_passes / max(1, total_audited)) * 100.0
    cloze_rate = (cloze_valid_passes / max(1, total_audited)) * 100.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    print("\n" + "=" * 80)
    print("SPAN-LEVEL CLASSIFICATION PERFORMANCE: " + args.model_id)
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Gold Support':<12} | {'Pred Support':<12}")
    print("-" * 80)

    f1_list = []
    summary_metrics = {}
    for lbl, _ in labels_to_eval:
        d = aggregate_counts[lbl]
        p, r, f1 = calc_f1(d["tp"], d["fp"], d["fn"])
        f1_list.append(f1)
        summary_metrics[lbl] = {
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1": round(f1 * 100, 2),
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "gold_support": d["gold_cnt"],
            "pred_support": d["pred_cnt"]
        }
        print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | {d['gold_cnt']:<12} | {d['pred_cnt']:<12}")

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    print("-" * 80)
    print(f"{'Macro Average F1':<20} | {'-':<10} | {'-':<10} | {macro_f1*100:8.2f}% | {'-':<12} | {'-':<12}")

    print("\n" + "=" * 80)
    print("INVARIANT AUDIT & OPERATIONAL METRICS")
    print("=" * 80)
    print(f"Zero-Corruption Invariant  : {zc_rate:6.2f}% ({zero_corruption_passes}/{total_audited} docs passed)")
    print(f"Cloze Stemless Invariant   : {cloze_rate:6.2f}% ({cloze_valid_passes}/{total_audited} docs passed)")
    print(f"Average CPU Inference Time : {avg_latency:6.1f} ms / document slice ({args.slice_chars} chars)")
    print("=" * 80)

    final_output = {
        "model_id": args.model_id,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents_audited": total_audited,
        "slice_chars_per_doc": args.slice_chars,
        "macro_f1": round(macro_f1 * 100, 2),
        "span_metrics": summary_metrics,
        "zero_corruption_rate": round(zc_rate, 2),
        "cloze_stemless_rate": round(cloze_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "per_document_results": doc_results
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed evaluation results saved to: {out_file}")


if __name__ == "__main__":
    main()
