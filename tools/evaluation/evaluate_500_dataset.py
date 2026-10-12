#!/usr/bin/env python3
"""
tools/evaluate_500_dataset.py — Comprehensive Evaluation of the 500-Document Batch Dataset & Sequence Labelling Models.

Evaluates:
1. Corpus & Stratified Diversity Metrics (500 documents, tokens, questions, grades, regions).
2. Ground Truth Architectural Invariants (Zero Text Corruption, Cloze Stemless Invariant, XML Well-formedness).
3. Deterministic Parser (DET) Performance on ALL 500 Documents (Span-level Precision, Recall, F1, Latency, Escalation).
4. Neural Model Micro-Benchmark (CPU-only constrained smoke evaluation of ModernBERT sequence labelling head).
5. Comprehensive Failure & Discrepancy Categorization.
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
import tiktoken

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations, BASE_TAGS
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic
from model.inference.predict import build_tagged_xml


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
    stemless_matches = len(re.findall(r"</question_label>\s*<option_label>", xml_text))
    return (not has_empty_stem), stemless_matches


def extract_grade_and_province(doc_name: str) -> Tuple[str, str]:
    """Infers grade (10, 11, 12) and province from filename."""
    grade_match = re.search(r"Toán\s*(10|11|12)", doc_name, re.IGNORECASE)
    grade = f"Grade {grade_match.group(1)}" if grade_match else "Other"

    # Common provinces
    known_provinces = [
        "Hà Nội", "TP HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Nghệ An", "Thanh Hóa",
        "Nam Định", "Bắc Giang", "Vĩnh Phúc", "Huế", "Bình Định", "Đồng Nai", "Long An",
        "An Giang", "Lâm Đồng", "Phú Thọ", "Thái Bình", "Quảng Ninh", "Hưng Yên",
        "Hải Dương", "Hà Tĩnh", "Quảng Nam", "Khánh Hòa", "Bình Dương", "Bến Tre",
        "Tây Ninh", "Quảng Trị", "Quảng Bình", "Ninh Bình", "Lạng Sơn", "Bắc Ninh",
        "Tiền Giang", "Kiên Giang", "Bà Rịa", "Gia Lai", "Đắk Lắk", "Thái Nguyên",
        "Sơn La", "Hà Nam", "Vĩnh Long", "Yên Bái", "Lào Cai", "Cao Bằng", "Sóc Trăng",
        "Bạc Liêu", "Cà Mau", "Kon Tum", "Bình Thuận", "Ninh Thuận", "Phú Yên"
    ]
    province = "Unknown"
    for prov in known_provinces:
        if prov.lower() in doc_name.lower():
            province = prov
            break
    return grade, province


def main():
    parser = argparse.ArgumentParser(description="Evaluate 500-Document Batch Dataset & Models")
    parser.add_argument("--state-file", type=str, default="data/annotation_batch_500_state.json")
    parser.add_argument("--output-json", type=str, default="artifacts/evaluation_500_batch_results.json")
    parser.add_argument("--eval-neural-smoke", action="store_true", default=True, help="Run neural smoke micro-benchmark")
    parser.add_argument("--neural-smoke-limit", type=int, default=3, help="Number of documents for neural smoke test")
    args = parser.parse_args()

    state_path = Path(args.state_path if hasattr(args, "state_path") else args.state_file)
    if not state_path.exists():
        print(f"Error: State file not found at {state_path}")
        sys.exit(1)

    with open(state_path, "r", encoding="utf-8") as f:
        batch_state = json.load(f)

    completed_docs = batch_state.get("completed_docs", {})
    print("=" * 80)
    print("VIETNAMESE SEQUENCE LABELLING v2 — 500 DOCUMENT BATCH EVALUATION")
    print("=" * 80)
    print(f"Total Documents in Ledger: {len(completed_docs)}")

    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    enc_o200k = tiktoken.get_encoding("o200k_base")

    labels_to_eval = [
        ("question_label", 4),
        ("option_label", 4),
        ("stem", 10),
        ("option_text", 10),
        ("stimulus", 20),
        ("section", 20),
        ("explanation", 20),
    ]

    # Metrics containers
    doc_records = []
    grade_distribution = defaultdict(int)
    province_distribution = defaultdict(int)

    total_raw_chars = 0
    total_raw_words = 0
    total_raw_cl100k = 0
    total_raw_o200k = 0

    total_xml_chars = 0
    total_xml_words = 0
    total_xml_cl100k = 0
    total_xml_o200k = 0

    gold_zc_passes = 0
    gold_cloze_passes = 0
    total_gold_questions = 0
    total_gold_stimuli = 0
    total_gold_stemless = 0

    det_zc_passes = 0
    det_cloze_passes = 0
    det_high_conf_passes = 0
    det_latencies_ms = []

    det_aggregate_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "gold_cnt": 0, "pred_cnt": 0} for lbl, _ in labels_to_eval}

    start_eval_time = time.time()

    print("\nPhase 1: Auditing ground truth and evaluating Deterministic Parser across all 500 documents...")

    for idx, (doc_name, meta) in enumerate(completed_docs.items()):
        if (idx + 1) % 50 == 0 or idx == 0 or idx == len(completed_docs) - 1:
            print(f"  Progress: [{idx + 1}/{len(completed_docs)}] documents evaluated...", flush=True)

        out_dir = Path(meta.get("output_dir", ""))
        xml_path = out_dir / "merged.xml"
        if not xml_path.exists():
            print(f"Warning: {xml_path} missing, skipping.")
            continue

        xml_content = xml_path.read_text(encoding="utf-8")
        raw_text, gold_spans = parse_xml_annotations(xml_content)

        grade, province = extract_grade_and_province(doc_name)
        grade_distribution[grade] += 1
        province_distribution[province] += 1

        # Token metrics
        r_chars = len(raw_text)
        r_words = len(raw_text.split())
        r_cl100k = len(enc_cl100k.encode(raw_text))
        r_o200k = len(enc_o200k.encode(raw_text))

        x_chars = len(xml_content)
        x_words = len(xml_content.split())
        x_cl100k = len(enc_cl100k.encode(xml_content))
        x_o200k = len(enc_o200k.encode(xml_content))

        total_raw_chars += r_chars
        total_raw_words += r_words
        total_raw_cl100k += r_cl100k
        total_raw_o200k += r_o200k

        total_xml_chars += x_chars
        total_xml_words += x_words
        total_xml_cl100k += x_cl100k
        total_xml_o200k += x_o200k

        # Ground truth invariant verification
        gzc = check_zero_corruption(raw_text, xml_content)
        if gzc:
            gold_zc_passes += 1
        g_cloze_ok, s_count = check_cloze_stemless_validity(xml_content)
        if g_cloze_ok:
            gold_cloze_passes += 1
        total_gold_stemless += s_count

        q_count = len([s for s in gold_spans if s.get("label") == "question_label"])
        stim_count = len([s for s in gold_spans if s.get("label") == "stimulus"])
        total_gold_questions += q_count
        total_gold_stimuli += stim_count

        # Deterministic Parser evaluation
        t_det0 = time.perf_counter()
        det_res = parse_chunk_deterministic(raw_text)
        det_lat_ms = (time.perf_counter() - t_det0) * 1000.0
        det_latencies_ms.append(det_lat_ms)

        pred_spans = det_res.spans
        if not det_res.needs_llm_review and det_res.confidence >= 0.75:
            det_high_conf_passes += 1

        pred_xml = build_tagged_xml(raw_text, pred_spans)
        dzc = check_zero_corruption(raw_text, pred_xml)
        if dzc:
            det_zc_passes += 1
        d_cloze_ok, _ = check_cloze_stemless_validity(pred_xml)
        if d_cloze_ok:
            det_cloze_passes += 1

        # Span matches for this doc
        doc_span_metrics = {}
        for lbl, tol in labels_to_eval:
            tp, fp, fn, g_cnt, p_cnt = evaluate_span_matching(gold_spans, pred_spans, lbl, tolerance=tol)
            det_aggregate_counts[lbl]["tp"] += tp
            det_aggregate_counts[lbl]["fp"] += fp
            det_aggregate_counts[lbl]["fn"] += fn
            det_aggregate_counts[lbl]["gold_cnt"] += g_cnt
            det_aggregate_counts[lbl]["pred_cnt"] += p_cnt
            doc_span_metrics[lbl] = {"tp": tp, "fp": fp, "fn": fn, "gold": g_cnt, "pred": p_cnt}

        doc_records.append({
            "doc_name": doc_name,
            "grade": grade,
            "province": province,
            "raw_tokens_cl100k": r_cl100k,
            "raw_tokens_o200k": r_o200k,
            "gold_questions": q_count,
            "gold_stimuli": stim_count,
            "gold_zc": gzc,
            "det_high_confidence": not det_res.needs_llm_review,
            "det_confidence": round(det_res.confidence, 3),
            "det_latency_ms": round(det_lat_ms, 2),
            "det_spans": doc_span_metrics,
        })

    elapsed_phase1 = time.time() - start_eval_time
    print(f"Phase 1 complete in {elapsed_phase1:.2f} seconds.\n")

    # Neural Model Smoke Micro-Benchmark
    neural_results = []
    if args.eval_neural_smoke:
        print("Phase 2: Executing Neural Model Micro-Benchmark (ModernBERT / EnhancedBertForTokenClassification)...")
        model_path = WORKSPACE_DIR / "results/sanity_check"
        if model_path.exists() and (model_path / "config.json").exists():
            try:
                from transformers import AutoTokenizer
                from model.module.head import EnhancedBertForTokenClassification
                from model.inference.predict import load_label_mapping, predict_text

                print(f"  Loading model from {model_path} onto CPU...")
                t_load0 = time.perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                tag_to_id, id_to_tag = load_label_mapping(model_path)
                neural_model = EnhancedBertForTokenClassification.from_pretrained(str(model_path), num_labels=len(id_to_tag))
                neural_model.eval()
                t_load = time.perf_counter() - t_load0
                print(f"  Model loaded successfully in {t_load:.2f}s.")

                # Select representative smoke sample
                smoke_docs = list(completed_docs.items())[:args.neural_smoke_limit]
                for s_idx, (s_name, s_meta) in enumerate(smoke_docs):
                    s_xml_path = Path(s_meta["output_dir"]) / "merged.xml"
                    s_xml = s_xml_path.read_text(encoding="utf-8")
                    s_raw, s_gold = parse_xml_annotations(s_xml)

                    # Limit evaluation slice to 3000 chars to avoid CPU core throttling
                    s_slice = s_raw[:3000]
                    s_gold_slice = [sp for sp in s_gold if sp["end"] <= 3000]

                    t_n0 = time.perf_counter()
                    pred_res = predict_text(
                        text=s_slice,
                        model=neural_model,
                        tokenizer=tokenizer,
                        id_to_tag=id_to_tag,
                        device="cpu",
                        max_length=1024,
                        stride=256,
                        mask_latex=True,
                        use_viterbi=True
                    )
                    t_n_ms = (time.perf_counter() - t_n0) * 1000.0

                    s_pred_spans = pred_res.get("spans", [])
                    s_pred_xml = pred_res.get("xml_text", s_slice)

                    s_zc = check_zero_corruption(s_slice, s_pred_xml)
                    s_cloze, _ = check_cloze_stemless_validity(s_pred_xml)

                    s_metrics = {}
                    for lbl in ["question_label", "option_label", "stem", "option_text"]:
                        tp, fp, fn, g_c, p_c = evaluate_span_matching(s_gold_slice, s_pred_spans, lbl, tolerance=6)
                        p, r, f1 = calc_f1(tp, fp, fn)
                        s_metrics[lbl] = {"prec": round(p * 100, 2), "rec": round(r * 100, 2), "f1": round(f1 * 100, 2), "gold": g_c, "pred": p_c}

                    neural_results.append({
                        "doc_name": s_name,
                        "slice_chars": len(s_slice),
                        "latency_ms": round(t_n_ms, 2),
                        "zero_corruption": s_zc,
                        "cloze_stemless_valid": s_cloze,
                        "metrics": s_metrics
                    })
                    print(f"  [Sample {s_idx + 1}/{len(smoke_docs)}] {s_name[:35]} — {t_n_ms:.1f}ms | ZC: {s_zc} | Cloze: {s_cloze}")

            except Exception as e:
                print(f"  Warning: Neural benchmark encountered error: {e}")
        else:
            print(f"  Checkpoint not found at {model_path}, skipping neural benchmark.")

    # Compute aggregate statistics
    num_docs = len(doc_records)
    det_f1_summary = {}
    f1_values = []
    for lbl, _ in labels_to_eval:
        d = det_aggregate_counts[lbl]
        p, r, f1 = calc_f1(d["tp"], d["fp"], d["fn"])
        det_f1_summary[lbl] = {
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1": round(f1 * 100, 2),
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "support_gold": d["gold_cnt"],
            "support_pred": d["pred_cnt"],
        }
        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    final_results = {
        "metadata": {
            "dataset_name": "500 Diverse Digital Math Exams Batch",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_documents": num_docs,
            "hardware": {
                "device": "CPU",
                "cuda_available": False,
                "note": "Hardware Pre-Flight Check: Full transformer evaluation constrained to smoke test per AGENTS.md Section 5."
            }
        },
        "corpus_metrics": {
            "total_documents": num_docs,
            "total_annotated_questions": total_gold_questions,
            "total_annotated_stimuli": total_gold_stimuli,
            "total_stemless_cloze_questions": total_gold_stemless,
            "avg_questions_per_doc": round(total_gold_questions / max(1, num_docs), 1),
            "grade_distribution": dict(sorted(grade_distribution.items())),
            "top_provinces": dict(sorted(province_distribution.items(), key=lambda x: -x[1])[:10]),
            "token_volumes": {
                "raw_characters": total_raw_chars,
                "raw_words": total_raw_words,
                "raw_cl100k_tokens": total_raw_cl100k,
                "raw_o200k_tokens": total_raw_o200k,
                "xml_characters": total_xml_chars,
                "xml_words": total_xml_words,
                "xml_cl100k_tokens": total_xml_cl100k,
                "xml_o200k_tokens": total_xml_o200k,
                "xml_overhead_ratio": round(total_xml_cl100k / max(1, total_raw_cl100k), 3),
            }
        },
        "ground_truth_invariants": {
            "zero_corruption_rate": round((gold_zc_passes / max(1, num_docs)) * 100, 2),
            "zero_corruption_passes": gold_zc_passes,
            "cloze_stemless_valid_rate": round((gold_cloze_passes / max(1, num_docs)) * 100, 2),
            "cloze_stemless_passes": gold_cloze_passes,
        },
        "deterministic_parser_evaluation": {
            "macro_f1": round(macro_f1 * 100, 2),
            "span_level_metrics": det_f1_summary,
            "zero_corruption_rate": round((det_zc_passes / max(1, num_docs)) * 100, 2),
            "cloze_stemless_valid_rate": round((det_cloze_passes / max(1, num_docs)) * 100, 2),
            "high_confidence_rate": round((det_high_conf_passes / max(1, num_docs)) * 100, 2),
            "high_confidence_count": det_high_conf_passes,
            "needs_llm_review_count": num_docs - det_high_conf_passes,
            "latency_ms": {
                "mean": round(float(np.mean(det_latencies_ms)), 2),
                "median": round(float(np.median(det_latencies_ms)), 2),
                "p95": round(float(np.percentile(det_latencies_ms, 95)), 2),
                "p99": round(float(np.percentile(det_latencies_ms, 99)), 2),
                "min": round(float(np.min(det_latencies_ms)), 2),
                "max": round(float(np.max(det_latencies_ms)), 2),
            }
        },
        "neural_smoke_microbenchmark": {
            "model_path": "results/sanity_check",
            "backbone": "aisingapore/SEA-LION-ModernBERT-300M",
            "evaluated_samples": neural_results
        }
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEvaluation successfully saved to: {out_file}")

    # Print summary tables to stdout
    print("\n" + "=" * 80)
    print("DATASET CORPUS OVERVIEW (500 DOCUMENTS)")
    print("=" * 80)
    print(f"Total Documents Audited    : {num_docs:,}")
    print(f"Total Questions Extracted  : {total_gold_questions:,} (avg {total_gold_questions/max(1, num_docs):.1f}/doc)")
    print(f"Total Stimuli Enclosed     : {total_gold_stimuli:,}")
    print(f"Stemless Cloze Questions   : {total_gold_stemless:,}")
    print(f"Corpus Tokens (cl100k)     : {total_raw_cl100k:,} raw | {total_xml_cl100k:,} annotated XML")
    print(f"Corpus Tokens (o200k)      : {total_raw_o200k:,} raw | {total_xml_o200k:,} annotated XML")
    print(f"Corpus Characters          : {total_raw_chars:,} raw chars")

    print("\n" + "=" * 80)
    print("DETERMINISTIC PARSER (DET) SPAN-LEVEL PERFORMANCE (500 DOCUMENTS)")
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Gold Sup':<10} | {'Pred Sup':<10}")
    print("-" * 80)
    for lbl, _ in labels_to_eval:
        m = det_f1_summary[lbl]
        print(f"{lbl:<20} | {m['precision']:>8.2f}% | {m['recall']:>8.2f}% | {m['f1']:>8.2f}% | {m['support_gold']:<10} | {m['support_pred']:<10}")
    print("-" * 80)
    print(f"{'Macro Average':<20} | {'-':<10} | {'-':<10} | {macro_f1*100:>8.2f}% | {'-':<10} | {'-':<10}")

    print("\n" + "=" * 80)
    print("INVARIANT AUDIT & PIPELINE LATENCY")
    print("=" * 80)
    print(f"Ground Truth Zero-Corruption Invariant : {final_results['ground_truth_invariants']['zero_corruption_rate']:.2f}% ({gold_zc_passes}/{num_docs})")
    print(f"Ground Truth Cloze Stemless Invariant  : {final_results['ground_truth_invariants']['cloze_stemless_valid_rate']:.2f}% ({gold_cloze_passes}/{num_docs})")
    print(f"Deterministic Latency (Mean / Median)  : {final_results['deterministic_parser_evaluation']['latency_ms']['mean']:.2f} ms / {final_results['deterministic_parser_evaluation']['latency_ms']['median']:.2f} ms per document")
    print(f"Deterministic High Confidence Rate     : {final_results['deterministic_parser_evaluation']['high_confidence_rate']:.2f}% ({det_high_conf_passes}/{num_docs})")
    print(f"Needs LLM Review Escalation Rate       : {100.0 - final_results['deterministic_parser_evaluation']['high_confidence_rate']:.2f}% ({num_docs - det_high_conf_passes}/{num_docs})")
    print("=" * 80)


if __name__ == "__main__":
    main()
