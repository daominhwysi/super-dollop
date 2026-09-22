#!/usr/bin/env python3
"""
Evaluation and Token Counting Script for Azozo Sequence Labelling Dataset.
Calculates token counts and evaluates Deterministic Parser (DET) F1 performance.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import tiktoken

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic

def evaluate_span_matching(gold_spans, pred_spans, label, tolerance=4):
    gold = [s for s in gold_spans if s.get("label") == label]
    pred = [s for s in pred_spans if s.get("label") == label]
    
    tp = 0
    fp = 0
    matched_gold_indices = set()
    
    for p in pred:
        p_start = p.get("start", 0)
        found_idx = None
        for idx, g in enumerate(gold):
            if idx not in matched_gold_indices and abs(g.get("start", 0) - p_start) <= tolerance:
                found_idx = idx
                break
        if found_idx is not None:
            tp += 1
            matched_gold_indices.add(found_idx)
        else:
            fp += 1
            
    fn = len(gold) - tp
    return tp, fp, fn, len(gold), len(pred)

def calc_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def get_stats(vals):
    if not vals:
        return {"total": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    arr = np.array(vals)
    return {
        "total": int(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "std": float(np.std(arr))
    }

def main():
    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    enc_o200k = tiktoken.get_encoding("o200k_base")
    
    dataset_dir = WORKSPACE_DIR / "data/sequence_labelling_annotated"
    xml_files = sorted(dataset_dir.rglob("merged.xml"))
    print(f"Discovered {len(xml_files)} merged.xml files in {dataset_dir}")
    
    docs_data = []
    
    # Target span labels to evaluate
    labels_to_eval = [
        ("question_label", 4),
        ("option_label", 4),
        ("stem", 10),
        ("option_text", 10),
        ("stimulus", 20),
        ("explanation", 20),
        ("figure", 10),
        ("section", 20),
    ]
    
    start_time = time.time()
    for idx, xml_file in enumerate(xml_files):
        doc_dir = xml_file.parent
        audit_file = doc_dir / "audit_report.json"
        
        audit_info = {}
        if audit_file.exists():
            try:
                audit_info = json.loads(audit_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading audit file {audit_file}: {e}")
                
        decision = audit_info.get("decision", "UNKNOWN")
        is_malfunctioned = audit_info.get("is_malfunctioned", False)
        grade = audit_info.get("grade", "UNKNOWN")
        overall_score = audit_info.get("overall_score", 0.0)
        
        # Category deduction
        rel_path = doc_dir.relative_to(dataset_dir)
        parts = rel_path.parts
        if len(parts) >= 2:
            category = f"{parts[0]}/{parts[1]}"
            root_category = parts[0]
        elif len(parts) == 1:
            category = "Root"
            root_category = "Root"
        else:
            category = "Root"
            root_category = "Root"
            
        xml_content = xml_file.read_text(encoding="utf-8")
        try:
            raw_text, gold_spans = parse_xml_annotations(xml_content)
        except Exception as e:
            print(f"Error parsing XML annotations for {xml_file}: {e}")
            continue
            
        # Token metrics
        raw_cl100k = len(enc_cl100k.encode(raw_text))
        raw_o200k = len(enc_o200k.encode(raw_text))
        raw_words = len(raw_text.split())
        raw_chars = len(raw_text)
        
        xml_cl100k = len(enc_cl100k.encode(xml_content))
        xml_o200k = len(enc_o200k.encode(xml_content))
        xml_words = len(xml_content.split())
        xml_chars = len(xml_content)
        
        # DET parser run
        det_start = time.perf_counter()
        det_res = parse_chunk_deterministic(raw_text)
        det_time_ms = (time.perf_counter() - det_start) * 1000
        
        pred_spans = det_res.spans
        high_conf = not det_res.needs_llm_review
        
        span_metrics = {}
        for lbl, tol in labels_to_eval:
            tp, fp, fn, g_cnt, p_cnt = evaluate_span_matching(gold_spans, pred_spans, lbl, tolerance=tol)
            span_metrics[lbl] = {
                "tp": tp, "fp": fp, "fn": fn, "gold_cnt": g_cnt, "pred_cnt": p_cnt
            }
            
        doc_entry = {
            "doc_id": doc_dir.name,
            "category": category,
            "root_category": root_category,
            "decision": decision,
            "grade": grade,
            "overall_score": overall_score,
            "is_malfunctioned": is_malfunctioned,
            "tokens": {
                "raw_cl100k": raw_cl100k,
                "raw_o200k": raw_o200k,
                "raw_words": raw_words,
                "raw_chars": raw_chars,
                "xml_cl100k": xml_cl100k,
                "xml_o200k": xml_o200k,
                "xml_words": xml_words,
                "xml_chars": xml_chars,
            },
            "det_parser": {
                "high_confidence": high_conf,
                "time_ms": det_time_ms,
                "questions_detected": len([s for s in pred_spans if s.get("label") == "question_label"]),
                "spans": span_metrics,
            }
        }
        docs_data.append(doc_entry)

    total_time = time.time() - start_time
    print(f"Processed {len(docs_data)} documents in {total_time:.2f}s")
    
    # Segment data into subsets
    passed_docs = [d for d in docs_data if d["decision"] == "PASS"]
    needs_revision_docs = [d for d in docs_data if d["decision"] == "NEEDS_REVISION"]
    discarded_docs = [d for d in docs_data if d["decision"] == "DISCARD"]
    non_discarded_docs = [d for d in docs_data if d["decision"] != "DISCARD"]
    all_docs = docs_data
    
    subsets = {
        "PASS": passed_docs,
        "NEEDS_REVISION": needs_revision_docs,
        "DISCARD": discarded_docs,
        "NON_DISCARDED (PASS + NEEDS_REVISION)": non_discarded_docs,
        "ALL": all_docs,
    }
    
    results = {}
    
    for subset_name, subset in subsets.items():
        if not subset:
            continue
            
        tok_stats = {
            "raw_cl100k": get_stats([d["tokens"]["raw_cl100k"] for d in subset]),
            "raw_o200k": get_stats([d["tokens"]["raw_o200k"] for d in subset]),
            "raw_words": get_stats([d["tokens"]["raw_words"] for d in subset]),
            "raw_chars": get_stats([d["tokens"]["raw_chars"] for d in subset]),
            "xml_cl100k": get_stats([d["tokens"]["xml_cl100k"] for d in subset]),
            "xml_o200k": get_stats([d["tokens"]["xml_o200k"] for d in subset]),
            "xml_words": get_stats([d["tokens"]["xml_words"] for d in subset]),
            "xml_chars": get_stats([d["tokens"]["xml_chars"] for d in subset]),
        }
        
        # High confidence pass count
        high_conf_cnt = sum(1 for d in subset if d["det_parser"]["high_confidence"])
        
        # Aggregate span metrics
        agg_spans = {}
        for lbl, _ in labels_to_eval:
            tp = sum(d["det_parser"]["spans"][lbl]["tp"] for d in subset)
            fp = sum(d["det_parser"]["spans"][lbl]["fp"] for d in subset)
            fn = sum(d["det_parser"]["spans"][lbl]["fn"] for d in subset)
            gold_cnt = sum(d["det_parser"]["spans"][lbl]["gold_cnt"] for d in subset)
            pred_cnt = sum(d["det_parser"]["spans"][lbl]["pred_cnt"] for d in subset)
            prec, rec, f1 = calc_f1(tp, fp, fn)
            agg_spans[lbl] = {
                "tp": tp, "fp": fp, "fn": fn,
                "gold_cnt": gold_cnt, "pred_cnt": pred_cnt,
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1": round(f1 * 100, 2),
            }
            
        # Category breakdown for this subset
        cat_map = defaultdict(list)
        for d in subset:
            cat_map[d["category"]].append(d)
            
        cat_breakdown = {}
        for cat, cat_docs in sorted(cat_map.items()):
            c_high = sum(1 for d in cat_docs if d["det_parser"]["high_confidence"])
            c_spans = {}
            for lbl, _ in labels_to_eval:
                tp = sum(d["det_parser"]["spans"][lbl]["tp"] for d in cat_docs)
                fp = sum(d["det_parser"]["spans"][lbl]["fp"] for d in cat_docs)
                fn = sum(d["det_parser"]["spans"][lbl]["fn"] for d in cat_docs)
                prec, rec, f1 = calc_f1(tp, fp, fn)
                c_spans[lbl] = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2),
                }
            cat_breakdown[cat] = {
                "doc_count": len(cat_docs),
                "raw_tokens_cl100k": sum(d["tokens"]["raw_cl100k"] for d in cat_docs),
                "xml_tokens_cl100k": sum(d["tokens"]["xml_cl100k"] for d in cat_docs),
                "high_conf_rate": round(c_high / len(cat_docs) * 100, 2),
                "spans": c_spans,
            }
            
        results[subset_name] = {
            "doc_count": len(subset),
            "token_statistics": tok_stats,
            "high_confidence_count": high_conf_cnt,
            "high_confidence_rate": round(high_conf_cnt / len(subset) * 100, 2),
            "span_f1_metrics": agg_spans,
            "category_breakdown": cat_breakdown,
        }

    # Save to JSON
    out_json = WORKSPACE_DIR / "artifacts" / "passed_documents_token_det_f1_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {out_json}")

if __name__ == "__main__":
    main()
