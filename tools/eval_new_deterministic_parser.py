import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

WORKSPACE_DIR = Path("/home/daominhwysi/project/azozo-experiment")
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic as parse_new_det
from sequence_labelling.parser.long_parser.old.deterministic_parser import parse_chunk_deterministic as parse_old_det

def evaluate_span_matching(gold_spans: List[Dict[str, Any]], pred_spans: List[Dict[str, Any]], label: str, tolerance: int = 4):
    gold = [s for s in gold_spans if s["label"] == label]
    pred = [s for s in pred_spans if s["label"] == label]
    
    tp = 0
    fp = 0
    matched_gold_indices = set()
    
    for p in pred:
        p_start = p["start"]
        found_idx = None
        for idx, g in enumerate(gold):
            if idx not in matched_gold_indices and abs(g["start"] - p_start) <= tolerance:
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

def main():
    dataset_dir = WORKSPACE_DIR / "data/sequence_labelling_annotated"
    xml_files = sorted(dataset_dir.rglob("merged.xml"))
    print(f"Benchmarking New vs Old Deterministic Parser on {len(xml_files)} documents...")
    
    old_totals = defaultdict(int)
    new_totals = defaultdict(int)
    
    cat_summary = defaultdict(lambda: {
        "docs": 0,
        "old_ql_tp": 0, "old_ql_fp": 0, "old_ql_fn": 0,
        "new_ql_tp": 0, "new_ql_fp": 0, "new_ql_fn": 0,
        "old_ol_tp": 0, "old_ol_fp": 0, "old_ol_fn": 0,
        "new_ol_tp": 0, "new_ol_fp": 0, "new_ol_fn": 0,
        "old_stem_tp": 0, "old_stem_fp": 0, "old_stem_fn": 0,
        "new_stem_tp": 0, "new_stem_fp": 0, "new_stem_fn": 0,
        "old_pass": 0, "new_pass": 0,
    })
    
    start_time = time.time()
    
    for i, xml_file in enumerate(xml_files):
        rel_path = xml_file.parent.relative_to(dataset_dir)
        top_cat = rel_path.parts[0] if len(rel_path.parts) > 1 else "Root"
        
        try:
            raw_xml = xml_file.read_text(encoding="utf-8")
            raw_text, gold_spans = parse_xml_annotations(raw_xml)
            if not raw_text.strip():
                continue
                
            cat_summary[top_cat]["docs"] += 1
            old_totals["docs"] += 1
            new_totals["docs"] += 1
            
            # 1. Evaluate Old Parser
            old_res = parse_old_det(raw_text)
            if not old_res.needs_llm_review:
                old_totals["high_confidence"] += 1
                cat_summary[top_cat]["old_pass"] += 1
                
            o_ql_tp, o_ql_fp, o_ql_fn, _, _ = evaluate_span_matching(gold_spans, old_res.spans, "question_label", tolerance=4)
            o_ol_tp, o_ol_fp, o_ol_fn, _, _ = evaluate_span_matching(gold_spans, old_res.spans, "option_label", tolerance=4)
            o_s_tp, o_s_fp, o_s_fn, _, _ = evaluate_span_matching(gold_spans, old_res.spans, "stem", tolerance=10)
            
            old_totals["ql_tp"] += o_ql_tp; old_totals["ql_fp"] += o_ql_fp; old_totals["ql_fn"] += o_ql_fn
            old_totals["ol_tp"] += o_ol_tp; old_totals["ol_fp"] += o_ol_fp; old_totals["ol_fn"] += o_ol_fn
            old_totals["stem_tp"] += o_s_tp; old_totals["stem_fp"] += o_s_fp; old_totals["stem_fn"] += o_s_fn
            
            cat_summary[top_cat]["old_ql_tp"] += o_ql_tp; cat_summary[top_cat]["old_ql_fp"] += o_ql_fp; cat_summary[top_cat]["old_ql_fn"] += o_ql_fn
            cat_summary[top_cat]["old_ol_tp"] += o_ol_tp; cat_summary[top_cat]["old_ol_fp"] += o_ol_fp; cat_summary[top_cat]["old_ol_fn"] += o_ol_fn
            cat_summary[top_cat]["old_stem_tp"] += o_s_tp; cat_summary[top_cat]["old_stem_fp"] += o_s_fp; cat_summary[top_cat]["old_stem_fn"] += o_s_fn
            
            # 2. Evaluate New Parser
            new_res = parse_new_det(raw_text)
            if not new_res.needs_llm_review:
                new_totals["high_confidence"] += 1
                cat_summary[top_cat]["new_pass"] += 1
                
            n_ql_tp, n_ql_fp, n_ql_fn, ql_g, ql_p = evaluate_span_matching(gold_spans, new_res.spans, "question_label", tolerance=4)
            n_ol_tp, n_ol_fp, n_ol_fn, ol_g, ol_p = evaluate_span_matching(gold_spans, new_res.spans, "option_label", tolerance=4)
            n_s_tp, n_s_fp, n_s_fn, s_g, s_p = evaluate_span_matching(gold_spans, new_res.spans, "stem", tolerance=10)
            n_ot_tp, n_ot_fp, n_ot_fn, ot_g, ot_p = evaluate_span_matching(gold_spans, new_res.spans, "option_text", tolerance=10)
            
            new_totals["ql_tp"] += n_ql_tp; new_totals["ql_fp"] += n_ql_fp; new_totals["ql_fn"] += n_ql_fn
            new_totals["ol_tp"] += n_ol_tp; new_totals["ol_fp"] += n_ol_fp; new_totals["ol_fn"] += n_ol_fn
            new_totals["stem_tp"] += n_s_tp; new_totals["stem_fp"] += n_s_fp; new_totals["stem_fn"] += n_s_fn
            new_totals["ot_tp"] += n_ot_tp; new_totals["ot_fp"] += n_ot_fp; new_totals["ot_fn"] += n_ot_fn
            
            cat_summary[top_cat]["new_ql_tp"] += n_ql_tp; cat_summary[top_cat]["new_ql_fp"] += n_ql_fp; cat_summary[top_cat]["new_ql_fn"] += n_ql_fn
            cat_summary[top_cat]["new_ol_tp"] += n_ol_tp; cat_summary[top_cat]["new_ol_fp"] += n_ol_fp; cat_summary[top_cat]["new_ol_fn"] += n_ol_fn
            cat_summary[top_cat]["new_stem_tp"] += n_s_tp; cat_summary[top_cat]["new_stem_fp"] += n_s_fp; cat_summary[top_cat]["new_stem_fn"] += n_s_fn

        except Exception as e:
            print(f"Error on {xml_file}: {e}")

    total_time = time.time() - start_time
    total_docs = old_totals["docs"]
    
    print("\n" + "="*80)
    print("COMPARATIVE BENCHMARK: NEW DETERMINISTIC PARSER (v2.5) vs OLD BASELINE")
    print("="*80)
    print(f"Total Documents Evaluated: {total_docs}")
    print(f"Benchmark Runtime: {total_time:.2f}s ({total_time/max(total_docs,1)*1000:.1f}ms/doc)\n")
    
    # Calculate Overall Metrics
    o_ql_p, o_ql_r, o_ql_f1 = calc_f1(old_totals["ql_tp"], old_totals["ql_fp"], old_totals["ql_fn"])
    o_ol_p, o_ol_r, o_ol_f1 = calc_f1(old_totals["ol_tp"], old_totals["ol_fp"], old_totals["ol_fn"])
    o_s_p, o_s_r, o_s_f1 = calc_f1(old_totals["stem_tp"], old_totals["stem_fp"], old_totals["stem_fn"])
    
    n_ql_p, n_ql_r, n_ql_f1 = calc_f1(new_totals["ql_tp"], new_totals["ql_fp"], new_totals["ql_fn"])
    n_ol_p, n_ol_r, n_ol_f1 = calc_f1(new_totals["ol_tp"], new_totals["ol_fp"], new_totals["ol_fn"])
    n_s_p, n_s_r, n_s_f1 = calc_f1(new_totals["stem_tp"], new_totals["stem_fp"], new_totals["stem_fn"])
    n_ot_p, n_ot_r, n_ot_f1 = calc_f1(new_totals["ot_tp"], new_totals["ot_fp"], new_totals["ot_fn"])

    print("--- 1. OVERALL ACCURACY & F1 COMPARISON ---")
    print(f"{'Metric / Tag':<22} | {'OLD Parser':<24} | {'NEW Parser (v2.5)':<24} | {'DELTA (F1)':<10}")
    print("-" * 88)
    print(f"High-Conf Pass Rate    | {old_totals['high_confidence']}/{total_docs} ({old_totals['high_confidence']/total_docs*100:4.1f}%)        | {new_totals['high_confidence']}/{total_docs} ({new_totals['high_confidence']/total_docs*100:4.1f}%)        | +{(new_totals['high_confidence']-old_totals['high_confidence'])/total_docs*100:+.1f}%")
    print(f"Question Label F1      | {o_ql_f1*100:5.2f}% (P={o_ql_p*100:4.1f}%, R={o_ql_r*100:4.1f}%) | {n_ql_f1*100:5.2f}% (P={n_ql_p*100:4.1f}%, R={n_ql_r*100:4.1f}%) | {n_ql_f1*100 - o_ql_f1*100:+5.2f}%")
    print(f"Option Label F1        | {o_ol_f1*100:5.2f}% (P={o_ol_p*100:4.1f}%, R={o_ol_r*100:4.1f}%) | {n_ol_f1*100:5.2f}% (P={n_ol_p*100:4.1f}%, R={n_ol_r*100:4.1f}%) | {n_ol_f1*100 - o_ol_f1*100:+5.2f}%")
    print(f"Question Stem F1       | {o_s_f1*100:5.2f}% (P={o_s_p*100:4.1f}%, R={o_s_r*100:4.1f}%) | {n_s_f1*100:5.2f}% (P={n_s_p*100:4.1f}%, R={n_s_r*100:4.1f}%) | {n_s_f1*100 - o_s_f1*100:+5.2f}%")
    print(f"Option Text F1         | {'---':<24} | {n_ot_f1*100:5.2f}% (P={n_ot_p*100:4.1f}%, R={n_ot_r*100:4.1f}%) | {'---':<10}")

    print("\n--- 2. CATEGORY-BY-CATEGORY QUESTION LABEL F1 COMPARISON ---")
    print(f"{'Category':<30} | {'Docs':<5} | {'Old QL F1':<10} | {'New QL F1':<10} | {'Old OL F1':<10} | {'New OL F1':<10}")
    print("-" * 88)
    for cat, s in sorted(cat_summary.items()):
        _, _, o_qf = calc_f1(s["old_ql_tp"], s["old_ql_fp"], s["old_ql_fn"])
        _, _, n_qf = calc_f1(s["new_ql_tp"], s["new_ql_fp"], s["new_ql_fn"])
        _, _, o_of = calc_f1(s["old_ol_tp"], s["old_ol_fp"], s["old_ol_fn"])
        _, _, n_of = calc_f1(s["new_ol_tp"], s["new_ol_fp"], s["new_ol_fn"])
        print(f"{cat:<30} | {s['docs']:<5} | {o_qf*100:6.2f}%    | {n_qf*100:6.2f}%    | {o_of*100:6.2f}%    | {n_of*100:6.2f}%")

    # Save detailed JSON benchmark
    bench_out = WORKSPACE_DIR / "backend/logs/artifact/new_deterministic_parser_benchmark.json"
    bench_out.parent.mkdir(parents=True, exist_ok=True)
    bench_out.write_text(json.dumps({
        "old_parser": {
            "high_confidence_pass": old_totals["high_confidence"],
            "question_label_f1": round(o_ql_f1, 4),
            "option_label_f1": round(o_ol_f1, 4),
            "stem_f1": round(o_s_f1, 4),
        },
        "new_parser": {
            "high_confidence_pass": new_totals["high_confidence"],
            "question_label_f1": round(n_ql_f1, 4),
            "option_label_f1": round(n_ol_f1, 4),
            "stem_f1": round(n_s_f1, 4),
            "option_text_f1": round(n_ot_f1, 4),
        },
        "category_breakdown": cat_summary
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved benchmark comparison to {bench_out}")

if __name__ == "__main__":
    main()
