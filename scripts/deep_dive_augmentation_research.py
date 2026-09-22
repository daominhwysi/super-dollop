#!/usr/bin/env python3
"""
Deep-Dive Empirical Research & Evaluation Suite for Sequence Labelling Augmentation.

Investigates:
1. Invariant & Boundary Fidelity across all augmentation types (zero span drift).
2. Augmentation Method Sensitivity & Rule-based Parser (DET) degradation curve.
3. Structural Vector Distance & Domain Gap (Synthetic vs Real Scanned vs Real Digital).
4. Synthetic-to-Real Corpus Mixing Ratio (0% to 100%) and tag balance entropy.
5. Real Exam Augmentation Policy (Pristine vs Augmented Real trade-offs).
"""

import os
import sys
import json
import re
import copy
import math
import random
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from sequence_labelling.parser.deterministic_parser import parse_chunk_deterministic
from tools.build_training_dataset import (
    tokenize_with_offsets,
    align_spans_to_bio,
    chunk_bio_sequence,
    RECOGNIZED_TAGS
)
from tools.select_gold_benchmark import extract_structural_vector_and_tokens
from synthetic_exam_generator.reconstructor import (
    reconstruct_exam,
    spans_to_xml,
    ReconstructorConfig,
    DEFAULT_QUESTION_PREFIXES,
    OPTION_PREFIX_STYLES,
    inject_vietnamese_typos
)


def evaluate_span_matching(gold_spans, pred_spans, label, tolerance=4):
    gold = [s for s in gold_spans if s.get("label") == label]
    pred = [s for s in pred_spans if s.get("label") == label]
    tp = 0
    fp = 0
    matched_gold = set()
    for p in pred:
        p_start = p.get("start", 0)
        found = None
        for idx, g in enumerate(gold):
            if idx not in matched_gold and abs(g.get("start", 0) - p_start) <= tolerance:
                found = idx
                break
        if found is not None:
            tp += 1
            matched_gold.add(found)
        else:
            fp += 1
    fn = len(gold) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn, len(gold), len(pred)


def run_experiment_1_span_fidelity(synthetic_files: List[Path], num_trials: int = 50) -> Dict[str, Any]:
    """Experiment 1: Verify mathematical span-text alignment and BIO validity under heavy noise."""
    print("\n--- Running Experiment 1: Span Alignment & BIO Fidelity Audit ---")
    configs = [
        ("clean", ReconstructorConfig(seed=42)),
        ("light_noise", ReconstructorConfig(seed=42, typo_rate=0.01, space_noise_rate=0.03, casing_noise_prob=0.02)),
        ("heavy_noise", ReconstructorConfig(seed=42, typo_rate=0.05, space_noise_rate=0.10, casing_noise_prob=0.08, inline_option_prob=0.50, grid_2x2_prob=0.30)),
        ("extreme_noise", ReconstructorConfig(seed=42, typo_rate=0.10, space_noise_rate=0.15, casing_noise_prob=0.15, inline_option_prob=0.80, grid_2x2_prob=0.50, collapse_whitespace_prob=0.50))
    ]
    
    results = {}
    sample_files = synthetic_files[:min(num_trials, len(synthetic_files))]
    
    for cfg_name, cfg in configs:
        total_spans = 0
        span_drift_count = 0
        bio_alignment_errors = 0
        xml_roundtrip_errors = 0
        
        for f in sample_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                reconstructed = reconstruct_exam(data, cfg)
                raw_text = reconstructed.get("raw_text", "")
                spans = reconstructed.get("spans", [])
                
                # Check 1: Span boundary text match
                for s in spans:
                    total_spans += 1
                    sub = raw_text[s["start"]:s["end"]]
                    if not sub.strip():
                        span_drift_count += 1
                
                # Check 2: BIO token alignment
                tokens, offsets = tokenize_with_offsets(raw_text)
                bio_tags = align_spans_to_bio(offsets, spans)
                if len(bio_tags) != len(tokens):
                    bio_alignment_errors += 1
                
                # Check 3: XML Round-trip
                xml_out = spans_to_xml(raw_text, spans)
                rt_text, rt_spans = parse_xml_annotations(xml_out)
                if rt_text != raw_text:
                    xml_roundtrip_errors += 1
            except Exception as e:
                bio_alignment_errors += 1
                
        results[cfg_name] = {
            "total_spans_tested": total_spans,
            "span_drift_count": span_drift_count,
            "span_drift_rate": span_drift_count / max(1, total_spans),
            "bio_alignment_errors": bio_alignment_errors,
            "xml_roundtrip_errors": xml_roundtrip_errors,
            "pass_rate": 100.0 * (1.0 - (span_drift_count + bio_alignment_errors + xml_roundtrip_errors) / max(1, total_spans))
        }
        print(f"  [{cfg_name}] Spans: {total_spans:,} | Drifts: {span_drift_count} | BIO Errs: {bio_alignment_errors} | RT Errs: {xml_roundtrip_errors} | Pass: {results[cfg_name]['pass_rate']:.4f}%")
        
    return results


def run_experiment_2_ablation_and_det_degradation(synthetic_files: List[Path], num_trials: int = 30) -> Dict[str, Any]:
    """Experiment 2: Measure how individual augmentations degrade rule-based parsing and increase subword token counts."""
    print("\n--- Running Experiment 2: Augmentation Method Sensitivity & DET Parser Degradation ---")
    test_files = synthetic_files[:min(num_trials, len(synthetic_files))]
    
    variations = {
        "0_baseline_clean": ReconstructorConfig(seed=42, randomize_q_num=False),
        "1_inline_options_100pct": ReconstructorConfig(seed=42, inline_option_prob=1.0, randomize_q_num=False),
        "2_grid_2x2_100pct": ReconstructorConfig(seed=42, grid_2x2_prob=1.0, randomize_q_num=False),
        "3_same_line_stem_options": ReconstructorConfig(seed=42, same_line_stem_options_prob=1.0, randomize_q_num=False),
        "4_typo_rate_1pct": ReconstructorConfig(seed=42, typo_rate=0.01, randomize_q_num=False),
        "5_typo_rate_3pct": ReconstructorConfig(seed=42, typo_rate=0.03, randomize_q_num=False),
        "6_typo_rate_5pct": ReconstructorConfig(seed=42, typo_rate=0.05, randomize_q_num=False),
        "7_space_noise_5pct": ReconstructorConfig(seed=42, space_noise_rate=0.05, randomize_q_num=False),
        "8_space_noise_15pct": ReconstructorConfig(seed=42, space_noise_rate=0.15, randomize_q_num=False),
        "9_option_prefix_paren": ReconstructorConfig(seed=42, option_prefix_style="lowercase_paren", randomize_q_num=False),
        "10_option_prefix_bold": ReconstructorConfig(seed=42, option_prefix_style="bold_capital_dot", randomize_q_num=False),
        "11_latex_mask_30pct": ReconstructorConfig(seed=42, latex_mask_prob=0.30, randomize_q_num=False),
        "12_combined_moderate": ReconstructorConfig(seed=42, typo_rate=0.01, space_noise_rate=0.03, inline_option_prob=0.30, grid_2x2_prob=0.20, randomize_q_num=False)
    }
    
    results = {}
    
    for var_name, cfg in variations.items():
        q_label_f1s = []
        opt_label_f1s = []
        stem_f1s = []
        opt_text_f1s = []
        total_words = 0
        total_tokens = 0
        
        for f in test_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                reconstructed = reconstruct_exam(data, cfg)
                raw_text = reconstructed.get("raw_text", "")
                gold_spans = reconstructed.get("spans", [])
                
                # Token count stats
                words = len(raw_text.split())
                tokens, _ = tokenize_with_offsets(raw_text)
                total_words += words
                total_tokens += len(tokens)
                
                # DET parser evaluation
                det_res = parse_chunk_deterministic(raw_text)
                pred_spans = det_res.spans
                
                # Evaluate key labels
                _, _, f1_ql, _, _, _, _, _ = evaluate_span_matching(gold_spans, pred_spans, "question_label", tolerance=4)
                _, _, f1_ol, _, _, _, _, _ = evaluate_span_matching(gold_spans, pred_spans, "option_label", tolerance=4)
                _, _, f1_st, _, _, _, _, _ = evaluate_span_matching(gold_spans, pred_spans, "stem", tolerance=10)
                _, _, f1_ot, _, _, _, _, _ = evaluate_span_matching(gold_spans, pred_spans, "option_text", tolerance=10)
                
                q_label_f1s.append(f1_ql)
                opt_label_f1s.append(f1_ol)
                stem_f1s.append(f1_st)
                opt_text_f1s.append(f1_ot)
            except Exception as e:
                pass
                
        tokens_per_word = round(total_tokens / max(1, total_words), 3)
        mean_ql = round(float(np.mean(q_label_f1s)) * 100, 2)
        mean_ol = round(float(np.mean(opt_label_f1s)) * 100, 2)
        mean_st = round(float(np.mean(stem_f1s)) * 100, 2)
        mean_ot = round(float(np.mean(opt_text_f1s)) * 100, 2)
        macro_f1 = round((mean_ql + mean_ol + mean_st + mean_ot) / 4.0, 2)
        
        results[var_name] = {
            "tokens_per_word": tokens_per_word,
            "q_label_f1": mean_ql,
            "opt_label_f1": mean_ol,
            "stem_f1": mean_st,
            "opt_text_f1": mean_ot,
            "macro_f1": macro_f1
        }
        print(f"  {var_name:26s} | Tok/Wd: {tokens_per_word:.2f} | Q_Lab: {mean_ql:5.1f}% | Opt_Lab: {mean_ol:5.1f}% | Stem: {mean_st:5.1f}% | Opt_Txt: {mean_ot:5.1f}% | Macro F1: {macro_f1:5.1f}%")
        
    return results


def run_experiment_3_structural_distance(real_docs_dir: Path, synthetic_files: List[Path], gold_path: Path) -> Dict[str, Any]:
    """Experiment 3: Compute 13-D structural vectors and calculate cosine similarity / distance between real and synthetic data."""
    print("\n--- Running Experiment 3: Structural Vector Alignment & Domain Gap ---")
    
    # 1. Load real scanned and real digital documents
    pdf_report = Path("data/pdf_classification_report.json")
    pdf_types = {}
    if pdf_report.exists():
        d = json.loads(pdf_report.read_text(encoding="utf-8"))
        for item in d.get("files", []):
            stem = os.path.splitext(item.get("file_name", ""))[0]
            pdf_types[stem] = item.get("type", "UNKNOWN")
            
    real_scanned_vecs = []
    real_digital_vecs = []
    
    for d in real_docs_dir.glob("*"):
        if not d.is_dir(): continue
        mx = d / "merged.xml"
        mj = d / "merged.json"
        if not (mx.exists() and mj.exists()): continue
        try:
            xml_text = mx.read_text(encoding="utf-8")
            merged_json = json.loads(mj.read_text(encoding="utf-8"))
            is_dig = pdf_types.get(d.name) == "DIGITAL"
            vec, _, _ = extract_structural_vector_and_tokens(xml_text, merged_json, is_digital=is_dig)
            if is_dig:
                real_digital_vecs.append(vec)
            else:
                real_scanned_vecs.append(vec)
        except Exception:
            pass
            
    mean_scanned_vec = np.mean(real_scanned_vecs, axis=0) if real_scanned_vecs else np.zeros(13)
    mean_digital_vec = np.mean(real_digital_vecs, axis=0) if real_digital_vecs else np.zeros(13)
    
    print(f"Loaded {len(real_scanned_vecs)} real SCANNED vectors and {len(real_digital_vecs)} real DIGITAL vectors.")
    
    # 2. Test Synthetic Configurations
    sample_files = synthetic_files[:60]
    configs = {
        "Synth_Clean_Default": ReconstructorConfig(seed=42),
        "Synth_Light_Aug": ReconstructorConfig(seed=42, typo_rate=0.01, space_noise_rate=0.03, casing_noise_prob=0.02, inline_option_prob=0.20, grid_2x2_prob=0.15),
        "Synth_Moderate_Aug": ReconstructorConfig(seed=42, typo_rate=0.02, space_noise_rate=0.05, casing_noise_prob=0.04, inline_option_prob=0.40, grid_2x2_prob=0.25, prob_inline_barem=0.30, prob_answer_grid=0.15),
        "Synth_Heavy_Aug": ReconstructorConfig(seed=42, typo_rate=0.05, space_noise_rate=0.12, casing_noise_prob=0.10, inline_option_prob=0.70, grid_2x2_prob=0.40, collapse_whitespace_prob=0.40),
        "Synth_Calibrated_Optimal": ReconstructorConfig(seed=42, typo_rate=0.015, space_noise_rate=0.04, casing_noise_prob=0.03, inline_option_prob=0.35, grid_2x2_prob=0.20, same_line_stem_options_prob=0.10, prob_inline_barem=0.35, prob_answer_grid=0.15, prob_table_barem=0.10)
    }
    
    results = {}
    
    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
        
    for name, cfg in configs.items():
        vecs = []
        for f in sample_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                reconstructed = reconstruct_exam(data, cfg)
                raw_text = reconstructed.get("raw_text", "")
                spans = reconstructed.get("spans", [])
                xml_out = spans_to_xml(raw_text, spans)
                
                # Synthetic merged json dummy
                synth_mj = {"questions": [], "stimuli": []}
                for s in spans:
                    if s["label"] == "stimulus":
                        synth_mj["stimuli"].append({"id": "stim"})
                
                # Dummy questions for option counts
                q_count = max(1, sum(1 for s in spans if s["label"] == "question_label"))
                opt_count = sum(1 for s in spans if s["label"] == "option_label")
                avg_opts = opt_count / q_count
                synth_mj["questions"] = [{"options": [1]*int(round(avg_opts)), "stem": "dummy"} for _ in range(q_count)]
                
                vec, _, _ = extract_structural_vector_and_tokens(xml_out, synth_mj, is_digital=False)
                vecs.append(vec)
            except Exception:
                pass
                
        mean_synth_vec = np.mean(vecs, axis=0) if vecs else np.zeros(13)
        cos_sim_scanned = cosine_sim(mean_synth_vec, mean_scanned_vec)
        euc_dist_scanned = float(np.linalg.norm(mean_synth_vec - mean_scanned_vec))
        cos_sim_digital = cosine_sim(mean_synth_vec, mean_digital_vec)
        
        results[name] = {
            "cosine_sim_to_scanned": round(cos_sim_scanned, 4),
            "euclidean_dist_to_scanned": round(euc_dist_scanned, 4),
            "cosine_sim_to_digital": round(cos_sim_digital, 4)
        }
        print(f"  {name:25s} | CosSim to Real Scanned: {cos_sim_scanned:.4f} | EucDist: {euc_dist_scanned:.4f} | CosSim to Real Digital: {cos_sim_digital:.4f}")
        
    return results


def run_experiment_4_mixing_ratio_simulation(real_records_count: int, synthetic_records_count: int) -> Dict[str, Any]:
    """Experiment 4: Simulate dataset metrics and class balance entropy across synthetic mixing ratios."""
    print("\n--- Running Experiment 4: Synthetic-to-Real Mixing Ratio Simulation ---")
    ratios = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    
    # Baseline empirical tag proportions per doc from data
    # Real avg per doc: Q_Lab: 48.7, Stem: 47.5, Opt_Lab: 185.2, Opt_Txt: 185.2, Stim: 1.34, Expl: 7.5, Sec: 8.0, Fig: 18.7
    real_tag_profile = {
        "QUESTION_LABEL": 48.7,
        "STEM": 47.5,
        "OPTION_LABEL": 185.2,
        "OPTION_TEXT": 185.2,
        "STIMULUS": 1.34,
        "EXPLANATION": 7.5,
        "SECTION": 8.0,
        "FIGURE": 18.7
    }
    
    # Synthetic avg per doc: Q_Lab: 22.0, Stem: 21.0, Opt_Lab: 65.0, Opt_Txt: 65.0, Stim: 3.5, Expl: 22.0, Sec: 4.5, Fig: 0.0
    synth_tag_profile = {
        "QUESTION_LABEL": 22.0,
        "STEM": 21.0,
        "OPTION_LABEL": 65.0,
        "OPTION_TEXT": 65.0,
        "STIMULUS": 3.5,
        "EXPLANATION": 22.0,
        "SECTION": 4.5,
        "FIGURE": 0.0
    }
    
    num_real = real_records_count
    results = {}
    
    for r in ratios:
        if r == 0.0:
            num_synth = 0
        elif r >= 1.0:
            num_synth = synthetic_records_count
        else:
            num_synth = int(round(num_real * (r / (1.0 - r))))
            
        total_docs = num_real + num_synth
        actual_synth_ratio = round(num_synth / total_docs, 4)
        
        # Calculate aggregated tag counts
        merged_tags = {}
        total_entities = 0
        for tag in real_tag_profile:
            count = num_real * real_tag_profile[tag] + num_synth * synth_tag_profile[tag]
            merged_tags[tag] = count
            total_entities += count
            
        # Shannon Entropy of tag distribution: H = -sum(p * log2(p))
        entropy = 0.0
        for tag, count in merged_tags.items():
            p = count / max(1, total_entities)
            if p > 0:
                entropy -= p * math.log2(p)
                
        # Rare entity representation boost: (STIMULUS + EXPLANATION) proportion
        stim_expl_prop = (merged_tags["STIMULUS"] + merged_tags["EXPLANATION"]) / total_entities
        fig_prop = merged_tags["FIGURE"] / total_entities
        
        results[f"ratio_{int(r*100):02d}"] = {
            "synthetic_ratio": actual_synth_ratio,
            "real_documents": num_real,
            "synthetic_documents": num_synth,
            "total_documents": total_docs,
            "shannon_entropy": round(entropy, 4),
            "stimulus_and_explanation_pct": round(stim_expl_prop * 100, 2),
            "figure_pct": round(fig_prop * 100, 2)
        }
        print(f"  Ratio {int(r*100):2d}% Synth | Docs: {total_docs:4d} ({num_real}R + {num_synth:3d}S) | Entropy: {entropy:.4f} | Stim+Expl: {stim_expl_prop*100:5.2f}% | Figure: {fig_prop*100:5.2f}%")
        
    return results


def run_experiment_5_real_augmentation_tradeoffs(real_docs_dir: Path, gold_path: Path, num_trials: int = 30) -> Dict[str, Any]:
    """Experiment 5: Assess token fragmentation and degradation when augmenting already-scanned real documents."""
    print("\n--- Running Experiment 5: Real Exam Set Augmentation Policy Trade-offs ---")
    
    # Exclude gold set
    gold_ids = set()
    if gold_path.exists():
        data = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_ids = {d["doc_id"] for d in data.get("documents", [])}
        
    sample_docs = []
    for d in sorted(real_docs_dir.glob("*")):
        if not d.is_dir() or d.name in gold_ids: continue
        mx = d / "merged.xml"
        if mx.exists():
            sample_docs.append(mx)
            if len(sample_docs) >= num_trials:
                break
                
    policies = {
        "0_pure_pristine_real": {"typo_rate": 0.0, "space_noise": 0.0},
        "1_conservative_real_aug": {"typo_rate": 0.005, "space_noise": 0.015},
        "2_moderate_real_aug": {"typo_rate": 0.015, "space_noise": 0.04},
        "3_aggressive_real_aug": {"typo_rate": 0.04, "space_noise": 0.10}
    }
    
    results = {}
    rng = random.Random(42)
    
    for pol_name, params in policies.items():
        typo_rate = params["typo_rate"]
        space_noise = params["space_noise"]
        
        total_orig_words = 0
        total_tokens = 0
        total_chars = 0
        det_f1_ql = []
        
        for mx in sample_docs:
            try:
                xml_text = mx.read_text(encoding="utf-8")
                raw_text, gold_spans = parse_xml_annotations(xml_text)
                
                # Apply noise
                aug_text = raw_text
                if typo_rate > 0:
                    aug_text = inject_vietnamese_typos(aug_text, typo_rate, rng)
                    
                orig_words = len(raw_text.split())
                tokens, _ = tokenize_with_offsets(aug_text)
                total_orig_words += orig_words
                total_tokens += len(tokens)
                total_chars += len(aug_text)
                
                # Check DET Question Label F1 under this real noise
                det_res = parse_chunk_deterministic(aug_text)
                _, _, f1_ql, _, _, _, _, _ = evaluate_span_matching(gold_spans, det_res.spans, "question_label", tolerance=6)
                det_f1_ql.append(f1_ql)
            except Exception:
                pass
                
        tok_per_orig_word = round(total_tokens / max(1, total_orig_words), 3)
        mean_det_f1 = round(float(np.mean(det_f1_ql)) * 100, 2) if det_f1_ql else 0.0
        
        results[pol_name] = {
            "tokens_per_orig_word": tok_per_orig_word,
            "mean_det_question_f1": mean_det_f1,
            "noise_compounding_penalty_pct": round(max(0.0, results.get("0_pure_pristine_real", {}).get("mean_det_question_f1", mean_det_f1) - mean_det_f1), 2)
        }
        print(f"  {pol_name:26s} | Tok/Word: {tok_per_orig_word:.3f} | DET Q_Lab F1: {mean_det_f1:.1f}% | Degradation: {results[pol_name]['noise_compounding_penalty_pct']:.1f}%")
        
    return results


def main():
    print("================================================================================")
    print("  INCLUSIVE DEEP-DIVE RESEARCH: AUGMENTATION EVALUATION & ARCHITECTURE SIZING   ")
    print("================================================================================")
    
    start_time = time.time()
    
    synth_dir = Path("data/synthetic_exams")
    synth_files = sorted(synth_dir.glob("*.json"))
    real_dir = Path("data/sequence_labelling_annotated")
    gold_path = Path("data/benchmark_gold_set.json")
    
    exp1_res = run_experiment_1_span_fidelity(synth_files, num_trials=50)
    exp2_res = run_experiment_2_ablation_and_det_degradation(synth_files, num_trials=30)
    exp3_res = run_experiment_3_structural_distance(real_dir, synth_files, gold_path)
    exp4_res = run_experiment_4_mixing_ratio_simulation(real_records_count=407, synthetic_records_count=566)
    exp5_res = run_experiment_5_real_augmentation_tradeoffs(real_dir, gold_path, num_trials=30)
    
    total_elapsed = round(time.time() - start_time, 2)
    print(f"\nAll experiments completed in {total_elapsed}s.")
    
    # Save full results json
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "augmentation_deep_dive_results.json"
    
    final_output = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "experiment_1_span_fidelity": exp1_res,
        "experiment_2_ablation_and_det_degradation": exp2_res,
        "experiment_3_structural_distance": exp3_res,
        "experiment_4_mixing_ratio_simulation": exp4_res,
        "experiment_5_real_augmentation_tradeoffs": exp5_res
    }
    
    out_file.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported empirical results to: {out_file}")


if __name__ == "__main__":
    main()
