#!/usr/bin/env python3
"""
Unified Training Dataset Builder for Sequence Labelling.

Combines real annotated examination documents with synthetic curriculum exams
while strictly isolating the frozen Gold Benchmark set.

Exports three standard ML training formats:
1. BIO Token Chunks (train_bio_chunks.jsonl): 512-token sliding windows with BIO labels
   for encoder-based token classification (PhoBERT, XLM-RoBERTa, mDeBERTa).
2. Document Spans (train_spans.jsonl): Document-level character offsets for span models (spaCy, LayoutLM).
3. Generative Instruction (train_generative.jsonl): Prompt-response pairs (Raw Markdown -> XML)
   for instruction fine-tuning small LLMs (Qwen2.5, Llama-3).
"""

import os
import sys
import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import yaml

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
from synthetic_exam_generator.reconstructor import (
    reconstruct_exam,
    spans_to_xml,
    ReconstructorConfig,
)

# Standard tags to include in sequence labeling
RECOGNIZED_TAGS = {
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "stimulus",
    "figure",
    "explanation",
    "section",
}


def load_gold_set_exclusion(gold_path: Path) -> set:
    """Loads document IDs that must be strictly excluded from training."""
    if not gold_path.exists():
        print(f"Warning: Gold set {gold_path} not found. Proceeding without exclusion.")
        return set()
    try:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
        doc_ids = {d["doc_id"] for d in data.get("documents", [])}
        print(f"Loaded {len(doc_ids)} Gold Benchmark document IDs to exclude from training.")
        return doc_ids
    except Exception as e:
        print(f"Error loading gold set exclusion: {e}")
        return set()


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenizes text into words and punctuation tokens with (start, end) character offsets."""
    tokens = []
    offsets = []
    pattern = re.compile(r"\w+|[^\w\s]")
    for m in pattern.finditer(text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


import bisect

def align_spans_to_bio(
    token_offsets: List[Tuple[int, int]],
    spans: List[Dict[str, Any]]
) -> List[str]:
    """Aligns character-level spans to token-level BIO tags in O(M log N) time."""
    num_tokens = len(token_offsets)
    bio_tags = ["O"] * num_tokens
    if num_tokens == 0 or not spans:
        return bio_tags

    # Extract start positions for binary search
    token_starts = [t[0] for t in token_offsets]

    # Process section first so fine-grained tags (question, stem, options) take precedence
    priority_order = {"section": 0}
    sorted_spans = sorted(
        spans,
        key=lambda s: (priority_order.get(s.get("label", "").lower(), 1), s.get("start", 0))
    )

    for span in sorted_spans:
        label = span.get("label", "").lower()
        if label not in RECOGNIZED_TAGS:
            continue
        s_start = span.get("start", 0)
        s_end = span.get("end", 0)
        tag_name = label.upper()

        # Find first token that might overlap: token_starts >= s_start (lookback 1)
        idx = bisect.bisect_right(token_starts, s_start)
        start_idx = max(0, idx - 1)

        first = True
        for i in range(start_idx, num_tokens):
            t_start, t_end = token_offsets[i]
            if t_start >= s_end:
                break
            if max(s_start, t_start) < min(s_end, t_end):
                bio_tags[i] = f"B-{tag_name}" if first else f"I-{tag_name}"
                first = False

    return bio_tags


def chunk_bio_sequence(
    doc_id: str,
    tokens: List[str],
    bio_tags: List[str],
    token_offsets: List[Tuple[int, int]],
    chunk_size: int = 512,
    stride: int = 448
) -> List[Dict[str, Any]]:
    """Chunks token sequences into sliding windows for transformer training."""
    chunks = []
    n = len(tokens)
    if n == 0:
        return chunks

    idx = 0
    chunk_idx = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        c_tokens = tokens[idx:end]
        c_tags = bio_tags[idx:end]
        c_offsets = token_offsets[idx:end]

        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{chunk_idx:03d}",
            "doc_id": doc_id,
            "start_token_idx": idx,
            "end_token_idx": end,
            "char_start": c_offsets[0][0] if c_offsets else 0,
            "char_end": c_offsets[-1][1] if c_offsets else 0,
            "tokens": c_tokens,
            "ner_tags": c_tags,
        })
        chunk_idx += 1
        if end >= n:
            break
        idx += stride

    return chunks


def augment_real_document(
    raw_text: str,
    spans: List[Dict[str, Any]],
    rng: random.Random,
    latex_mask_prob: float = 0.05,
    latex_placeholder: str = "[LATEX]",
    latex_vary_delimiters_prob: float = 0.10,
    flatten_newlines_prob: float = 0.20,
    newline_to_space_prob: float = 0.20,
    tab_to_space: bool = True
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """
    Safely applies whitespace compression/jitter, tab normalization to space,
    newline-to-space flattening, and LaTeX formula masking/delimiter jitter
    to a real examination document.
    Protects markdown tables and LaTeX math intervals from disruption.
    Remaps span offsets monotonically and filters out degenerate or zero-length spans.
    """
    if not raw_text or not spans:
        return None

    # Identify LaTeX formula intervals and Markdown table intervals
    latex_matches = []
    for m in re.finditer(r'\$[^$\n]+\$', raw_text):
        latex_matches.append((m.start(), m.end(), m.group(0)))
    latex_lookup = {m[0]: m for m in latex_matches}

    table_intervals = []
    for m in re.finditer(r'\|[^\n]+\|', raw_text):
        table_intervals.append((m.start(), m.end()))

    def is_in_table(pos: int) -> bool:
        for s, e in table_intervals:
            if s <= pos < e:
                return True
        return False

    def is_table_row_boundary(pos: int) -> bool:
        for s, e in table_intervals:
            if pos == e or pos + 1 == s:
                return True
        return False

    is_doc_flatten = (flatten_newlines_prob > 0.0 and rng.random() < flatten_newlines_prob)
    mode = rng.choice(['collapse', 'jitter', 'mixed'])
    new_chars = []
    old_to_new = {}
    i = 0
    n = len(raw_text)
    new_pos = 0

    while i < n:
        old_to_new[i] = new_pos

        # Check if i starts a LaTeX math formula
        if i in latex_lookup:
            start_m, end_m, formula = latex_lookup[i]
            if latex_mask_prob > 0.0 and rng.random() < latex_mask_prob:
                replacement = latex_placeholder
            elif latex_vary_delimiters_prob > 0.0 and rng.random() < latex_vary_delimiters_prob:
                inner = formula[1:-1].strip()
                replacement = f"\\( {inner} \\)"
            else:
                replacement = formula

            for ch in replacement:
                new_chars.append(ch)
                new_pos += 1
            for k in range(start_m + 1, end_m + 1):
                old_to_new[k] = new_pos
            i = end_m
            continue

        if is_in_table(i):
            new_chars.append(raw_text[i])
            new_pos += 1
            old_to_new[i + 1] = new_pos
            i += 1
            continue

        # Handle line breaks / newlines
        if raw_text[i] in '\r\n':
            if is_table_row_boundary(i):
                new_chars.append('\n')
                new_pos += 1
                if raw_text[i] == '\r' and i + 1 < n and raw_text[i + 1] == '\n':
                    old_to_new[i + 1] = new_pos
                    old_to_new[i + 2] = new_pos
                    i += 2
                else:
                    old_to_new[i + 1] = new_pos
                    i += 1
                continue

            j = i
            while j < n and raw_text[j] in '\r\n \t' and j not in latex_lookup and not is_in_table(j) and not is_table_row_boundary(j):
                j += 1

            if is_doc_flatten:
                replacement = ' '
            elif newline_to_space_prob > 0.0 and rng.random() < newline_to_space_prob:
                replacement = ' '
            else:
                num_newlines = sum(1 for ch in raw_text[i:j] if ch == '\n')
                if num_newlines >= 2:
                    replacement = '\n\n' if mode != 'collapse' else '\n'
                else:
                    replacement = '\n'

            for ch in replacement:
                new_chars.append(ch)
                new_pos += 1
            for k in range(i + 1, j + 1):
                old_to_new[k] = new_pos
            i = j
            continue

        c = raw_text[i]
        if c in ' \t':
            j = i
            while j < n and raw_text[j] in ' \t' and j not in latex_lookup and not is_in_table(j):
                j += 1
            span_len = j - i
            has_tab = '\t' in raw_text[i:j]

            if is_doc_flatten or mode in ['collapse', 'mixed']:
                if tab_to_space and has_tab:
                    replacement = ' '
                elif span_len >= 2:
                    replacement = ' ' if rng.random() < 0.85 else ('\t' if rng.random() < 0.3 else '  ')
                else:
                    if has_tab and tab_to_space:
                        replacement = ' '
                    elif mode == 'mixed' and rng.random() < 0.08:
                        replacement = '  ' if rng.random() < 0.7 else '\t'
                    else:
                        replacement = c
            else: # jitter mode
                if span_len >= 2:
                    replacement = '\t' if rng.random() < 0.3 else '  '
                else:
                    if rng.random() < 0.08:
                        replacement = '  ' if rng.random() < 0.7 else '\t'
                    else:
                        replacement = c

            for ch in replacement:
                new_chars.append(ch)
                new_pos += 1
            for k in range(i + 1, j + 1):
                old_to_new[k] = new_pos
            i = j
        else:
            new_chars.append(c)
            new_pos += 1
            old_to_new[i + 1] = new_pos
            i += 1

    old_to_new[n] = new_pos
    final_text = ''.join(new_chars)

    valid_spans = []
    for s in spans:
        orig_start = s.get('start', 0)
        orig_end = s.get('end', 0)
        new_start = old_to_new.get(orig_start, orig_start)
        new_end = old_to_new.get(orig_end, orig_end)

        if new_end <= new_start:
            continue

        sub = final_text[new_start:new_end].strip()
        if not sub:
            continue

        valid_spans.append({
            'start': new_start,
            'end': new_end,
            'label': s.get('label', ''),
            'text': sub
        })

    if not valid_spans:
        return None

    return final_text, valid_spans


def process_real_annotated_documents(
    annotated_dir: Path,
    exclude_doc_ids: set,
    min_score: float = 85.0,
    whitespace_aug_prob: float = 0.0,
    latex_mask_prob: float = 0.05,
    flatten_newlines_prob: float = 0.20,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Loads and formats real annotated documents with optional paired whitespace, tab & newline normalization."""
    records = []
    pristine_count = 0
    augmented_count = 0
    rng = random.Random(seed)

    for d in sorted(annotated_dir.glob("*")):
        if not d.is_dir():
            continue
        doc_id = d.name
        if doc_id in exclude_doc_ids:
            continue

        audit_file = d / "audit_report.json"
        xml_file = d / "merged.xml"
        if not (audit_file.exists() and xml_file.exists()):
            continue

        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            if audit.get("decision") == "DISCARD":
                continue
            if float(audit.get("overall_score", 0.0)) < min_score:
                continue

            xml_content = xml_file.read_text(encoding="utf-8")
            raw_text, spans = parse_xml_annotations(xml_content)
            if not raw_text.strip() or not spans:
                continue

            # 1. Primary Pristine Document
            records.append({
                "doc_id": doc_id,
                "source": "real",
                "raw_text": raw_text,
                "xml_content": xml_content,
                "spans": spans,
                "score": float(audit.get("overall_score", 0.0))
            })
            pristine_count += 1

            # 2. Stochastic Whitespace, Tab & Newline Augmented Paired Variant
            if whitespace_aug_prob > 0.0 and rng.random() < whitespace_aug_prob:
                aug_res = augment_real_document(
                    raw_text, spans, rng,
                    latex_mask_prob=latex_mask_prob,
                    flatten_newlines_prob=flatten_newlines_prob,
                    newline_to_space_prob=0.20,
                    tab_to_space=True
                )
                if aug_res is not None:
                    aug_text, aug_spans = aug_res
                    aug_xml = spans_to_xml(aug_text, aug_spans)
                    records.append({
                        "doc_id": f"{doc_id}_aug",
                        "source": "real_augmented",
                        "raw_text": aug_text,
                        "xml_content": aug_xml,
                        "spans": aug_spans,
                        "score": float(audit.get("overall_score", 0.0))
                    })
                    augmented_count += 1
        except Exception as e:
            print(f"Error processing real doc {doc_id}: {e}")

    return records, pristine_count, augmented_count


def process_synthetic_exams(
    synthetic_dir: Path,
    target_count: Optional[int] = None,
    seed: int = 42,
    config: Optional[ReconstructorConfig] = None
) -> List[Dict[str, Any]]:
    """Loads and compiles synthetic curriculum exams into sequence-labelled documents."""
    exam_files = sorted(synthetic_dir.glob("*.json"))
    if not exam_files:
        return []

    rng = random.Random(seed)
    rng.shuffle(exam_files)
    if target_count is not None and target_count < len(exam_files):
        exam_files = exam_files[:target_count]

    cfg = config or ReconstructorConfig(
        seed=str(seed),
        inline_option_prob=0.35,
        grid_2x2_prob=0.20,
        same_line_stem_options_prob=0.10,
        flatten_newlines_prob=0.20,
        collapse_whitespace_prob=0.15,
        enable_permutations=True,
        prob_inline_barem=0.35,
        prob_answer_grid=0.15,
        prob_table_barem=0.10,
        prob_no_barem=0.40,
        space_noise_rate=0.035,
        casing_noise_prob=0.03,
        typo_rate=0.015,
        latex_mask_prob=0.05,
        include_span_text=False
    )

    records = []
    for f in exam_files:
        try:
            exam_data = json.loads(f.read_text(encoding="utf-8"))
            reconstructed = reconstruct_exam(exam_data, cfg)
            raw_text = reconstructed.get("raw_text", "")
            spans = reconstructed.get("spans", [])
            if not raw_text.strip() or not spans:
                continue

            xml_content = spans_to_xml(raw_text, spans)
            records.append({
                "doc_id": f"synthetic_{exam_data.get('exam_id', f.stem)}",
                "source": "synthetic",
                "subject": exam_data.get("subject", "general"),
                "raw_text": raw_text,
                "xml_content": xml_content,
                "spans": spans,
                "score": 100.0
            })
        except Exception as e:
            print(f"Error compiling synthetic exam {f.name}: {e}")

    return records


def process_gold_benchmark_documents(
    annotated_dir: Path,
    gold_path: Path
) -> List[Dict[str, Any]]:
    """Loads the 63 Gold Benchmark documents strictly in pristine ground-truth condition."""
    if not gold_path.exists():
        print(f"Warning: Gold benchmark path {gold_path} does not exist.")
        return []
    try:
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_docs = gold_data.get("documents", [])
    except Exception as e:
        print(f"Error loading gold benchmark for export: {e}")
        return []

    records = []
    for g in gold_docs:
        doc_id = g.get("doc_id")
        doc_dir = annotated_dir / doc_id
        xml_file = doc_dir / "merged.xml"
        if not xml_file.exists():
            continue
        try:
            xml_content = xml_file.read_text(encoding="utf-8")
            raw_text, spans = parse_xml_annotations(xml_content)
            if not raw_text.strip() or not spans:
                continue
            records.append({
                "doc_id": doc_id,
                "source": "gold_benchmark",
                "tier": g.get("tier", "UNKNOWN"),
                "score": float(g.get("score", 100.0)),
                "raw_text": raw_text,
                "xml_content": xml_content,
                "spans": spans,
            })
        except Exception as e:
            print(f"Error processing gold benchmark doc {doc_id}: {e}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare Unified Training Set for Sequence Labelling.")
    parser.add_argument(
        "--gold-set",
        type=str,
        default="data/benchmark_gold_set.json",
        help="Path to Gold Benchmark manifest to exclude from training"
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Path to real annotated documents directory"
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="data/synthetic_exams",
        help="Path to synthetic curriculum exams directory"
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.40,
        help="Target proportion of synthetic documents in training set (e.g. 0.40 = 60%% real, 40%% synthetic)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Sliding window chunk token size for transformer BIO chunks (default: 512)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=448,
        help="Stride for sliding window chunking (default: 448 = 64 token overlap)"
    )
    parser.add_argument(
        "--min-real-score",
        type=float,
        default=85.0,
        help="Minimum review score for real annotated documents (default: 85.0)"
    )
    parser.add_argument(
        "--synthetic-inline-opt-prob",
        type=float,
        default=0.35,
        help="Probability of collapsing options horizontally into inline layout (default: 0.35)"
    )
    parser.add_argument(
        "--synthetic-grid-2x2-prob",
        type=float,
        default=0.20,
        help="Probability of 2x2 grid option layout (default: 0.20)"
    )
    parser.add_argument(
        "--synthetic-same-line-stem-prob",
        type=float,
        default=0.10,
        help="Probability of option A starting on the same line as stem (default: 0.10)"
    )
    parser.add_argument(
        "--synthetic-typo-rate",
        type=float,
        default=0.015,
        help="Rate of Vietnamese tone/telex typo injection in synthetic text (default: 0.015)"
    )
    parser.add_argument(
        "--synthetic-space-noise",
        type=float,
        default=0.035,
        help="Rate of space jitter and tabulation noise in synthetic text (default: 0.035)"
    )
    parser.add_argument(
        "--synthetic-casing-prob",
        type=float,
        default=0.03,
        help="Probability of label casing perturbation in synthetic text (default: 0.03)"
    )
    parser.add_argument(
        "--real-whitespace-aug-prob",
        type=float,
        default=0.25,
        help="Probability of generating a paired whitespace-augmented variant for real documents (default: 0.25)"
    )
    parser.add_argument(
        "--real-latex-mask-prob",
        type=float,
        default=0.05,
        help="Probability of masking LaTeX formulas with placeholder in augmented real documents (default: 0.05)"
    )
    parser.add_argument(
        "--real-flatten-newlines-prob",
        type=float,
        default=None,
        help="Probability of flattening newlines into spaces in augmented real documents (default: from config/profile)"
    )
    parser.add_argument(
        "--synthetic-flatten-newlines-prob",
        type=float,
        default=None,
        help="Probability of flattening all newlines into spaces across synthetic exams (default: from config/profile)"
    )
    parser.add_argument(
        "--synthetic-collapse-whitespace-prob",
        type=float,
        default=None,
        help="Probability of collapsing multiple spaces and tabs into single space in synthetic exams (default: from config/profile)"
    )
    parser.add_argument(
        "--window-configs",
        type=str,
        default=None,
        help="Comma-separated multi-scale sliding window configurations (e.g. '512:128,768:192,1024:256,2048:512')"
    )
    parser.add_argument(
        "--synthetic-admin-header-prob",
        type=float,
        default=None,
        help="Probability of prepending untagged administrative school headers in synthetic exams"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/augmentation_config.yaml",
        help="Path to YAML augmentation configuration file (default: configs/augmentation_config.yaml)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Name of configuration profile to use from YAML (e.g. balanced_production, conservative_clean, aggressive_stress_test)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and sampling"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/training_dataset",
        help="Directory to save output training datasets"
    )
    args = parser.parse_args()

    # Load YAML Configuration if present
    cfg_file = WORKSPACE_DIR / args.config
    if not cfg_file.exists() and (WORKSPACE_DIR / "augmentation_config.yaml").exists():
        cfg_file = WORKSPACE_DIR / "augmentation_config.yaml"
    yaml_cfg = {}
    if cfg_file.exists():
        try:
            yaml_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            print(f"Loaded augmentation configuration from {cfg_file.name}")
        except Exception as e:
            print(f"Warning: Could not parse YAML config {cfg_file}: {e}")

    # Resolve active profile
    profile_name = args.profile or yaml_cfg.get("active_profile", "balanced_production")
    profile_data = yaml_cfg.get("profiles", {}).get(profile_name, {})
    if profile_data:
        print(f"Active Augmentation Profile: [{profile_name}]")

    # Resolve parameters (CLI flag overrides profile, which overrides YAML corpus/defaults)
    synthetic_ratio = args.synthetic_ratio
    if "synthetic_ratio" in profile_data:
        synthetic_ratio = float(profile_data["synthetic_ratio"])
    elif "corpus" in yaml_cfg and "synthetic_ratio" in yaml_cfg["corpus"]:
        synthetic_ratio = float(yaml_cfg["corpus"]["synthetic_ratio"])

    real_ws_prob = args.real_whitespace_aug_prob
    if "real_whitespace_aug_prob" in profile_data:
        real_ws_prob = float(profile_data["real_whitespace_aug_prob"])
    elif "real_data" in yaml_cfg and "whitespace_augmentation" in yaml_cfg["real_data"]:
        real_ws_prob = float(yaml_cfg["real_data"]["whitespace_augmentation"].get("paired_variant_prob", real_ws_prob))

    real_latex_prob = args.real_latex_mask_prob
    if "real_latex_mask_prob" in profile_data:
        real_latex_prob = float(profile_data["real_latex_mask_prob"])
    elif "real_data" in yaml_cfg and "latex_masking" in yaml_cfg["real_data"]:
        real_latex_prob = float(yaml_cfg["real_data"]["latex_masking"].get("mask_prob", real_latex_prob))

    synth_inline_prob = args.synthetic_inline_opt_prob
    if "inline_option_prob" in profile_data:
        synth_inline_prob = float(profile_data["inline_option_prob"])

    synth_grid_prob = args.synthetic_grid_2x2_prob
    if "grid_2x2_prob" in profile_data:
        synth_grid_prob = float(profile_data["grid_2x2_prob"])

    synth_typo_rate = args.synthetic_typo_rate
    if "typo_rate" in profile_data:
        synth_typo_rate = float(profile_data["typo_rate"])

    synth_space_noise = args.synthetic_space_noise
    if "space_noise_rate" in profile_data:
        synth_space_noise = float(profile_data["space_noise_rate"])

    real_flatten_prob = 0.20
    if args.real_flatten_newlines_prob is not None:
        real_flatten_prob = args.real_flatten_newlines_prob
    elif "real_flatten_newlines_prob" in profile_data:
        real_flatten_prob = float(profile_data["real_flatten_newlines_prob"])
    elif "real_data" in yaml_cfg and "whitespace_augmentation" in yaml_cfg["real_data"]:
        real_flatten_prob = float(yaml_cfg["real_data"]["whitespace_augmentation"].get("flatten_newlines_prob", 0.20))

    synth_flatten_prob = 0.20
    if args.synthetic_flatten_newlines_prob is not None:
        synth_flatten_prob = args.synthetic_flatten_newlines_prob
    elif "synthetic_flatten_newlines_prob" in profile_data:
        synth_flatten_prob = float(profile_data["synthetic_flatten_newlines_prob"])
    elif "synthetic_data" in yaml_cfg and "layout" in yaml_cfg["synthetic_data"]:
        synth_flatten_prob = float(yaml_cfg["synthetic_data"]["layout"].get("flatten_newlines_prob", 0.20))

    synth_collapse_prob = 0.15
    if args.synthetic_collapse_whitespace_prob is not None:
        synth_collapse_prob = args.synthetic_collapse_whitespace_prob
    elif "synthetic_collapse_whitespace_prob" in profile_data:
        synth_collapse_prob = float(profile_data["synthetic_collapse_whitespace_prob"])
    elif "synthetic_data" in yaml_cfg and "layout" in yaml_cfg["synthetic_data"]:
        synth_collapse_prob = float(yaml_cfg["synthetic_data"]["layout"].get("collapse_whitespace_prob", 0.15))

    # Resolve synthetic administrative header probability
    synth_admin_prob = 0.40
    if args.synthetic_admin_header_prob is not None:
        synth_admin_prob = args.synthetic_admin_header_prob
    elif "synthetic_data" in yaml_cfg and "administrative_headers" in yaml_cfg["synthetic_data"]:
        admin_sec = yaml_cfg["synthetic_data"]["administrative_headers"]
        synth_admin_prob = float(admin_sec.get("probability", 0.40)) if admin_sec.get("enabled", True) else 0.0

    # Resolve multi-scale sliding window configurations (512 to 2048)
    window_configs = []
    if args.window_configs:
        for item in args.window_configs.split(","):
            parts = [int(p.strip()) for p in item.split(":") if p.strip()]
            if len(parts) == 2:
                window_configs.append((parts[0], parts[1]))
            elif len(parts) == 1:
                window_configs.append((parts[0], parts[0] // 4))
    elif "corpus" in yaml_cfg and "sliding_windows" in yaml_cfg["corpus"]:
        for item in yaml_cfg["corpus"]["sliding_windows"]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                window_configs.append((int(item[0]), int(item[1])))
    if not window_configs:
        window_configs = [(args.chunk_size, args.stride)]

    print(f"Active Multi-Scale Window Configs: {window_configs}")
    print(f"Synthetic Administrative Header Probability: {synth_admin_prob:.2f}")
    print(f"Real Data Whitespace/Flatten Probabilities: WS={real_ws_prob:.2f}, Flatten={real_flatten_prob:.2f}")
    print(f"Synthetic Layout Probabilities: Inline={synth_inline_prob:.2f}, Grid2x2={synth_grid_prob:.2f}, Flatten={synth_flatten_prob:.2f}, Collapse={synth_collapse_prob:.2f}")

    gold_path = WORKSPACE_DIR / args.gold_set
    annotated_dir = WORKSPACE_DIR / args.annotated_dir
    synthetic_dir = WORKSPACE_DIR / args.synthetic_dir
    out_dir = WORKSPACE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Gold Set Exclusions
    exclude_doc_ids = load_gold_set_exclusion(gold_path)

    # 2. Process Real Documents (with safe whitespace, tab, and newline augmentation)
    print(f"\nProcessing real annotated documents from {annotated_dir}...")
    real_records, num_pristine, num_augmented = process_real_annotated_documents(
        annotated_dir,
        exclude_doc_ids,
        min_score=args.min_real_score,
        whitespace_aug_prob=real_ws_prob,
        latex_mask_prob=real_latex_prob,
        flatten_newlines_prob=real_flatten_prob,
        seed=args.seed
    )
    print(f"Loaded {len(real_records)} real training documents ({num_pristine} pristine, {num_augmented} whitespace-augmented paired variants; Gold benchmark excluded).")

    # 3. Calculate target synthetic count based on ratio
    num_real_unique = num_pristine
    target_synthetic = int(round(num_real_unique * (synthetic_ratio / (1.0 - synthetic_ratio))))
    print(f"\nCompiling synthetic exams from {synthetic_dir} (target: {target_synthetic} exams, ratio: {synthetic_ratio:.2f})...")
    
    synth_cfg = ReconstructorConfig(
        seed=str(args.seed),
        inline_option_prob=synth_inline_prob,
        grid_2x2_prob=synth_grid_prob,
        same_line_stem_options_prob=args.synthetic_same_line_stem_prob,
        flatten_newlines_prob=synth_flatten_prob,
        collapse_whitespace_prob=synth_collapse_prob,
        enable_permutations=True,
        prob_inline_barem=0.35,
        prob_answer_grid=0.15,
        prob_table_barem=0.10,
        prob_no_barem=0.40,
        space_noise_rate=synth_space_noise,
        casing_noise_prob=args.synthetic_casing_prob,
        typo_rate=synth_typo_rate,
        latex_mask_prob=0.05,
        include_span_text=False,
        admin_header_prob=synth_admin_prob
    )
    
    synthetic_records = process_synthetic_exams(
        synthetic_dir,
        target_count=target_synthetic,
        seed=args.seed,
        config=synth_cfg
    )
    print(f"Compiled {len(synthetic_records)} synthetic training documents.")

    # 4. Merge and Shuffle Documents
    all_documents = real_records + synthetic_records
    rng = random.Random(args.seed)
    rng.shuffle(all_documents)
    print(f"\nTotal Unified Training Documents: {len(all_documents)} ({len(real_records)} Real [{num_pristine} pristine + {num_augmented} augmented], {len(synthetic_records)} Synthetic)")

    # 5. Build Training Outputs
    spans_jsonl_path = out_dir / "train_spans.jsonl"
    bio_jsonl_path = out_dir / "train_bio_chunks.jsonl"
    gen_jsonl_path = out_dir / "train_generative.jsonl"
    manifest_path = out_dir / "dataset_manifest.json"

    total_tokens = 0
    total_chunks = 0
    tag_counts = Counter()

    print(f"\nWriting dataset formats to {out_dir}...")
    with open(spans_jsonl_path, "w", encoding="utf-8") as f_spans, \
         open(bio_jsonl_path, "w", encoding="utf-8") as f_bio, \
         open(gen_jsonl_path, "w", encoding="utf-8") as f_gen:

        for doc in all_documents:
            doc_id = doc["doc_id"]
            raw_text = doc["raw_text"]
            xml_content = doc["xml_content"]
            spans = doc["spans"]

            # Format 1: Spans JSONL
            span_entry = {
                "doc_id": doc_id,
                "source": doc["source"],
                "text": raw_text,
                "spans": spans
            }
            f_spans.write(json.dumps(span_entry, ensure_ascii=False) + "\n")

            # Format 2: Generative Instruction JSONL
            gen_entry = {
                "doc_id": doc_id,
                "source": doc["source"],
                "prompt": f"Sequence label the following examination text using standard XML structural tags:\n\n{raw_text}",
                "response": xml_content
            }
            f_gen.write(json.dumps(gen_entry, ensure_ascii=False) + "\n")

            # Format 3: BIO Token Chunks across Multi-Scale Windows
            tokens, offsets = tokenize_with_offsets(raw_text)
            total_tokens += len(tokens)
            bio_tags = align_spans_to_bio(offsets, spans)

            for tag in bio_tags:
                tag_counts[tag] += 1

            for chunk_size, stride in window_configs:
                chunks = chunk_bio_sequence(
                    doc_id,
                    tokens,
                    bio_tags,
                    offsets,
                    chunk_size=chunk_size,
                    stride=stride
                )
                total_chunks += len(chunks)

                for c in chunks:
                    c["window_size"] = chunk_size
                    c["stride"] = stride
                    f_bio.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 6. Build Gold Benchmark Test/Validation Outputs (100% pristine, 0 noise)
    test_spans_path = out_dir / "test_spans.jsonl"
    test_bio_path = out_dir / "test_bio_chunks.jsonl"
    test_gen_path = out_dir / "test_generative.jsonl"

    test_records = process_gold_benchmark_documents(annotated_dir, gold_path)
    test_total_tokens = 0
    test_total_chunks = 0
    test_tag_counts = Counter()

    if test_records:
        print(f"\nWriting Gold Benchmark test/validation formats ({len(test_records)} docs) to {out_dir}...")
        with open(test_spans_path, "w", encoding="utf-8") as f_spans, \
             open(test_bio_path, "w", encoding="utf-8") as f_bio, \
             open(test_gen_path, "w", encoding="utf-8") as f_gen:

            for doc in test_records:
                doc_id = doc["doc_id"]
                raw_text = doc["raw_text"]
                xml_content = doc["xml_content"]
                spans = doc["spans"]

                span_entry = {
                    "doc_id": doc_id,
                    "source": doc["source"],
                    "tier": doc.get("tier"),
                    "score": doc.get("score"),
                    "text": raw_text,
                    "spans": spans
                }
                f_spans.write(json.dumps(span_entry, ensure_ascii=False) + "\n")

                gen_entry = {
                    "doc_id": doc_id,
                    "source": doc["source"],
                    "tier": doc.get("tier"),
                    "prompt": f"Sequence label the following examination text using standard XML structural tags:\n\n{raw_text}",
                    "response": xml_content
                }
                f_gen.write(json.dumps(gen_entry, ensure_ascii=False) + "\n")

                tokens, offsets = tokenize_with_offsets(raw_text)
                test_total_tokens += len(tokens)
                bio_tags = align_spans_to_bio(offsets, spans)
                for tag in bio_tags:
                    test_tag_counts[tag] += 1

                for chunk_size, stride in window_configs:
                    chunks = chunk_bio_sequence(
                        doc_id,
                        tokens,
                        bio_tags,
                        offsets,
                        chunk_size=chunk_size,
                        stride=stride
                    )
                    test_total_chunks += len(chunks)
                    for c in chunks:
                        c["window_size"] = chunk_size
                        c["stride"] = stride
                        c["tier"] = doc.get("tier")
                        f_bio.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 7. Save Manifest & Summary
    manifest = {
        "train": {
            "total_documents": len(all_documents),
            "real_documents_total": len(real_records),
            "real_pristine_documents": num_pristine,
            "real_augmented_documents": num_augmented,
            "real_whitespace_aug_prob": args.real_whitespace_aug_prob,
            "synthetic_documents": len(synthetic_records),
            "synthetic_ratio": round(len(synthetic_records) / len(all_documents), 4) if all_documents else 0.0,
            "synthetic_admin_header_prob": synth_admin_prob,
            "total_tokens": total_tokens,
            "total_bio_chunks": total_chunks,
            "window_configs": window_configs,
            "tag_distribution": dict(tag_counts.most_common()),
            "output_files": {
                "spans": str(spans_jsonl_path),
                "bio_chunks": str(bio_jsonl_path),
                "generative": str(gen_jsonl_path)
            }
        },
        "test_validation": {
            "source": str(gold_path),
            "total_documents": len(test_records),
            "total_tokens": test_total_tokens,
            "total_bio_chunks": test_total_chunks,
            "window_configs": window_configs,
            "tag_distribution": dict(test_tag_counts.most_common()),
            "output_files": {
                "spans": str(test_spans_path),
                "bio_chunks": str(test_bio_path),
                "generative": str(test_gen_path)
            }
        },
        "total_documents": len(all_documents),
        "total_tokens": total_tokens,
        "total_bio_chunks": total_chunks,
        "window_configs": window_configs,
        "tag_distribution": dict(tag_counts.most_common()),
        "output_files": {
            "spans": str(spans_jsonl_path),
            "bio_chunks": str(bio_jsonl_path),
            "generative": str(gen_jsonl_path),
            "test_spans": str(test_spans_path),
            "test_bio_chunks": str(test_bio_path),
            "test_generative": str(test_gen_path)
        }
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDataset preparation completed successfully!")
    print(f"  - Training Documents: {len(all_documents):,}")
    print(f"  - Training Tokens: {total_tokens:,}")
    print(f"  - Training Chunks: {total_chunks:,}")
    if test_records:
        print(f"  - Gold Benchmark Test Documents: {len(test_records):,}")
        print(f"  - Gold Benchmark Test Tokens: {test_total_tokens:,}")
        print(f"  - Gold Benchmark Test Chunks: {test_total_chunks:,}")
    print(f"  - Window Configurations: {window_configs}")
    print(f"  - Saved to: {out_dir}")
    print(f"    * {spans_jsonl_path.name}")
    print(f"    * {bio_jsonl_path.name}")
    print(f"    * {gen_jsonl_path.name}")
    if test_records:
        print(f"    * {test_spans_path.name}")
        print(f"    * {test_bio_path.name}")
        print(f"    * {test_gen_path.name}")
    print(f"    * {manifest_path.name}")


if __name__ == "__main__":
    main()
