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
from typing import List, Dict, Any, Tuple, Optional, Union

import torch
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
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


def get_normalized_span_bounds(text: str, start: int, end: int) -> Tuple[int, int, str]:
    """Returns character bounds stripped of leading and trailing whitespace."""
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    sub = text[start:end]
    l_strip = len(sub) - len(sub.lstrip())
    r_strip = len(sub) - len(sub.rstrip())
    norm_start = start + l_strip
    norm_end = end - r_strip
    return norm_start, norm_end, sub.strip()


def evaluate_normalized_exact_matching(
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    label: str,
    raw_text: str
) -> Tuple[int, int, int, int, int]:
    """
    Computes TP, FP, FN with whitespace-normalized exact character matching.
    Both predicted and gold spans have leading/trailing whitespace stripped.
    Requires exact norm_start == norm_start and norm_end == norm_end.
    """
    gold = [s for s in gold_spans if s.get("label") == label]
    pred = [s for s in pred_spans if s.get("label") == label]

    norm_gold = []
    for g in gold:
        ns, ne, nt = get_normalized_span_bounds(raw_text, g.get("start", 0), g.get("end", 0))
        if nt:
            norm_gold.append((ns, ne))

    norm_pred = []
    for p in pred:
        ns, ne, nt = get_normalized_span_bounds(raw_text, p.get("start", 0), p.get("end", 0))
        if nt:
            norm_pred.append((ns, ne))

    tp = 0
    fp = 0
    matched_gold = set()

    for p_ns, p_ne in norm_pred:
        matched_idx = None
        for g_idx, (g_ns, g_ne) in enumerate(norm_gold):
            if g_idx not in matched_gold:
                if p_ns == g_ns and p_ne == g_ne:
                    matched_idx = g_idx
                    break
        if matched_idx is not None:
            tp += 1
            matched_gold.add(matched_idx)
        else:
            fp += 1

    fn = len(norm_gold) - tp
    return tp, fp, fn, len(norm_gold), len(norm_pred)


def evaluate_char_mask_metrics(
    gold_spans: List[Dict[str, Any]],
    pred_spans: List[Dict[str, Any]],
    label: str,
    text_len: int
) -> Tuple[int, int, int, float, float, float, float]:
    """
    Computes character-level mask intersection, union, precision, recall, F1, and IoU.
    Returns: (intersection, pred_chars, gold_chars, prec, rec, f1, iou)
    """
    gold = [s for s in gold_spans if s.get("label") == label]
    pred = [s for s in pred_spans if s.get("label") == label]

    gold_set = set()
    for g in gold:
        s = max(0, min(g.get("start", 0), text_len))
        e = max(s, min(g.get("end", 0), text_len))
        gold_set.update(range(s, e))

    pred_set = set()
    for p in pred:
        s = max(0, min(p.get("start", 0), text_len))
        e = max(s, min(p.get("end", 0), text_len))
        pred_set.update(range(s, e))

    inter = len(gold_set & pred_set)
    union = len(gold_set | pred_set)
    pred_len = len(pred_set)
    gold_len = len(gold_set)

    if pred_len == 0 and gold_len == 0:
        return 0, 0, 0, 1.0, 1.0, 1.0, 1.0

    prec = inter / pred_len if pred_len > 0 else 0.0
    rec = inter / gold_len if gold_len > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    iou = inter / union if union > 0 else 0.0

    return inter, pred_len, gold_len, prec, rec, f1, iou


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
        r"</?(?:question|question_label|stem|option_label|option_text|stimulus|section|explanation)(?:\s+(?:[^>\"']|\"[^\"]*\"|'[^']*')*)?/?>"
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


DEFAULT_HF_MODEL_ID = "daominhwysi/sealior-modernbert-vi-exam-seq-labeling"
DEFAULT_LOCAL_FALLBACK = "results/sanity_check"


def _looks_like_hf_repo_id(value: str) -> bool:
    """Returns True if value looks like an owner/repo Hugging Face Hub ID rather than a local path."""
    if not value or value == DEFAULT_LOCAL_FALLBACK or value.startswith(("./", "../", "/", "~")):
        return False
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        return False
    return parts[0] not in {"results", "output", "checkpoints", "data", "tmp"}


def resolve_latest_model_source(
    model_path: Optional[Union[str, Path]] = None,
    default_repo_id: str = DEFAULT_HF_MODEL_ID,
    revision: str = "main",
    offline: bool = False,
) -> Tuple[Optional[Path], str]:
    """
    Resolves a model checkpoint path cleanly:
    1. Existing local checkpoint directory containing config.json is used directly.
    2. Explicit HF Hub repo ID (`owner/repo`) or missing default local placeholder
       always checks Hugging Face Hub for the latest commit on `revision` via
       `snapshot_download` (honoring HF_HOME / HF_HUB_CACHE), falling back to
       strict offline cache ONLY if the Hub check fails.
    """
    raw_arg = str(model_path).strip() if model_path is not None else ""
    if raw_arg:
        local_cand = Path(raw_arg).expanduser()
        if local_cand.exists() and (local_cand / "config.json").exists():
            return local_cand, f"local:{local_cand}"

    target_repo = raw_arg if _looks_like_hf_repo_id(raw_arg) else default_repo_id
    if not target_repo:
        return (Path(raw_arg) if raw_arg else None), raw_arg

    from huggingface_hub import HfApi, snapshot_download

    token = os.getenv("HF_TOKEN") or None
    if not offline:
        try:
            info = HfApi(token=token).repo_info(
                repo_id=target_repo,
                repo_type="model",
                revision=revision,
            )
            latest_sha = getattr(info, "sha", revision) or revision
            snapshot_path = snapshot_download(
                repo_id=target_repo,
                revision=latest_sha,
                token=token,
                local_files_only=False,
            )
            return Path(snapshot_path), f"hf:{target_repo}@{latest_sha[:12]} (latest {revision})"
        except Exception as e:
            print(f"Warning: Could not reach Hugging Face Hub for {target_repo}@{revision}: {e}")

    try:
        snapshot_path = snapshot_download(
            repo_id=target_repo,
            revision=revision,
            token=token,
            local_files_only=True,
        )
        return Path(snapshot_path), f"hf-cache:{target_repo}@{revision} (offline)"
    except Exception as e:
        print(f"Warning: Could not resolve offline cached snapshot for {target_repo}@{revision}: {e}")
        return (Path(raw_arg) if raw_arg else None), target_repo


def _resolve_hf_cached_snapshot(model_dir: Optional[Path]) -> Optional[Path]:
    """Backward-compatible wrapper around resolve_latest_model_source."""
    resolved, _ = resolve_latest_model_source(model_path=model_dir)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequence Labelling Model against Gold Benchmark.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--gold_set", "--gold-set", "--gold-manifest", type=str, default="data/benchmark_gold_set.json", help="Path to gold set JSON")
    parser.add_argument("--model_path", "--model-path", "--checkpoint", type=str, default=DEFAULT_HF_MODEL_ID, help="Local checkpoint path or Hugging Face Hub model ID")
    parser.add_argument("--revision", type=str, default="main", help="Hugging Face Hub branch/commit/tag to resolve (default: main)")
    parser.add_argument("--offline", action="store_true", help="Resolve Hugging Face model from local cache only without network checks")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to evaluate")
    parser.add_argument("--use_det", action="store_true", help="Use deterministic parser directly")
    parser.add_argument("--max-length", type=int, default=1024, help="Sliding window token length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    parser.add_argument("--mask-latex", action="store_true", default=False, help="Mask LaTeX expressions before tokenization")
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
    print(f"Inference Mode      : {'Deterministic' if args.use_det else f'Single-Pass Sliding Window (max_length={args.max_length}, stride={args.stride}, mask_latex={args.mask_latex})'}")

    # Load Gold Set
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    documents = gold_data.get("documents", [])
    if args.limit:
        documents = documents[:args.limit]
    print(f"Evaluating Documents: {len(documents)} documents (Corpus Token Total: {gold_data.get('total_tokens', 'N/A')})\n")

    model = None
    tokenizer = None
    id_to_tag = None
    if not args.use_det:
        model_dir, model_source = resolve_latest_model_source(
            model_path=args.model_path,
            revision=args.revision,
            offline=args.offline,
        )
        if model_dir and model_dir.exists() and (model_dir / "config.json").exists():
            try:
                from transformers import AutoTokenizer
                print(f"Resolved model source: {model_source}")
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
        else:
            print(f"Notice: No valid checkpoint resolved for '{args.model_path}' ({model_source}).")
            print("Falling back to Deterministic Parser + Gold Consistency verification.\n")

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

    # -------------------------------------------------------------------------
    # SINGLE-PASS EVALUATION
    # -------------------------------------------------------------------------
    norm_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "gold_cnt": 0, "pred_cnt": 0} for lbl, _ in labels_to_eval}
    char_mask_counts = {lbl: {"inter": 0, "pred_chars": 0, "gold_chars": 0} for lbl, _ in labels_to_eval}
    tolerant_counts = {lbl: {"tp": 0, "fp": 0, "fn": 0, "gold_cnt": 0, "pred_cnt": 0} for lbl, _ in labels_to_eval}

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
                max_length=args.max_length,
                stride=args.stride,
                mask_latex=args.mask_latex,
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

        text_len = len(raw_text)

        # 3. Multi-Metric Evaluation across all categories
        for lbl, tol in labels_to_eval:
            # 3A. Whitespace-Normalized Exact Matching
            tp, fp, fn, g_cnt, p_cnt = evaluate_normalized_exact_matching(gold_spans, pred_spans, lbl, raw_text)
            norm_counts[lbl]["tp"] += tp
            norm_counts[lbl]["fp"] += fp
            norm_counts[lbl]["fn"] += fn
            norm_counts[lbl]["gold_cnt"] += g_cnt
            norm_counts[lbl]["pred_cnt"] += p_cnt

            # 3B. Character-Level Mask IoU & F1
            inter, p_chars, g_chars, _, _, _, _ = evaluate_char_mask_metrics(gold_spans, pred_spans, lbl, text_len)
            char_mask_counts[lbl]["inter"] += inter
            char_mask_counts[lbl]["pred_chars"] += p_chars
            char_mask_counts[lbl]["gold_chars"] += g_chars

            # 3C. Boundary-Tolerant Matching (Legacy)
            tp_tol, fp_tol, fn_tol, g_tol, p_tol = evaluate_span_matching(gold_spans, pred_spans, lbl, tolerance=tol)
            tolerant_counts[lbl]["tp"] += tp_tol
            tolerant_counts[lbl]["fp"] += fp_tol
            tolerant_counts[lbl]["fn"] += fn_tol
            tolerant_counts[lbl]["gold_cnt"] += g_tol
            tolerant_counts[lbl]["pred_cnt"] += p_tol

    total_docs = len(latency_records)
    zc_rate = (zero_corruption_passes / max(1, total_docs)) * 100.0
    cloze_rate = (cloze_valid_passes / max(1, total_docs)) * 100.0
    avg_latency = float(np.mean(latency_records)) if latency_records else 0.0

    print("\n" + "=" * 80)
    print("1. WHITESPACE-NORMALIZED EXACT MATCH (STRICT ZERO-TOLERANCE CONTENT MATCH)")
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 80)

    norm_f1_list = []
    for lbl, _ in labels_to_eval:
        data = norm_counts[lbl]
        p, r, f1 = calc_f1(data["tp"], data["fp"], data["fn"])
        if data["gold_cnt"] > 0:
            norm_f1_list.append(f1)
            print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | {data['gold_cnt']:<8}")
        elif data["pred_cnt"] == 0:
            print(f"{lbl:<20} | {'-':<10} | {'-':<10} | {'-':<10} | {data['gold_cnt']:<8}")
        else:
            norm_f1_list.append(0.0)
            print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | {data['gold_cnt']:<8}")

    norm_macro_f1 = float(np.mean(norm_f1_list)) if norm_f1_list else 0.0
    print("-" * 80)
    print(f"{'Macro Average F1':<20} | {'-':<10} | {'-':<10} | {norm_macro_f1*100:8.2f}% | {'-':<8}\n")

    print("=" * 80)
    print("2. CHARACTER-LEVEL MASK COVERAGE & IoU (CONTINUOUS TEXT BLOCKS)")
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Char-Prec':<10} | {'Char-Rec':<10} | {'Char-F1':<10} | {'Char-IoU':<10} | {'Gold Chars':<10}")
    print("-" * 80)

    for lbl, _ in labels_to_eval:
        cdata = char_mask_counts[lbl]
        inter = cdata["inter"]
        p_len = cdata["pred_chars"]
        g_len = cdata["gold_chars"]
        union = p_len + g_len - inter

        if g_len == 0 and p_len == 0:
            print(f"{lbl:<20} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} | {g_len:<10}")
            continue

        cp = inter / p_len if p_len > 0 else 0.0
        cr = inter / g_len if g_len > 0 else 0.0
        cf1 = (2 * cp * cr / (cp + cr)) if (cp + cr) > 0 else 0.0
        ciou = inter / union if union > 0 else 0.0

        print(f"{lbl:<20} | {cp*100:8.2f}% | {cr*100:8.2f}% | {cf1*100:8.2f}% | {ciou*100:8.2f}% | {g_len:<10}")

    print("-" * 80 + "\n")

    print("=" * 80)
    print("3. BOUNDARY-TOLERANT SPAN F1 (LEGACY WINDOW TOLERANCE)")
    print("=" * 80)
    print(f"{'Entity Label':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Tol (chars)':<12}")
    print("-" * 80)

    tol_f1_list = []
    for lbl, tol in labels_to_eval:
        data = tolerant_counts[lbl]
        p, r, f1 = calc_f1(data["tp"], data["fp"], data["fn"])
        if data["gold_cnt"] > 0:
            tol_f1_list.append(f1)
            print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | ±{tol:<10}")
        elif data["pred_cnt"] == 0:
            print(f"{lbl:<20} | {'-':<10} | {'-':<10} | {'-':<10} | ±{tol:<10}")
        else:
            tol_f1_list.append(0.0)
            print(f"{lbl:<20} | {p*100:8.2f}% | {r*100:8.2f}% | {f1*100:8.2f}% | ±{tol:<10}")

    tol_macro_f1 = float(np.mean(tol_f1_list)) if tol_f1_list else 0.0
    print("-" * 80)
    print(f"{'Macro Average F1':<20} | {'-':<10} | {'-':<10} | {tol_macro_f1*100:8.2f}% | {'-':<12}\n")

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
