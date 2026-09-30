#!/usr/bin/env python3
"""
High-Throughput Sliding-Window Sequence Labelling Inference Engine.
Segments long Vietnamese examination documents into structured components
(question_label, stem, option_label, option_text, stimulus, section, explanation)
with zero character corruption.
"""

import os
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import re
import json
import math
import bisect
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

from model.module.head import EnhancedBertForTokenClassification


def is_valid_latex(content: str) -> bool:
    """Validates if content enclosed in $...$ or $$...$$ is likely math/latex."""
    content_stripped = content.strip()
    if not content_stripped:
        return False
    if len(content_stripped) == 1:
        return content_stripped.isalnum()
    if re.match(r'^[\[\(][^\[\]\(\)]+[\]\)]$', content_stripped) and (';' in content_stripped or ',' in content_stripped):
        return True
    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []
    for char in content_stripped:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack:
                return False
            last = stack.pop()
            if brackets[last] != char:
                return False
    if stack:
        return False
    math_indicators = ['\\', '^', '_', '+', '-', '*', '/', '=', '<', '>', '{', '}', '[', ']']
    if any(ind in content_stripped for ind in math_indicators):
        return True
    if len(content_stripped) < 10 and re.match(r'^[a-zA-Z0-9]+$', content_stripped):
        return True
    return False


def get_latex_spans(text: str) -> List[Tuple[int, int]]:
    """Finds non-overlapping character index ranges for math formulas."""
    spans = []
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        content = match.group(0)[2:-2]
        if is_valid_latex(content):
            spans.append(match.span())
    for match in re.finditer(r"\$(?!\s)[^\$\n]+?(?<!\s)\$", text):
        span = match.span()
        content = match.group(0)[1:-1]
        if not is_valid_latex(content):
            continue
        overlap = False
        for d_start, d_end in spans:
            if not (span[1] <= d_start or span[0] >= d_end):
                overlap = True
                break
        if not overlap:
            spans.append(span)
    spans.sort(key=lambda x: x[0])
    return spans


def build_tagged_xml(raw_text: str, segments: List[Dict[str, Any]]) -> str:
    """
    Constructs inline-tagged XML string without mutating original characters:
        <question_label>Câu 1.</question_label> <stem>Nội dung...</stem>
    """
    sorted_segs = sorted(segments, key=lambda x: (x.get("start", 0), -x.get("end", 0)))
    result = []
    cursor = 0
    for seg in sorted_segs:
        start = seg.get("start", -1)
        end = seg.get("end", -1)
        label = seg["label"]
        text = seg.get("text", "")
        if start >= 0 and end >= 0:
            if start < cursor:
                continue
            if start > cursor:
                result.append(raw_text[cursor:start])
            sub_text = raw_text[start:end]
            result.append(f"<{label}>{sub_text}</{label}>")
            cursor = end
        else:
            idx = raw_text.find(text, cursor)
            if idx == -1:
                continue
            if idx > cursor:
                result.append(raw_text[cursor:idx])
            result.append(f"<{label}>{text}</{label}>")
            cursor = idx + len(text)
    if cursor < len(raw_text):
        result.append(raw_text[cursor:])
    return "".join(result)


reconstruct_xml_from_predictions = build_tagged_xml



def load_label_mapping(model_dir: Union[str, Path], revision: Optional[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Loads tag-to-id and id-to-tag mappings from model directory or dataset outputs."""
    model_path = Path(model_dir)
    mapping_candidates = [
        model_path / "label_mapping.json",
        Path("data/training_dataset/label_mapping.json"),
        Path("output/dataset/label_mapping.json")
    ]
    for p in mapping_candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                tag_to_id = data["tag_to_id"]
                id_to_tag = {int(k): v for k, v in data["id_to_tag"].items()}
                return tag_to_id, id_to_tag
            except Exception:
                pass

    if not os.path.isdir(str(model_dir)):
        try:
            from transformers.utils.hub import cached_file
            cached_map = cached_file(str(model_dir), "label_mapping.json", revision=revision)
            if cached_map and os.path.exists(cached_map):
                with open(cached_map, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tag_to_id = data["tag_to_id"]
                    id_to_tag = {int(k): v for k, v in data["id_to_tag"].items()}
                    return tag_to_id, id_to_tag
        except Exception:
            pass

    # Default fallback mapping
    base_tags = ["question_label", "stem", "option_label", "option_text", "stimulus", "section", "explanation"]
    tag_to_id = {"O": 0}
    for tag in base_tags:
        tag_to_id[f"B-{tag}"] = len(tag_to_id)
        tag_to_id[f"I-{tag}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag




def predict_text(
    text: str,
    model: Any,
    tokenizer: Any,
    id_to_tag: Dict[int, str],
    device: str = "cpu",
    max_length: int = 1024,
    stride: int = 256,
    batch_size: int = 8,
    mask_latex: bool = False,
    use_viterbi: bool = True
) -> Dict[str, Any]:
    """
    Runs sequence labeling on a raw text document with sliding window, batched GPU execution,
    center-weighted overlapping logit pooling, and Constrained Viterbi BIO decoding.
    """
    model.eval()
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    if mask_latex:
        latex_spans = get_latex_spans(clean_text)
        processed_text = ""
        last_idx = 0
        for start, end in latex_spans:
            processed_text += clean_text[last_idx:start] + "[LATEX]"
            last_idx = end
        processed_text += clean_text[last_idx:]

        # Precompute interval mapping boundaries for O(log M) binary search
        mod_intervals = []
        cur_mod = 0
        cur_orig = 0
        latex_len = len("[LATEX]")
        for o_start, o_end in latex_spans:
            text_seg_len = o_start - cur_orig
            if text_seg_len > 0:
                mod_intervals.append((cur_mod, cur_mod + text_seg_len, cur_orig, False))
                cur_mod += text_seg_len
                cur_orig = o_start
            mod_intervals.append((cur_mod, cur_mod + latex_len, cur_orig, True))
            cur_mod += latex_len
            cur_orig = o_end
        mod_intervals.append((cur_mod, float("inf"), cur_orig, False))
        mod_starts = [x[0] for x in mod_intervals]

        def map_idx(idx: int) -> int:
            i = bisect.bisect_right(mod_starts, idx) - 1
            if i < 0:
                return 0
            m_start, _, o_start, is_latex = mod_intervals[i]
            if is_latex:
                return o_start
            return o_start + (idx - m_start)
    else:
        processed_text = clean_text
        map_idx = lambda x: x

    enc = tokenizer(
        processed_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    all_input_ids = enc["input_ids"]
    all_offsets = enc["offset_mapping"]
    all_attention_mask = enc.get("attention_mask", [1] * len(all_input_ids))

    chunks_ids = []
    chunks_offsets = []
    chunks_attention_mask = []

    idx = 0
    n = len(all_input_ids)
    if n == 0:
        chunks_ids = [[]]
        chunks_offsets = [[]]
        chunks_attention_mask = [[]]
    else:
        while idx < n:
            end = min(idx + max_length, n)
            chunks_ids.append(all_input_ids[idx:end])
            chunks_offsets.append(all_offsets[idx:end])
            chunks_attention_mask.append(all_attention_mask[idx:end])
            if end >= n:
                break
            idx += stride

    num_chunks = len(chunks_ids)
    span_logits: Dict[Tuple[int, int], List[Tuple[torch.Tensor, float]]] = {}

    with torch.no_grad():
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for b_start in range(0, num_chunks, batch_size):
            b_end = min(b_start + batch_size, num_chunks)
            chunk_slice_ids = chunks_ids[b_start:b_end]
            chunk_slice_att = chunks_attention_mask[b_start:b_end]
            b_max_len = max(len(c) for c in chunk_slice_ids) if chunk_slice_ids else 0

            b_input_ids = torch.full((len(chunk_slice_ids), b_max_len), fill_value=pad_id, dtype=torch.long, device=device)
            b_attention_mask = torch.zeros((len(chunk_slice_ids), b_max_len), dtype=torch.long, device=device)

            for i, (inp, att) in enumerate(zip(chunk_slice_ids, chunk_slice_att)):
                b_input_ids[i, :len(inp)] = torch.tensor(inp, dtype=torch.long, device=device)
                b_attention_mask[i, :len(att)] = torch.tensor(att, dtype=torch.long, device=device)

            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            b_logits = outputs.logits.cpu()  # [B, seq_len, num_labels]

            for chunk_offset_in_batch, chunk_idx in enumerate(range(b_start, b_end)):
                chunk_offsets = chunks_offsets[chunk_idx]
                logits = b_logits[chunk_offset_in_batch]
                chunk_len = len(chunk_offsets)

                for tok_idx, (start, end) in enumerate(chunk_offsets):
                    if start == end:
                        continue
                    orig_start = map_idx(start)
                    orig_end = map_idx(end)
                    if orig_start >= orig_end:
                        continue

                    # Center-weighted pooling: edge tokens receive lower weight than central tokens
                    dist_to_edge = min(tok_idx, chunk_len - 1 - tok_idx)
                    weight = min(1.0, (dist_to_edge + 1) / 32.0)

                    key = (orig_start, orig_end)
                    if key not in span_logits:
                        span_logits[key] = []
                    span_logits[key].append((logits[tok_idx], weight))

    # Weighted average logits for overlapping subwords
    sorted_spans = sorted(span_logits.keys(), key=lambda x: x[0])
    if not sorted_spans:
        return {
            "raw_text": clean_text,
            "xml_text": clean_text,
            "spans": [],
            "num_tokens": 0
        }

    stacked_logits = []
    for span in sorted_spans:
        items = span_logits[span]
        total_w = sum(w for _, w in items)
        if total_w > 0.0:
            weighted_logits = sum(l * w for l, w in items) / total_w
        else:
            weighted_logits = items[0][0]
        stacked_logits.append(weighted_logits)

    if use_viterbi:
        from model.inference.viterbi import ConstrainedViterbiDecoder
        decoder = ConstrainedViterbiDecoder(id_to_tag)
        pred_ids = decoder.decode(torch.stack(stacked_logits))
    else:
        pred_ids = [int(torch.argmax(l).item()) for l in stacked_logits]

    token_predictions = []
    for span, pred_id in zip(sorted_spans, pred_ids):
        tag = id_to_tag.get(pred_id, "O")
        token_predictions.append({
            "start": span[0],
            "end": span[1],
            "tag": tag,
            "text": clean_text[span[0]:span[1]]
        })

    # Word-level BIO consistency:
    # A continuation subword inside an alphanumeric word cannot start a new entity or switch labels.
    for i in range(1, len(token_predictions)):
        prev_tok = token_predictions[i - 1]
        curr_tok = token_predictions[i]
        if curr_tok["start"] == prev_tok["end"]:
            if (curr_tok["start"] > 0 and 
                clean_text[curr_tok["start"] - 1].isalnum() and 
                clean_text[curr_tok["start"]].isalnum()):
                prev_tag = prev_tok["tag"]
                if prev_tag != "O":
                    prev_label = prev_tag.split("-")[-1]
                    curr_tok["tag"] = f"I-{prev_label}"
                else:
                    curr_tok["tag"] = "O"

    # Group BIO tokens into entity segments
    segments = []
    current_seg: Optional[Dict[str, Any]] = None

    for tok in token_predictions:
        tag = tok["tag"]
        start = tok["start"]
        end = tok["end"]

        if tag.startswith("B-"):
            label = tag[2:]
            # Subword continuity: merge if same label and directly adjacent without separation
            if current_seg and current_seg["label"] == label and start == current_seg["end"]:
                current_seg["end"] = end
                current_seg["text"] = clean_text[current_seg["start"]:end]
            else:
                if current_seg:
                    segments.append(current_seg)
                current_seg = {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": clean_text[start:end]
                }
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_seg and current_seg["label"] == label:
                current_seg["end"] = end
                current_seg["text"] = clean_text[current_seg["start"]:end]
            else:
                if current_seg:
                    segments.append(current_seg)
                current_seg = {
                    "label": label,
                    "start": start,
                    "end": end,
                    "text": clean_text[start:end]
                }
        else:  # 'O'
            if current_seg:
                segments.append(current_seg)
                current_seg = None

    if current_seg:
        segments.append(current_seg)

    xml_text = build_tagged_xml(clean_text, segments)

    return {
        "raw_text": clean_text,
        "xml_text": xml_text,
        "spans": segments,
        "num_tokens": len(sorted_spans)
    }


class SequenceLabelPredictor:
    """
    High-level predictor class for token classification and XML reconstruction on raw texts.
    """
    def __init__(
        self,
        model_dir: Union[str, Path] = "results/mmbert_seqlabel_v2",
        device: Optional[str] = None,
        max_length: int = 1024,
        stride: int = 256,
        batch_size: int = 8,
        mask_latex: bool = False,
        use_viterbi: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.mask_latex = mask_latex
        self.use_viterbi = use_viterbi
        self.model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.tag_to_id, self.id_to_tag = load_label_mapping(self.model_dir)
        self.model = EnhancedBertForTokenClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.tag_to_id),
            id2label=self.id_to_tag,
            label2id=self.tag_to_id
        ).to(self.device)

    def predict(self, text: str) -> Dict[str, Any]:
        return predict_text(
            text=text,
            model=self.model,
            tokenizer=self.tokenizer,
            id_to_tag=self.id_to_tag,
            device=self.device,
            max_length=self.max_length,
            stride=self.stride,
            batch_size=self.batch_size,
            mask_latex=self.mask_latex,
            use_viterbi=self.use_viterbi
        )


def main():
    parser = argparse.ArgumentParser(description="Run sliding-window sequence labelling inference.")
    parser.add_argument("--model-dir", type=str, default="results/mmbert_seqlabel_v2", help="Path to model directory")
    parser.add_argument("--text", type=str, default=None, help="Input raw exam string")
    parser.add_argument("--file", type=str, default=None, help="Input markdown/text file path")
    parser.add_argument("--output", type=str, default=None, help="Path to save output XML file")
    parser.add_argument("--max-length", type=int, default=1024, help="Sliding window token length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for sliding window chunks")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-viterbi", action="store_true", default=False, help="Disable Constrained Viterbi decoding and use naive argmax")
    args = parser.parse_args()

    input_text = args.text
    if not input_text and args.file:
        input_text = Path(args.file).read_text(encoding="utf-8")
    if not input_text:
        input_text = "Câu 1: Cho hàm số $y = f(x)$. Giá trị nhỏ nhất là?\nA. 1\nB. 2\nC. 3\nD. 4"

    print(f"Loading model and tokenizer from '{args.model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tag_to_id, id_to_tag = load_label_mapping(args.model_dir)

    model = EnhancedBertForTokenClassification.from_pretrained(
        args.model_dir,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id
    ).to(args.device)

    result = predict_text(
        input_text,
        model,
        tokenizer,
        id_to_tag,
        device=args.device,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
        use_viterbi=not args.no_viterbi
    )

    print("\n--- Predicted XML Output ---")
    print(result["xml_text"])

    if args.output:
        Path(args.output).write_text(result["xml_text"], encoding="utf-8")
        print(f"Saved predicted XML to '{args.output}'")


if __name__ == "__main__":
    main()
