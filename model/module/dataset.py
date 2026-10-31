#!/usr/bin/env python3
"""
PyTorch Dataset Loaders for Sequence Labelling:
1. MultiWindowBIODataset: Streaming BIO loader with word-level sliding windows for records longer than max_length.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


DEFAULT_BASE_TAGS = [
    "question_label",
    "stem",
    "option_label",
    "option_text",
    "stimulus",
    "section",
    "explanation"
]


def get_tag_mappings(base_tags: Optional[List[str]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generates standard BIO tag mapping dictionaries."""
    tags = base_tags or DEFAULT_BASE_TAGS
    tag_to_id = {"O": 0}
    for t in tags:
        tag_to_id[f"B-{t}"] = len(tag_to_id)
        tag_to_id[f"I-{t}"] = len(tag_to_id)
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag


def normalize_bio_tag(tag: str) -> str:
    """
    Normalizes BIO tags to canonical format:
    e.g. 'B-QUESTION_LABEL' -> 'B-question_label', 'I-STEM' -> 'I-stem', 'O' -> 'O'.
    Protects against case mismatches between dataset generation and tag mappings.
    """
    if not tag or str(tag).strip().upper() == "O":
        return "O"
    tag_str = str(tag).strip()
    if tag_str.startswith(("B-", "b-", "I-", "i-")):
        return f"{tag_str[:2].upper()}{tag_str[2:].lower()}"
    return tag_str.lower()


def classify_latex(formula_str: str) -> str:
    r"""
    Classify LaTeX expression into typed tokens based on structural role and length:
    - [MATH_CALC]: contains '=', r'\approx', r'\equiv', etc.
    - [MATH_VAR_SHORT]: len <= 3 (e.g. $x$, $y_1$)
    - [MATH_VAR_MED]: 4 <= len <= 10 (e.g. $f(x)$, $x^2+1$)
    - [MATH_EXPR_SHORT]: 11 <= len <= 30
    - [MATH_EXPR_LONG]: len > 30
    """
    content = formula_str.strip("$").strip()
    if any(ind in content for ind in ["=", r"\approx", r"\equiv"]):
        return "[MATH_CALC]"
    length = len(content)
    if length <= 3:
        return "[MATH_VAR_SHORT]"
    elif length <= 10:
        return "[MATH_VAR_MED]"
    elif length <= 30:
        return "[MATH_EXPR_SHORT]"
    else:
        return "[MATH_EXPR_LONG]"


def _word_windows(word_lengths: List[int], budget: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Greedy word-level windows over per-word subword counts. Each window holds at most
    `budget` subword tokens; the next window starts about `overlap` tokens before the end
    of the previous one. A single word longer than the budget is kept alone.
    """
    n = len(word_lengths)
    if n == 0:
        return []
    if sum(word_lengths) <= budget:
        return [(0, n)]

    windows: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = start
        total = 0
        while end < n and total + word_lengths[end] <= budget:
            total += word_lengths[end]
            end += 1
        if end == start:
            end = start + 1
        windows.append((start, end))
        if end >= n:
            break

        # Step back from `end` while the overlapped words fit in the overlap budget
        next_start = end
        back = 0
        while next_start - 1 > start and back + word_lengths[next_start - 1] <= overlap:
            next_start -= 1
            back += word_lengths[next_start]
        start = max(next_start, start + 1)
    return windows


class MultiWindowBIODataset(Dataset):
    """
    Memory-efficient, lazy offset-indexed BIO dataset loader. Records longer than
    `max_length` subword tokens are split into overlapping word-level windows.
    Maintains a small index in RAM and reads lines on-demand, preventing out-of-memory
    crashes in multi-process/DDP distributed environments.
    """
    def __init__(
        self,
        chunks_file_path: Union[str, Path],
        tag_to_id: Dict[str, int],
        tokenizer: Optional[Any] = None,
        pad_token_id: int = 0,
        ignore_label_id: int = -100,
        max_samples: Optional[int] = None,
        max_length: int = 8192,
        label_all_subwords: bool = True,
        window_overlap_tokens: int = 1024,
        doc_filter: Optional[Callable[[str], bool]] = None,
        record_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.chunks_file_path = Path(chunks_file_path)
        self.tag_to_id = tag_to_id
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length
        self.label_all_subwords = label_all_subwords
        self.window_overlap_tokens = window_overlap_tokens
        self._file = None

        if not self.chunks_file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file_path}")

        # Build binary byte offsets and cached sequence lengths for zero-copy lazy loading
        offsets = []
        raw_lengths = []
        with open(self.chunks_file_path, "rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    item = json.loads(line)
                    doc_ok = (doc_filter is None or doc_filter(item.get("doc_id", "")))
                    rec_ok = (record_filter is None or record_filter(item))
                    if doc_ok and rec_ok:
                        offsets.append(offset)
                        if "input_ids" in item:
                            raw_lengths.append(len(item["input_ids"]))
                        else:
                            raw_lengths.append(int(len(item.get("tokens", [])) * 1.35))
                        if max_samples and len(offsets) >= max_samples:
                            break
                offset = f.tell()
                line = f.readline()

        import numpy as np
        # Each index row is (byte_offset, word_start, word_end). word_start == -1 means the
        # whole record is used as-is (pre-tokenized records, pre-chunked records, or no tokenizer).
        is_pretokenized = offsets and self._is_pretokenized(offsets[0])
        is_already_chunked = offsets and self._is_already_chunked(offsets[0])

        if tokenizer is not None and offsets and not is_pretokenized and not is_already_chunked:
            self.index = np.array(self._build_windows(offsets), dtype=np.int64).reshape(-1, 3)
            self._seq_lengths = np.array(
                [int((we - ws) * 1.35) if ws >= 0 and we >= ws else 0 for _, ws, we in self.index],
                dtype=np.int32,
            )
        else:
            self.index = np.array([(off, -1, -1) for off in offsets], dtype=np.int64).reshape(-1, 3)
            self._seq_lengths = np.array(raw_lengths, dtype=np.int32)
        self._pid = None
        print(
            f"Indexed {len(offsets)} BIO chunks into {len(self.index)} windows (re-slicing={'DISABLED' if is_already_chunked or is_pretokenized else 'ENABLED'}) "
            f"from '{self.chunks_file_path.name}' (RAM: {self.index.nbytes / (1024 * 1024):.2f} MB)."
        )

    def _is_pretokenized(self, offset: int) -> bool:
        with open(self.chunks_file_path, "r", encoding="utf-8") as f:
            f.seek(int(offset))
            return "input_ids" in json.loads(f.readline())

    def _is_already_chunked(self, offset: int) -> bool:
        """
        Detects if records have already been sliced into sliding chunks at dataset build stage
        (e.g., contains 'chunk_id', 'window_size', or 'char_start'), avoiding redundant 2nd slicing.
        """
        with open(self.chunks_file_path, "r", encoding="utf-8") as f:
            f.seek(int(offset))
            item = json.loads(f.readline())
            return "chunk_id" in item or "window_size" in item or "start_token_idx" in item

    def _build_windows(self, offsets: List[int], batch_size: int = 256) -> List[Tuple[int, int, int]]:
        """
        Split every record into word-level windows that fit within `max_length` subword tokens.
        Consecutive windows overlap by about `window_overlap_tokens`, so no word is seen only
        at a window edge. Records that already fit produce exactly one window.
        """
        budget = self.max_length - self.tokenizer.num_special_tokens_to_add(False)
        if budget <= 0:
            raise ValueError(f"max_length={self.max_length} leaves no room for content tokens")

        windows: List[Tuple[int, int, int]] = []
        pending: List[Tuple[int, List[str]]] = []

        def flush():
            if not pending:
                return
            encoded = self.tokenizer(
                [tokens for _, tokens in pending],
                is_split_into_words=True,
                add_special_tokens=False,
            )
            for batch_idx, (off, tokens) in enumerate(pending):
                counts = [0] * len(tokens)
                for word_idx in encoded.word_ids(batch_idx):
                    if word_idx is not None:
                        counts[word_idx] += 1
                for ws, we in _word_windows(counts, budget, self.window_overlap_tokens):
                    windows.append((off, ws, we))
            pending.clear()

        with open(self.chunks_file_path, "r", encoding="utf-8") as f:
            for off in offsets:
                f.seek(int(off))
                item = json.loads(f.readline())
                tokens = item.get("tokens", [])
                if not tokens:
                    windows.append((off, 0, 0))
                    continue
                pending.append((off, tokens))
                if len(pending) >= batch_size:
                    flush()
        flush()
        return windows

    def _get_file(self):
        current_pid = os.getpid()
        if getattr(self, "_pid", None) != current_pid:
            self._file = None
            self._pid = current_pid
        if self._file is None or self._file.closed:
            self._file = open(self.chunks_file_path, "r", encoding="utf-8")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None
        self._pid = None

    def _lookup_tag_id(self, tag: str) -> int:
        norm = normalize_bio_tag(tag)
        if norm in self.tag_to_id:
            return self.tag_to_id[norm]
        if tag in self.tag_to_id:
            return self.tag_to_id[tag]
        return self.tag_to_id.get("O", 0)

    def __len__(self) -> int:
        return len(self.index)

    def get_sequence_length(self, idx: int) -> int:
        """
        Fast approximate sequence length for bucketing without full tokenization.
        Uses cached sequence length computed during initial index pass.
        """
        if hasattr(self, "_seq_lengths") and idx < len(self._seq_lengths) and self._seq_lengths[idx] > 0:
            return int(self._seq_lengths[idx])
        _, word_start, word_end = self.index[idx]
        if word_start >= 0 and word_end >= word_start:
            # Approximate subwords ~ 1.35x words
            return int((word_end - word_start) * 1.35)
        # Fallback for whole items
        f = self._get_file()
        offset = int(self.index[idx][0])
        f.seek(offset)
        line = f.readline()
        item = json.loads(line)
        if "input_ids" in item:
            return len(item["input_ids"])
        return int(len(item.get("tokens", [])) * 1.35)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_file()
        offset, word_start, word_end = (int(v) for v in self.index[idx])
        f.seek(offset)
        line = f.readline()
        item = json.loads(line)

        tokens = item.get("tokens", [])
        bio_tags = item.get("ner_tags", item.get("tags", item.get("labels", [])))
        if word_start >= 0:
            tokens = tokens[word_start:word_end]
            bio_tags = bio_tags[word_start:word_end]

        if "input_ids" in item:
            input_ids = item["input_ids"]
            if bio_tags and isinstance(bio_tags[0], int):
                label_ids = bio_tags
            else:
                label_ids = [self._lookup_tag_id(t) for t in bio_tags]
            attention_mask = item.get("attention_mask", [1] * len(input_ids))
        elif self.tokenizer is not None:
            tokenized = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=False
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            try:
                word_ids = tokenized.word_ids()
            except Exception:
                word_ids = None

            if word_ids is not None:
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(self.ignore_label_id)
                    elif word_idx != previous_word_idx:
                        if word_idx < len(bio_tags):
                            tag_str = bio_tags[word_idx]
                            label_ids.append(self._lookup_tag_id(tag_str))
                        else:
                            label_ids.append(self.ignore_label_id)
                    else:
                        if self.label_all_subwords and word_idx < len(bio_tags):
                            tag_str = bio_tags[word_idx]
                            norm = normalize_bio_tag(tag_str)
                            cont_tag = f"I-{norm[2:]}" if norm.startswith("B-") else norm
                            label_ids.append(self._lookup_tag_id(cont_tag))
                        else:
                            label_ids.append(self.ignore_label_id)
                    previous_word_idx = word_idx
            else:
                label_ids = [
                    self._lookup_tag_id(bio_tags[min(i, len(bio_tags) - 1)])
                    for i in range(len(input_ids))
                ]
        else:
            raise KeyError("Item must contain 'input_ids' or MultiWindowBIODataset must be initialized with a tokenizer.")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

    def __del__(self):
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
