#!/usr/bin/env python3
"""
PyTorch Dataset Loaders for Sequence Labelling:
1. MultiWindowBIODataset: Fast streaming loader for pre-tokenized multi-scale BIO chunks.
2. OnlineAugmentedDataset: Dynamic on-the-fly multi-scale sliding window dataset.
"""

import os
import re
import json
import bisect
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

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


def extract_layout_features(
    text: str,
    token_offsets: List[Tuple[int, int]]
) -> Dict[str, List[int]]:
    """
    Extracts 2D layout features for each token offset in raw markdown text:
    1. column_indices: character column index of token start on its line.
    2. col_buckets: bucket indices:
       - 0: col <= 1
       - 1: col <= 5
       - 2: col <= 10
       - 3: col <= 20
       - 4: col > 20
    3. is_line_start: 1 if token is the first non-whitespace element on its line, else 0.
    4. is_blank_preceded: 1 if current line is preceded by a blank line, else 0.
    """
    lines = text.split("\n")
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    column_indices = []
    col_buckets = []
    is_line_start = []
    is_blank_preceded = []

    for start, end in token_offsets:
        if start == end and start == 0:
            column_indices.append(0)
            col_buckets.append(0)
            is_line_start.append(0)
            is_blank_preceded.append(0)
            continue

        line_idx = bisect.bisect_right(line_starts, start) - 1
        if line_idx < 0:
            line_idx = 0
        line_offset = start - line_starts[line_idx]
        column_indices.append(line_offset)

        if line_offset <= 1:
            bucket = 0
        elif line_offset <= 5:
            bucket = 1
        elif line_offset <= 10:
            bucket = 2
        elif line_offset <= 20:
            bucket = 3
        else:
            bucket = 4
        col_buckets.append(bucket)

        line_text = lines[line_idx] if line_idx < len(lines) else ""
        first = 1 if line_text[:line_offset].strip() == "" else 0
        is_line_start.append(first)

        blank = 1 if (line_idx > 0 and lines[line_idx - 1].strip() == "") else 0
        is_blank_preceded.append(blank)

    return {
        "column_indices": column_indices,
        "col_buckets": col_buckets,
        "is_line_start": is_line_start,
        "is_blank_preceded": is_blank_preceded
    }


class MultiWindowBIODataset(Dataset):
    """
    Memory-efficient, lazy offset-indexed BIO dataset loader supporting multi-scale
    sliding window chunks (512, 768, 1024, 2048 tokens).
    Maintains a tiny index in RAM (< 1 MB) and reads lines on-demand, preventing
    out-of-memory (OOM) crashes in multi-process/DDP distributed environments.
    """
    def __init__(
        self,
        chunks_file_path: Union[str, Path],
        tag_to_id: Dict[str, int],
        tokenizer: Optional[Any] = None,
        pad_token_id: int = 0,
        ignore_label_id: int = -100,
        max_samples: Optional[int] = None,
        max_length: int = 2048,
        label_all_subwords: bool = True
    ):
        self.chunks_file_path = Path(chunks_file_path)
        self.tag_to_id = tag_to_id
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length
        self.label_all_subwords = label_all_subwords
        self._file = None

        if not self.chunks_file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file_path}")

        # Build binary byte offsets for zero-copy lazy loading
        offsets = []
        with open(self.chunks_file_path, "rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    offsets.append(offset)
                    if max_samples and len(offsets) >= max_samples:
                        break
                offset = f.tell()
                line = f.readline()

        import numpy as np
        self.offsets = np.array(offsets, dtype=np.int64)
        self._pid = None
        print(f"Indexed {len(self.offsets)} multi-scale BIO chunks from '{self.chunks_file_path.name}' (RAM: {self.offsets.nbytes / (1024 * 1024):.2f} MB).")

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
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_file()
        f.seek(int(self.offsets[idx]))
        line = f.readline()
        item = json.loads(line)

        tokens = item.get("tokens", [])
        bio_tags = item.get("ner_tags", item.get("tags", item.get("labels", [])))

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

        batch_dict = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }
        if "col_buckets" in item and "is_line_start" in item and "is_blank_preceded" in item:
            batch_dict["col_buckets"] = torch.tensor(item["col_buckets"], dtype=torch.long)
            batch_dict["is_line_start"] = torch.tensor(item["is_line_start"], dtype=torch.long)
            batch_dict["is_blank_preceded"] = torch.tensor(item["is_blank_preceded"], dtype=torch.long)
        elif "raw_text" in item and "offset_mapping" in item:
            layout_feats = extract_layout_features(item["raw_text"], item["offset_mapping"])
            batch_dict["col_buckets"] = torch.tensor(layout_feats["col_buckets"], dtype=torch.long)
            batch_dict["is_line_start"] = torch.tensor(layout_feats["is_line_start"], dtype=torch.long)
            batch_dict["is_blank_preceded"] = torch.tensor(layout_feats["is_blank_preceded"], dtype=torch.long)
        return batch_dict

    def __del__(self):
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


class OnlineAugmentedDataset(Dataset):
    """
    Dynamic Online PyTorch Dataset that reconstructs, augments, tokenizes,
    and aligns character spans on-the-fly across Multi-Scale Sliding Windows
    ((512, 128), (768, 192), (1024, 256), (2048, 512)).
    """
    def __init__(
        self,
        raw_items: List[Dict[str, Any]],
        tokenizer: Any,
        tag_to_id: Dict[str, int],
        is_train: bool = True,
        window_configs: Optional[List[Tuple[int, int]]] = None
    ):
        self.raw_items = raw_items
        self._tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.is_train = is_train
        self.window_configs = window_configs or [
            (512, 128),
            (768, 192),
            (1024, 256),
            (2048, 512)
        ]
        self.index_map = []
        if self._tokenizer is not None:
            self._build_index_map()

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok):
        self._tokenizer = tok
        if tok is not None and not self.index_map:
            self._build_index_map()

    def _build_index_map(self):
        from synthetic_exam_generator.reconstructor import reconstruct_exam, ReconstructorConfig
        base_config = ReconstructorConfig(randomize_q_num=False)
        self.index_map = []
        
        for doc_idx, item in enumerate(self.raw_items):
            if item.get("is_real", False) and "raw_text" in item:
                raw_text = item["raw_text"]
            elif "sections" in item:
                rec = reconstruct_exam(item, base_config)
                raw_text = rec["raw_text"]
            else:
                raw_text = item.get("raw_text", "")

            if not raw_text.strip():
                continue

            for max_len, stride in self.window_configs:
                tokenized = self._tokenizer(
                    raw_text,
                    return_offsets_mapping=False,
                    truncation=True,
                    max_length=max_len,
                    stride=stride,
                    return_overflowing_tokens=True,
                    add_special_tokens=True
                )
                num_chunks = len(tokenized["input_ids"])
                for chunk_idx in range(num_chunks):
                    self.index_map.append((doc_idx, max_len, stride, chunk_idx))

        split_name = "Train" if self.is_train else "Eval/Test"
        print(f"[{split_name}] Initialized {len(self.index_map)} multi-scale window chunks across {len(self.raw_items)} documents.")

    def __len__(self) -> int:
        return len(self.index_map) if self.index_map else len(self.raw_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from synthetic_exam_generator.reconstructor import reconstruct_exam, ReconstructorConfig
        from tools.dataset.build_training_dataset import align_spans_to_bio

        if not self.index_map:
            doc_idx = idx
            max_len, stride = (512, 128)
            target_chunk_idx = 0
        else:
            doc_idx, max_len, stride, target_chunk_idx = self.index_map[idx]

        item = self.raw_items[doc_idx]

        if self.is_train:
            aug_config = ReconstructorConfig(
                inline_option_prob=0.35,
                grid_2x2_prob=0.20,
                same_line_stem_options_prob=0.10,
                enable_permutations=True,
                prob_inline_barem=0.35,
                prob_answer_grid=0.15,
                prob_table_barem=0.10,
                prob_no_barem=0.40,
                space_noise_rate=0.035,
                casing_noise_prob=0.03,
                typo_rate=0.015,
                latex_mask_prob=0.05,
                admin_header_prob=0.40
            )
        else:
            aug_config = ReconstructorConfig(randomize_q_num=False, admin_header_prob=0.0)

        # Handle real vs synthetic data
        if item.get("is_real", False) and "raw_text" in item and "spans" in item:
            raw_text = item["raw_text"]
            spans = item["spans"]
        elif "sections" in item:
            rec = reconstruct_exam(item, aug_config)
            raw_text = rec["raw_text"]
            spans = rec["spans"]
        else:
            raw_text = item.get("raw_text", "")
            spans = item.get("spans", [])

        tokenized = self._tokenizer(
            raw_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            add_special_tokens=True
        )

        num_chunks = len(tokenized["input_ids"])
        chunk_idx = target_chunk_idx % max(1, num_chunks)

        input_ids = tokenized["input_ids"][chunk_idx]
        attention_mask = tokenized["attention_mask"][chunk_idx]
        offset_mapping = tokenized["offset_mapping"][chunk_idx]

        bio_tags = align_spans_to_bio(offset_mapping, spans)
        labels = [self.tag_to_id.get(normalize_bio_tag(t), self.tag_to_id.get(t, 0)) for t in bio_tags]
        layout_feats = extract_layout_features(raw_text, offset_mapping)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "col_buckets": torch.tensor(layout_feats["col_buckets"], dtype=torch.long),
            "is_line_start": torch.tensor(layout_feats["is_line_start"], dtype=torch.long),
            "is_blank_preceded": torch.tensor(layout_feats["is_blank_preceded"], dtype=torch.long)
        }
