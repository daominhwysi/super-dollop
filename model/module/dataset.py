#!/usr/bin/env python3
"""
PyTorch Dataset Loaders for Sequence Labelling:
1. MultiWindowBIODataset: Fast streaming loader for pre-tokenized multi-scale BIO chunks.
2. OnlineAugmentedDataset: Dynamic on-the-fly multi-scale sliding window dataset.
"""

import os
import json
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


class MultiWindowBIODataset(Dataset):
    """
    Fast, pre-tokenized BIO dataset loader supporting multi-scale sliding window chunks
    (512, 768, 1024, 2048 tokens).
    """
    def __init__(
        self,
        chunks_file_path: Union[str, Path],
        tag_to_id: Dict[str, int],
        tokenizer: Optional[Any] = None,
        pad_token_id: int = 0,
        ignore_label_id: int = -100,
        max_samples: Optional[int] = None,
        max_length: int = 2048
    ):
        self.chunks_file_path = Path(chunks_file_path)
        self.tag_to_id = tag_to_id
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length
        self.samples = []

        if not self.chunks_file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file_path}")

        with open(self.chunks_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    self.samples.append(data)
                    if max_samples and len(self.samples) >= max_samples:
                        break
                except Exception:
                    pass

        print(f"Loaded {len(self.samples)} multi-scale BIO chunks from '{self.chunks_file_path.name}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        tokens = item.get("tokens", [])
        bio_tags = item.get("ner_tags", item.get("tags", item.get("labels", [])))

        if "input_ids" in item:
            input_ids = item["input_ids"]
            if bio_tags and isinstance(bio_tags[0], int):
                label_ids = bio_tags
            else:
                label_ids = [self.tag_to_id.get(t, self.tag_to_id.get("O", 0)) for t in bio_tags]
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
                            label_ids.append(self.tag_to_id.get(tag_str, self.tag_to_id.get("O", 0)))
                        else:
                            label_ids.append(self.ignore_label_id)
                    else:
                        label_ids.append(self.ignore_label_id)
                    previous_word_idx = word_idx
            else:
                label_ids = [
                    self.tag_to_id.get(bio_tags[min(i, len(bio_tags) - 1)], 0)
                    for i in range(len(input_ids))
                ]
        else:
            raise KeyError("Item must contain 'input_ids' or MultiWindowBIODataset must be initialized with a tokenizer.")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


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
        from tools.build_training_dataset import align_spans_to_bio

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
        labels = [self.tag_to_id.get(t, 0) for t in bio_tags]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
