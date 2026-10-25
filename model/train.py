#!/usr/bin/env python3
"""
Full Fine-Tuning Trainer for Vietnamese Sequence Labelling.
Optimized for aisingapore/SEA-LION-ModernBERT-300M (300M parameters, 8192 context window) with
word-level sliding windows, Focal Loss, Distributed Training (DDP/torchrun),
and Hugging Face Hub integration.
"""

import os
import sys
import hashlib
from pathlib import Path

# Configure PyTorch memory allocator to avoid fragmentation on T4 / constrained GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Ensure repository root is in sys.path regardless of execution entrypoint
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import json
import yaml
import math
import random
import argparse
import contextlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv(dotenv_path=WORKSPACE_DIR / ".env")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()

from model.module.head import EnhancedBertForTokenClassification, FocalLoss
from model.module.dataset import get_tag_mappings, MultiWindowBIODataset
from model.inference.viterbi import ConstrainedViterbiDecoder


# ModernBERT names its normalisation layers `norm`, `attn_norm`, `mlp_norm`, `final_norm`
NO_DECAY_KEYWORDS = ("bias", "norm", "layer_pooling")


class LengthGroupedBatchSampler(Sampler[List[int]]):
    """
    BatchSampler that groups samples by sequence length into buckets before batching,
    preventing wasted GPU computation on pad tokens when training across multi-scale
    windows (512, 768, 1024, 2048).
    
    Batches are shuffled randomly at the batch level each epoch to preserve SGD randomness.
    """
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Sort indices by length with small random jitter to avoid deterministic tie-breaks
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        
        # Bucket indices by length (exact or coarse bucket)
        # Using coarse buckets of 64 tokens groups sequences of similar length together
        indices.sort(key=lambda idx: (self.lengths[idx] // 64, rng.random()))

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class DistributedLengthGroupedBatchSampler(Sampler[List[int]]):
    """Distributed version of LengthGroupedBatchSampler for DDP."""
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: (self.lengths[idx] // 64, rng.random()))

        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        # Distribute batches evenly across replicas
        if not self.drop_last:
            padding_size = (self.num_replicas - (len(batches) % self.num_replicas)) % self.num_replicas
            if padding_size > 0:
                batches.extend(batches[:padding_size])
            num_samples = len(batches) // self.num_replicas
        else:
            num_samples = len(batches) // self.num_replicas
            batches = batches[: num_samples * self.num_replicas]

        batches = batches[self.rank * num_samples : (self.rank + 1) * num_samples]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            total_batches = len(self.lengths) // self.batch_size
            return total_batches // self.num_replicas
        total_batches = (len(self.lengths) + self.batch_size - 1) // self.batch_size
        return (total_batches + self.num_replicas - 1) // self.num_replicas


def parse_args():
    parser = argparse.ArgumentParser(description="Train mmBERT-base for Vietnamese Sequence Labelling.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config YAML (default: configs/train_config.yaml)")

    # Model configuration
    parser.add_argument("--model-name", "--model_name", type=str, default=None, help="Hugging Face model backbone")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--enhanced-head", "--enhanced_head", action="store_true", default=True, help="Enable enhanced head with multi-sample dropout and layer pooling")
    parser.add_argument("--attn-implementation", "--attn_implementation", type=str, default=None, help="Attention implementation ('sdpa', 'flash_attention_2', or 'eager')")
    parser.add_argument("--no-lora", "--no_lora", action="store_true", default=False, help="Full fine-tuning without LoRA")
    parser.add_argument("--focal-gamma", "--focal_gamma", type=float, default=None, help="Focal loss gamma parameter")
    parser.add_argument("--label-smoothing", "--label_smoothing", type=float, default=None, help="Label smoothing rate")

    # Dataset options
    parser.add_argument("--dataset-repo-id", "--dataset_repo_id", type=str, default=None, help="Hugging Face dataset repo ID (e.g. daominhwysi/synthetic-seq-labelling-vi-exam-v2)")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None, help="Directory containing training dataset")
    parser.add_argument("--val-file", "--val_file", type=str, default=None, help="Path to evaluation/validation chunks JSONL file (e.g. data/training_dataset/test_bio_chunks.jsonl)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", "--backbone-lr", "--backbone_lr", type=float, default=None, help="Backbone learning rate")
    parser.add_argument("--head-lr", "--head_lr", type=float, default=None, help="Classifier head learning rate")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", type=int, default=None, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", "--gradient_accumulation_steps", "--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--gradient-checkpointing", "--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save GPU memory")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--lr-scheduler-type", "--lr_scheduler_type", type=str, default=None, help="Learning rate scheduler type (cosine or linear)")
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--logs-per-epoch", "--logs_per_epoch", type=int, default=None, help="Number of progress logs per epoch")
    parser.add_argument("--dataloader-num-workers", "--dataloader_num_workers", type=int, default=None, help="Dataloader num workers")

    # Checkpoint and Hub options
    parser.add_argument("--resume-from-checkpoint", "--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint ('auto', directory path, or Hugging Face Hub repo ID)")
    parser.add_argument("--eval-steps", "--eval_steps", type=int, default=None, help="Evaluate validation set every N optimization steps (0 to disable mid-epoch evaluation)")
    parser.add_argument("--save-steps", "--save_steps", type=int, default=None, help="Save a rolling checkpoint every N steps (0 to disable)")
    parser.add_argument("--save-total-limit", "--save_total_limit", type=int, default=None, help="Maximum number of rolling step checkpoints to keep (default: 2)")
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true", default=False, help="Push checkpoint to Hugging Face Hub during and after training")
    parser.add_argument("--hub-model-id", "--hub_model_id", type=str, default=None, help="Hugging Face model repository ID")

    # Hardware precision & Mixed Precision (AMP)
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable FP16 mixed precision training (recommended for NVIDIA T4/V100 GPUs)")
    parser.add_argument("--bf16", action="store_true", default=None, help="Enable BF16 mixed precision training (for Ampere+ GPUs)")
    parser.add_argument("--no-amp", "--no_amp", action="store_true", default=False, help="Disable Automatic Mixed Precision (run in full FP32)")
    parser.add_argument("--max-length", "--max_length", "--max-seq-length", type=int, default=None, help="Maximum token sequence length (defaults to model.max_sequence_length, 8192)")
    parser.add_argument("--empty-cache-freq", "--empty_cache_freq", type=int, default=0, help="Periodically empty CUDA cache every N steps (0 to disable)")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-dagshub", "--no_dagshub", action="store_true", default=False, help="Disable DagsHub tracking even if enabled in config")
    parser.add_argument("--no-mlflow", "--no_mlflow", action="store_true", default=False, help="Disable MLflow tracking completely")
    parser.add_argument("--balance-class-weights", "--balance_class_weights", action="store_true", default=False, help="Enable inverse-frequency balanced class weights for rare tags")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 5-step test without full training")

    return parser.parse_args()


def compute_entity_metrics(
    all_preds: List[List[str]],
    all_labels: List[List[str]]
) -> Dict[str, float]:
    """Computes entity-level precision, recall, and F1 score."""
    try:
        from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average="macro")
        prec = precision_score(all_labels, all_preds, average="macro")
        rec = recall_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")
        return {
            "macro_f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "micro_f1": float(micro_f1)
        }
    except ImportError:
        correct = 0
        total = 0
        for p_seq, l_seq in zip(all_preds, all_labels):
            for p, l in zip(p_seq, l_seq):
                if l != "O" or p != "O":
                    if p == l:
                        correct += 1
                    total += 1
        acc = correct / max(1, total)
        return {"macro_f1": acc, "precision": acc, "recall": acc, "micro_f1": acc}


def save_training_checkpoint(
    save_dir: Path,
    model: nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_macro_f1: float,
    patience_counter: int,
    tag_to_id: Dict[str, int],
    id_to_tag: Dict[int, str],
    step_in_epoch: int = 0,
    is_main_process: bool = True
):
    """Saves model weights, config, tokenizer, label map, and full trainer state for continuity."""
    if not is_main_process:
        return

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model

    # 1. Save model weights, config, enhanced head config, safetensors, pytorch_model.bin
    raw_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 2. Save label mapping
    label_map_file = save_dir / "label_mapping.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2)

    # 3. Save complete trainer state bundle (optimizer, scheduler, scaler, steps, epochs)
    trainer_state = {
        "epoch": epoch,
        "global_step": global_step,
        "step_in_epoch": step_in_epoch,
        "best_macro_f1": best_macro_f1,
        "patience_counter": patience_counter,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    torch.save(trainer_state, save_dir / "trainer_state.pt")
    del trainer_state
    gc.collect()


def get_free_disk_space_gb(path: Path) -> float:
    """Returns free disk space in gigabytes for the given filesystem path."""
    try:
        import shutil
        target = path if path.exists() else (path.parent if path.parent.exists() else Path.cwd())
        total, used, free = shutil.disk_usage(target)
        return free / (1024 ** 3)
    except Exception:
        return 999.0


def update_latest_checkpoint_link(output_dir: Path, target_dir: Path, is_main_process: bool = True):
    """Updates latest_checkpoint symlink or pointer metadata without duplicating gigabytes of model files."""
    if not is_main_process:
        return
    import shutil
    latest_link = output_dir / "latest_checkpoint"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        rel_target = target_dir.relative_to(output_dir) if target_dir.is_relative_to(output_dir) else target_dir
        latest_link.symlink_to(rel_target, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        # Fallback for environments without symlink support: store a lightweight metadata pointer
        pointer_file = output_dir / "latest_checkpoint.json"
        with open(pointer_file, "w", encoding="utf-8") as f:
            json.dump({"target_dir": str(target_dir.resolve()), "name": target_dir.name}, f, indent=2)


def prune_checkpoints(output_dir: Path, save_total_limit: int, is_main_process: bool = True):
    """Prunes older step checkpoints to respect save_total_limit and conserve disk space."""
    if not is_main_process or save_total_limit < 0:
        return

    import shutil
    step_dirs = []
    for d in output_dir.glob("checkpoint-*"):
        if d.is_dir() and not d.is_symlink() and d.name.replace("checkpoint-", "").isdigit():
            step_dirs.append((int(d.name.replace("checkpoint-", "")), d))

    step_dirs.sort(key=lambda x: x[0])
    while len(step_dirs) > save_total_limit:
        oldest_step, oldest_dir = step_dirs.pop(0)
        try:
            print(f"Pruning old checkpoint: {oldest_dir.name} (enforcing save_total_limit={save_total_limit})")
            shutil.rmtree(oldest_dir)
        except Exception as e:
            print(f"Warning: Could not prune {oldest_dir}: {e}")


def generate_model_readme(
    repo_id: str,
    metrics: Optional[Dict[str, float]] = None,
    args: Optional[Any] = None,
    tag_to_id: Optional[Dict[str, int]] = None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    base_model_name: Optional[str] = None
) -> str:
    """Generates comprehensive Hugging Face Model Card README.md with YAML metadata, benchmarks, and usage."""
    metrics = metrics or {}
    macro_f1 = metrics.get("macro_f1", 0.0)
    precision = metrics.get("precision", None)
    recall = metrics.get("recall", None)
    micro_f1 = metrics.get("micro_f1", None)

    base_model = base_model_name or (getattr(args, "model_name_or_path", "aisingapore/SEA-LION-ModernBERT-300M") if args else "aisingapore/SEA-LION-ModernBERT-300M")
    lr = getattr(args, "learning_rate", 3e-5) if args else 3e-5
    batch_size = getattr(args, "batch_size", 16) if args else 16
    epochs_val = total_epochs or (getattr(args, "epochs", 10) if args else 10)
    max_len = getattr(args, "max_length", 1024) if args else 1024

    metrics_yaml = f"""    - name: Validation Macro F1
      type: f1
      value: {macro_f1 * 100:.2f}"""
    if precision is not None and precision > 0:
        metrics_yaml += f"""\n    - name: Validation Precision
      type: precision
      value: {precision * 100:.2f}"""
    if recall is not None and recall > 0:
        metrics_yaml += f"""\n    - name: Validation Recall
      type: recall
      value: {recall * 100:.2f}"""
    if micro_f1 is not None and micro_f1 > 0:
        metrics_yaml += f"""\n    - name: Validation Micro F1
      type: f1
      value: {micro_f1 * 100:.2f}"""

    readme = f"""---
language:
- vi
- en
license: apache-2.0
base_model: {base_model}
tags:
- sequence-labeling
- token-classification
- modernbert
- sea-lion
- vietnamese
- exam-parser
- educational
- zero-character-mutation
datasets:
- daominhwysi/sequence-labelling-vi-exam-v3
metrics:
- f1
- precision
- recall
widget:
- text: "Câu 1: Cho hàm số $y = f(x)$ liên tục trên \\\\mathbb{{R}}. Giá trị lớn nhất trên đoạn [-1; 2] là?\\nA. 1\\nB. 2\\nC. 3\\nD. 4"
  example_title: "Vietnamese Math Exam"
- text: "Read the following passage and mark the letter A, B, C or D on your answer sheet to indicate the correct answer.\\nDigital innovations are revolutionizing modern educational assessment.\\nQuestion 1: What is the main idea of the passage?\\nA. Traditional classrooms\\nB. Digital transformation in education\\nC. Examination scoring\\nD. Student attendance"
  example_title: "English Reading Comprehension Exam"
pipeline_tag: token-classification
model-index:
- name: {repo_id}
  results:
  - task:
      type: token-classification
      name: Token Classification (Sequence Labelling)
    dataset:
      name: Vietnamese Exam Sequence Labelling Dataset v3
      type: daominhwysi/sequence-labelling-vi-exam-v3
    metrics:
{metrics_yaml}
---

# {repo_id.split('/')[-1]}

Production-grade token classification and sequence labelling model fine-tuned from **[{base_model}](https://huggingface.co/{base_model})** on the comprehensive [Vietnamese Exam Sequence Labelling Dataset v3](https://huggingface.co/datasets/daominhwysi/sequence-labelling-vi-exam-v3).

Designed for granular structural extraction from Vietnamese educational examinations (grades 8–12 across Math, Physics, Chemistry, Biology, History, Geography, Literature, and English), supporting both scanned and digital documents with **Zero Character Mutation guarantee**.

---

## Model Architecture

- **Base Encoder**: `{base_model}` (Multilingual ModernBERT, 8192 max context, unpad flash attention)
- **Classification Head**: `EnhancedBertForTokenClassification` featuring:
  - **SigmoidWeightedLayerPooling**: 4-layer sigmoid-gated layer pooling replacing competing softmax
  - **1D Depthwise Convolutional Refinement**: `kernel_size=7` capturing local boundary n-grams
  - **Multi-Sample Dropout (MSD)**: `[0.05, 0.10, 0.15, 0.20, 0.25]`
- **Context Length**: Trained with dynamic multi-scale windows (`768`, `1024`, `1536`, `2048`) with `LengthGroupedBatchSampler`.

---

## Entity Schema (7 Core Examination Entities)

The model performs token-level sequence labelling using standard BIO tagging:

| Entity Tag | Description | Example |
| :--- | :--- | :--- |
| `QUESTION_LABEL` | Question prefix and numbering | `Câu 1:`, `Question 20.`, `Bài 3:` |
| `STEM` | Main question prompt or problem statement | `Cho hàm số $y=f(x)$...` |
| `OPTION_LABEL` | Multiple-choice identifier | `A.`, `B.`, `C.`, `D.` |
| `OPTION_TEXT` | Content text of the choice | `x = 2`, `increase rapidly` |
| `STIMULUS` | Shared reading comprehension text, poem, or stimulus passage | Reading passages, dialogues, texts |
| `SECTION` | Examination headers, directions, and instructions | `PHẦN I. TRẮC NGHIỆM`, `Read the following...` |
| `EXPLANATION` | Detailed solutions, answer keys, or scoring barems | `Lời giải: Áp dụng định lý...` |

### Architectural Invariants
1. **Stemless Cloze Questions**: Full support for cloze / fill-in questions where `<question_label>` directly precedes `<option_label>` without an explicit `<stem>`.
2. **Zero Character Mutation Guarantee**: Sequence labeling and XML reconstruction wrap tags directly around input substrings with 0% character deletion, alteration, or hallucination.

---

## Evaluation Results (Gold Benchmark Test Set)

Evaluated on the strictly isolated 4-tier **Gold Benchmark** (51 pristine real exam documents, ~458k tokens):

| Metric | Score |
| :--- | :--- |
| **Validation Macro F1** | **{macro_f1 * 100:.2f}%** |
| **Precision** | **{f"{precision * 100:.2f}%" if precision is not None else "N/A"}** |
| **Recall** | **{f"{recall * 100:.2f}%" if recall is not None else "N/A"}** |
| **Micro F1** | **{f"{micro_f1 * 100:.2f}%" if micro_f1 is not None else "N/A"}** |

*Checkpoint synced at Epoch {epoch or epochs_val}/{epochs_val}.*

---

## Quickstart & Usage

### 1. Installation

```bash
pip install torch transformers huggingface_hub
```

### 2. Python Inference

```python
import torch
from transformers import AutoTokenizer
from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping

repo_id = "{repo_id}"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
tag_to_id, id_to_tag = load_label_mapping(repo_id)

model = EnhancedBertForTokenClassification.from_pretrained(
    repo_id,
    num_labels=len(tag_to_id),
    id2label=id_to_tag,
    label2id=tag_to_id
).to(device)

text = \"\"\"
Read the following passage and mark the letter A, B, C or D on your answer sheet.
Digital tools help modern students collaborate more effectively across borders.
Question 1: What do digital tools help students do?
A. Play games
B. Collaborate effectively
C. Avoid homework
D. Work alone
\"\"\"

res = predict_text(text, model=model, tokenizer=tokenizer, id_to_tag=id_to_tag, device=device)
print(res["xml_text"])
```

### Output:
```xml
<section>Read the following passage and mark the letter A, B, C or D on your answer sheet.</section>
<stimulus>Digital tools help modern students collaborate more effectively across borders.</stimulus>
<question_label>Question 1:</question_label> <stem>What do digital tools help students do?</stem>
<option_label>A.</option_label> <option_text>Play games</option_text>
<option_label>B.</option_label> <option_text>Collaborate effectively</option_text>
<option_label>C.</option_label> <option_text>Avoid homework</option_text>
<option_label>D.</option_label> <option_text>Work alone</option_text>
```

---

## Training Details

- **Dataset**: `daominhwysi/synthetic-seq-labelling-vi-exam-v2` (776 documents, 5.2M tokens, 90k multi-scale sliding chunks).
- **Optimizer**: AdamW (Learning Rate: `{lr}`) with Linear Warmup and Cosine Decay.
- **Batch Size**: `{batch_size}`
- **Sequence Length**: `{max_len}`
- **License**: Apache 2.0
"""
    return readme.strip() + "\n"


def push_checkpoint_to_hub(
    repo_id: str,
    folder_path: Path,
    commit_message: str,
    metrics: Optional[Dict[str, float]] = None,
    args: Optional[Any] = None,
    tag_to_id: Optional[Dict[str, int]] = None,
    epoch: Optional[int] = None
):
    """Pushes the primary model checkpoint and rewritten Model Card README to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        print(f"\nPushing checkpoint to Hugging Face Hub: '{repo_id}'...")
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Generate and rewrite README.md model card before uploading
        try:
            readme_content = generate_model_readme(
                repo_id=repo_id,
                metrics=metrics,
                args=args,
                tag_to_id=tag_to_id,
                epoch=epoch,
                total_epochs=getattr(args, "epochs", None) if args else None,
                base_model_name=getattr(args, "model_name_or_path", "aisingapore/SEA-LION-ModernBERT-300M") if args else "aisingapore/SEA-LION-ModernBERT-300M"
            )
            readme_file = folder_path / "README.md"
            readme_file.write_text(readme_content, encoding="utf-8")
            print(f"Generated and updated Model Card README.md ({len(readme_content):,} bytes) at '{readme_file}'")
        except Exception as err:
            print(f"Warning: Could not generate/rewrite README.md before push: {err}")

        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["checkpoint-*", "checkpoint-*/*", "step_*", "latest_checkpoint*", "*.tmp"]
        )
        print(f"Checkpoint successfully synchronized to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Notice: Failed to push checkpoint to Hugging Face Hub: {e}")


def evaluate_split(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    id_to_tag: Dict[int, str],
) -> Tuple[float, List[List[str]], List[List[str]]]:
    """
    Average loss plus decoded tag sequences for `loader`. Predictions are decoded with the same
    constrained Viterbi decoder used at inference, so the metrics describe the deployed system.
    """
    decoder = ConstrainedViterbiDecoder(id_to_tag=id_to_tag)
    model.eval()
    total_loss = 0.0
    all_preds: List[List[str]] = []
    all_labels: List[List[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs.loss.item()

            logits = outputs.logits.float().cpu().numpy()
            label_ids = labels.cpu().numpy()
            for logit_seq, label_seq in zip(logits, label_ids):
                valid = label_seq != -100
                if not valid.any():
                    continue
                pred_ids = decoder.decode(logit_seq[valid])
                all_preds.append([id_to_tag.get(int(p), "O") for p in pred_ids])
                all_labels.append([id_to_tag.get(int(l), "O") for l in label_seq[valid]])

    num_batches = len(loader)
    if torch.distributed.is_initialized():
        # Synchronize loss and sample counts across all DDP ranks
        metrics_tensor = torch.tensor([total_loss, float(num_batches)], device=device)
        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_loss = metrics_tensor[0].item() / max(1.0, metrics_tensor[1].item())

        # Gather predictions and labels across all ranks so Macro F1 evaluates on full split
        world_size = torch.distributed.get_world_size()
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        torch.distributed.all_gather_object(gathered_preds, all_preds)
        torch.distributed.all_gather_object(gathered_labels, all_labels)
        all_preds = [p for rank_list in gathered_preds if rank_list for p in rank_list]
        all_labels = [l for rank_list in gathered_labels if rank_list for l in rank_list]
    else:
        avg_loss = total_loss / max(1, num_batches)

    return avg_loss, all_preds, all_labels


def load_checkpoint_weights(model: nn.Module, checkpoint_dir: Path):
    """Load saved weights (safetensors preferred) from a checkpoint directory into `model`."""
    raw_model = model.module if hasattr(model, "module") else model
    safetensors_file = checkpoint_dir / "model.safetensors"
    if safetensors_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_file))
    else:
        state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: loading '{checkpoint_dir}' left {len(missing)} missing and {len(unexpected)} unexpected keys.")


def run_evaluation(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    id_to_tag: Dict[int, str],
    tag_to_id: Dict[str, int],
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    epochs: int,
    global_step: int,
    output_dir: Path,
    is_main_process: bool,
    is_distributed: bool,
    best_macro_f1: float,
    patience_counter: int,
    max_patience: int,
    args: Any,
    mlflow_client: Any = None,
    eval_title: str = "Evaluation",
    force_push_hub: bool = False,
    save_total_limit: int = 1,
) -> Tuple[float, int, bool]:
    """
    Runs full evaluation over the validation/gold test set, computes entity-level metrics,
    logs results to console and MLflow, and saves the best model checkpoint if Macro F1 peaks.
    
    Returns:
        (best_macro_f1, patience_counter, stop_training)
    """
    avg_val_loss, all_preds, all_labels = evaluate_split(model, val_loader, device, use_amp, amp_dtype, id_to_tag)

    stop_training = False
    if is_main_process:
        metrics = compute_entity_metrics(all_preds, all_labels)
        current_macro_f1 = metrics["macro_f1"]

        print(
            f"\n[{eval_title} (Dev Set)] Step {global_step} | Val Loss: {avg_val_loss:.4f} "
            f"| Macro F1: {current_macro_f1 * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | Recall: {metrics['recall'] * 100:.2f}%"
        )
        if mlflow_client is not None:
            try:
                mlflow_client.log_metrics(
                    {
                        "validation/loss": avg_val_loss,
                        "validation/macro_f1": metrics["macro_f1"],
                        "validation/precision": metrics["precision"],
                        "validation/recall": metrics["recall"],
                        "validation/micro_f1": metrics["micro_f1"],
                        "epoch": epoch,
                        "best_validation/macro_f1": max(best_macro_f1, current_macro_f1),
                    },
                    step=global_step,
                )
            except Exception as e:
                print(f"Warning: MLflow metric logging failed; disabling tracking: {e}")
                try:
                    mlflow_client.end_run(status="FAILED")
                except Exception:
                    pass
                mlflow_client = None

        is_best = current_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = current_macro_f1
            patience_counter = 0
            print(f"  --> New best model checkpoint! Macro F1: {best_macro_f1 * 100:.2f}%. Saving to '{output_dir}/best'...")
            save_training_checkpoint(
                save_dir=output_dir / "best",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_macro_f1=best_macro_f1,
                patience_counter=patience_counter,
                tag_to_id=tag_to_id,
                id_to_tag=id_to_tag,
                is_main_process=is_main_process
            )
            with open(output_dir / "best" / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            with open(output_dir / "best_dev_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            update_latest_checkpoint_link(output_dir, output_dir / "best", is_main_process=is_main_process)

            if args.push_to_hub and args.hub_model_id:
                push_checkpoint_to_hub(
                    repo_id=args.hub_model_id,
                    folder_path=output_dir / "best",
                    commit_message=f"Step {global_step} (Epoch {epoch}/{epochs}) - Best Macro F1: {best_macro_f1 * 100:.2f}%",
                    metrics=metrics,
                    args=args,
                    tag_to_id=tag_to_id,
                    epoch=epoch
                )
        else:
            patience_counter += 1
            if patience_counter >= max_patience and not args.dry_run:
                print(f"Early stopping triggered after {patience_counter} evaluations without improvement.")
                stop_training = True

            eval_step_dir = output_dir / f"checkpoint-{global_step}"
            if not eval_step_dir.exists():
                save_training_checkpoint(
                    save_dir=eval_step_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_macro_f1=best_macro_f1,
                    patience_counter=patience_counter,
                    tag_to_id=tag_to_id,
                    id_to_tag=id_to_tag,
                    is_main_process=is_main_process
                )
            update_latest_checkpoint_link(output_dir, eval_step_dir, is_main_process=is_main_process)
            effective_save_limit = (
                save_total_limit
                if save_total_limit is not None
                else getattr(args, "save_total_limit", None) or 1
            )
            prune_checkpoints(output_dir, effective_save_limit, is_main_process=is_main_process)
            if force_push_hub and args.push_to_hub and args.hub_model_id:
                push_checkpoint_to_hub(
                    repo_id=args.hub_model_id,
                    folder_path=output_dir,
                    commit_message=f"Epoch {epoch}/{epochs} End - Macro F1: {current_macro_f1 * 100:.2f}% (Best: {best_macro_f1 * 100:.2f}%)",
                    metrics=metrics,
                    args=args,
                    tag_to_id=tag_to_id,
                    epoch=epoch
                )

    if is_distributed:
        best_tensor = torch.tensor([best_macro_f1, float(patience_counter), 1.0 if stop_training else 0.0], device=device)
        torch.distributed.broadcast(best_tensor, src=0)
        best_macro_f1 = float(best_tensor[0].item())
        patience_counter = int(best_tensor[1].item())
        stop_training = bool(best_tensor[2].item() == 1.0)

    model.train()
    return best_macro_f1, patience_counter, stop_training


def main():
    args = parse_args()

    # Distributed Training Initialization (DDP / torchrun)
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (local_rank == 0)

    # 1. Load Configuration
    cfg_path = Path(args.config)
    if not cfg_path.exists() and (WORKSPACE_DIR / "configs" / "train_config.yaml").exists():
        cfg_path = WORKSPACE_DIR / "configs" / "train_config.yaml"
    elif not cfg_path.exists() and Path("train_config.yaml").exists():
        cfg_path = Path("train_config.yaml")

    cfg = {}
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if is_main_process:
                print(f"Loaded training config from '{cfg_path}'.")
        except Exception as e:
            if is_main_process:
                print(f"Warning: Could not parse config '{cfg_path}': {e}")

    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    model_head_cfg = cfg.get("model_head", {})
    head_cfg = cfg.get("head", {})
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    model_name = args.model_name or model_cfg.get("name_or_path", "aisingapore/SEA-LION-ModernBERT-300M")
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "results/mmbert_seqlabel_v2"))
    epochs = args.epochs or train_cfg.get("num_train_epochs", 4)
    lr = args.lr or float(train_cfg.get("learning_rate", 1.5e-5))
    head_lr = args.head_lr or float(train_cfg.get("head_learning_rate", 5.0e-5))
    batch_size = args.batch_size or int(train_cfg.get("per_device_train_batch_size", 4))
    eval_batch_size = args.eval_batch_size or int(train_cfg.get("per_device_eval_batch_size", 4))
    grad_accum = args.gradient_accumulation_steps or int(train_cfg.get("gradient_accumulation_steps", 4))
    weight_decay = args.weight_decay or float(train_cfg.get("weight_decay", 0.05))
    scheduler_type = args.lr_scheduler_type or train_cfg.get("lr_scheduler_type", "cosine")
    warmup_ratio = args.warmup_ratio or float(train_cfg.get("warmup_ratio", 0.10))
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else float(loss_cfg.get("focal_gamma", head_cfg.get("focal_gamma", 2.0)))
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else float(loss_cfg.get("label_smoothing", head_cfg.get("label_smoothing", 0.05)))
    seed = args.seed or int(data_cfg.get("seed", 42))

    # Determine Mixed Precision (AMP)
    if args.no_amp:
        use_amp = False
        amp_dtype = torch.float32
        precision_str = "FP32 (AMP disabled)"
    elif args.fp16 is True:
        use_amp = True
        amp_dtype = torch.float16
        precision_str = "FP16 (via --fp16)"
    elif args.bf16 is True:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
            precision_str = "BF16 (via --bf16)"
        else:
            if is_main_process:
                print("Notice: BF16 requested but GPU lacks native BF16 support (e.g. NVIDIA T4). Falling back to FP16.")
            use_amp = True
            amp_dtype = torch.float16
            precision_str = "FP16 (fallback from --bf16: GPU lacks native bf16)"
    else:
        # Resolve from train_config.yaml
        cfg_bf16 = bool(train_cfg.get("bf16", False))
        cfg_fp16 = bool(train_cfg.get("fp16", False))
        if cfg_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_amp = True
            amp_dtype = torch.bfloat16
            precision_str = "BF16 (from config)"
        elif cfg_fp16 or cfg_bf16:
            use_amp = True
            amp_dtype = torch.float16
            precision_str = "FP16 (from config)" if cfg_fp16 else "FP16 (fallback from config bf16: GPU lacks native bf16)"
        else:
            use_amp = False
            amp_dtype = torch.float32
            precision_str = "FP32 (config bf16/fp16 both false)"

    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16 and torch.cuda.is_available()))
    max_length = args.max_length or int(model_cfg.get("max_sequence_length", 8192))
    window_overlap_tokens = int(data_cfg.get("window_overlap_tokens", 1024))
    save_steps = args.save_steps if args.save_steps is not None else int(train_cfg.get("save_steps", 0))
    eval_steps = args.eval_steps if args.eval_steps is not None else int(train_cfg.get("eval_steps", 0))
    save_total_limit = args.save_total_limit if args.save_total_limit is not None else int(train_cfg.get("save_total_limit", 2))
    logging_cfg = cfg.get("logging", {})

    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)

    if is_main_process:
        print(f"\n============================================================")
        print(f"Starting mmBERT Sequence Labelling Full Fine-Tuning")
        print(f"  Backbone Model       : {model_name}")
        print(f"  Distributed          : {is_distributed} (World Size: {world_size}, Local Rank: {local_rank})")
        print(f"  Device               : {device}")
        print(f"  Precision            : {precision_str}")
        print(f"  GradScaler Enabled   : {scaler.is_enabled()}")
        print(f"  Epochs               : {epochs}")
        print(f"  Backbone LR          : {lr}")
        print(f"  Head LR              : {head_lr}")
        print(f"  Batch Size           : {batch_size} (effective: {batch_size * grad_accum * world_size})")
        print(f"  Max Sequence Length  : {max_length}")
        if save_steps > 0:
            print(f"  Save Checkpoints     : Every {save_steps} steps (Keep last {save_total_limit})")
        if eval_steps > 0:
            print(f"  Eval Validation      : Every {eval_steps} steps")
        if args.resume_from_checkpoint:
            print(f"  Resume Source        : {args.resume_from_checkpoint}")
        free_gb = get_free_disk_space_gb(output_dir)
        print(f"  Output Dir           : {output_dir}")
        print(f"  Free Disk Space      : {free_gb:.2f} GB")
        if free_gb < 5.0:
            print(f"  WARNING: Low free disk space ({free_gb:.2f} GB)! Checkpoint pruning will aggressively maintain headroom.")
        print(f"  Focal Gamma          : {focal_gamma}")
        print(f"  Label Smoothing      : {label_smoothing}")
        print(f"============================================================\n")

    # 2. Setup Tokenizer and Tag Mappings
    tag_to_id, id_to_tag = get_tag_mappings()
    num_labels = len(tag_to_id)

    if is_main_process:
        label_map_file = output_dir / "label_mapping.json"
        with open(label_map_file, "w", encoding="utf-8") as f:
            json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2)

    if is_main_process:
        print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    latex_placeholder = model_cfg.get("latex_placeholder", "[LATEX]")
    special_tokens = ["<blank />", "<blank/>", "[BLANK]"]
    if latex_placeholder and latex_placeholder not in special_tokens:
        special_tokens.append(latex_placeholder)

    # Capture constituent subwords before adding atomic special tokens
    constituent_subwords: Dict[str, List[int]] = {}
    for st in special_tokens:
        sub_ids = tokenizer.encode(st, add_special_tokens=False)
        if sub_ids:
            constituent_subwords[st] = sub_ids

    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    if is_main_process:
        tokenizer.save_pretrained(output_dir)

    # 3. Resolve Dataset Files (Local or Hugging Face Hub)
    if args.dataset_repo_id:
        from huggingface_hub import hf_hub_download
        if is_main_process:
            print(f"Downloading dataset from Hugging Face repository: '{args.dataset_repo_id}'...")
        chunks_file = Path(hf_hub_download(
            repo_id=args.dataset_repo_id,
            filename="train_bio_chunks.jsonl",
            repo_type="dataset"
        ))
        val_chunks_file = Path(hf_hub_download(
            repo_id=args.dataset_repo_id,
            filename="test_bio_chunks.jsonl",
            repo_type="dataset"
        ))
        try:
            hf_hub_download(
                repo_id=args.dataset_repo_id,
                filename="dataset_manifest.json",
                repo_type="dataset",
                local_dir=str(chunks_file.parent)
            )
        except Exception:
            pass
    else:
        chunks_path = args.data_dir or data_cfg.get("bio_chunks_file", "data/training_dataset/train_bio_chunks.jsonl")
        chunks_file = Path(chunks_path)
        if not chunks_file.is_absolute():
            chunks_file = WORKSPACE_DIR / chunks_file

        val_chunks_path = args.val_file or data_cfg.get("val_bio_chunks_file")
        if val_chunks_path:
            val_chunks_file = Path(val_chunks_path)
            if not val_chunks_file.is_absolute():
                val_chunks_file = WORKSPACE_DIR / val_chunks_file
        else:
            val_chunks_file = None

    if not chunks_file.exists():
        if is_main_process:
            print(f"Error: BIO chunks file '{chunks_file}' does not exist.")
            print("Please pass --dataset_repo_id or run 'tools/dataset/build_training_dataset.py' first.")
        sys.exit(1)

    # 4. Load Datasets
    # The dev set is a document-level split of the training file, used for checkpoint
    # selection and early stopping. The gold benchmark is evaluated once, after training,
    # with the selected checkpoint, so it never influences model selection.
    dev_ratio = float(data_cfg.get("dev_ratio", 0.05))

    def is_dev_doc(doc_id: str) -> bool:
        # Strip chunk suffixes and augmentation flags to ensure the base document
        # and all its augmented/chunked variants strictly reside in the SAME split.
        # e.g. "doc_123::chunk_001_aug" -> "doc_123"
        base_id = doc_id.split("::")[0]
        if base_id.endswith("_aug"):
            base_id = base_id[:-4]
        digest = hashlib.md5(f"{seed}:{base_id}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF < dev_ratio

    dataset_kwargs = dict(
        tag_to_id=tag_to_id,
        tokenizer=tokenizer,
        max_length=max_length,
        window_overlap_tokens=window_overlap_tokens,
        max_samples=20 if args.dry_run else None,
    )
    if is_main_process:
        print(f"Loading training chunks from '{chunks_file.name}' (document-level dev split, dev_ratio={dev_ratio})...")
    train_dataset = MultiWindowBIODataset(
        chunks_file_path=chunks_file,
        doc_filter=lambda doc_id: not is_dev_doc(doc_id),
        **dataset_kwargs,
    )
    # Deduplicate validation chunks: for the dev set used during training,
    # evaluate on a single canonical window size (e.g. 1024 or the largest available)
    # to avoid 16x redundant forward passes and skewed F1 metrics.
    def dev_record_filter(item: Dict[str, Any]) -> bool:
        w_size = item.get("window_size")
        # If dataset has multiple window sizes, keep only 1024 (or keep all if not multi-scale)
        if w_size is not None and w_size != 1024:
            return False
        return True

    val_dataset = MultiWindowBIODataset(
        chunks_file_path=chunks_file,
        doc_filter=is_dev_doc,
        record_filter=dev_record_filter if not args.dry_run else None,
        **dataset_kwargs,
    )
    if len(val_dataset) == 0:
        # Fallback to unfiltered dev if no 1024-scale chunk was matched
        val_dataset = MultiWindowBIODataset(
            chunks_file_path=chunks_file,
            doc_filter=is_dev_doc,
            **dataset_kwargs,
        )
    if len(val_dataset) == 0:
        raise ValueError(f"Dev split is empty; increase dataset.dev_ratio (currently {dev_ratio})")

    test_dataset = None
    if val_chunks_file and val_chunks_file.exists():
        if is_main_process:
            print(f"Loading gold benchmark chunks from '{val_chunks_file.name}' (final report only)...")
        test_dataset = MultiWindowBIODataset(
            chunks_file_path=val_chunks_file,
            record_filter=dev_record_filter if not args.dry_run else None,
            **dataset_kwargs,
        )
        if len(test_dataset) == 0:
            # Fallback to unfiltered gold test if no 1024-scale chunk was matched
            test_dataset = MultiWindowBIODataset(
                chunks_file_path=val_chunks_file,
                **dataset_kwargs,
            )

    if is_main_process:
        print(
            f"Dataset split: {len(train_dataset)} training windows, {len(val_dataset)} dev windows, "
            f"{len(test_dataset) if test_dataset is not None else 0} gold test windows."
        )

    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        max_len = max(len(x) for x in input_ids)
        padded_ids = torch.stack([nn.functional.pad(x, (0, max_len - len(x)), value=tokenizer.pad_token_id or 0) for x in input_ids])
        padded_mask = torch.stack([nn.functional.pad(x, (0, max_len - len(x)), value=0) for x in attention_mask])
        padded_labels = torch.stack([nn.functional.pad(x, (0, max_len - len(x)), value=-100) for x in labels])

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_mask,
            "labels": padded_labels
        }

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    workers = int(args.dataloader_num_workers if args.dataloader_num_workers is not None else train_cfg.get("dataloader_num_workers", 2))
    if os.name == "nt":
        workers = 0

    # Build sequence lengths for length-grouped bucketing
    if is_main_process:
        print("Pre-calculating sequence lengths for length-grouped batch sampling...")
    train_lengths = [train_dataset.get_sequence_length(i) for i in range(len(train_dataset))]

    # Distributed or standard samplers
    if is_distributed:
        train_batch_sampler = DistributedLengthGroupedBatchSampler(
            lengths=train_lengths,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=local_rank,
            drop_last=False,
            shuffle=True,
            seed=seed,
        )
        train_sampler = None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_batch_sampler = LengthGroupedBatchSampler(
            lengths=train_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            seed=seed,
        )
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=workers
    )

    # 5. Initialize Model
    attn_impl = args.attn_implementation or model_cfg.get("attn_implementation", "sdpa")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_tag,
        label2id=tag_to_id,
        _attn_implementation=attn_impl,
    )
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions and max_length > max_positions:
        raise ValueError(f"max_length={max_length} exceeds the backbone's max_position_embeddings={max_positions}")
    config.output_hidden_states = True
    config._attn_implementation = attn_impl
    config.torch_dtype = torch.float32

    try:
        base_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
        )
    except TypeError:
        base_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
        )
    base_model = base_model.to(torch.float32)

    orig_vocab_size = getattr(config, "vocab_size", len(tokenizer))
    base_model.resize_token_embeddings(len(tokenizer))
    if len(tokenizer) > orig_vocab_size or any(tokenizer.convert_tokens_to_ids(st) >= orig_vocab_size for st in special_tokens):
        embeddings = base_model.get_input_embeddings()
        with torch.no_grad():
            unused_id = None
            for candidate in ["<unused0>", "[unused0]", "<unused_0>"]:
                c_id = tokenizer.convert_tokens_to_ids(candidate)
                if c_id is not None and c_id != tokenizer.unk_token_id and c_id < orig_vocab_size:
                    unused_id = c_id
                    break

            for st in special_tokens:
                tid = tokenizer.convert_tokens_to_ids(st)
                if tid is None or tid < orig_vocab_size:
                    continue

                sub_ids = constituent_subwords.get(st, [])
                sub_ids = [idx for idx in sub_ids if idx < orig_vocab_size and idx != tokenizer.unk_token_id]
                if sub_ids:
                    semantic_vec = embeddings.weight.data[sub_ids].mean(dim=0)
                    embeddings.weight.data[tid].copy_(semantic_vec)
                    if is_main_process:
                        print(f"Semantically initialized special token '{st}' (id={tid}) from {len(sub_ids)} constituent subwords.")
                elif unused_id is not None:
                    embeddings.weight.data[tid].copy_(embeddings.weight.data[unused_id])
                    if is_main_process:
                        print(f"Initialized special token '{st}' (id={tid}) from fallback <unused*> token (id={unused_id}).")
                else:
                    embeddings.weight.data[tid].copy_(embeddings.weight.data[:orig_vocab_size].mean(dim=0))
                    if is_main_process:
                        print(f"Initialized special token '{st}' (id={tid}) from global pretrained vocab mean fallback.")
    if is_main_process:
        print(f"Base model attention implementation configured: '{getattr(config, '_attn_implementation', attn_impl)}'")

    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if is_main_process:
            print("Gradient checkpointing enabled.")

    # 4b. Optional Class Weights for Severe Tag Imbalance (e.g. B-STIMULUS, B-SECTION)
    class_weights = None
    if (args.balance_class_weights 
        or train_cfg.get("balance_class_weights", False) 
        or loss_cfg.get("focal_alpha") == "smoothed_inverse_freq"
        or head_cfg.get("focal_alpha") == "smoothed_inverse_freq"):
        manifest_file = chunks_file.parent / "dataset_manifest.json"
        if manifest_file.exists():
            try:
                manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
                tag_dist = manifest_data.get("tag_distribution", {})
                if tag_dist:
                    total_cnt = sum(tag_dist.values())
                    weights = []
                    for i in range(num_labels):
                        tag_name = id_to_tag.get(i, "O")
                        cnt = tag_dist.get(tag_name.upper(), tag_dist.get(tag_name, 1))
                        # Smooth inverse frequency with power 0.25 (calibrated RFC specification)
                        w = (total_cnt / max(cnt, 1)) ** 0.25
                        weights.append(w)
                    weights_tensor = torch.tensor(weights, dtype=torch.float)
                    weights_tensor = weights_tensor / weights_tensor.mean()
                    class_weights = weights_tensor.to(device)
                    if is_main_process:
                        print("Balanced class weights enabled (power=0.25). Normalized weights per tag:")
                        for i in range(num_labels):
                            print(f"  {id_to_tag[i]:<18}: {class_weights[i].item():.3f}")
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not compute balanced class weights from {manifest_file}: {e}")

    num_layers_to_fuse = int(model_head_cfg.get("num_pooling_layers", head_cfg.get("layers_to_pool", 4)))
    use_conv_refinement = bool(model_head_cfg.get("use_conv_refinement", head_cfg.get("use_conv_refinement", True)))
    conv_kernel_size = int(model_head_cfg.get("conv_kernel_size", head_cfg.get("conv_kernel_size", 7)))
    dropout_rates = tuple(model_head_cfg.get("dropout_rates", head_cfg.get("dropout_rates", [0.05, 0.10, 0.15, 0.20, 0.25])))
    per_branch_focal_loss = bool(model_head_cfg.get("per_branch_focal_loss", head_cfg.get("per_branch_focal_loss", True)))

    model = EnhancedBertForTokenClassification(
        config=config,
        base_model=base_model,
        num_layers_to_fuse=num_layers_to_fuse,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        dropout_rates=dropout_rates,
        use_conv_refinement=use_conv_refinement,
        conv_kernel_size=conv_kernel_size,
        per_branch_focal_loss=per_branch_focal_loss
    ).to(device=device, dtype=torch.float32)

    # 5b. Checkpoint Resume Resolution (Auto, Local Path, or Hugging Face Hub)
    resume_dir = None
    if args.resume_from_checkpoint:
        raw_resume = args.resume_from_checkpoint.strip()
        if raw_resume == "auto":
            # 1. Prefer highest step checkpoint in output_dir
            step_checkpoints = []
            for d in output_dir.glob("checkpoint-*"):
                if d.is_dir() and d.name.replace("checkpoint-", "").isdigit():
                    if (d / "trainer_state.pt").exists() or (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
                        step_checkpoints.append((int(d.name.replace("checkpoint-", "")), d))
            if step_checkpoints:
                step_checkpoints.sort(key=lambda x: x[0], reverse=True)
                resume_dir = step_checkpoints[0][1]
            elif (output_dir / "latest_checkpoint" / "trainer_state.pt").exists() or (output_dir / "latest_checkpoint" / "model.safetensors").exists() or (output_dir / "latest_checkpoint" / "pytorch_model.bin").exists():
                resume_dir = output_dir / "latest_checkpoint"
            elif (output_dir / "trainer_state.pt").exists() or (output_dir / "model.safetensors").exists() or (output_dir / "pytorch_model.bin").exists():
                resume_dir = output_dir
            else:
                if is_main_process:
                    print(f"Notice: --resume-from-checkpoint 'auto' specified, but no previous checkpoint found in '{output_dir}'. Starting fresh.")
        elif "/" in raw_resume and not Path(raw_resume).exists():
            # Hugging Face Hub repository ID!
            if is_main_process:
                print(f"Downloading checkpoint from Hugging Face Hub repository: '{raw_resume}'...")
                from huggingface_hub import snapshot_download
                # Only download the primary checkpoint & state, skipping redundant intermediate step dirs
                dl_path = snapshot_download(
                    repo_id=raw_resume,
                    ignore_patterns=["checkpoint-*", "checkpoint-*/*", "latest_checkpoint/*"]
                )
            else:
                dl_path = None
            if is_distributed:
                obj_list = [dl_path]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                resume_dir = Path(obj_list[0])
            else:
                resume_dir = Path(dl_path)
        else:
            p = Path(raw_resume)
            if not p.is_absolute():
                p = WORKSPACE_DIR / p
            if p.exists():
                resume_dir = p
            else:
                if is_main_process:
                    print(f"Warning: Checkpoint path '{p}' does not exist. Starting fresh.")

    # Load Model Weights
    if resume_dir is not None:
        # If resume_dir points to a directory without weights at root, inspect subfolders
        if not (resume_dir / "model.safetensors").exists() and not (resume_dir / "pytorch_model.bin").exists():
            if (resume_dir / "latest_checkpoint" / "model.safetensors").exists() or (resume_dir / "latest_checkpoint" / "pytorch_model.bin").exists():
                resume_dir = resume_dir / "latest_checkpoint"
            else:
                step_subs = []
                for sub in resume_dir.glob("checkpoint-*"):
                    if sub.is_dir() and sub.name.replace("checkpoint-", "").isdigit():
                        if (sub / "model.safetensors").exists() or (sub / "pytorch_model.bin").exists():
                            step_subs.append((int(sub.name.replace("checkpoint-", "")), sub))
                if step_subs:
                    step_subs.sort(key=lambda x: x[0], reverse=True)
                    resume_dir = step_subs[0][1]

        weights_file = resume_dir / "model.safetensors"
        bin_file = resume_dir / "pytorch_model.bin"
        if weights_file.exists():
            from safetensors.torch import load_file
            if is_main_process:
                print(f"Resuming model weights from '{weights_file}'...")
            state_dict = load_file(weights_file, device=str(device))
            model.load_state_dict(state_dict, strict=False)
        elif bin_file.exists():
            if is_main_process:
                print(f"Resuming model weights from '{bin_file}'...")
            try:
                state_dict = torch.load(bin_file, map_location=device, weights_only=True)
            except Exception:
                try:
                    state_dict = torch.load(bin_file, map_location=device, weights_only=False)
                except TypeError:
                    state_dict = torch.load(bin_file, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            if is_main_process:
                print(f"Notice: No model weights file found in '{resume_dir}'. Initializing from base model.")

    # Wrap model with DistributedDataParallel if running multi-GPU
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # 6. Optimizer & Scheduler
    raw_model = model.module if hasattr(model, "module") else model
    def build_param_groups(module: nn.Module, group_lr: float) -> List[Dict[str, Any]]:
        decay, no_decay_params = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in NO_DECAY_KEYWORDS):
                no_decay_params.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay, "lr": group_lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": group_lr},
        ]

    optimizer_grouped_parameters = (
        build_param_groups(raw_model.base_model, lr) + build_param_groups(raw_model.head, head_lr)
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    total_training_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    # 7. Training State & Continuity Restoration
    start_epoch = 1
    resume_step_in_epoch = 0
    global_step = 0
    best_macro_f1 = -1.0
    patience_counter = 0
    max_patience = int(train_cfg.get("early_stopping_patience", 3))

    if resume_dir is not None:
        trainer_state_file = resume_dir / "trainer_state.pt"
        if trainer_state_file.exists():
            try:
                if is_main_process:
                    print(f"Loading full trainer continuity state from '{trainer_state_file}'...")
                try:
                    trainer_state = torch.load(trainer_state_file, map_location=device, weights_only=False)
                except TypeError:
                    trainer_state = torch.load(trainer_state_file, map_location=device)

                # 1. Restore optimizer states and tensors
                if "optimizer_state_dict" in trainer_state:
                    optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
                    for state_val in optimizer.state.values():
                        for k, v in state_val.items():
                            if isinstance(v, torch.Tensor):
                                state_val[k] = v.to(device)

                # 2. Restore scheduler states
                if "scheduler_state_dict" in trainer_state:
                    scheduler.load_state_dict(trainer_state["scheduler_state_dict"])

                # 3. Restore scaler states
                if "scaler_state_dict" in trainer_state and trainer_state["scaler_state_dict"] is not None and scaler.is_enabled():
                    scaler.load_state_dict(trainer_state["scaler_state_dict"])

                # 4. Restore loop counters
                saved_epoch = trainer_state.get("epoch", 1)
                global_step = trainer_state.get("global_step", 0)
                best_macro_f1 = trainer_state.get("best_macro_f1", -1.0)
                patience_counter = trainer_state.get("patience_counter", 0)
                resume_step_in_epoch = trainer_state.get("step_in_epoch", 0)

                # Check if checkpoint was saved mid-epoch or at epoch boundary
                steps_per_epoch = len(train_loader) if len(train_loader) > 0 else 1
                if resume_step_in_epoch > 0 and resume_step_in_epoch < steps_per_epoch:
                    start_epoch = saved_epoch
                    if is_main_process:
                        print(f"Resuming mid-epoch at Epoch {start_epoch}, Step {resume_step_in_epoch}/{steps_per_epoch} (global step {global_step}).")
                else:
                    start_epoch = saved_epoch + 1
                    resume_step_in_epoch = 0
                    if is_main_process:
                        print(f"Resuming at Epoch boundary: starting Epoch {start_epoch} (global step {global_step}).")

                # 5. Restore RNG states if present
                if "rng_state" in trainer_state:
                    try:
                        rng = trainer_state["rng_state"]
                        random.setstate(rng["python"])
                        np.random.set_state(rng["numpy"])
                        torch.set_rng_state(rng["torch"])
                        if torch.cuda.is_available() and rng.get("cuda") is not None:
                            torch.cuda.set_rng_state_all(rng["cuda"])
                    except Exception:
                        pass

                if is_main_process:
                    print(f"Successfully restored training continuity!")
                    print(f"  Resuming at Epoch   : {start_epoch} (Completed: {saved_epoch}/{epochs})")
                    print(f"  Current Global Step : {global_step}/{total_training_steps}")
                    print(f"  Best Prior Macro F1 : {best_macro_f1 * 100:.2f}%")
                    print(f"  Current LR          : {scheduler.get_last_lr()[0]:.2e}")
            except Exception as e:
                if is_main_process:
                    print(f"Warning: Could not fully restore trainer_state.pt: {e}. Proceeding with resumed model weights only.")

    if start_epoch > epochs:
        if is_main_process:
            print(f"\nNotice: Checkpoint has already completed epoch {start_epoch - 1} of {epochs}.")
            print(f"To train further, specify a higher --epochs value (e.g. --epochs {start_epoch}).")

    mlflow_client = None
    if is_main_process:
        dagshub_config_path = WORKSPACE_DIR / "configs" / "dagshub_config.yaml"
        dagshub_config = {}
        if dagshub_config_path.exists():
            dagshub_config = yaml.safe_load(
                dagshub_config_path.read_text(encoding="utf-8")
            ) or {}
        dagshub_settings = dagshub_config.get("dagshub", {})

        if args.no_mlflow or args.no_dagshub:
            dagshub_settings["enabled"] = False

        report_to = str(logging_cfg.get("report_to", "none")).lower()
        mlflow_enabled = not args.no_mlflow and (
            report_to == "mlflow"
            or dagshub_settings.get("enabled", False)
            or bool(os.getenv("MLFLOW_TRACKING_URI"))
        )

        if mlflow_enabled:
            try:
                import mlflow

                if dagshub_settings.get("enabled", False) and not args.no_dagshub:
                    username_env = dagshub_settings.get(
                        "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_USERNAME"
                    )
                    password_env = dagshub_settings.get(
                        "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_PASSWORD"
                    )
                    tracking_username = os.getenv(username_env)
                    tracking_password = os.getenv(password_env)
                    if not tracking_username or not tracking_password:
                        raise RuntimeError(
                            f"Set {username_env} and {password_env} to enable DagsHub tracking."
                        )
                    os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password

                    tracking_uri = dagshub_settings.get("set_tracking_uri", "")
                    if "dagshub.com" in tracking_uri and not tracking_uri.endswith(".mlflow"):
                        tracking_uri = tracking_uri.rstrip("/") + ".mlflow"
                    mlflow.set_tracking_uri(tracking_uri)
                    experiment_base = dagshub_settings.get("set_experiment", "vietnamese-sequence-labelling")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    experiment_name = f"{experiment_base}_{timestamp}"
                else:
                    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                    if tracking_uri:
                        mlflow.set_tracking_uri(tracking_uri)
                    else:
                        local_mlruns = output_dir / "mlruns"
                        mlflow.set_tracking_uri(f"file://{local_mlruns.resolve()}")
                    experiment_name = os.getenv(
                        "MLFLOW_EXPERIMENT_NAME",
                        logging_cfg.get("mlflow_experiment_name", "vietnamese-sequence-labelling"),
                    )

                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=output_dir.name)
                mlflow_client = mlflow
                run_params = {
                    "model_name": model_name,
                    "output_dir": str(output_dir),
                    "epochs": epochs,
                    "learning_rate": lr,
                    "head_lr": head_lr,
                    "batch_size": batch_size,
                    "eval_batch_size": eval_batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "weight_decay": weight_decay,
                    "scheduler_type": scheduler_type,
                    "warmup_ratio": warmup_ratio,
                    "max_length": max_length,
                    "focal_gamma": focal_gamma,
                    "label_smoothing": label_smoothing,
                    "precision": precision_str,
                    "world_size": world_size,
                    "seed": seed,
                }
                mlflow.log_params({key: value for key, value in run_params.items() if value is not None})
                print(f"MLflow tracking enabled (experiment: {experiment_name}).")
            except Exception as e:
                err_msg = str(e)
                if "<html" in err_msg.lower() or "<!doctype" in err_msg.lower():
                    err_summary = "Tracking server returned an HTML error page (verify repository exists on DagsHub and URI ends in .mlflow)"
                else:
                    err_summary = err_msg.split("\n")[0]
                print(f"Warning: MLflow could not be initialized; continuing without tracking: {err_summary}")
                if mlflow_client is not None:
                    try:
                        mlflow_client.end_run(status="FAILED")
                    except Exception:
                        pass
                    mlflow_client = None

    last_eval_step = -1
    stop_training = False

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        elif train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        if args.logs_per_epoch and args.logs_per_epoch > 0:
            progress_interval = max(1, math.ceil(len(train_loader) / args.logs_per_epoch))
        else:
            progress_interval = max(1, int(logging_cfg.get("logging_steps", 25)))
        interval_loss_sum = 0.0
        interval_batch_count = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < resume_step_in_epoch:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            is_accumulating = is_distributed and ((step + 1) % grad_accum != 0) and ((step + 1) != len(train_loader))
            sync_context = model.no_sync() if is_accumulating else contextlib.nullcontext()

            with sync_context:
                if use_amp and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_accum
                    loss.backward()

            train_loss += loss.item() * grad_accum
            interval_loss_sum += loss.item() * grad_accum
            interval_batch_count += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if use_amp and amp_dtype == torch.float16 and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    # Only advance learning rate schedule if optimizer was not skipped during FP16 scale calibration
                    if scale_before <= scale_after:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                if args.empty_cache_freq > 0 and global_step % args.empty_cache_freq == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Periodic mid-epoch rolling checkpoint save
                if save_steps > 0 and global_step % save_steps == 0:
                    # Prune older checkpoints BEFORE saving to ensure disk headroom
                    free_gb = get_free_disk_space_gb(output_dir)
                    if free_gb < 3.0:
                        print(f"\nWarning: Low free disk space ({free_gb:.2f} GB). Aggressively pruning old checkpoints...")
                    prune_checkpoints(output_dir, max(0, save_total_limit - 1), is_main_process=is_main_process)

                    step_dir = output_dir / f"checkpoint-{global_step}"
                    save_training_checkpoint(
                        save_dir=step_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_macro_f1=best_macro_f1,
                        patience_counter=patience_counter,
                        tag_to_id=tag_to_id,
                        id_to_tag=id_to_tag,
                        step_in_epoch=step + 1,
                        is_main_process=is_main_process
                    )
                    # Update symlink/pointer to latest checkpoint without duplicating gigabytes
                    update_latest_checkpoint_link(output_dir, step_dir, is_main_process=is_main_process)
                    prune_checkpoints(output_dir, save_total_limit, is_main_process=is_main_process)

                # Periodic mid-epoch validation evaluation
                if eval_steps > 0 and global_step % eval_steps == 0:
                    best_macro_f1, patience_counter, stop_training = run_evaluation(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        id_to_tag=id_to_tag,
                        tag_to_id=tag_to_id,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        epochs=epochs,
                        global_step=global_step,
                        output_dir=output_dir,
                        is_main_process=is_main_process,
                        is_distributed=is_distributed,
                        best_macro_f1=best_macro_f1,
                        patience_counter=patience_counter,
                        max_patience=max_patience,
                        args=args,
                        mlflow_client=mlflow_client,
                        eval_title=f"Step {global_step} Evaluation",
                        save_total_limit=save_total_limit,
                    )
                    last_eval_step = global_step
                    if stop_training:
                        break

            if is_main_process and ((step + 1) % progress_interval == 0 or (step + 1) == len(train_loader)):
                avg_train_loss = interval_loss_sum / max(1, interval_batch_count)
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Epoch {epoch}/{epochs}] Batch {step + 1}/{len(train_loader)} "
                    f"| Step {global_step}/{total_training_steps} "
                    f"| Loss {avg_train_loss:.4f} | LR {current_lr:.2e}"
                )
                if mlflow_client is not None:
                    try:
                        mlflow_client.log_metrics(
                            {"train/loss": avg_train_loss, "train/learning_rate": current_lr},
                            step=global_step,
                        )
                    except Exception as e:
                        print(f"Warning: MLflow metric logging failed; disabling tracking: {e}")
                        try:
                            mlflow_client.end_run(status="FAILED")
                        except Exception:
                            pass
                        mlflow_client = None
                interval_loss_sum = 0.0
                interval_batch_count = 0

            if args.dry_run and global_step >= 5:
                if is_main_process:
                    print("Dry run test steps completed!")
                break

        if stop_training:
            break

        # End of epoch evaluation (run if not already evaluated on this exact global_step)
        if global_step != last_eval_step:
            best_macro_f1, patience_counter, stop_training = run_evaluation(
                model=model,
                val_loader=val_loader,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                id_to_tag=id_to_tag,
                tag_to_id=tag_to_id,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                epochs=epochs,
                global_step=global_step,
                output_dir=output_dir,
                is_main_process=is_main_process,
                is_distributed=is_distributed,
                best_macro_f1=best_macro_f1,
                patience_counter=patience_counter,
                max_patience=max_patience,
                args=args,
                mlflow_client=mlflow_client,
                eval_title=f"Epoch {epoch} Milestone",
                force_push_hub=True,
                save_total_limit=save_total_limit,
            )
            last_eval_step = global_step

        resume_step_in_epoch = 0
        if stop_training:
            break

        if args.dry_run:
            break

    final_metrics: Dict[str, float] = {}
    if test_loader is not None:
        best_dir = output_dir / "best"
        if best_dir.exists():
            load_checkpoint_weights(model, best_dir)
        _, test_preds, test_labels = evaluate_split(model, test_loader, device, use_amp, amp_dtype, id_to_tag)
        if is_main_process:
            test_metrics = compute_entity_metrics(test_preds, test_labels)
            final_metrics = test_metrics
            print(
                f"\n[Final Gold Benchmark] Macro F1: {test_metrics['macro_f1'] * 100:.2f}% | "
                f"Precision: {test_metrics['precision'] * 100:.2f}% | Recall: {test_metrics['recall'] * 100:.2f}% | "
                f"Micro F1: {test_metrics['micro_f1'] * 100:.2f}%"
            )
            with open(output_dir / "gold_test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            if best_dir.exists():
                with open(best_dir / "gold_test_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, indent=2)

    if is_main_process:
        if not final_metrics and (output_dir / "best_dev_metrics.json").exists():
            try:
                final_metrics = json.loads((output_dir / "best_dev_metrics.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        if not final_metrics:
            final_metrics = {"macro_f1": best_macro_f1}

        print(f"\nTraining completed! Best Dev Macro F1: {best_macro_f1 * 100:.2f}%")
        if mlflow_client is not None and mlflow_client.active_run() is not None:
            try:
                mlflow_client.end_run()
            except Exception as e:
                print(f"Warning: Could not close MLflow run cleanly: {e}")

        # Final push to Hugging Face Hub if requested
        if args.push_to_hub and args.hub_model_id:
            push_folder = output_dir / "best" if (output_dir / "best").exists() else output_dir
            macro_score = final_metrics.get("macro_f1", best_macro_f1)
            push_checkpoint_to_hub(
                repo_id=args.hub_model_id,
                folder_path=push_folder,
                commit_message=f"Final SEA-LION-ModernBERT on Vietnamese Exam Sequence Labelling (Macro F1: {macro_score * 100:.2f}%)",
                metrics=final_metrics,
                args=args,
                tag_to_id=tag_to_id,
                epoch=epochs
            )

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
