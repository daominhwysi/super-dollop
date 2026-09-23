#!/usr/bin/env python3
"""
Full Fine-Tuning Trainer for Vietnamese Sequence Labelling.
Optimized for jhu-clsp/mmBERT-base (280M parameters, 8192 context window) with
Multi-Scale Sliding Windows, Focal Loss, Distributed Training (DDP/torchrun),
and Hugging Face Hub integration.
"""

import os
import sys
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
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv(dotenv_path=WORKSPACE_DIR / ".env")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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
from model.module.dataset import get_tag_mappings, MultiWindowBIODataset, OnlineAugmentedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train mmBERT-base for Vietnamese Sequence Labelling.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config YAML (default: configs/train_config.yaml)")
    
    # Model configuration
    parser.add_argument("--model-name", "--model_name", type=str, default=None, help="Hugging Face model backbone")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--enhanced-head", "--enhanced_head", action="store_true", default=True, help="Enable enhanced head with multi-sample dropout and layer pooling")
    parser.add_argument("--no-lora", "--no_lora", action="store_true", default=False, help="Full fine-tuning without LoRA")
    parser.add_argument("--focal-gamma", "--focal_gamma", type=float, default=None, help="Focal loss gamma parameter")
    parser.add_argument("--label-smoothing", "--label_smoothing", type=float, default=None, help="Label smoothing rate")
    
    # Dataset options
    parser.add_argument("--dataset-repo-id", "--dataset_repo_id", type=str, default=None, help="Hugging Face dataset repo ID (e.g. daominhwysi/synthetic-seq-labelling-vi-exam-v2)")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None, help="Directory containing training dataset")
    parser.add_argument("--val-file", "--val_file", type=str, default=None, help="Path to evaluation/validation chunks JSONL file (e.g. data/training_dataset/test_bio_chunks.jsonl)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Backbone learning rate")
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
    parser.add_argument("--save-steps", "--save_steps", type=int, default=None, help="Save a rolling checkpoint every N steps (0 to disable)")
    parser.add_argument("--save-total-limit", "--save_total_limit", type=int, default=None, help="Maximum number of rolling step checkpoints to keep (default: 2)")
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true", default=False, help="Push checkpoint to Hugging Face Hub during and after training")
    parser.add_argument("--hub-model-id", "--hub_model_id", type=str, default=None, help="Hugging Face model repository ID")
    
    # Hardware precision & Mixed Precision (AMP)
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable FP16 mixed precision training (recommended for NVIDIA T4/V100 GPUs)")
    parser.add_argument("--bf16", action="store_true", default=None, help="Enable BF16 mixed precision training (for Ampere+ GPUs)")
    parser.add_argument("--no-amp", "--no_amp", action="store_true", default=False, help="Disable Automatic Mixed Precision (run in full FP32)")
    parser.add_argument("--max-length", "--max_length", "--max-seq-length", type=int, default=None, help="Maximum token sequence length (defaults to 2048)")
    parser.add_argument("--empty-cache-freq", "--empty_cache_freq", type=int, default=0, help="Periodically empty CUDA cache every N steps (0 to disable)")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-dagshub", "--no_dagshub", action="store_true", default=False, help="Disable DagsHub tracking even if enabled in config")
    parser.add_argument("--no-mlflow", "--no_mlflow", action="store_true", default=False, help="Disable MLflow tracking completely")
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
    is_main_process: bool = True
):
    """Saves model weights, config, tokenizer, label map, and full trainer state for continuity."""
    if not is_main_process:
        return

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
        "best_macro_f1": best_macro_f1,
        "patience_counter": patience_counter,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    torch.save(trainer_state, save_dir / "trainer_state.pt")


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


def push_checkpoint_to_hub(repo_id: str, folder_path: Path, commit_message: str):
    """Pushes the primary model checkpoint to Hugging Face Hub (excluding intermediate checkpoints)."""
    try:
        from huggingface_hub import HfApi
        print(f"\nPushing checkpoint to Hugging Face Hub: '{repo_id}'...")
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["checkpoint-*", "checkpoint-*/*", "*.tmp"]
        )
        print(f"Checkpoint successfully synchronized to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Notice: Failed to push checkpoint to Hugging Face Hub: {e}")


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
    head_cfg = cfg.get("head", {})
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    model_name = args.model_name or model_cfg.get("name_or_path", "jhu-clsp/mmBERT-base")
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "results/mmbert_seqlabel_v2"))
    epochs = args.epochs or train_cfg.get("num_train_epochs", 4)
    lr = args.lr or float(train_cfg.get("learning_rate", 2.5e-5))
    head_lr = args.head_lr or float(train_cfg.get("head_learning_rate", 5.0e-5))
    batch_size = args.batch_size or int(train_cfg.get("per_device_train_batch_size", 4))
    eval_batch_size = args.eval_batch_size or int(train_cfg.get("per_device_eval_batch_size", 4))
    grad_accum = args.gradient_accumulation_steps or int(train_cfg.get("gradient_accumulation_steps", 4))
    weight_decay = args.weight_decay or float(train_cfg.get("weight_decay", 0.01))
    scheduler_type = args.lr_scheduler_type or train_cfg.get("lr_scheduler_type", "cosine")
    warmup_ratio = args.warmup_ratio or float(train_cfg.get("warmup_ratio", 0.10))
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else float(head_cfg.get("focal_gamma", 1.5))
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else float(head_cfg.get("label_smoothing", 0.0))
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
    max_length = args.max_length or 2048
    save_steps = args.save_steps if args.save_steps is not None else int(train_cfg.get("save_steps", 0))
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
    if latex_placeholder:
        special_tokens.append(latex_placeholder)
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
            print("Please pass --dataset_repo_id or run 'tools/build_training_dataset.py' first.")
        sys.exit(1)

    # 4. Load Datasets
    if val_chunks_file and val_chunks_file.exists():
        if is_main_process:
            print(f"Loading 100% of training chunks from '{chunks_file.name}' for training...")
        train_dataset = MultiWindowBIODataset(
            chunks_file_path=chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=20 if args.dry_run else None
        )
        if is_main_process:
            print(f"Loading evaluation chunks from Gold Benchmark test set '{val_chunks_file.name}'...")
        val_dataset = MultiWindowBIODataset(
            chunks_file_path=val_chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=20 if args.dry_run else None
        )
        if is_main_process:
            print(f"Dataset split: {len(train_dataset)} training chunks (100%), {len(val_dataset)} validation chunks (Gold Benchmark).")
    else:
        full_dataset = MultiWindowBIODataset(
            chunks_file_path=chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=20 if args.dry_run else None
        )
        val_ratio = float(data_cfg.get("val_ratio", 0.0))
        if val_ratio > 0.0:
            n_val = max(1, int(len(full_dataset) * val_ratio))
            n_train = len(full_dataset) - n_val
            train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
            if is_main_process:
                print(f"Dataset split: {n_train} training chunks, {n_val} validation chunks.")
        else:
            train_dataset = full_dataset
            val_dataset = full_dataset
            if is_main_process:
                print(f"Notice: Validating on training set ({len(train_dataset)} chunks).")

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

    # Distributed or standard samplers
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    workers = int(args.dataloader_num_workers if args.dataloader_num_workers is not None else train_cfg.get("dataloader_num_workers", 2))
    if os.name == "nt":
        workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    config.output_hidden_states = True

    base_model = AutoModel.from_pretrained(model_name, config=config)
    base_model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if is_main_process:
            print("Gradient checkpointing enabled.")

    model = EnhancedBertForTokenClassification(
        config=config,
        base_model=base_model,
        num_layers_to_fuse=int(head_cfg.get("layers_to_pool", 4)),
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing
    ).to(device)

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
    no_decay = ["bias", "LayerNorm.weight", "layer_weights"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in raw_model.base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr
        },
        {
            "params": [p for n, p in raw_model.base_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr
        },
        {
            "params": [p for n, p in raw_model.head.named_parameters()],
            "weight_decay": weight_decay,
            "lr": head_lr
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    total_training_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    # 7. Training State & Continuity Restoration
    start_epoch = 1
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
                saved_epoch = trainer_state.get("epoch", 0)
                start_epoch = saved_epoch + 1
                global_step = trainer_state.get("global_step", 0)
                best_macro_f1 = trainer_state.get("best_macro_f1", -1.0)
                patience_counter = trainer_state.get("patience_counter", 0)

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

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
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
                        is_main_process=is_main_process
                    )
                    # Update symlink/pointer to latest checkpoint without duplicating gigabytes
                    update_latest_checkpoint_link(output_dir, step_dir, is_main_process=is_main_process)
                    prune_checkpoints(output_dir, save_total_limit, is_main_process=is_main_process)

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

        # Validation at end of epoch
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_amp and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                lbls = labels.cpu().numpy()

                for p_seq, l_seq in zip(preds, lbls):
                    seq_pred = []
                    seq_lbl = []
                    for p, l in zip(p_seq, l_seq):
                        if l != -100:
                            seq_pred.append(id_to_tag.get(p, "O"))
                            seq_lbl.append(id_to_tag.get(l, "O"))
                    all_preds.append(seq_pred)
                    all_labels.append(seq_lbl)

        stop_training = False
        if is_main_process:
            avg_val_loss = val_loss / max(1, len(val_loader))
            metrics = compute_entity_metrics(all_preds, all_labels)
            current_macro_f1 = metrics["macro_f1"]

            print(f"\n[Epoch {epoch} Evaluation (Gold Test Set)] Val Loss: {avg_val_loss:.4f} | Macro F1: {current_macro_f1 * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | Recall: {metrics['recall'] * 100:.2f}%")
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
                print(f"  --> New best model checkpoint! Saving to '{output_dir}'...")
                save_training_checkpoint(
                    save_dir=output_dir,
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
                update_latest_checkpoint_link(output_dir, output_dir, is_main_process=is_main_process)
            else:
                patience_counter += 1
                if patience_counter >= max_patience and not args.dry_run:
                    print(f"Early stopping triggered after {patience_counter} evaluations without improvement.")
                    stop_training = True

                # Only save to latest_checkpoint if output_dir was not updated with a new best
                save_training_checkpoint(
                    save_dir=output_dir / "latest_checkpoint",
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

            # Auto-sync to Hugging Face Hub at end of each epoch if push_to_hub is active
            if args.push_to_hub and args.hub_model_id:
                push_checkpoint_to_hub(
                    repo_id=args.hub_model_id,
                    folder_path=output_dir,
                    commit_message=f"Epoch {epoch}/{epochs} - Macro F1: {current_macro_f1 * 100:.2f}% (Best: {best_macro_f1 * 100:.2f}%)"
                )

        if is_distributed:
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item() == 1)

        if stop_training:
            break

        if args.dry_run:
            break

    if is_main_process:
        print(f"\nTraining completed! Best Validation Macro F1: {best_macro_f1 * 100:.2f}%")
        if mlflow_client is not None and mlflow_client.active_run() is not None:
            try:
                mlflow_client.end_run()
            except Exception as e:
                print(f"Warning: Could not close MLflow run cleanly: {e}")

        # Final push to Hugging Face Hub if requested
        if args.push_to_hub and args.hub_model_id:
            push_checkpoint_to_hub(
                repo_id=args.hub_model_id,
                folder_path=output_dir,
                commit_message=f"Final mmBERT on Vietnamese Exam Sequence Labelling (Macro F1: {best_macro_f1 * 100:.2f}%)"
            )

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
