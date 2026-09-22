#!/usr/bin/env python3
"""
Full Fine-Tuning Trainer for Vietnamese Sequence Labelling.
Optimized for jhu-clsp/mmBERT-base (280M parameters, 8192 context window) with
Multi-Scale Sliding Windows, Focal Loss, and seqeval entity metrics.
"""

import os
import sys
import json
import yaml
import math
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

from model.module.head import EnhancedBertForTokenClassification, FocalLoss
from model.module.dataset import get_tag_mappings, MultiWindowBIODataset, OnlineAugmentedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train mmBERT-base for Vietnamese Sequence Labelling.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config YAML (default: configs/train_config.yaml)")
    parser.add_argument("--model-name", type=str, default=None, help="Hugging Face model backbone")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing training dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Backbone learning rate")
    parser.add_argument("--head-lr", type=float, default=None, help="Classifier head learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--val-file", type=str, default=None, help="Path to evaluation/validation chunks JSONL file (e.g. data/training_dataset/test_bio_chunks.jsonl)")
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
        # Simple token-level fallback if seqeval not installed
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


def main():
    args = parse_args()
    
    # 1. Load Configuration
    cfg_path = Path(args.config)
    if not cfg_path.exists() and Path("train_config.yaml").exists():
        cfg_path = Path("train_config.yaml")
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        print(f"Loaded training config from '{cfg_path}'.")

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
    grad_accum = args.grad_accum or int(train_cfg.get("gradient_accumulation_steps", 4))
    seed = args.seed or int(data_cfg.get("seed", 42))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n============================================================")
    print(f"Starting mmBERT Sequence Labelling Full Fine-Tuning (NO LoRA)")
    print(f"  Backbone Model  : {model_name}")
    print(f"  Device          : {device}")
    print(f"  Epochs          : {epochs}")
    print(f"  Backbone LR     : {lr}")
    print(f"  Head LR         : {head_lr}")
    print(f"  Batch Size      : {batch_size} (effective: {batch_size * grad_accum})")
    print(f"  Output Dir      : {output_dir}")
    print(f"============================================================\n")

    # 2. Setup Tokenizer and Tag Mappings
    tag_to_id, id_to_tag = get_tag_mappings()
    num_labels = len(tag_to_id)

    # Save label mapping to output dir
    label_map_file = output_dir / "label_mapping.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump({"tag_to_id": tag_to_id, "id_to_tag": id_to_tag}, f, indent=2)

    print(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    latex_placeholder = model_cfg.get("latex_placeholder", "[LATEX]")
    special_tokens = ["<blank />", "<blank/>", "[BLANK]"]
    if latex_placeholder:
        special_tokens.append(latex_placeholder)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.save_pretrained(output_dir)

    # 3. Load Dataset
    chunks_file = Path(data_cfg.get("bio_chunks_file", "data/training_dataset/train_bio_chunks.jsonl"))
    if not chunks_file.exists():
        print(f"Error: BIO chunks file '{chunks_file}' does not exist. Run 'tools/build_training_dataset.py' first.")
        sys.exit(1)

    val_chunks_path = args.val_file or data_cfg.get("val_bio_chunks_file")
    val_chunks_file = Path(val_chunks_path) if val_chunks_path else None

    if val_chunks_file and val_chunks_file.exists():
        print(f"Loading 100% of training chunks from '{chunks_file.name}' for model training...")
        train_dataset = MultiWindowBIODataset(
            chunks_file_path=chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_samples=20 if args.dry_run else None
        )
        print(f"Loading Gold Benchmark test/validation chunks from '{val_chunks_file.name}'...")
        val_dataset = MultiWindowBIODataset(
            chunks_file_path=val_chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_samples=20 if args.dry_run else None
        )
        print(f"Dataset configuration: {len(train_dataset)} training chunks (100%), {len(val_dataset)} validation chunks from Gold Benchmark.")
    else:
        full_dataset = MultiWindowBIODataset(
            chunks_file_path=chunks_file,
            tag_to_id=tag_to_id,
            tokenizer=tokenizer,
            max_samples=20 if args.dry_run else None
        )
        val_ratio = float(data_cfg.get("val_ratio", 0.10))
        if val_ratio > 0.0:
            n_val = max(1, int(len(full_dataset) * val_ratio))
            n_train = len(full_dataset) - n_val
            train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
            print(f"Dataset split (random holdout): {n_train} training chunks, {n_val} validation chunks.")
        else:
            train_dataset = full_dataset
            val_dataset = full_dataset
            print(f"Notice: val_ratio is 0.0 and no separate val_bio_chunks_file found; validating on training set.")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 if os.name == "nt" else int(train_cfg.get("dataloader_num_workers", 2)),
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0 if os.name == "nt" else int(train_cfg.get("dataloader_num_workers", 2))
    )

    # 4. Initialize Enhanced Model
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_tag,
        label2id=tag_to_id
    )
    config.output_hidden_states = True

    base_model = AutoModel.from_pretrained(model_name, config=config)
    base_model.resize_token_embeddings(len(tokenizer))

    model = EnhancedBertForTokenClassification(
        config=config,
        base_model=base_model,
        num_layers_to_fuse=int(head_cfg.get("layers_to_pool", 4)),
        focal_gamma=float(head_cfg.get("focal_gamma", 1.5)),
        label_smoothing=float(head_cfg.get("label_smoothing", 0.0))
    ).to(device)

    # 5. Differential Optimizer (Backbone vs Head)
    no_decay = ["bias", "LayerNorm.weight", "layer_weights"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "lr": lr
        },
        {
            "params": [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr
        },
        {
            "params": [p for n, p in model.head.named_parameters()],
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "lr": head_lr
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    total_training_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_training_steps * float(train_cfg.get("warmup_ratio", 0.10)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    use_amp = bool(train_cfg.get("bf16", False)) or bool(train_cfg.get("fp16", False))
    amp_dtype = torch.bfloat16 if train_cfg.get("bf16", False) and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16 and torch.cuda.is_available()))

    # 6. Training & Evaluation Loop
    best_macro_f1 = -1.0
    global_step = 0
    patience_counter = 0
    max_patience = int(train_cfg.get("early_stopping_patience", 3))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
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

                pbar.set_postfix({
                    "loss": f"{loss.item() * grad_accum:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

            if args.dry_run and global_step >= 5:
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

        avg_val_loss = val_loss / max(1, len(val_loader))
        metrics = compute_entity_metrics(all_preds, all_labels)
        current_macro_f1 = metrics["macro_f1"]

        print(f"\n[Epoch {epoch} Results] Val Loss: {avg_val_loss:.4f} | Macro F1: {current_macro_f1 * 100:.2f}% | Precision: {metrics['precision'] * 100:.2f}% | Recall: {metrics['recall'] * 100:.2f}%")

        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            patience_counter = 0
            print(f"  --> New best model! Saving checkpoint to '{output_dir}'...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= max_patience and not args.dry_run:
                print(f"Early stopping triggered after {patience_counter} evaluations without improvement.")
                break

        if args.dry_run:
            break

    print(f"\nTraining completed! Best Validation Macro F1: {best_macro_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
