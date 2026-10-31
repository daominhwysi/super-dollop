#!/usr/bin/env python3
"""
Benchmark real-world CPU throughput (Tokens/second) of SEA-LION-ModernBERT-300M
on local hardware (Intel Core i3-4130T @ 2.90GHz).
Measures:
1. Pure Model Forward Pass Throughput (Tokens/s)
2. Full Pipeline End-to-End Throughput (Tokenize -> Forward -> Viterbi -> Spans)
"""

import sys
import time
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import torch
from transformers import AutoTokenizer
from model.module.head import EnhancedBertForTokenClassification
from model.inference.predict import predict_text, load_label_mapping

def main():
    model_name = "daominhwysi/sealior-modernbert-vi-exam-seq-labeling"
    device = "cpu"

    print("=" * 70)
    print("  CPU THROUGHPUT BENCHMARK - POTATO RIG EDITION")
    print("  Hardware: Intel(R) Core(TM) i3-4130T @ 2.90GHz (2C/4T)")
    print(f"  Model   : {model_name} (~300M parameters)")
    print("=" * 70 + "\n")

    print("[1/3] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tag_to_id, id_to_tag = load_label_mapping(model_name)
    
    model = EnhancedBertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tag_to_id),
        id2label=id_to_tag,
        label2id=tag_to_id,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # 1. Pure Forward Pass Benchmark
    print("[2/3] Benchmarking Pure Forward Pass Throughput...")
    test_lengths = [128, 256, 512, 1024]
    
    # Warmup
    dummy = torch.randint(100, 1000, (1, 128), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)

    print(f"{'Seq Length':<12} | {'Time/Pass (ms)':<15} | {'Throughput (Tokens/s)':<22}")
    print("-" * 55)

    with torch.no_grad():
        for seq_len in test_lengths:
            inp = torch.randint(100, 1000, (1, seq_len), dtype=torch.long, device=device)
            att = torch.ones((1, seq_len), dtype=torch.long, device=device)
            
            # Run 3 repeats
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = model(inp, attention_mask=att)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            avg_time = sum(times) / len(times)
            tokens_per_sec = seq_len / avg_time
            print(f"{seq_len:<12} | {avg_time * 1000:<15.1f} | {tokens_per_sec:<22.1f}")

    # 2. End-to-End Pipeline Benchmark
    print("\n[3/3] Benchmarking Full End-to-End Pipeline (Text -> Tokenize -> Model -> Viterbi -> XML)...")
    sample_file = WORKSPACE_DIR / "artifacts" / "samples" / "official_math_exam_2025.txt"
    if not sample_file.exists():
        sample_file = WORKSPACE_DIR / "artifacts" / "samples" / "sample_01_math_latex.txt"
    
    raw_text = sample_file.read_text(encoding="utf-8")
    num_tokens = len(tokenizer.tokenize(raw_text))

    print(f"Test File : {sample_file.name}")
    print(f"File Size : {len(raw_text)} chars (~{num_tokens} tokens)")

    # Run without Viterbi (Argmax)
    t0 = time.perf_counter()
    res_argmax = predict_text(
        raw_text,
        model=model,
        tokenizer=tokenizer,
        id_to_tag=id_to_tag,
        device=device,
        max_length=1024,
        stride=512,
        batch_size=1,
        use_viterbi=False
    )
    t_argmax = time.perf_counter() - t0
    tps_argmax = num_tokens / t_argmax

    # Run with Constrained Viterbi
    t0 = time.perf_counter()
    res_viterbi = predict_text(
        raw_text,
        model=model,
        tokenizer=tokenizer,
        id_to_tag=id_to_tag,
        device=device,
        max_length=1024,
        stride=512,
        batch_size=1,
        use_viterbi=True
    )
    t_viterbi = time.perf_counter() - t0
    tps_viterbi = num_tokens / t_viterbi

    print(f"\nResults on {sample_file.name}:")
    print(f"  - Argmax Decoding     : {t_argmax:.2f}s ({tps_argmax:.1f} tokens/s)")
    print(f"  - Constrained Viterbi : {t_viterbi:.2f}s ({tps_viterbi:.1f} tokens/s)")
    print(f"  - Entities Detected   : {len(res_viterbi['spans'])} spans")
    print("=" * 70)

if __name__ == "__main__":
    main()
