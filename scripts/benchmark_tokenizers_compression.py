#!/usr/bin/env python3
"""
Comprehensive Tokenizer Compression Benchmark:
aisingapore/SEA-LION-ModernBERT-300M vs. jhu-clsp/mmBERT-base
with control baselines (answerdotai/ModernBERT-base, vinai/phobert-base-v2, xlm-roberta-base).

Evaluates:
1. 63 Gold Benchmark Documents (annotated exams, tiers, subjects)
2. Stratified Diverse Raw Exams (Math, Science, Literature, Social Science, English)
3. Granular Linguistic Probes (Prose, Diacritics, LaTeX, Layout, Bilingual, Whitespace)
4. Comprehensive Vietnamese Syllable Dictionary Fragmentation Audit
5. Context Window Capacity & Sliding Window Simulation
6. Tokenization Latency & Throughput
"""

import os
import sys
import re
import time
import json
import glob
from collections import defaultdict
from transformers import AutoTokenizer

def strip_tags(text: str) -> str:
    """Strip XML-like tags to evaluate on raw source document text."""
    return re.sub(r'<[^>]+>', '', text)

def count_words(text: str) -> int:
    return len(text.split())

def main():
    print("=" * 80)
    print("TOKENIZER COMPRESSION BENCHMARK")
    print("SEA-LION-ModernBERT-300M vs. mmBERT-base on Vietnamese")
    print("=" * 80)

    # 1. Load Tokenizers
    print("\n[1/6] Loading tokenizers...")
    model_ids = {
        "SEA-LION-300M": "aisingapore/SEA-LION-ModernBERT-300M",
        "mmBERT-base": "jhu-clsp/mmBERT-base",
        "ModernBERT-base": "answerdotai/ModernBERT-base",
        "PhoBERT-base": "vinai/phobert-base-v2",
        "XLM-RoBERTa-base": "xlm-roberta-base"
    }

    tokenizers = {}
    for name, m_id in model_ids.items():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m_id, trust_remote_code=True)
        tokenizers[name] = tok
        print(f"  Loaded {name:16s} ({m_id}) | Vocab: {tok.vocab_size:,} in {time.time()-t0:.2f}s")

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {k: {"id": v, "vocab_size": tokenizers[k].vocab_size} for k, v in model_ids.items()}
        }
    }

    # 2. Benchmark Corpus 1: 63 Gold Benchmark Documents
    print("\n[2/6] Benchmarking 63 Gold Benchmark Documents...")
    gold_set_path = "data/benchmark_gold_set.json"
    if not os.path.exists(gold_set_path):
        print(f"Error: {gold_set_path} not found!")
        sys.exit(1)

    with open(gold_set_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_docs = gold_data.get("documents", [])
    print(f"  Found {len(gold_docs)} gold documents.")

    gold_stats = {
        "total_docs": 0,
        "total_words": 0,
        "total_chars": 0,
        "total_bytes": 0,
        "tokens": {k: 0 for k in tokenizers},
        "by_tier": defaultdict(lambda: {"docs": 0, "words": 0, "tokens": {k: 0 for k in tokenizers}})
    }

    for doc in gold_docs:
        doc_dir = doc.get("path", "")
        tier = str(doc.get("tier", "unknown"))
        xml_path = os.path.join(doc_dir, "merged.xml")
        if not os.path.exists(xml_path):
            continue

        with open(xml_path, "r", encoding="utf-8") as f:
            raw_text = strip_tags(f.read())

        words = count_words(raw_text)
        chars = len(raw_text)
        bytes_len = len(raw_text.encode("utf-8"))

        gold_stats["total_docs"] += 1
        gold_stats["total_words"] += words
        gold_stats["total_chars"] += chars
        gold_stats["total_bytes"] += bytes_len

        gold_stats["by_tier"][tier]["docs"] += 1
        gold_stats["by_tier"][tier]["words"] += words

        for name, tok in tokenizers.items():
            tok_count = len(tok.encode(raw_text, add_special_tokens=False))
            gold_stats["tokens"][name] += tok_count
            gold_stats["by_tier"][tier]["tokens"][name] += tok_count

    print(f"  Successfully processed {gold_stats['total_docs']} documents:")
    print(f"  Words: {gold_stats['total_words']:,} | Chars: {gold_stats['total_chars']:,} | Bytes: {gold_stats['total_bytes']:,}")
    print("\n  Gold Set Overall Results:")
    for name in tokenizers:
        t_count = gold_stats["tokens"][name]
        fert = t_count / gold_stats["total_words"]
        c_per_t = gold_stats["total_chars"] / t_count
        b_per_t = gold_stats["total_bytes"] / t_count
        print(f"    {name:16s}: {t_count:8,d} tokens | Fertility: {fert:.3f} tok/word | Chars/tok: {c_per_t:.2f} | Bytes/tok: {b_per_t:.2f}")

    mm_t = gold_stats["tokens"]["mmBERT-base"]
    sl_t = gold_stats["tokens"]["SEA-LION-300M"]
    mod_t = gold_stats["tokens"]["ModernBERT-base"]
    print(f"  --> SEA-LION vs mmBERT compression: {(mm_t - sl_t) / mm_t * 100:+.2f}%")
    print(f"  --> SEA-LION vs ModernBERT compression: {(mod_t - sl_t) / mod_t * 100:+.2f}%")
    print(f"  --> mmBERT vs ModernBERT compression: {(mod_t - mm_t) / mod_t * 100:+.2f}%")

    results["gold_set"] = gold_stats

    # 3. Benchmark Corpus 2: Stratified Diverse Raw Exams (500 Raw Docs Directory)
    print("\n[3/6] Benchmarking Diverse Subject Raw Exams (data/batch_500_diverse_raw/)...")
    raw_files = sorted(glob.glob("data/batch_500_diverse_raw/*.md"))
    print(f"  Found {len(raw_files)} raw documents.")

    subject_keywords = {
        "Toán (Math)": ["toán", "toan"],
        "Vật lý (Physics)": ["vật lý", "vật lí", "vat ly"],
        "Hóa học (Chemistry)": ["hóa học", "hóa", "hoa hoc"],
        "Sinh học (Biology)": ["sinh học", "sinh", "sinh hoc"],
        "Ngữ văn (Literature)": ["ngữ văn", "văn", "ngu van"],
        "Lịch sử / Địa lý (Social)": ["lịch sử", "địa lý", "địa lí", "lich su", "dia ly", "gdcd", "kinh tế"],
        "Tiếng Anh (English)": ["tiếng anh", "anh văn", "english", "tieng anh"]
    }

    # Group files into subjects
    subject_docs = defaultdict(list)
    for fpath in raw_files:
        fname_lower = os.path.basename(fpath).lower()
        matched = False
        for subj, kw_list in subject_keywords.items():
            if any(kw in fname_lower for kw in kw_list):
                subject_docs[subj].append(fpath)
                matched = True
                break
        if not matched:
            subject_docs["Khác (Other)"].append(fpath)

    print("  Subject distribution across raw documents:")
    for subj, flist in subject_docs.items():
        print(f"    {subj:26s}: {len(flist)} files")

    # Select stratified sample (up to 15 per subject)
    subject_stats = {}
    for subj, flist in subject_docs.items():
        sampled_files = flist[:15]
        s_words = 0
        s_chars = 0
        s_bytes = 0
        s_tokens = {k: 0 for k in tokenizers}

        for fpath in sampled_files:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            w = count_words(text)
            c = len(text)
            b = len(text.encode("utf-8"))
            s_words += w
            s_chars += c
            s_bytes += b

            for name, tok in tokenizers.items():
                s_tokens[name] += len(tok.encode(text, add_special_tokens=False))

        fert = {k: s_tokens[k] / s_words if s_words > 0 else 0 for k in tokenizers}
        comp_vs_mm = (s_tokens["mmBERT-base"] - s_tokens["SEA-LION-300M"]) / s_tokens["mmBERT-base"] * 100 if s_tokens["mmBERT-base"] > 0 else 0
        comp_vs_mod = (s_tokens["ModernBERT-base"] - s_tokens["SEA-LION-300M"]) / s_tokens["ModernBERT-base"] * 100 if s_tokens["ModernBERT-base"] > 0 else 0

        subject_stats[subj] = {
            "doc_count": len(sampled_files),
            "words": s_words,
            "chars": s_chars,
            "bytes": s_bytes,
            "tokens": s_tokens,
            "fertility": fert,
            "compression_vs_mmbert_pct": comp_vs_mm,
            "compression_vs_modernbert_pct": comp_vs_mod
        }
        print(f"\n  [{subj}] ({len(sampled_files)} docs, {s_words:,} words):")
        print(f"    SEA-LION:    {s_tokens['SEA-LION-300M']:7,d} tokens ({fert['SEA-LION-300M']:.3f} tok/w)")
        print(f"    mmBERT:      {s_tokens['mmBERT-base']:7,d} tokens ({fert['mmBERT-base']:.3f} tok/w)")
        print(f"    ModernBERT:  {s_tokens['ModernBERT-base']:7,d} tokens ({fert['ModernBERT-base']:.3f} tok/w)")
        print(f"    PhoBERT:     {s_tokens['PhoBERT-base']:7,d} tokens ({fert['PhoBERT-base']:.3f} tok/w)")
        print(f"    XLM-R:       {s_tokens['XLM-RoBERTa-base']:7,d} tokens ({fert['XLM-RoBERTa-base']:.3f} tok/w)")
        print(f"    -> SEA-LION vs mmBERT: {comp_vs_mm:+.2f}% | vs ModBERT: {comp_vs_mod:+.2f}%")

    results["subject_breakdown"] = subject_stats

    # 4. Benchmark Corpus 3: Micro-Linguistic & Domain Probes
    print("\n[4/6] Benchmarking Micro-Linguistic & Domain Probes...")
    probes = {
        "vietnamese_prose_standard": (
            "Tiếng Việt là một ngôn ngữ có thanh điệu, giàu sức biểu cảm và có truyền thống văn học lâu đời. "
            "Từ những câu ca dao, dân ca mượt mà đến các tác phẩm văn học hiện đại của Thạch Lam, Nam Cao, "
            "vẻ đẹp của tiếng mẹ đẻ luôn là niềm tự hào to lớn của mỗi người dân Việt Nam."
        ),
        "vietnamese_complex_diacritics": (
            "chim phượng hoàng bay qua đỉnh núi cheo leo giữa mây mù hoằng hoá khuếch đại thanh thế. "
            "tiếng suối róc rách len lỏi qua từng kẽ đá quằn quại khúc khuỷu ngoằn ngoèo. "
            "nghiêng nghiêng bóng trúc quyến rũ tuyệt trần, trăng khuyết lấp lánh sương khuya hoang hoải."
        ),
        "exam_layout_delimiters": (
            "## Câu 1: Cho hàm số y = f(x) liên tục trên đoạn [-1; 3] và có đồ thị như hình vẽ.\n"
            "A. Hàm số đạt cực đại tại x = 1.    B. Giá trị cực tiểu của hàm số bằng -2.\n"
            "C. Đồ thị hàm số có hai điểm cực trị.    D. Hàm số đồng biến trên khoảng (1; 3).\n"
            "## Câu 2: Trong không gian Oxyz, tìm tọa độ tâm I và bán kính R của mặt cầu (S):\n"
            "| x | -\\infty | 0 | 2 | +\\infty |\n"
            "|---|---|---|---|---|\n"
            "| y' | + | 0 | - | 0 | + |\n"
        ),
        "stem_pure_latex": (
            r"\int_{0}^{\pi/2} \frac{\sin^3(x)}{\sin^3(x) + \cos^3(x)} dx = \frac{\pi}{4}. "
            r"\lim_{x \to 0} \frac{\sqrt{1+2x} - \sqrt[3]{1+3x}}{x^2} = \frac{1}{2}. "
            r"\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos(\alpha). "
            r"\Delta = b^2 - 4ac > 0 \implies x_{1,2} = \frac{-b \pm \sqrt{\Delta}}{2a}."
        ),
        "english_cloze_bilingual": (
            "Read the following passage and mark the letter A, B, C, or D on your answer sheet.\n"
            "Climate change is undoubtedly one of the most pressing challenges of our time. "
            "Rising temperatures have triggered unprecedented weather anomalies, leading to severe droughts, "
            "catastrophic floods, and widespread biodiversity loss across all continents.\n"
            "Question 31: According to paragraph 1, what has climate change triggered?"
        ),
        "consecutive_whitespace_and_tabs": (
            "Câu 1.    A. x = 1        B. x = 2        C. x = 3        D. x = 4\n"
            "Câu 2.    A. Đúng         B. Sai          C. Không xác định\n"
            "                    Phần I: Trắc nghiệm khách quan (7,0 điểm)\n"
        )
    }

    probe_stats = {}
    for p_name, p_text in probes.items():
        w = count_words(p_text)
        c = len(p_text)
        tokens = {k: len(tokenizers[k].encode(p_text, add_special_tokens=False)) for k in tokenizers}
        fert = {k: tokens[k] / w for k in tokenizers}
        comp_mm = (tokens["mmBERT-base"] - tokens["SEA-LION-300M"]) / tokens["mmBERT-base"] * 100
        comp_mod = (tokens["ModernBERT-base"] - tokens["SEA-LION-300M"]) / tokens["ModernBERT-base"] * 100

        probe_stats[p_name] = {
            "words": w,
            "chars": c,
            "tokens": tokens,
            "fertility": fert,
            "compression_vs_mmbert_pct": comp_mm,
            "compression_vs_modernbert_pct": comp_mod
        }
        print(f"\n  [Probe: {p_name}] ({w} words):")
        print(f"    SEA-LION:    {tokens['SEA-LION-300M']} tok ({fert['SEA-LION-300M']:.2f} t/w)")
        print(f"    mmBERT:      {tokens['mmBERT-base']} tok ({fert['mmBERT-base']:.2f} t/w)")
        print(f"    ModernBERT:  {tokens['ModernBERT-base']} tok ({fert['ModernBERT-base']:.2f} t/w)")
        print(f"    PhoBERT:     {tokens['PhoBERT-base']} tok ({fert['PhoBERT-base']:.2f} t/w)")
        print(f"    XLM-R:       {tokens['XLM-RoBERTa-base']} tok ({fert['XLM-RoBERTa-base']:.2f} t/w)")
        print(f"    -> SEA-LION vs mmBERT: {comp_mm:+.2f}% | vs ModBERT: {comp_mod:+.2f}%")

    results["linguistic_probes"] = probe_stats

    # 5. Benchmark Corpus 4: Vietnamese Syllable Preservability & Fragmentation Audit
    print("\n[5/6] Vietnamese Syllable Preservability & Fragmentation Audit...")
    # Generate canonical list of ~3,500 common Vietnamese syllables by combining initials, nuclei, finals, and tones
    # Or extract all unique syllables appearing across our 63 gold documents
    gold_syllables = set()
    for doc in gold_docs:
        xml_path = os.path.join(doc.get("path", ""), "merged.xml")
        if os.path.exists(xml_path):
            with open(xml_path, "r", encoding="utf-8") as f:
                raw_text = strip_tags(f.read())
            # Extract vietnamese words
            words = re.findall(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+', raw_text)
            for w in words:
                if len(w) <= 7: # Standard Vietnamese syllable length
                    gold_syllables.add(w.lower())

    print(f"  Extracted {len(gold_syllables):,} unique real-world Vietnamese syllables from Gold Set corpus.")

    syllable_audit = {
        "total_syllables": len(gold_syllables),
        "models": {}
    }

    for name in ["SEA-LION-300M", "mmBERT-base", "ModernBERT-base", "PhoBERT-base", "XLM-RoBERTa-base"]:
        tok = tokenizers[name]
        frag_counts = defaultdict(int) # 1: single token, 2: 2 tokens, 3: 3 tokens, etc.
        byte_fallback_count = 0

        for syl in gold_syllables:
            # Test in standard mid-sentence position: " " + syllable
            pieces = tok.tokenize(" " + syl)
            frag_counts[len(pieces)] += 1
            if any("<0x" in p or "" in p or len(p.encode("utf-8", "ignore")) != len(p.encode("utf-8")) for p in pieces):
                byte_fallback_count += 1

        total = len(gold_syllables)
        p1 = frag_counts[1] / total * 100
        p2 = frag_counts[2] / total * 100
        p3plus = sum(frag_counts[k] for k in frag_counts if k >= 3) / total * 100

        syllable_audit["models"][name] = {
            "unfragmented_1_token_pct": p1,
            "fragmented_2_tokens_pct": p2,
            "fragmented_3plus_tokens_pct": p3plus,
            "byte_fallbacks": byte_fallback_count,
            "distribution": dict(frag_counts)
        }
        print(f"\n  [{name}] Syllable Fragmentation Audit ({total:,} syllables):")
        print(f"    Preserved as 1 token (Crisp Boundary): {frag_counts[1]:5,d} ({p1:5.2f}%)")
        print(f"    Split into 2 tokens:                  {frag_counts[2]:5,d} ({p2:5.2f}%)")
        print(f"    Split into 3+ tokens:                 {sum(frag_counts[k] for k in frag_counts if k >= 3):5,d} ({p3plus:5.2f}%)")
        print(f"    Byte fallbacks:                       {byte_fallback_count:5,d}")

    results["syllable_audit"] = syllable_audit

    # 6. Context Window & Sliding Window Simulation
    print("\n[6/6] Context Window & Sliding Window Simulation (Max Window = 8,192 tokens)...")
    # ModernBERT window = 8,192
    context_sim = {}
    max_ctx = 8192

    for name in ["SEA-LION-300M", "mmBERT-base", "ModernBERT-base", "PhoBERT-base"]:
        fert = gold_stats["tokens"][name] / gold_stats["total_words"]
        est_words_in_window = int(max_ctx / fert) if fert > 0 else 0
        est_chars_in_window = int(est_words_in_window * (gold_stats["total_chars"] / gold_stats["total_words"]))

        # Sliding window counts for standard exam lengths: 2000, 4000, 6000 words
        # Window size 1024, stride 768
        w_counts = {}
        for doc_w in [2000, 4000, 6000]:
            tok_len = int(doc_w * fert)
            # Sliding window formula: ceil(max(0, tok_len - window) / stride) + 1
            if tok_len <= 1024:
                n_wins = 1
            else:
                n_wins = 1 + (tok_len - 1024 + 767) // 768
            w_counts[f"doc_{doc_w}_words"] = {"tokens": tok_len, "windows_1024": n_wins}

        context_sim[name] = {
            "fertility_gold": fert,
            "words_in_8192_window": est_words_in_window,
            "chars_in_8192_window": est_chars_in_window,
            "sliding_windows": w_counts
        }

    results["context_simulation"] = context_sim

    # Tokenizer Throughput benchmark
    print("\n  Measuring Tokenizer Throughput (100k words test)...")
    sample_corpus_text = ""
    for doc in gold_docs[:10]:
        xml_p = os.path.join(doc.get("path", ""), "merged.xml")
        if os.path.exists(xml_p):
            with open(xml_p, "r", encoding="utf-8") as f:
                sample_corpus_text += strip_tags(f.read()) + "\n"

    test_words = count_words(sample_corpus_text)
    throughput_stats = {}
    for name in ["SEA-LION-300M", "mmBERT-base", "ModernBERT-base"]:
        tok = tokenizers[name]
        # Warmup
        tok.encode(sample_corpus_text[:2000], add_special_tokens=False)
        t_start = time.time()
        iters = 5
        for _ in range(iters):
            tok.encode(sample_corpus_text, add_special_tokens=False)
        elapsed = time.time() - t_start
        wps = (test_words * iters) / elapsed
        throughput_stats[name] = {
            "words_per_sec": int(wps),
            "elapsed_s": elapsed
        }
        print(f"    {name:16s}: {int(wps):,d} words/sec (took {elapsed:.2f}s for {test_words*iters:,} words)")

    results["throughput"] = throughput_stats

    # Save complete JSON
    out_json = "artifacts/reports/benchmark_tokenizers_compression_data.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Saved comprehensive benchmark data to {out_json}")

if __name__ == "__main__":
    main()
