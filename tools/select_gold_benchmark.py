#!/usr/bin/env python3
"""
Structural Diversity-Driven Gold Benchmark Selection on Token Scale Across Difficulty Tiers.

Dynamically selects a non-overfitted, structurally diverse Gold Benchmark set sized
by a TARGET TOKEN BUDGET (rather than arbitrary document counts) by:
1. Measuring exact ground-truth source token counts per document.
2. Stratifying documents across difficulty tiers based on audit scores:
   - Tier 1: Pristine Baseline [98-100] (Target: 30% of tokens)
   - Tier 2: Strong Real-World [95-98) (Target: 30% of tokens)
   - Tier 3: Moderate / Noisy [90-95)  (Target: 25% of tokens)
   - Tier 4: Stress / Edge Cases [80-90) (Target: 15% of tokens)
3. Vectorizing 13-D document structural features (tag bigrams, cloze ratio, stimuli, figures, LaTeX, modality).
4. Applying Constrained Farthest-First Traversal (Max-Min Facility Dispersion) on the token scale until each tier's token quota is filled.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def load_pdf_type_mapping(report_path: Path) -> Dict[str, str]:
    """Loads document modality (DIGITAL vs SCANNED) from pdf_classification_report.json."""
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        mapping = {}
        for item in data.get("files", []):
            stem = os.path.splitext(item.get("file_name", ""))[0]
            if stem:
                mapping[stem] = item.get("type", "UNKNOWN")
        return mapping
    except Exception as e:
        print(f"Warning: could not load PDF classification report: {e}")
        return {}


def extract_structural_vector_and_tokens(
    xml_text: str,
    merged_json: dict,
    is_digital: bool
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Extracts ground truth source token count and a 13-dimensional structural vector.
    """
    # Ground truth source text (stripping XML tags)
    raw_text = re.sub(r"<[^>]+>", "", xml_text)
    # Token count matching NLP token classification (words + punctuation)
    token_count = len(re.findall(r"\w+|[^\w\s]", raw_text))

    qs = merged_json.get("questions", [])
    stims = merged_json.get("stimuli", [])
    q_count = max(1, len(qs))

    # Option statistics
    opt_lens = [len(q.get("options", [])) for q in qs]
    avg_opts = float(np.mean(opt_lens)) if opt_lens else 0.0
    std_opts = float(np.std(opt_lens)) if opt_lens else 0.0

    # Cloze / stemless questions
    stemless_count = sum(1 for q in qs if not q.get("stem", "").strip() and q.get("options"))
    stemless_ratio = stemless_count / q_count

    # Stimuli & passage density
    stim_count = len(stims)
    stim_ratio = stim_count / q_count

    # Figures, tables, sections, explanations
    fig_count = xml_text.count("<figure")
    fig_ratio = fig_count / q_count

    table_count = xml_text.count("|---|")
    table_ratio = table_count / q_count

    sec_count = xml_text.count("<section")
    sec_ratio = sec_count / q_count

    expl_count = xml_text.count("<explanation")
    expl_ratio = expl_count / q_count

    # LaTeX math formula density
    latex_matches = re.findall(r"\$[^$]+\$", xml_text)
    latex_density = len(latex_matches) / max(1, len(xml_text.split()))

    # Tag transition bigrams
    tags = [t for t in re.findall(r"</?([a-z_]+)", xml_text) if not t.startswith("/")]
    pairs = list(zip(tags[:-1], tags[1:]))
    total_pairs = max(1, len(pairs))
    pair_counts = Counter(pairs)

    p_q_to_opt = pair_counts.get(("question_label", "option_label"), 0) / total_pairs
    p_stim_to_q = pair_counts.get(("stimulus", "question_label"), 0) / total_pairs
    p_opt_to_expl = pair_counts.get(("option_text", "explanation"), 0) / total_pairs

    vec = np.array([
        avg_opts,
        std_opts,
        stemless_ratio,
        stim_ratio,
        fig_ratio,
        table_ratio,
        sec_ratio,
        expl_ratio,
        latex_density,
        p_q_to_opt,
        p_stim_to_q,
        p_opt_to_expl,
        1.0 if is_digital else 0.0
    ], dtype=float)

    meta = {
        "token_count": token_count,
        "word_count": len(raw_text.split()),
        "q_count": len(qs),
        "stim_count": stim_count,
        "stemless_count": stemless_count,
        "fig_count": fig_count,
        "table_count": table_count,
        "expl_count": expl_count,
        "sec_count": sec_count,
        "latex_count": len(latex_matches),
        "avg_opts": round(avg_opts, 2)
    }
    return vec, token_count, meta


def select_diverse_token_budget_stratified(
    candidates: List[Dict[str, Any]],
    tier_token_quotas: Dict[str, int],
    max_doc_tokens: int = 20000
) -> List[Dict[str, Any]]:
    """
    Applies Constrained Farthest-First Traversal on token budgets.
    Picks structurally diverse candidates until each tier's token quota is filled.
    """
    X = np.array([c["vec"] for c in candidates])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-6
    X_norm = (X - mean) / std

    # Start with the structural medoid of Tier 1 (within size cap)
    t1_name = "Tier 1 [98-100]"
    t1_indices = [
        i for i, c in enumerate(candidates)
        if c["tier"] == t1_name and c["token_count"] <= max_doc_tokens
    ]
    if not t1_indices:
        t1_indices = list(range(len(candidates)))

    t1_sub = X_norm[t1_indices]
    dist_sub = np.linalg.norm(t1_sub[:, None, :] - t1_sub[None, :, :], axis=-1)
    medoid_idx = t1_indices[int(np.argmin(dist_sub.sum(axis=1)))]

    selected_indices = [medoid_idx]
    tier_accum_tokens = defaultdict(int)
    tier_accum_tokens[candidates[medoid_idx]["tier"]] += candidates[medoid_idx]["token_count"]

    min_dists = np.linalg.norm(X_norm - X_norm[medoid_idx], axis=1)

    while True:
        # Eligible: not selected yet, tier not full, and respects max doc tokens
        eligible = [
            i for i, c in enumerate(candidates)
            if i not in selected_indices
            and tier_accum_tokens[c["tier"]] < tier_token_quotas.get(c["tier"], 0)
            and c["token_count"] <= max_doc_tokens
        ]
        if not eligible:
            # Relax max doc tokens if a tier still needs tokens and has no smaller docs
            eligible = [
                i for i, c in enumerate(candidates)
                if i not in selected_indices
                and tier_accum_tokens[c["tier"]] < tier_token_quotas.get(c["tier"], 0)
            ]
            if not eligible:
                break

        # Pick candidate with max minimum distance to already selected candidates
        best_candidate = max(eligible, key=lambda idx: min_dists[idx])
        selected_indices.append(best_candidate)
        tier_accum_tokens[candidates[best_candidate]["tier"]] += candidates[best_candidate]["token_count"]

        # Update min_dists
        new_dists = np.linalg.norm(X_norm - X_norm[best_candidate], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return [candidates[i] for i in selected_indices]


def generate_token_scale_report(
    selected: List[Dict[str, Any]],
    tier_token_quotas: Dict[str, int],
    total_corpus_tokens: int
) -> str:
    """Generates an evaluation benchmark report based on token scale metrics."""
    total_docs = len(selected)
    total_tokens = sum(s["token_count"] for s in selected)
    total_qs = sum(s["meta"]["q_count"] for s in selected)
    total_stims = sum(s["meta"]["stim_count"] for s in selected)
    total_cloze = sum(s["meta"]["stemless_count"] for s in selected)
    total_figs = sum(s["meta"]["fig_count"] for s in selected)
    avg_score = np.mean([s["score"] for s in selected]) if selected else 0.0

    lines = [
        "# 🏆 Gold Benchmark Set Manifest (Token-Scale Stratified & Structurally Diverse)",
        "",
        f"- **Total Documents Selected**: `{total_docs}`",
        f"- **Total Source Tokens**: **{total_tokens:,}** ({total_tokens / total_corpus_tokens * 100:.2f}% of total corpus)",
        f"- **Total Structured Questions**: `{total_qs:,}`",
        f"- **Passage Stimuli Included**: `{total_stims}` (in {sum(1 for s in selected if s['meta']['stim_count'] > 0)} documents)",
        f"- **Cloze (Stemless) Questions**: `{total_cloze}` (in {sum(1 for s in selected if s['meta']['stemless_count'] > 0)} documents)",
        f"- **Inline Figures**: `{total_figs}`",
        f"- **Average Audit Score**: **{avg_score:.2f} / 100**",
        "",
        "## 📊 Distribution Across Difficulty Tiers (Token Scale)",
        "",
        "| Difficulty Tier | Target Tokens | Selected Tokens | Docs | Avg Score | Scanned | Digital | Stimuli | Cloze | Expl |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for tier, target_toks in tier_token_quotas.items():
        t_docs = [s for s in selected if s["tier"] == tier]
        t_toks = sum(s["token_count"] for s in t_docs)
        t_score = np.mean([s["score"] for s in t_docs]) if t_docs else 0.0
        scanned = sum(1 for s in t_docs if s["pdf_type"] == "SCANNED")
        digital = sum(1 for s in t_docs if s["pdf_type"] == "DIGITAL")
        stims = sum(1 for s in t_docs if s["meta"]["stim_count"] > 0)
        cloze = sum(1 for s in t_docs if s["meta"]["stemless_count"] > 0)
        expl = sum(1 for s in t_docs if s["meta"]["expl_count"] > 0)
        lines.append(
            f"| `{tier}` | {target_toks:,} | **{t_toks:,}** | {len(t_docs)} | **{t_score:.1f}** | {scanned} | {digital} | {stims} | {cloze} | {expl} |"
        )

    lines.extend([
        "",
        "## 📑 Detailed Manifest of Selected Documents",
        "",
        "| # | Document ID | Tier | Score | Type | Tokens | Questions | Stimuli | Cloze | Figures | Expl |",
        "| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ])

    for i, s in enumerate(selected, start=1):
        m = s["meta"]
        doc_link = f"`{s['doc_id']}`"
        lines.append(
            f"| {i} | {doc_link} | {s['tier']} | {s['score']:.1f} | {s['pdf_type']} | {s['token_count']:,} | {m['q_count']} | {m['stim_count']} | {m['stemless_count']} | {m['fig_count']} | {m['expl_count']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Select Gold Benchmark Set using Stratified Diverse Top-K on a Token Scale."
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=350000,
        help="Target total token budget for gold benchmark (default: 350,000 tokens ~11% of corpus)"
    )
    parser.add_argument(
        "--token-ratio",
        type=float,
        default=None,
        help="Target token budget as ratio of total corpus tokens (e.g. 0.10 for 10%)"
    )
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=20000,
        help="Maximum tokens allowed for a single document to prevent multi-exam mega-books from monopolizing a tier (default: 20,000)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=80.0,
        help="Minimum audit score threshold (default: 80.0)"
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Path to annotated data"
    )
    parser.add_argument(
        "--pdf-report",
        type=str,
        default="data/pdf_classification_report.json",
        help="Path to PDF classification report"
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="data/benchmark_gold_set.json",
        help="Output path for benchmark JSON"
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="artifacts/benchmark_gold_set_report.md",
        help="Output path for Markdown report"
    )
    args = parser.parse_args()

    annotated_dir = WORKSPACE_DIR / args.annotated_dir
    pdf_report_path = WORKSPACE_DIR / args.pdf_report
    out_json_path = WORKSPACE_DIR / args.out_json
    out_report_path = WORKSPACE_DIR / args.out_report

    pdf_type_map = load_pdf_type_mapping(pdf_report_path)

    doc_paths = sorted(annotated_dir.glob("*"))
    print(f"Scanning {len(doc_paths)} annotated documents in {annotated_dir}...")

    candidates = []
    for d in doc_paths:
        if not d.is_dir():
            continue
        doc_id = d.name
        audit_file = d / "audit_report.json"
        merged_json_file = d / "merged.json"
        merged_xml_file = d / "merged.xml"

        if not (audit_file.exists() and merged_json_file.exists() and merged_xml_file.exists()):
            continue

        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            merged_json = json.loads(merged_json_file.read_text(encoding="utf-8"))
            merged_xml = merged_xml_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading {doc_id}: {e}")
            continue

        score = float(audit.get("overall_score", 0.0))
        if score < args.min_score:
            continue

        qs = merged_json.get("questions", [])
        if len(qs) < 3:
            continue

        # Score Tiers
        if score >= 98.0:
            tier = "Tier 1 [98-100]"
        elif score >= 95.0:
            tier = "Tier 2 [95-98)"
        elif score >= 90.0:
            tier = "Tier 3 [90-95)"
        else:
            tier = "Tier 4 [80-90)"

        pdf_type = pdf_type_map.get(doc_id, "UNKNOWN")
        is_digital = (pdf_type == "DIGITAL")

        vec, token_count, meta = extract_structural_vector_and_tokens(
            merged_xml, merged_json, is_digital
        )

        candidates.append({
            "doc_id": doc_id,
            "tier": tier,
            "score": score,
            "pdf_type": pdf_type,
            "token_count": token_count,
            "vec": vec,
            "meta": meta,
            "dir_path": str(d.resolve())
        })

    total_corpus_tokens = sum(c["token_count"] for c in candidates)
    print(f"Found {len(candidates)} valid candidate documents (Total Tokens: {total_corpus_tokens:,}).")

    target_tokens = args.target_tokens
    if args.token_ratio is not None:
        target_tokens = int(total_corpus_tokens * args.token_ratio)

    print(f"Targeting Gold Benchmark Token Budget: {target_tokens:,} ({target_tokens / total_corpus_tokens * 100:.2f}% of corpus)")

    tier_token_quotas = {
        "Tier 1 [98-100]": int(target_tokens * 0.30),
        "Tier 2 [95-98)": int(target_tokens * 0.30),
        "Tier 3 [90-95)": int(target_tokens * 0.25),
        "Tier 4 [80-90)": int(target_tokens * 0.15),
    }
    print(f"Tier Token Budgets: {tier_token_quotas}")

    selected = select_diverse_token_budget_stratified(
        candidates,
        tier_token_quotas,
        max_doc_tokens=args.max_doc_tokens
    )

    total_sel_tokens = sum(s["token_count"] for s in selected)

    # Save JSON manifest
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "total_selected_documents": len(selected),
        "total_tokens": total_sel_tokens,
        "token_ratio_of_corpus": round(total_sel_tokens / total_corpus_tokens, 4),
        "target_tokens": target_tokens,
        "tier_token_quotas": tier_token_quotas,
        "total_questions": sum(s["meta"]["q_count"] for s in selected),
        "average_score": round(float(np.mean([s["score"] for s in selected])), 2),
        "documents": [
            {
                "doc_id": s["doc_id"],
                "tier": s["tier"],
                "score": s["score"],
                "pdf_type": s["pdf_type"],
                "token_count": s["token_count"],
                "metrics": s["meta"],
                "path": s["dir_path"]
            }
            for s in selected
        ]
    }
    out_json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON manifest to {out_json_path}")

    # Save Markdown report
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = generate_token_scale_report(selected, tier_token_quotas, total_corpus_tokens)
    out_report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved Markdown report to {out_report_path}")

    print("\nSelection completed successfully:")
    print(f"  Selected Documents: {len(selected)}")
    print(f"  Total Source Tokens: {total_sel_tokens:,} ({total_sel_tokens / total_corpus_tokens * 100:.2f}% of corpus)")
    print(f"  Total Questions: {manifest['total_questions']:,}")
    print(f"  Average Score: {manifest['average_score']}")


if __name__ == "__main__":
    main()
