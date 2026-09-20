"""
Test Gemini 3.8 intra-model consistency by running independent reviews twice
on a dozen discarded documents to compute Cohen's Kappa coefficient.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sequence_labelling.annotator.reviewer import AnnotationReviewerAgent, ReviewDecision

TARGET_DOCS = [
    "exam-007--333e58e1279b",
    "exam-019--6bf349fe480f",
    "exam-018--0ccd581501db",
    "exam-025--4a19980384f5",
    "exam-035--2eb29d0b34d2",
    "exam-052--42b985ae82c1",
    "exam-059--7a93d83d93f6",
    "exam-072--f0dc5e8b607b",
    "exam-084--6e7ee9ea82f4",
    "exam-102--f5ce32c113f2",
    "exam-120--7c8adfd6794f",
    "exam-160--0d168c762b05",
]

OUTPUT_JSON = Path("backend/logs/gemini_intra_model_kappa.json")


def load_doc_data(doc_id: str) -> Tuple[str, str]:
    ann_dir = Path("data/sequence_labelling_annotated")
    raw_dir = Path("data/sequence_labelling_input_data")

    xml_path = ann_dir / doc_id / "merged.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")

    raw_path = raw_dir / f"{doc_id}.md"
    if not raw_path.exists():
        raw_path = raw_dir / f"{doc_id}.txt"
    if not raw_path.exists():
        raw_path = raw_dir / doc_id / "raw_ocr.md"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw OCR for {doc_id}")

    return xml_path.read_text(encoding="utf-8"), raw_path.read_text(encoding="utf-8")


def compute_cohen_kappa(
    pairs: List[Tuple[str, str]], labels: List[str] = ["PASS", "NEEDS_REVISION", "DISCARD"]
) -> Tuple[float, float, float]:
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0

    po = sum(1 for a, b in pairs if a == b) / n
    c1 = Counter(a for a, b in pairs)
    c2 = Counter(b for a, b in pairs)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)

    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return kappa, po, pe


def run_experiment():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    agent = AnnotationReviewerAgent(
        model="gemini-3.8-flash", provider="agy", thinking="medium"
    )

    # Resume support if partially executed
    results_map: Dict[str, Dict[str, Any]] = {}
    if OUTPUT_JSON.exists():
        try:
            cached = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and "documents" in cached:
                results_map = {d["doc_id"]: d for d in cached["documents"]}
        except Exception:
            results_map = {}

    print("=" * 80)
    print("GEMINI 3.8 FLASH (MEDIUM THINKING) - INTRA-MODEL KAPPA EXPERIMENT")
    print(f"Target Documents : {len(TARGET_DOCS)} discarded documents")
    print(f"Model            : gemini-3.8-flash (provider: agy, effort: medium)")
    print(f"Output Checkpoint: {OUTPUT_JSON}")
    print("=" * 80)

    for idx, doc_id in enumerate(TARGET_DOCS, 1):
        doc_entry = results_map.get(doc_id, {"doc_id": doc_id})

        xml_text, raw_text = load_doc_data(doc_id)

        # ── RUN 1 ─────────────────────────────────────────────────────────────
        if "run1" not in doc_entry:
            print(f"\n[{idx}/{len(TARGET_DOCS)}] {doc_id} -> Executing Run 1...", end="", flush=True)
            t0 = time.time()
            rep1 = agent.review_document(
                xml_content=xml_text,
                raw_ocr_text=raw_text,
                doc_id=doc_id,
                use_llm=True,
            )
            elapsed1 = time.time() - t0
            print(f" Done ({elapsed1:.1f}s) => {rep1.decision.value} (Score: {rep1.overall_score})")

            doc_entry["run1"] = {
                "decision": rep1.decision.value,
                "overall_score": rep1.overall_score,
                "deterministic_score": rep1.deterministic_score,
                "llm_score": rep1.llm_score,
                "confirmed_issues_count": len(rep1.confirmed_issues),
                "issues_count": len(rep1.issues),
                "discard_reasons": rep1.discard_reasons,
                "issues": [
                    {
                        "category": iss.category,
                        "severity": iss.severity.value,
                        "message": iss.message,
                    }
                    for iss in rep1.issues
                ],
                "duration_sec": round(elapsed1, 1),
            }
            results_map[doc_id] = doc_entry
            # Checkpoint save
            OUTPUT_JSON.write_text(
                json.dumps({"documents": list(results_map.values())}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print(f"[{idx}/{len(TARGET_DOCS)}] {doc_id} -> Run 1 cached: {doc_entry['run1']['decision']}")

        # ── RUN 2 ─────────────────────────────────────────────────────────────
        if "run2" not in doc_entry:
            print(f"[{idx}/{len(TARGET_DOCS)}] {doc_id} -> Executing Run 2...", end="", flush=True)
            t0 = time.time()
            rep2 = agent.review_document(
                xml_content=xml_text,
                raw_ocr_text=raw_text,
                doc_id=doc_id,
                use_llm=True,
            )
            elapsed2 = time.time() - t0
            print(f" Done ({elapsed2:.1f}s) => {rep2.decision.value} (Score: {rep2.overall_score})")

            doc_entry["run2"] = {
                "decision": rep2.decision.value,
                "overall_score": rep2.overall_score,
                "deterministic_score": rep2.deterministic_score,
                "llm_score": rep2.llm_score,
                "confirmed_issues_count": len(rep2.confirmed_issues),
                "issues_count": len(rep2.issues),
                "discard_reasons": rep2.discard_reasons,
                "issues": [
                    {
                        "category": iss.category,
                        "severity": iss.severity.value,
                        "message": iss.message,
                    }
                    for iss in rep2.issues
                ],
                "duration_sec": round(elapsed2, 1),
            }
            doc_entry["decision_match"] = doc_entry["run1"]["decision"] == doc_entry["run2"]["decision"]
            doc_entry["score_diff"] = abs(
                round(doc_entry["run1"]["overall_score"] - doc_entry["run2"]["overall_score"], 1)
            )
            results_map[doc_id] = doc_entry
            # Checkpoint save
            OUTPUT_JSON.write_text(
                json.dumps({"documents": list(results_map.values())}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print(f"[{idx}/{len(TARGET_DOCS)}] {doc_id} -> Run 2 cached: {doc_entry['run2']['decision']}")

    # ── KAPPA CALCULATION ──────────────────────────────────────────────────────
    doc_list = [results_map[doc_id] for doc_id in TARGET_DOCS if doc_id in results_map]
    pairs = [(d["run1"]["decision"], d["run2"]["decision"]) for d in doc_list]

    kappa_3way, po_3way, pe_3way = compute_cohen_kappa(pairs, ["PASS", "NEEDS_REVISION", "DISCARD"])

    def to_bin(d: str) -> str:
        return "PASS" if d == "PASS" else "DEFECTIVE"

    bin_pairs = [(to_bin(a), to_bin(b)) for a, b in pairs]
    kappa_bin, po_bin, pe_bin = compute_cohen_kappa(bin_pairs, ["PASS", "DEFECTIVE"])

    avg_score_diff = sum(d["score_diff"] for d in doc_list) / max(1, len(doc_list))

    stats = {
        "total_documents": len(doc_list),
        "cohen_kappa_3way": round(kappa_3way, 4),
        "observed_agreement_3way": round(po_3way, 4),
        "expected_agreement_3way": round(pe_3way, 4),
        "cohen_kappa_binary": round(kappa_bin, 4),
        "observed_agreement_binary": round(po_bin, 4),
        "expected_agreement_binary": round(pe_bin, 4),
        "average_score_diff": round(avg_score_diff, 2),
    }

    final_payload = {
        "stats": stats,
        "documents": doc_list,
    }
    OUTPUT_JSON.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED - INTRA-MODEL RESULTS")
    print(f"Total Documents Evaluated : {len(doc_list)}")
    print(f"Observed Agreement (Po)   : {po_3way:.1%}")
    print(f"Expected Agreement (Pe)   : {pe_3way:.1%}")
    print(f"Cohen's Kappa (3-way)     : {kappa_3way:.4f}")
    print(f"Binary Agreement (Po)     : {po_bin:.1%}")
    print(f"Binary Cohen's Kappa      : {kappa_bin:.4f}")
    print(f"Average Score Difference  : {avg_score_diff:.1f} pts")
    print("=" * 80)

    # Print confusion matrix
    decisions = ["PASS", "NEEDS_REVISION", "DISCARD"]
    print("\nConfusion Matrix (Run 1 rows x Run 2 columns):")
    print(f"{'Run 1 \\ Run 2':<16} | {'PASS':<8} | {'REVISION':<8} | {'DISCARD':<8}")
    print("-" * 48)
    for r1 in decisions:
        row_counts = [sum(1 for a, b in pairs if a == r1 and b == r2) for r2 in decisions]
        print(f"{r1:<16} | {row_counts[0]:<8} | {row_counts[1]:<8} | {row_counts[2]:<8}")


if __name__ == "__main__":
    run_experiment()
