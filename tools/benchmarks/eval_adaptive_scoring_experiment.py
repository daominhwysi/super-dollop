#!/usr/bin/env python3
"""Experimental Adaptive Scoring System Benchmark and Necessity Audit.

Evaluates three scoring models against 488 audit reports in data/review_report.json:
1. Legacy Baseline (fixed penalties, linear sum, static thresholds)
2. Full Adaptive Engine (scale-normalized saturation, defect density, isolated vs systemic)
3. Targeted Minimal Heuristics (scale-aware discard buffer + isolated defect bypass)

Outputs statistical differentials and exports artifacts/adaptive_scoring_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_labelling.annotator.reviewer import (
    AuditIssue,
    IssueSeverity,
    PARSER_ERROR_PENALTIES,
    ReviewDecision,
)


@dataclass
class DocumentScaleContext:
    """Document scale metrics and normalization factors."""

    total_questions: int = 1
    total_options: int = 0
    token_count: int = 0
    stimuli_count: int = 0
    raw_char_count: int = 0

    @classmethod
    def from_report_metrics(cls, metrics: Dict[str, Any]) -> DocumentScaleContext:
        q_count = max(1, int(metrics.get("questions_count") or metrics.get("det_total_questions") or 1))
        opt_count = int(metrics.get("option_labels_count") or 0)
        stim_count = int(metrics.get("stimuli_count") or 0)
        char_count = int(metrics.get("raw_char_count") or metrics.get("annotated_char_count") or 0)
        word_count = int(metrics.get("raw_word_count") or metrics.get("annotated_word_count") or (char_count // 6))
        return cls(
            total_questions=q_count,
            total_options=opt_count,
            token_count=word_count,
            stimuli_count=stim_count,
            raw_char_count=char_count,
        )

    @property
    def scale_factor(self) -> float:
        """Sublinear scale dampening factor based on baseline of 10 questions."""
        return max(1.0, math.sqrt(self.total_questions / 10.0))

    def defect_density(self, defect_count: int) -> float:
        return defect_count / max(1, self.total_questions)


# Category saturation caps (maximum deduction a single category can inflict)
CATEGORY_PENALTY_CAPS: Dict[str, float] = {
    "unannotated_question": 60.0,
    "stimulus_nesting": 100.0,
    "stimulus_missing_citation": 25.0,
    "single_question_stimulus": 20.0,
    "stimulus_anchor_invalid": 35.0,
    "broken_system_tag": 50.0,
    "missing_question_label": 40.0,
    "missing_stem": 35.0,
    "missing_options": 40.0,
    "absorbed_subquestions": 30.0,
    "mislabelled_element": 35.0,
    "prohibited_tags": 40.0,
    "truncated_output": 60.0,
    "xml_syntax": 50.0,
    "schema_conformance": 30.0,
    "stimulus": 35.0,
    "question_structure": 40.0,
    "option_structure": 35.0,
    "verbatim_fidelity": 40.0,
    "llm_semantic": 30.0,
}


@dataclass
class ScoredOutcome:
    doc_id: str
    overall_score: float
    deterministic_score: float
    llm_score: float
    decision: str
    discard_reasons: List[str] = field(default_factory=list)
    reclassified_issues: List[Dict[str, Any]] = field(default_factory=list)
    defect_density: float = 0.0


class ScoringEvaluator:
    """Evaluates audit reports using Legacy, Full Adaptive, and Minimal Heuristic models."""

    @staticmethod
    def score_legacy(report: Dict[str, Any]) -> ScoredOutcome:
        """Legacy scoring baseline reproducing reviewer.py."""
        doc_id = report.get("doc_id", "")
        det_score = float(report.get("deterministic_score", 100.0))
        confirmed_issues = report.get("confirmed_issues", [])
        discard_reasons = list(report.get("discard_reasons", []))

        total_penalty = 0.0
        for iss in confirmed_issues:
            cat = iss.get("category", "llm_semantic").lower().strip()
            sev_str = iss.get("severity", "MAJOR")
            sev = IssueSeverity(sev_str) if sev_str in IssueSeverity.__members__ else IssueSeverity.MAJOR
            cat_map = PARSER_ERROR_PENALTIES.get(cat, PARSER_ERROR_PENALTIES["llm_semantic"])
            total_penalty += cat_map.get(sev, 5.0)

        llm_score = max(0.0, 100.0 - total_penalty)
        overall_score = round(det_score * 0.40 + llm_score * 0.60, 1)

        has_critical = any(iss.get("severity") == "CRITICAL" for iss in confirmed_issues)
        has_major = any(iss.severity == IssueSeverity.MAJOR if hasattr(iss, "severity") else iss.get("severity") == "MAJOR" for iss in confirmed_issues)

        is_malfunctioned = len(discard_reasons) > 0 or overall_score < 75.0 or has_critical
        if is_malfunctioned:
            decision = ReviewDecision.DISCARD.value
        elif has_major:
            decision = ReviewDecision.NEEDS_REVISION.value
        elif overall_score >= 80.0:
            decision = ReviewDecision.PASS.value
        else:
            decision = ReviewDecision.NEEDS_REVISION.value

        return ScoredOutcome(
            doc_id=doc_id,
            overall_score=overall_score,
            deterministic_score=det_score,
            llm_score=round(llm_score, 1),
            decision=decision,
            discard_reasons=discard_reasons,
        )

    @staticmethod
    def score_full_adaptive(report: Dict[str, Any]) -> ScoredOutcome:
        """Full adaptive model with scale-normalized saturation, defect density, and isolated-defect grading."""
        doc_id = report.get("doc_id", "")
        det_score = float(report.get("deterministic_score", 100.0))
        metrics = report.get("metrics", {})
        context = DocumentScaleContext.from_report_metrics(metrics)
        confirmed_issues = report.get("confirmed_issues", [])
        raw_discard_reasons = report.get("discard_reasons", [])

        # Categorize defects and check for isolated vs systemic occurrences
        category_counts = Counter(iss.get("category", "").lower().strip() for iss in confirmed_issues)
        reclassified_issues = []
        has_critical = False
        has_systemic_major = False
        has_isolated_major = False

        # Group raw penalties by category for diminishing saturation
        cat_raw_penalties: Dict[str, float] = defaultdict(float)

        for iss in confirmed_issues:
            cat = iss.get("category", "llm_semantic").lower().strip()
            sev_str = iss.get("severity", "MAJOR")
            sev = IssueSeverity(sev_str) if sev_str in IssueSeverity.__members__ else IssueSeverity.MAJOR

            # Invariant: true architectural fatal flaws are unconditionally CRITICAL
            if sev == IssueSeverity.CRITICAL or cat in ["stimulus_nesting", "truncated_output"]:
                has_critical = True
                cat_raw_penalties[cat] += 50.0
                reclassified_issues.append({**iss, "effective_severity": "CRITICAL"})
                continue

            # Check if this is an isolated occurrence in a large document
            count_in_cat = category_counts[cat]
            defect_rate = count_in_cat / max(1, context.total_questions)

            is_hard_category = cat in ["unannotated_question", "broken_system_tag"]
            is_isolated = (
                not is_hard_category
                and context.total_questions >= 25
                and count_in_cat <= 2
                and defect_rate < 0.05
            )

            if sev == IssueSeverity.MAJOR:
                if is_isolated:
                    effective_sev = IssueSeverity.MINOR
                    has_isolated_major = True
                else:
                    effective_sev = IssueSeverity.MAJOR
                    has_systemic_major = True
            else:
                effective_sev = sev

            cat_map = PARSER_ERROR_PENALTIES.get(cat, PARSER_ERROR_PENALTIES["llm_semantic"])
            penalty = cat_map.get(effective_sev, 5.0)
            cat_raw_penalties[cat] += penalty

            reclassified_issues.append({**iss, "effective_severity": effective_sev.value})

        # Apply category-level diminishing saturation math:
        # P_cat = Cap_cat * (1 - exp(-raw_sum / (Cap_cat * scale_factor)))
        total_adaptive_penalty = 0.0
        for cat, raw_sum in cat_raw_penalties.items():
            cap = CATEGORY_PENALTY_CAPS.get(cat, 35.0)
            # Dampened by scale_factor so large documents with low defect density have reduced penalty
            dampened_cap = cap
            exponent = -raw_sum / max(1.0, (dampened_cap * context.scale_factor))
            cat_penalty = dampened_cap * (1.0 - math.exp(exponent))
            total_adaptive_penalty += cat_penalty

        llm_score = max(0.0, 100.0 - total_adaptive_penalty)
        overall_score = round(det_score * 0.40 + llm_score * 0.60, 1)

        # Discard reasons evaluation
        adaptive_discards = []
        for r in raw_discard_reasons:
            # Filter out legacy static "below threshold 75" messages
            if "below minimum threshold" not in r:
                adaptive_discards.append(r)

        if has_critical:
            decision = ReviewDecision.DISCARD.value
            if not adaptive_discards:
                adaptive_discards.append("Critical architectural defect detected.")
        else:
            # Scale-aware discard floor: large documents (Q >= 30) have floor at 68.0 instead of 75.0
            discard_floor = 68.0 if context.total_questions >= 30 else 75.0
            if overall_score < discard_floor or len(adaptive_discards) > 0:
                decision = ReviewDecision.DISCARD.value
                if not adaptive_discards:
                    adaptive_discards.append(f"Score {overall_score:.1f} below adaptive threshold {discard_floor:.1f}")
            elif has_systemic_major:
                decision = ReviewDecision.NEEDS_REVISION.value
            elif overall_score >= 80.0:
                decision = ReviewDecision.PASS.value
            elif overall_score >= 78.0 and context.defect_density(len(confirmed_issues)) < 0.02 and not has_isolated_major:
                decision = ReviewDecision.PASS.value
            else:
                decision = ReviewDecision.NEEDS_REVISION.value

        return ScoredOutcome(
            doc_id=doc_id,
            overall_score=overall_score,
            deterministic_score=det_score,
            llm_score=round(llm_score, 1),
            decision=decision,
            discard_reasons=adaptive_discards,
            reclassified_issues=reclassified_issues,
            defect_density=round(context.defect_density(len(confirmed_issues)), 4),
        )

    @staticmethod
    def score_minimal_heuristic(report: Dict[str, Any]) -> ScoredOutcome:
        """Targeted minimal heuristic: legacy scores + scale-aware discard buffer + isolated single-defect pass."""
        doc_id = report.get("doc_id", "")
        det_score = float(report.get("deterministic_score", 100.0))
        metrics = report.get("metrics", {})
        q_count = max(1, int(metrics.get("questions_count") or metrics.get("det_total_questions") or 1))
        confirmed_issues = report.get("confirmed_issues", [])
        raw_discard_reasons = report.get("discard_reasons", [])

        # Standard legacy penalty summation
        total_penalty = 0.0
        has_critical = False
        major_issues = []

        for iss in confirmed_issues:
            cat = iss.get("category", "llm_semantic").lower().strip()
            sev_str = iss.get("severity", "MAJOR")
            sev = IssueSeverity(sev_str) if sev_str in IssueSeverity.__members__ else IssueSeverity.MAJOR

            if sev == IssueSeverity.CRITICAL or cat in ["stimulus_nesting"]:
                has_critical = True

            if sev == IssueSeverity.MAJOR:
                major_issues.append(iss)

            cat_map = PARSER_ERROR_PENALTIES.get(cat, PARSER_ERROR_PENALTIES["llm_semantic"])
            total_penalty += cat_map.get(sev, 5.0)

        llm_score = max(0.0, 100.0 - total_penalty)
        overall_score = round(det_score * 0.40 + llm_score * 0.60, 1)

        # Rule 1: Scale-aware discard buffer
        # For Q >= 30 with 0 critical defects, relax discard threshold from 75 to 68 to route to revision
        discard_floor = 68.0 if (q_count >= 30 and not has_critical) else 75.0

        # Filter out static threshold discard reasons if within buffer
        minimal_discards = []
        for r in raw_discard_reasons:
            if "below minimum threshold" in r:
                if overall_score < discard_floor:
                    minimal_discards.append(r)
            else:
                minimal_discards.append(r)

        if has_critical or len(minimal_discards) > 0 or overall_score < discard_floor:
            decision = ReviewDecision.DISCARD.value
        else:
            # Rule 2: Isolated single-defect bypass
            # If Q >= 30, overall_score >= 88.0, and exactly 1 MAJOR defect that is non-structural
            is_isolated_single_defect = False
            if len(major_issues) == 1 and q_count >= 30 and overall_score >= 88.0:
                single_cat = major_issues[0].get("category", "")
                if single_cat not in ["unannotated_question", "stimulus_nesting", "broken_system_tag"]:
                    is_isolated_single_defect = True

            if len(major_issues) > 0 and not is_isolated_single_defect:
                decision = ReviewDecision.NEEDS_REVISION.value
            elif overall_score >= 80.0:
                decision = ReviewDecision.PASS.value
            else:
                decision = ReviewDecision.NEEDS_REVISION.value

        return ScoredOutcome(
            doc_id=doc_id,
            overall_score=overall_score,
            deterministic_score=det_score,
            llm_score=round(llm_score, 1),
            decision=decision,
            discard_reasons=minimal_discards,
        )


def run_benchmark(report_path: Path) -> Dict[str, Any]:
    """Runs all 3 models on the review report and compiles comparative statistics."""
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    reports = data.get("reports", [])
    total_docs = len(reports)

    legacy_outcomes = []
    adaptive_outcomes = []
    minimal_outcomes = []

    for r in reports:
        legacy_outcomes.append(ScoringEvaluator.score_legacy(r))
        adaptive_outcomes.append(ScoringEvaluator.score_full_adaptive(r))
        minimal_outcomes.append(ScoringEvaluator.score_minimal_heuristic(r))

    def count_decisions(outcomes: List[ScoredOutcome]) -> Dict[str, int]:
        c = Counter(o.decision for o in outcomes)
        return {k: c.get(k, 0) for k in ["PASS", "NEEDS_REVISION", "DISCARD"]}

    def average_score(outcomes: List[ScoredOutcome]) -> float:
        return round(sum(o.overall_score for o in outcomes) / max(1, len(outcomes)), 2)

    legacy_counts = count_decisions(legacy_outcomes)
    adaptive_counts = count_decisions(adaptive_outcomes)
    minimal_counts = count_decisions(minimal_outcomes)

    # Compute Transition Matrices
    matrix_legacy_to_adaptive: Dict[str, Counter] = defaultdict(Counter)
    matrix_legacy_to_minimal: Dict[str, Counter] = defaultdict(Counter)

    for leg, adp, minm in zip(legacy_outcomes, adaptive_outcomes, minimal_outcomes):
        matrix_legacy_to_adaptive[leg.decision][adp.decision] += 1
        matrix_legacy_to_minimal[leg.decision][minm.decision] += 1

    # Detailed inspection of documents that changed decisions
    transition_details_adaptive = []
    for r, leg, adp in zip(reports, legacy_outcomes, adaptive_outcomes):
        if leg.decision != adp.decision:
            q_cnt = r.get("metrics", {}).get("questions_count", 0)
            transition_details_adaptive.append({
                "doc_id": r.get("doc_id"),
                "questions_count": q_cnt,
                "legacy_decision": leg.decision,
                "adaptive_decision": adp.decision,
                "legacy_score": leg.overall_score,
                "adaptive_score": adp.overall_score,
                "confirmed_issues_count": len(r.get("confirmed_issues", [])),
                "reclassified_issues": adp.reclassified_issues,
            })

    transition_details_minimal = []
    for r, leg, minm in zip(reports, legacy_outcomes, minimal_outcomes):
        if leg.decision != minm.decision:
            q_cnt = r.get("metrics", {}).get("questions_count", 0)
            transition_details_minimal.append({
                "doc_id": r.get("doc_id"),
                "questions_count": q_cnt,
                "legacy_decision": leg.decision,
                "minimal_decision": minm.decision,
                "legacy_score": leg.overall_score,
                "minimal_score": minm.overall_score,
                "confirmed_issues_count": len(r.get("confirmed_issues", [])),
            })

    # Length vs Score Correlation Analysis
    # Does penalty severity scale disproportionately with question count?
    q_vs_penalty_legacy = []
    q_vs_penalty_adaptive = []
    for r, leg, adp in zip(reports, legacy_outcomes, adaptive_outcomes):
        q = max(1, r.get("metrics", {}).get("questions_count", 1))
        pen_leg = 100.0 - leg.llm_score
        pen_adp = 100.0 - adp.llm_score
        q_vs_penalty_legacy.append((q, pen_leg))
        q_vs_penalty_adaptive.append((q, pen_adp))

    return {
        "total_documents": total_docs,
        "summary": {
            "legacy": {
                "counts": legacy_counts,
                "average_score": average_score(legacy_outcomes),
            },
            "full_adaptive": {
                "counts": adaptive_counts,
                "average_score": average_score(adaptive_outcomes),
            },
            "minimal_heuristic": {
                "counts": minimal_counts,
                "average_score": average_score(minimal_outcomes),
            },
        },
        "transitions": {
            "legacy_to_adaptive": {k: dict(v) for k, v in matrix_legacy_to_adaptive.items()},
            "legacy_to_minimal": {k: dict(v) for k, v in matrix_legacy_to_minimal.items()},
        },
        "adaptive_transitions_sample": transition_details_adaptive[:25],
        "minimal_transitions_sample": transition_details_minimal[:25],
        "all_adaptive_transitions": transition_details_adaptive,
        "all_minimal_transitions": transition_details_minimal,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Scoring Models")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/review_report.json",
        help="Path to review report JSON (default: data/review_report.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="artifacts/adaptive_scoring_benchmark.json",
        help="Path to output benchmark JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running adaptive scoring benchmark on {input_path}...")
    results = run_benchmark(input_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved benchmark results to {output_path}\n")

    # Print clean summary table
    print("=" * 70)
    print("📊 SCORING SYSTEM COMPARISON SUMMARY (488 Documents)")
    print("=" * 70)
    print(f"{'Model':<25} | {'PASS':<8} | {'REVISION':<10} | {'DISCARD':<8} | {'Avg Score':<10}")
    print("-" * 70)
    leg = results["summary"]["legacy"]
    adp = results["summary"]["full_adaptive"]
    minm = results["summary"]["minimal_heuristic"]
    print(f"{'1. Legacy Baseline':<25} | {leg['counts']['PASS']:<8} | {leg['counts']['NEEDS_REVISION']:<10} | {leg['counts']['DISCARD']:<8} | {leg['average_score']:<10.2f}")
    print(f"{'2. Full Adaptive':<25} | {adp['counts']['PASS']:<8} | {adp['counts']['NEEDS_REVISION']:<10} | {adp['counts']['DISCARD']:<8} | {adp['average_score']:<10.2f}")
    print(f"{'3. Minimal Heuristic':<25} | {minm['counts']['PASS']:<8} | {minm['counts']['NEEDS_REVISION']:<10} | {minm['counts']['DISCARD']:<8} | {minm['average_score']:<10.2f}")
    print("=" * 70)

    print("\n🔄 DECISION TRANSITIONS (Legacy -> Full Adaptive):")
    for src, dsts in results["transitions"]["legacy_to_adaptive"].items():
        dst_str = ", ".join(f"{dst}: {cnt}" for dst, cnt in dsts.items() if dst != src)
        unchanged = dsts.get(src, 0)
        print(f"  {src} ({sum(dsts.values())} docs): {unchanged} unchanged | Shifts -> {dst_str if dst_str else 'none'}")

    print("\n🔄 DECISION TRANSITIONS (Legacy -> Minimal Heuristic):")
    for src, dsts in results["transitions"]["legacy_to_minimal"].items():
        dst_str = ", ".join(f"{dst}: {cnt}" for dst, cnt in dsts.items() if dst != src)
        unchanged = dsts.get(src, 0)
        print(f"  {src} ({sum(dsts.values())} docs): {unchanged} unchanged | Shifts -> {dst_str if dst_str else 'none'}")


if __name__ == "__main__":
    main()
