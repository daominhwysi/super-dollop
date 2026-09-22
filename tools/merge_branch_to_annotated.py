#!/usr/bin/env python3
"""
Merge Branch to Annotated Dataset.

Safely merges candidate XML/JSON, revision records, and audit reports from
data/sequence_labelling_annotated_branch into the canonical
data/sequence_labelling_annotated directory.
Updates data/review_report.json and data/review_progress.json accordingly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sequence_labelling_annotated_branch into canonical annotated dataset."
    )
    parser.add_argument(
        "--branch-dir",
        type=Path,
        default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch",
        help="Path to branch directory containing repaired files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated",
        help="Path to canonical annotated directory.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=WORKSPACE_DIR / "data" / "review_report.json",
        help="Path to global review report JSON.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=WORKSPACE_DIR / "data" / "review_progress.json",
        help="Path to review progress JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview merge actions without modifying any files.",
    )
    return parser.parse_args()


def extract_latest_review(branch_doc_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """Extract the review payload and status from the latest round in the branch folder."""
    state_file = branch_doc_dir / "revision_state.json"
    audit_file = branch_doc_dir / "audit_report.json"

    # If audit_report.json exists, load it
    audit_report = None
    if audit_file.exists():
        try:
            audit_report = json.loads(audit_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not state_file.exists():
        if audit_report and audit_report.get("decision"):
            return audit_report, audit_report["decision"]
        return None, "UNKNOWN"

    try:
        state_data = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ⚠️ Could not read {state_file}: {e}")
        if audit_report and audit_report.get("decision"):
            return audit_report, audit_report["decision"]
        return None, "ERROR"

    status = state_data.get("status", "UNKNOWN")
    rounds = state_data.get("rounds", [])

    latest_review = None
    if rounds:
        last_round = rounds[-1]
        r_path = Path(last_round.get("path", ""))
        candidates = [r_path, branch_doc_dir / "revisions" / r_path.name]
        for cp in candidates:
            if cp.exists():
                try:
                    rd = json.loads(cp.read_text(encoding="utf-8"))
                    latest_review = rd.get("review")
                    if latest_review:
                        break
                except Exception:
                    pass

    # If status is IN_PROGRESS, infer from latest round or audit report
    if status == "IN_PROGRESS":
        if rounds and rounds[-1].get("decision"):
            status = rounds[-1]["decision"]
        elif audit_report and audit_report.get("decision"):
            status = audit_report["decision"]

    res = state_data.get("result", {})
    if not latest_review:
        if audit_report:
            latest_review = audit_report
            latest_review["decision"] = status
        else:
            latest_review = {
                "doc_id": branch_doc_dir.name,
                "decision": status,
                "overall_score": res.get("final_score") or res.get("initial_score", 0.0),
            }
    else:
        latest_review["decision"] = status
        if res.get("final_score") is not None:
            latest_review["overall_score"] = res["final_score"]

    return latest_review, status


def main() -> None:
    args = parse_args()
    branch_dir: Path = args.branch_dir
    target_dir: Path = args.target_dir
    report_path: Path = args.report_path
    progress_path: Path = args.progress_path
    dry_run: bool = args.dry_run

    if not branch_dir.exists():
        print(f"Error: Branch directory does not exist: {branch_dir}")
        sys.exit(1)
    if not target_dir.exists():
        print(f"Error: Target directory does not exist: {target_dir}")
        sys.exit(1)

    branch_doc_dirs = sorted([d for d in branch_dir.iterdir() if d.is_dir()])
    print(f"Discovered {len(branch_doc_dirs)} document directories in branch: {branch_dir}")
    if dry_run:
        print("🔍 [DRY-RUN MODE ENABLED] - No files will be modified on disk.\n")

    # Track actions
    merged_count = 0
    status_counts: dict[str, int] = {}
    updated_reviews: dict[str, dict[str, Any]] = {}

    for doc_dir in branch_doc_dirs:
        doc_id = doc_dir.name
        dest_doc_dir = target_dir / doc_id

        latest_review, status = extract_latest_review(doc_dir)
        status_counts[status] = status_counts.get(status, 0) + 1

        branch_xml = doc_dir / "merged.xml"
        branch_json = doc_dir / "merged.json"
        branch_state = doc_dir / "revision_state.json"
        branch_revisions = doc_dir / "revisions"

        if not dry_run:
            dest_doc_dir.mkdir(parents=True, exist_ok=True)

            # 1. Copy merged.xml
            if branch_xml.exists():
                shutil.copy2(branch_xml, dest_doc_dir / "merged.xml")

            # 2. Copy merged.json
            if branch_json.exists():
                shutil.copy2(branch_json, dest_doc_dir / "merged.json")

            # 3. Copy revision_state.json
            if branch_state.exists():
                shutil.copy2(branch_state, dest_doc_dir / "revision_state.json")

            # 4. Copy revisions/ directory
            if branch_revisions.exists() and branch_revisions.is_dir():
                dest_revs = dest_doc_dir / "revisions"
                shutil.copytree(branch_revisions, dest_revs, dirs_exist_ok=True)

            # 5. Synchronize audit_report.json
            dest_audit = dest_doc_dir / "audit_report.json"
            if latest_review:
                review_to_save = dict(latest_review)
                review_to_save["file_path"] = str(dest_doc_dir / "merged.xml")
                dest_audit.write_text(
                    json.dumps(review_to_save, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                updated_reviews[doc_id] = review_to_save
        else:
            if latest_review:
                updated_reviews[doc_id] = latest_review

        merged_count += 1

    print(f"Processed {merged_count} documents from branch:")
    for st, cnt in sorted(status_counts.items()):
        print(f"  - {st}: {cnt}")

    # 6. Update global review report and progress
    print("\nUpdating global review reports...")
    all_reports: list[dict[str, Any]] = []
    
    # Read existing review report if present
    existing_report_data: dict[str, Any] = {}
    if report_path.exists():
        try:
            existing_report_data = json.loads(report_path.read_text(encoding="utf-8"))
            all_reports = existing_report_data.get("reports", [])
        except Exception as e:
            print(f"  ⚠️ Could not read existing {report_path}: {e}")

    # Index existing reports by doc_id
    report_by_doc_id: dict[str, dict[str, Any]] = {
        r.get("doc_id"): dict(r) for r in all_reports if r.get("doc_id")
    }

    # If canonical dataset has documents not yet in report, add them
    for d in target_dir.iterdir():
        if not d.is_dir():
            continue
        doc_id = d.name
        if doc_id not in report_by_doc_id:
            audit_file = d / "audit_report.json"
            if audit_file.exists():
                try:
                    audit_entry = json.loads(audit_file.read_text(encoding="utf-8"))
                    report_by_doc_id[doc_id] = audit_entry
                except Exception:
                    pass

    # Update with new branch reviews
    for doc_id, rev in updated_reviews.items():
        if doc_id in report_by_doc_id:
            report_by_doc_id[doc_id].update(rev)
        else:
            report_by_doc_id[doc_id] = rev

    # Recompute global counts and stats across all documents
    total_docs = len(report_by_doc_id)
    passed_docs = [r for r in report_by_doc_id.values() if r.get("decision") == "PASS"]
    rev_docs = [r for r in report_by_doc_id.values() if r.get("decision") == "NEEDS_REVISION"]
    disc_docs = [r for r in report_by_doc_id.values() if r.get("decision") == "DISCARD"]

    passed_count = len(passed_docs)
    needs_revision_count = len(rev_docs)
    discarded_count = len(disc_docs)

    scores = [
        float(r.get("overall_score", 0.0))
        for r in report_by_doc_id.values()
        if r.get("overall_score") is not None
    ]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    discarded_paths = [
        str(r.get("file_path")) for r in disc_docs if r.get("file_path")
    ]

    print(f"Recalculated Dataset Metrics ({total_docs} total docs):")
    print(f"  - PASS          : {passed_count} ({passed_count/total_docs*100:.1f}%)")
    print(f"  - NEEDS_REVISION: {needs_revision_count} ({needs_revision_count/total_docs*100:.1f}%)")
    print(f"  - DISCARD       : {discarded_count} ({discarded_count/total_docs*100:.1f}%)")
    print(f"  - Average Score : {avg_score}/100")

    if not dry_run:
        # Update review_report.json
        final_report_data = dict(existing_report_data)
        final_report_data["total_documents"] = total_docs
        final_report_data["passed_count"] = passed_count
        final_report_data["needs_revision_count"] = needs_revision_count
        final_report_data["discarded_count"] = discarded_count
        final_report_data["discarded_paths"] = discarded_paths
        final_report_data["average_score"] = avg_score
        final_report_data["reports"] = sorted(
            report_by_doc_id.values(), key=lambda x: str(x.get("doc_id", ""))
        )

        report_path.write_text(
            json.dumps(final_report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✅ Saved updated review report: {report_path}")

        # Update review_progress.json
        if progress_path.exists():
            try:
                progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                progress_data["passed_count"] = passed_count
                progress_data["needs_revision_count"] = needs_revision_count
                progress_data["discarded_count"] = discarded_count
                progress_data["average_score"] = avg_score
                progress_path.write_text(
                    json.dumps(progress_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  ✅ Saved updated review progress: {progress_path}")
            except Exception as e:
                print(f"  ⚠️ Could not update progress file: {e}")

    print("\nMerge complete.")


if __name__ == "__main__":
    main()
