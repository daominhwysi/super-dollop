#!/usr/bin/env python3
"""
CLI Runner for isolated Editor/Reviewer revision loops.

Discovers all documents in NEEDS_REVISION status from review_report.json,
applies deterministic and targeted LLM surgical repairs, validates score elevation,
and optionally saves repaired ground truth to merged.xml and merged.json.
 uv run python tools/edit_review_set.py --help
 usage: edit_review_set.py [-h] [--report REPORT] [--raw-dir RAW_DIR] [--annotated-dir ANNOTATED_DIR] [--doc-id TARGET_DOC] [--model MODEL]
                          [--provider PROVIDER] [--reviewer-model REVIEWER_MODEL] [--reviewer-provider REVIEWER_PROVIDER]
                          [--editor-thinking EDITOR_THINKING] [--reviewer-thinking REVIEWER_THINKING] [--concurrency CONCURRENCY] [--limit LIMIT]
                          [--max-passes MAX_PASSES] [--filter FILTER_DECISION] [--branch-dir BRANCH_DIR] [--auto-save] [--dry-run]
                          [--out-report OUT_REPORT] [--reuse-prerun-logs] [--no-reuse-prerun-logs] [--max-doc-size MAX_DOC_SIZE] [--force-large]

Batch Revision Resolution Engine using EditorAgent (Search/Replace Diff Blocks)

options:
  -h, --help            show this help message and exit
  --report REPORT, -r REPORT
                        Path to review report JSON (default: backend/logs/review_report.json)
  --raw-dir RAW_DIR     Path to raw markdown directory (default: data/sequence_labelling_input_data)
  --annotated-dir ANNOTATED_DIR
                        Path to annotated XML directory (default: data/sequence_labelling_annotated)
  --doc-id TARGET_DOC, --target-doc TARGET_DOC
                        Filter resolution to a specific document ID or substring (e.g. exam_254)
  --model MODEL         Editor model name (default: gemini-3.8-flash)
  --provider PROVIDER   Editor LLM provider (default: agy)
  --reviewer-model REVIEWER_MODEL
                        Reviewer model name (default: gpt-5.6-luna)
  --reviewer-provider REVIEWER_PROVIDER
                        Reviewer provider (default: codex)
  --editor-thinking EDITOR_THINKING, --thinking EDITOR_THINKING
                        Editor reasoning/thinking effort (default: high)
  --reviewer-thinking REVIEWER_THINKING
                        Reviewer reasoning/thinking effort (default: high)
  --concurrency CONCURRENCY, -c CONCURRENCY
                        Concurrency worker threads (default: 4)
  --limit LIMIT, -l LIMIT
                        Limit number of documents to repair
  --max-passes MAX_PASSES, --max-rounds MAX_PASSES
                        Maximum repair passes per document (default: 2)
  --filter FILTER_DECISION, --only FILTER_DECISION
                        Review decisions to repair: needs_revision, discards, or discards,needs_revision (default: needs_revision)
  --branch-dir BRANCH_DIR, --output-dir BRANCH_DIR
                        Branch output directory to save repaired files (default: data/sequence_labelling_annotated_branch). Guarantees original files
                        are never overwritten.
  --auto-save, --save-branch
                        Automatically save repaired merged.xml and merged.json to the branch folder on success (never overwrites original files)
  --dry-run             Simulate repairs in memory without writing to disk
  --out-report OUT_REPORT
                        Path to output resolution report JSON (default: backend/logs/revision_resolution_report.json)
  --reuse-prerun-logs   Directly consume prerun error logs from report on Round 0 to seed Editor (skips duplicate Reviewer LLM call; default: True)
  --no-reuse-prerun-logs, --fresh-review
                        Force fresh Reviewer LLM evaluation on Round 0 instead of reusing prerun error logs
  --max-doc-size MAX_DOC_SIZE
                        Maximum XML file size in bytes to process (default: 300000 / 300KB)
  --force-large         Force processing documents exceeding --max-doc-size
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure workspace root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tqdm import tqdm

from sequence_labelling.config import (
    EDITOR_MODEL,
    EDITOR_PROVIDER,
    EDITOR_THINKING,
    REVIEWER_MODEL,
    REVIEWER_PROVIDER,
    REVIEWER_THINKING,
    WORKSPACE_DIR,
    get_provider_base_url,
)
from sequence_labelling.annotator.editor import EditorAgent
from sequence_labelling.annotator.reviewer import (
    AnnotationReviewerAgent,
    AuditIssue,
    IssueSeverity,
    ReviewReport,
    PARSER_ERROR_PENALTIES,
)
from sequence_labelling.annotator.revision_loop import EditorReviewerLoop
from sequence_labelling.parser.parser import parse_spans_into_structured_questions
from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import (
    parse_xml_with_anchors,
)


def load_revision_targets(
    report_path: Path,
    raw_dir: Path,
    annotated_dir: Path,
    decisions: Set[str],
) -> List[Dict[str, Any]]:
    """Load documents matching the requested review decisions."""
    json_path = report_path.with_suffix(".json") if report_path.suffix == ".md" else report_path
    if not json_path.exists():
        print(f"❌ Error: Report file '{json_path}' not found.")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = []
    reports = data.get("reports", [])

    for r in reports:
        if r.get("decision") not in decisions:
            continue

        doc_id = r.get("doc_id")
        xml_p = r.get("file_path")
        raw_p = r.get("raw_file_path")

        # Always prefer canonical source in annotated_dir to prevent branch collisions
        cand_annot = None
        for cand in [
            annotated_dir / f"{doc_id}/merged.xml",
            annotated_dir / f"{doc_id}.xml",
        ]:
            if cand.exists():
                cand_annot = cand
                break

        if cand_annot:
            xml_path = cand_annot
        else:
            xml_path = Path(xml_p) if xml_p else None

        raw_path = Path(raw_p) if raw_p else None
        if not (raw_path and raw_path.exists()):
            # Try to resolve relative to raw_dir
            if xml_path:
                try:
                    rel_to_annot = xml_path.relative_to(annotated_dir.resolve())
                except ValueError:
                    rel_to_annot = Path(xml_p)
                    if str(rel_to_annot).startswith("data/sequence_labelling_annotated/"):
                        rel_to_annot = Path(str(rel_to_annot)[len("data/sequence_labelling_annotated/"):])

                doc_stem = rel_to_annot.parent if rel_to_annot.name in ["merged.xml", "merged.json"] else rel_to_annot.with_suffix("")
                cand_raw = raw_dir / f"{doc_stem}.md"
                if cand_raw.exists():
                    raw_path = cand_raw

        if xml_path and xml_path.exists() and raw_path and raw_path.exists():
            issues_raw = r.get("issues", [])
            parsed_issues = []
            for iss in issues_raw:
                parsed_issues.append(
                    AuditIssue(
                        category=iss.get("category", "general"),
                        severity=IssueSeverity(iss.get("severity", "MAJOR")),
                        message=iss.get("message", ""),
                        line_number=iss.get("line_number"),
                        context_snippet=iss.get("context_snippet"),
                    )
                )

            try:
                rel_xml = xml_path.resolve().relative_to(annotated_dir.resolve())
            except ValueError:
                if xml_path.name in ["merged.xml", "merged.json"]:
                    rel_xml = Path(doc_id) / xml_path.name
                else:
                    rel_xml = Path(xml_path.name)

            prerun_report = None
            try:
                prerun_report = ReviewReport.model_validate(r)
            except Exception:
                pass

            cached_llm = None
            if r.get("llm_score") is not None or r.get("llm_confirmations") or r.get("issues"):
                cached_llm = {
                    "score": r.get("llm_score"),
                    "rubric_scores": r.get("rubric_scores"),
                    "issues": [
                        iss for iss in r.get("issues", [])
                        if (
                            iss.get("source") == "llm"
                            or (
                                "source" not in iss
                                and (
                                    iss.get("category") in PARSER_ERROR_PENALTIES
                                    or iss.get("category") == "llm_semantic"
                                )
                            )
                        )
                    ],
                    "parser_error_confirmations": r.get("llm_confirmations", []),
                    "summary": r.get("summary", ""),
                    "is_malfunctioned": r.get("is_malfunctioned", False),
                    "discard_reasons": [
                        disc for disc in r.get("discard_reasons", [])
                        if "[MALFUNCTION]" in disc
                    ],
                }

            # If not present in report, check per-document audit_report.json on disk
            if prerun_report is None or not cached_llm:
                audit_file = (
                    xml_path.parent / "audit_report.json"
                    if xml_path.name in ["merged.xml", "merged.json"]
                    else xml_path.with_suffix(".audit.json")
                )
                if audit_file.exists():
                    try:
                        with open(audit_file, "r", encoding="utf-8") as f_aud:
                            audit_data = json.load(f_aud)
                            if prerun_report is None:
                                prerun_report = ReviewReport.model_validate(audit_data)
                            if not cached_llm and audit_data.get("llm_score") is not None:
                                cached_llm = {
                                    "score": audit_data.get("llm_score"),
                                    "rubric_scores": audit_data.get("rubric_scores"),
                                    "issues": audit_data.get("issues", []),
                                    "parser_error_confirmations": audit_data.get("llm_confirmations", []),
                                    "summary": audit_data.get("summary", ""),
                                    "is_malfunctioned": audit_data.get("is_malfunctioned", False),
                                    "discard_reasons": audit_data.get("discard_reasons", []),
                                }
                    except Exception:
                        pass

            targets.append({
                "doc_id": doc_id,
                "source_decision": r.get("decision"),
                "xml_path": xml_path.resolve(),
                "raw_path": raw_path.resolve(),
                "rel_path": raw_path.resolve().relative_to(raw_dir.resolve()),
                "rel_xml_path": rel_xml,
                "initial_score": r.get("overall_score", 0.0),
                "issues": parsed_issues,
                "initial_report": prerun_report,
                "cached_llm_result": cached_llm,
            })
        else:
            print(f"⚠️ [Warning] Skipping '{doc_id}': could not locate XML ({xml_path}) or Raw ({raw_path})")

    return targets


CHECKPOINT_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_path(branch_dir: Path, target: Dict[str, Any]) -> Path:
    return branch_dir / target["rel_xml_path"].parent / "revision_state.json"


def apply_branch_resume(
    targets: List[Dict[str, Any]],
    branch_dir: Path,
    resume: bool,
    retry_unresolved: bool = False,
) -> List[Dict[str, Any]]:
    """Use compatible branch candidates as inputs for a subsequent run."""
    if not resume:
        return targets

    resumed: List[Dict[str, Any]] = []
    for target in targets:
        state_path = _checkpoint_path(branch_dir, target)
        if not state_path.exists():
            resumed.append(target)
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                resumed.append(target)
                continue
            if (
                state.get("original_xml_sha256") != _sha256_file(target["xml_path"])
                or state.get("raw_sha256") != _sha256_file(target["raw_path"])
            ):
                resumed.append(target)
                continue
            status = state.get("status")
            if status == "PASS" and not retry_unresolved:
                target["resume_skip_state"] = state
                resumed.append(target)
                continue
            candidate = branch_dir / target["rel_xml_path"]
            if candidate.exists() and status in {"NEEDS_REVISION", "DISCARD", "ERROR"}:
                target = dict(target)
                target["xml_path"] = candidate.resolve()
                target["resume_from_branch"] = True
            resumed.append(target)
        except (OSError, json.JSONDecodeError):
            resumed.append(target)
    return resumed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Revision Resolution Engine using EditorAgent (Search/Replace Diff Blocks)"
    )
    parser.add_argument(
        "--report",
        "-r",
        type=str,
        default="backend/logs/review_report.json",
        help="Path to review report JSON (default: backend/logs/review_report.json)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_input_data"),
        help="Path to raw markdown directory (default: data/sequence_labelling_input_data)",
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_annotated"),
        help="Path to annotated XML directory (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--doc-id",
        "--target-doc",
        dest="target_doc",
        type=str,
        default=None,
        help="Filter resolution to a specific document ID or substring (e.g. exam_254)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EDITOR_MODEL,
        help=f"Editor model name (default: {EDITOR_MODEL})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=EDITOR_PROVIDER,
        help=f"Editor LLM provider (default: {EDITOR_PROVIDER})",
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default=REVIEWER_MODEL,
        help=f"Reviewer model name (default: {REVIEWER_MODEL})",
    )
    parser.add_argument(
        "--reviewer-provider",
        type=str,
        default=REVIEWER_PROVIDER,
        help=f"Reviewer provider (default: {REVIEWER_PROVIDER})",
    )
    parser.add_argument(
        "--editor-thinking",
        "--thinking",
        dest="editor_thinking",
        type=str,
        default=EDITOR_THINKING,
        help=f"Editor reasoning/thinking effort (default: {EDITOR_THINKING})",
    )
    parser.add_argument(
        "--reviewer-thinking",
        type=str,
        default=REVIEWER_THINKING,
        help=f"Reviewer reasoning/thinking effort (default: {REVIEWER_THINKING})",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Concurrency worker threads (default: 4)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of documents to repair",
    )
    parser.add_argument(
        "--max-passes",
        "--max-rounds",
        dest="max_passes",
        type=int,
        default=2,
        help="Maximum repair passes per document (default: 2)",
    )
    parser.add_argument(
        "--filter",
        "--only",
        dest="filter_decision",
        default="needs_revision",
        help=(
            "Review decisions to repair: needs_revision, discards, or "
            "discards,needs_revision (default: needs_revision)"
        ),
    )
    parser.add_argument(
        "--branch-dir",
        "--output-dir",
        dest="branch_dir",
        type=str,
        default=str(WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch"),
        help=(
            "Branch output directory to save repaired files "
            "(default: data/sequence_labelling_annotated_branch). "
            "Guarantees original files are never overwritten."
        ),
    )
    parser.add_argument(
        "--auto-save",
        "--save-branch",
        dest="auto_save",
        action="store_true",
        default=True,
        help="Automatically save repaired merged.xml and merged.json to the branch folder on success (never overwrites original files; default: True)",
    )
    parser.add_argument(
        "--no-auto-save",
        dest="auto_save",
        action="store_false",
        help="Disable saving repaired files to branch folder (report-only mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate repairs in memory without writing to disk",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume_branch",
        action="store_false",
        default=True,
        help="Ignore compatible branch checkpoints and start from original XML.",
    )
    parser.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="Retry branch checkpoints that previously ended without PASS.",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="backend/logs/revision_resolution_report.json",
        help="Path to output resolution report JSON (default: backend/logs/revision_resolution_report.json)",
    )
    parser.add_argument(
        "--reuse-prerun-logs",
        dest="reuse_prerun_logs",
        action="store_true",
        default=True,
        help="Directly consume prerun error logs from report on Round 0 to seed Editor (skips duplicate Reviewer LLM call; default: True)",
    )
    parser.add_argument(
        "--no-reuse-prerun-logs",
        "--fresh-review",
        dest="reuse_prerun_logs",
        action="store_false",
        help="Force fresh Reviewer LLM evaluation on Round 0 instead of reusing prerun error logs",
    )
    parser.add_argument(
        "--max-doc-size",
        type=int,
        default=300000,
        help="Maximum XML file size in bytes to process (default: 300000 / 300KB)",
    )
    parser.add_argument(
        "--force-large",
        action="store_true",
        help="Force processing documents exceeding --max-doc-size",
    )
    return parser.parse_args()


def process_single_revision(
    target: Dict[str, Any],
    revision_loop: EditorReviewerLoop,
    auto_save: bool,
    dry_run: bool,
    branch_dir: Path,
    annotated_dir: Path,
    reuse_prerun_logs: bool = True,
) -> Dict[str, Any]:
    """Processes repair for a single document target, saving to branch_dir if enabled."""
    doc_id = target["doc_id"]
    xml_path: Path = target["xml_path"]
    raw_path: Path = target["raw_path"]
    annotated_xml = xml_path.read_text(encoding="utf-8")
    raw_ocr_text = raw_path.read_text(encoding="utf-8")

    use_cached_review = reuse_prerun_logs and not target.get("resume_from_branch")
    initial_report = target.get("initial_report") if use_cached_review else None
    cached_llm_result = target.get("cached_llm_result") if use_cached_review else None

    rel_xml = target.get("rel_xml_path")
    if not rel_xml:
        try:
            rel_xml = xml_path.resolve().relative_to(annotated_dir.resolve())
        except ValueError:
            rel_xml = Path(doc_id) / xml_path.name

    branch_xml_path = branch_dir / rel_xml
    if branch_xml_path.resolve() == xml_path.resolve():
        # Defensive fallback: if xml_path was somehow resolved into branch_dir, point to canonical annotated_dir
        cand_in_annot = annotated_dir / rel_xml
        if cand_in_annot.exists() and cand_in_annot.resolve() != branch_xml_path.resolve():
            xml_path = cand_in_annot
        else:
            raise RuntimeError(
                f"Branch path {branch_xml_path} matches original path {xml_path}! "
                "Direct overwriting of original files is strictly prohibited."
            )

    state_path = branch_xml_path.with_name("revision_state.json")
    state: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "doc_id": doc_id,
        "status": "IN_PROGRESS",
        "original_xml_sha256": _sha256_file(target["original_xml_path"] if target.get("original_xml_path") else xml_path),
        "raw_sha256": _sha256_file(raw_path),
        "source_xml_path": str(target.get("original_xml_path", xml_path)),
        "raw_path": str(raw_path),
        "branch_xml_path": str(branch_xml_path),
        "rounds": [],
    }

    def checkpoint(event: Dict[str, Any]) -> None:
        if not auto_save or dry_run:
            return
        branch_xml_path.parent.mkdir(parents=True, exist_ok=True)
        branch_xml_path.write_text(event["candidate_xml"], encoding="utf-8")
        round_number = int(event.get("round_number", 0))
        event_name = "initial_review.json" if event.get("stage") == "initial_review" else f"round_{round_number:03d}.json"
        event_path = branch_xml_path.parent / "revisions" / event_name
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        state["latest_round"] = round_number
        state["latest_event"] = str(event_path)
        state["rounds"] = [
            item for item in state["rounds"]
            if item.get("round_number") != round_number or item.get("stage") != event.get("stage")
        ]
        state["rounds"].append({
            "stage": event.get("stage"),
            "round_number": round_number,
            "path": str(event_path),
            "decision": event.get("review", {}).get("decision"),
            "score": event.get("review", {}).get("overall_score"),
        })
        _atomic_write_json(state_path, state)

    run_kwargs = {
        "annotated_xml": annotated_xml,
        "raw_ocr_text": raw_ocr_text,
        "doc_id": doc_id,
        "initial_report": initial_report,
        "cached_llm_result": cached_llm_result,
    }
    if auto_save and not dry_run:
        run_kwargs["checkpoint_callback"] = checkpoint
    res = revision_loop.run(**run_kwargs)

    already_passed = (
        res.success
        and res.initial_decision == "PASS"
        and res.final_decision == "PASS"
        and res.applied_patches_count == 0
        and res.rounds_completed == 0
    )

    saved = False
    if auto_save and not dry_run and not already_passed and res.repaired_xml:
        # Create branch subdirectories as needed
        branch_xml_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the best candidate even when unresolved so the next run can resume.
        branch_xml_path.write_text(res.repaired_xml, encoding="utf-8")

        # Update accompanying merged.json in branch folder if original was a merged.xml structure
        if xml_path.name == "merged.xml" or xml_path.with_name("merged.json").exists():
            branch_json_path = branch_xml_path.with_name("merged.json")
            try:
                spans, stimuli, questions = parse_xml_with_anchors(
                    raw_ocr_text, res.repaired_xml
                )
                json_data = {
                    "document_id": doc_id,
                    "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "questions_count": len(questions),
                    "questions": questions,
                    "stimuli": stimuli,
                }
                branch_json_path.write_text(
                    json.dumps(json_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                state["merged_json_error"] = str(e)
                print(f"⚠️ [Warning] Could not update {branch_json_path}: {e}")

        saved = True

        state["status"] = res.final_decision
        state["result"] = {
            "doc_id": doc_id,
            "initial_score": res.initial_score,
            "final_score": res.final_score,
            "initial_decision": res.initial_decision,
            "final_decision": res.final_decision,
            "success": res.success,
            "applied_patches": res.applied_patches_count,
            "failed_patches": res.failed_patches_count,
            "rounds_completed": res.rounds_completed,
            "rounds": [round_result.model_dump(mode="json") for round_result in res.rounds],
            "diff_summary": res.diff_summary,
            "duration_seconds": res.duration_seconds,
        }
        _atomic_write_json(state_path, state)
    elif auto_save and not dry_run and already_passed:
        state["status"] = "PASS"
        state["result"] = {
            "doc_id": doc_id,
            "initial_score": res.initial_score,
            "final_score": res.final_score,
            "initial_decision": res.initial_decision,
            "final_decision": res.final_decision,
            "success": res.success,
            "already_passed": True,
            "rounds_completed": res.rounds_completed,
        }
        _atomic_write_json(state_path, state)

    return {
        "doc_id": doc_id,
        "source_decision": target["source_decision"],
        "rel_path": str(target["rel_path"]),
        "initial_score": res.initial_score,
        "final_score": res.final_score,
        "initial_decision": res.initial_decision,
        "final_decision": res.final_decision,
        "success": res.success,
        "already_passed": already_passed,
        "applied_patches": res.applied_patches_count,
        "failed_patches": res.failed_patches_count,
        "rounds_completed": res.rounds_completed,
        "rounds": [round_result.model_dump(mode="json") for round_result in res.rounds],
        "saved_to_disk": saved,
        "saved_to_branch": saved,
        "branch_xml_path": str(branch_xml_path) if saved else None,
        "duration_seconds": res.duration_seconds,
        "diff_summary": res.diff_summary,
    }


def main():
    args = parse_args()

    if args.model:
        args.model = args.model.strip()
    if args.provider:
        args.provider = args.provider.strip()
    if args.reviewer_model:
        args.reviewer_model = args.reviewer_model.strip()
    if args.reviewer_provider:
        args.reviewer_provider = args.reviewer_provider.strip()

    if args.editor_thinking:
        args.editor_thinking = args.editor_thinking.strip()
    if args.reviewer_thinking:
        args.reviewer_thinking = args.reviewer_thinking.strip()

    decision_aliases = {
        "needs_revision": "NEEDS_REVISION",
        "revision": "NEEDS_REVISION",
        "discards": "DISCARD",
        "discard": "DISCARD",
    }
    requested_decisions = {
        decision_aliases.get(value.strip().lower(), value.strip().upper())
        for value in args.filter_decision.split(",")
        if value.strip()
    }
    allowed_decisions = {"NEEDS_REVISION", "DISCARD"}
    if not requested_decisions or not requested_decisions <= allowed_decisions:
        raise SystemExit(
            "error: --filter must be needs_revision, discards, or "
            "discards,needs_revision"
        )

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = WORKSPACE_DIR / report_path
    if not report_path.exists() and (WORKSPACE_DIR / "data" / "review_report.json").exists():
        report_path = WORKSPACE_DIR / "data" / "review_report.json"

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = WORKSPACE_DIR / raw_dir

    annotated_dir = Path(args.annotated_dir)
    if not annotated_dir.is_absolute():
        annotated_dir = WORKSPACE_DIR / annotated_dir

    branch_dir = Path(args.branch_dir)
    if not branch_dir.is_absolute():
        branch_dir = WORKSPACE_DIR / branch_dir

    if branch_dir.resolve() == annotated_dir.resolve():
        raise SystemExit(
            "❌ Error: --branch-dir cannot be the same as --annotated-dir.\n"
            "Directly overwriting original files is disabled. Please specify a separate branch directory\n"
            "(e.g. data/sequence_labelling_annotated_branch)."
        )

    targets = load_revision_targets(
        report_path,
        raw_dir,
        annotated_dir,
        decisions=requested_decisions,
    )

    if args.target_doc:
        query = args.target_doc.strip()
        targets = [
            t for t in targets
            if query in t["doc_id"] or query in str(t["rel_path"])
        ]

    if args.max_doc_size and not args.force_large:
        normal_targets = []
        for t in targets:
            size = t["xml_path"].stat().st_size
            if size > args.max_doc_size:
                print(
                    f"⚠️  Skipping oversized document {t['doc_id']} "
                    f"({size/1024:.1f} KB > {args.max_doc_size/1024:.1f} KB). "
                    f"Use --force-large to include."
                )
            else:
                normal_targets.append(t)
        targets = normal_targets

    if args.limit:
        targets = targets[: args.limit]

    for target in targets:
        target["original_xml_path"] = target["xml_path"]
    targets = apply_branch_resume(
        targets,
        branch_dir=branch_dir,
        resume=args.resume_branch,
        retry_unresolved=args.retry_unresolved,
    )
    resumed_passes = [t for t in targets if t.get("resume_skip_state")]
    targets = [t for t in targets if not t.get("resume_skip_state")]
    total_target_count = len(targets) + len(resumed_passes)

    base_url = get_provider_base_url(args.provider)

    print("=" * 70)
    print("🛠️  AZOZO EDITOR / REVIEWER REVISION LOOP")
    print("=" * 70)
    print(f"  Source Report   : {report_path}")
    print(f"  Raw Input Dir   : {raw_dir}")
    print(f"  Annotated Dir   : {annotated_dir} (READ-ONLY PROTECTED)")
    print(f"  Branch Dir      : {branch_dir}")
    print(f"  Total Targets   : {total_target_count} document(s) ({len(resumed_passes)} resumed PASS skipped)")
    print(f"  Decision Filter : {', '.join(sorted(requested_decisions))}")
    print(f"  Editor Model    : {args.model} (effort: {args.editor_thinking})")
    print(f"  Editor Provider : {args.provider}")
    print(f"  Reviewer Model  : {args.reviewer_model} (effort: {args.reviewer_thinking})")
    print(f"  Reviewer Provider: {args.reviewer_provider}")
    print(f"  Base URL        : {base_url if base_url else '(Codex / SDK Native)'}")
    print(f"  Concurrency     : {args.concurrency} worker thread(s)")
    print(f"  Max Doc Size    : {args.max_doc_size / 1024:.0f} KB")
    print(f"  Prerun Logs     : {'DIRECT CONSUME (Cached Round 0)' if args.reuse_prerun_logs else 'FRESH REVIEW (Re-running Reviewer Round 0)'}")
    print(f"  Auto-Save       : {'ENABLED -> ' + str(branch_dir) if args.auto_save else 'DISABLED (Report Only)'}")
    print(f"  Mode            : {'DRY RUN (In-Memory)' if args.dry_run else 'LIVE REPAIR'}")
    print("=" * 70)

    if not targets and not resumed_passes:
        print("✅ No target documents found to repair.")
        sys.exit(0)

    editor = EditorAgent(
        model=args.model,
        provider=args.provider,
        thinking=args.editor_thinking,
    )
    reviewer = AnnotationReviewerAgent(
        model=args.reviewer_model,
        provider=args.reviewer_provider,
        thinking=args.reviewer_thinking,
    )
    revision_loop = EditorReviewerLoop(
        editor=editor,
        reviewer=reviewer,
        max_rounds=args.max_passes,
    )
    results = []
    for target in resumed_passes:
        state = target["resume_skip_state"]
        saved_result = dict(state.get("result") or {})
        saved_result.setdefault("doc_id", target["doc_id"])
        saved_result.setdefault("source_decision", target["source_decision"])
        saved_result.setdefault("rel_path", str(target["rel_path"]))
        saved_result.setdefault("success", True)
        saved_result["already_passed"] = True
        saved_result["saved_to_branch"] = True
        saved_result["saved_to_disk"] = True
        saved_result["branch_xml_path"] = str(branch_dir / target["rel_xml_path"])
        results.append(saved_result)
    success_count = len(resumed_passes)
    fail_count = 0
    already_passed_count = len(resumed_passes)

    out_rep_path = Path(args.out_report)
    if not out_rep_path.is_absolute():
        out_rep_path = WORKSPACE_DIR / out_rep_path
    out_rep_path.parent.mkdir(parents=True, exist_ok=True)

    def save_partial_report(status: str = "IN_PROGRESS") -> None:
        _atomic_write_json(
            out_rep_path,
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": status,
                "total_targets": total_target_count,
                "completed_targets": len(results),
                "success_count": success_count,
                "already_passed_count": already_passed_count,
                "fail_count": fail_count,
                "success_rate": round(success_count / max(1, total_target_count) * 100, 1),
                "annotated_dir": str(annotated_dir),
                "branch_dir": str(branch_dir),
                "auto_save": args.auto_save,
                "reuse_prerun_logs": args.reuse_prerun_logs,
                "resume_branch": args.resume_branch,
                "retry_unresolved": args.retry_unresolved,
                "results": results,
            },
        )

    save_partial_report()

    pbar = tqdm(total=len(targets), desc="Resolving Revisions", unit="doc")

    max_workers = max(1, min(args.concurrency, len(targets)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_revision,
                target,
                revision_loop,
                args.auto_save,
                args.dry_run,
                branch_dir,
                annotated_dir,
                args.reuse_prerun_logs,
            ): target
            for target in targets
        }

        for future in as_completed(futures):
            target = futures[future]
            doc_id = target["doc_id"]
            try:
                res = future.result()
                results.append(res)
                if res.get("already_passed"):
                    success_count += 1
                    already_passed_count += 1
                    tqdm.write(
                        f"  ✅ [Already PASS after fresh review] {doc_id}: {res['initial_score']:.1f} ({res['initial_decision']}) — no edit needed"
                    )
                elif res["success"]:
                    success_count += 1
                    saved_info = f" [Saved to branch: {res['branch_xml_path']}]" if res.get("saved_to_branch") else ""
                    tqdm.write(
                        f"  ✅ [Repaired] {doc_id}: {res['initial_score']:.1f} ({res['initial_decision']}) -> {res['final_score']:.1f} ({res['final_decision']}) [{res['applied_patches']} patches in {res['duration_seconds']:.1f}s]{saved_info}"
                    )
                else:
                    fail_count += 1
                    tqdm.write(
                        f"  ⚠️ [Partial/Unresolved] {doc_id}: {res['initial_score']:.1f} -> {res['final_score']:.1f} ({res['final_decision']})"
                    )
                save_partial_report()
            except Exception as e:
                fail_count += 1
                results.append({
                    "doc_id": doc_id,
                    "source_decision": target["source_decision"],
                    "rel_path": str(target["rel_path"]),
                    "success": False,
                    "error": str(e),
                })
                save_partial_report()
                tqdm.write(f"\n❌ [Error] Failed repairing {doc_id}: {e}")

            pbar.update(1)
            pbar.set_postfix({"ok": success_count, "pass": already_passed_count, "fail": fail_count})

    pbar.close()

    summary_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "COMPLETED",
        "total_targets": total_target_count,
        "completed_targets": len(results),
        "success_count": success_count,
        "already_passed_count": already_passed_count,
        "fail_count": fail_count,
        "success_rate": round(success_count / max(1, total_target_count) * 100, 1),
        "annotated_dir": str(annotated_dir),
        "branch_dir": str(branch_dir),
        "auto_save": args.auto_save,
        "reuse_prerun_logs": args.reuse_prerun_logs,
        "resume_branch": args.resume_branch,
        "retry_unresolved": args.retry_unresolved,
        "results": results,
    }

    _atomic_write_json(out_rep_path, summary_data)

    print("\n" + "=" * 70)
    print("🎉 REVISION RESOLUTION COMPLETED!")
    print(f"  Total Targets Processed : {total_target_count}")
    print(f"  Successfully Upgraded   : {success_count} ({summary_data['success_rate']}%)")
    print(f"  Unresolved / Partial    : {fail_count}")
    if args.auto_save:
        print(f"  Branch Directory        : {branch_dir}")
    print(f"  Summary Report          : {out_rep_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
