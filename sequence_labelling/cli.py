"""
Unified CLI Entry Point for Vietnamese Sequence Labelling (vsl / sequence-label).

Consolidates all operational commands across document ingestion, top-down cartography,
batch sequence labelling, quality auditing, surgical repair, branch staging, dataset building,
evaluation, and telemetry observability.
Includes native background TMUX orchestration (--tmux).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sequence_labelling.config import (
    DATA_DIR,
    MODEL_DIR,
    WORKSPACE_DIR,
    EDITOR_MODEL,
    EDITOR_PROVIDER,
    EDITOR_THINKING,
    REVIEWER_MODEL,
    REVIEWER_PROVIDER,
    REVIEWER_THINKING,
    PARSER_MODEL,
    PARSER_PROVIDER,
    OCR_MODEL,
)


def _launch_in_tmux(session_name: str, subcommand: str, force: bool = False) -> None:
    """Launch any VSL command inside a detached background TMUX session with automatic logging."""
    # Filter sys.argv to remove --tmux and --force-tmux
    cleaned_argv: List[str] = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg == "--force-tmux":
            continue
        if arg == "--tmux":
            if i + 1 < len(sys.argv[1:]) and not sys.argv[1:][i + 1].startswith("-"):
                skip_next = True
            continue
        if arg.startswith("--tmux="):
            continue
        cleaned_argv.append(arg)

    has_res = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
    if has_res.returncode == 0:
        if not force:
            print("=" * 70)
            print(f"⚠️  [TMUX] Session '{session_name}' is already running!")
            print("=" * 70)
            print(f"  • To attach:  tmux attach -t {session_name}")
            print(f"  • To tail:    tail -f artifacts/ocr_logs/{session_name}.log")
            print(f"  • To restart: Add --force-tmux to kill and restart this session.")
            print("=" * 70)
            return
        else:
            print(f"⚠️  [TMUX] Terminating existing session '{session_name}' (--force-tmux specified)...")
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
            time.sleep(0.5)

    log_dir = WORKSPACE_DIR / "artifacts" / "ocr_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_name}.log"

    cmd_inner = (
        f"cd '{WORKSPACE_DIR}' && "
        f"PYTHONUNBUFFERED=1 uv run python -m sequence_labelling.cli {' '.join(shlex.quote(a) for a in cleaned_argv)} "
        f"2>&1 | tee -a '{log_path}'"
    )

    launch_res = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, cmd_inner],
        capture_output=True,
        text=True,
    )
    if launch_res.returncode != 0:
        print(f"❌ [TMUX Error] Failed to launch session '{session_name}': {launch_res.stderr.strip()}")
        return

    print("=" * 70)
    print(f"🚀 [TMUX] Successfully launched background session: '{session_name}'")
    print("=" * 70)
    print(f"  Command     : vsl {' '.join(cleaned_argv)}")
    print(f"  Log File    : {log_path}")
    print(f"  To Attach   : tmux attach -t {session_name}")
    print(f"  To Tail Log : tail -f {log_path}")
    print(f"  To Kill     : tmux kill-session -t {session_name}")
    print(f"  To Inspect  : uv run vsl status")
    print("=" * 70)


def _cmd_paths(args: argparse.Namespace) -> None:
    print(f"workspace={WORKSPACE_DIR}")
    print(f"data={DATA_DIR}")
    print(f"models={MODEL_DIR}")


def _cmd_ocr(args: argparse.Namespace) -> None:
    """Autonomous batch OCR using AgyPDFOCRConverter with rate limits and 5h-cap recovery."""
    from tools.ocr.run_agy_batch_ocr import main as ocr_main
    ocr_args = []
    if args.raw_dir:
        ocr_args.extend(["--raw-dir", str(args.raw_dir)])
    if args.out_dir:
        ocr_args.extend(["--out-dir", str(args.out_dir)])
    if args.concurrency:
        ocr_args.extend(["--concurrency", str(args.concurrency)])
    if args.limit:
        ocr_args.extend(["--limit", str(args.limit)])
    if args.overwrite:
        ocr_args.append("--overwrite")
    if args.sleep_on_cap:
        ocr_args.append("--sleep-on-cap")

    old_argv = sys.argv
    try:
        sys.argv = ["run_agy_batch_ocr.py"] + ocr_args
        ocr_main()
    finally:
        sys.argv = old_argv


def _cmd_cartography(args: argparse.Namespace) -> None:
    """Generate top-down topological page manifests and clean-cut seams."""
    from tools.cartography.generate_dataset_cartography import main as carto_main
    carto_args = []
    if args.input_dir:
        carto_args.extend(["--input-dir", str(args.input_dir)])
    if args.out_dir:
        carto_args.extend(["--output-dir", str(args.out_dir)])
    if args.model:
        carto_args.extend(["--model", str(args.model)])
    if args.provider:
        carto_args.extend(["--provider", str(args.provider)])
    if args.limit:
        carto_args.extend(["--limit", str(args.limit)])
    if args.overwrite:
        carto_args.append("--overwrite")

    old_argv = sys.argv
    try:
        sys.argv = ["generate_dataset_cartography.py"] + carto_args
        carto_main()
    finally:
        sys.argv = old_argv


def _cmd_metadata(args: argparse.Namespace) -> None:
    """Batch document metadata mining (subject, grade, year)."""
    from tools.metadata.generate_dataset_metadata import main as meta_main
    meta_args = []
    if args.input_dir:
        meta_args.extend(["--input-dir", str(args.input_dir)])
    if args.out_file:
        meta_args.extend(["--output-file", str(args.out_file)])
    if args.model:
        meta_args.extend(["--model", str(args.model)])
    if args.provider:
        meta_args.extend(["--provider", str(args.provider)])
    if args.limit:
        meta_args.extend(["--limit", str(args.limit)])

    old_argv = sys.argv
    try:
        sys.argv = ["generate_dataset_metadata.py"] + meta_args
        meta_main()
    finally:
        sys.argv = old_argv


def _cmd_parse(args: argparse.Namespace) -> None:
    """Sequence labelling parser pipeline (Greedy Chunker + Parser Workers + Source Merger)."""
    from tools.annotation.run_agy_batch_annotation import main as parse_main
    p_args = []
    if args.raw_dir:
        p_args.extend(["--raw-dir", str(args.raw_dir)])
    if args.annotated_dir:
        p_args.extend(["--annotated-dir", str(args.annotated_dir)])
    if args.model:
        p_args.extend(["--model", str(args.model)])
    if args.provider:
        p_args.extend(["--provider", str(args.provider)])
    if args.concurrency:
        p_args.extend(["--concurrency", str(args.concurrency)])
    if args.limit:
        p_args.extend(["--limit", str(args.limit)])
    if args.overwrite:
        p_args.append("--overwrite")
    if args.sleep_on_cap:
        p_args.append("--sleep-on-cap")

    old_argv = sys.argv
    try:
        sys.argv = ["run_agy_batch_annotation.py"] + p_args
        parse_main()
    finally:
        sys.argv = old_argv


def _cmd_audit(args: argparse.Namespace) -> None:
    """Hybrid deterministic & LLM semantic review with composite scoring [0-100]."""
    from sequence_labelling.annotator.reviewer import main as review_main
    r_args = []
    if args.annotated_dir:
        r_args.extend(["--annotated-dir", str(args.annotated_dir)])
    if args.raw_dir:
        r_args.extend(["--raw-dir", str(args.raw_dir)])
    if args.model:
        r_args.extend(["--model", str(args.model)])
    if args.provider:
        r_args.extend(["--provider", str(args.provider)])
    if args.thinking:
        r_args.extend(["--thinking", str(args.thinking)])
    if args.limit:
        r_args.extend(["--limit", str(args.limit)])
    if args.out_report:
        r_args.extend(["--output-report", str(args.out_report)])
    if args.overwrite:
        r_args.append("--overwrite")
    if args.no_llm:
        r_args.append("--no-llm")
    if args.granularity:
        r_args.extend(["--granularity", str(args.granularity)])

    old_argv = sys.argv
    try:
        sys.argv = ["reviewer.py"] + r_args
        review_main()
    finally:
        sys.argv = old_argv


def _cmd_repair(args: argparse.Namespace) -> None:
    """Multi-turn surgical EditorAgent repair loop routing to branch staging."""
    from sequence_labelling.annotator.revision_loop import run_batch_revision
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = WORKSPACE_DIR / report_path

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = WORKSPACE_DIR / raw_dir

    annotated_dir = Path(args.annotated_dir)
    if not annotated_dir.is_absolute():
        annotated_dir = WORKSPACE_DIR / annotated_dir

    branch_dir = Path(args.branch_dir)
    if not branch_dir.is_absolute():
        branch_dir = WORKSPACE_DIR / branch_dir

    out_report = Path(args.out_report) if args.out_report else None
    if out_report and not out_report.is_absolute():
        out_report = WORKSPACE_DIR / out_report

    run_batch_revision(
        report_path=report_path,
        raw_dir=raw_dir,
        annotated_dir=annotated_dir,
        branch_dir=branch_dir,
        filter_decision=args.filter_decision,
        model=args.model,
        provider=args.provider,
        reviewer_model=args.reviewer_model,
        reviewer_provider=args.reviewer_provider,
        editor_thinking=args.editor_thinking,
        reviewer_thinking=args.reviewer_thinking,
        concurrency=args.concurrency,
        limit=args.limit,
        max_passes=args.max_passes,
        granularity=args.granularity,
        target_doc=args.target_doc,
        auto_save=args.auto_save,
        dry_run=args.dry_run,
        out_report=out_report,
        reuse_prerun_logs=args.reuse_prerun_logs,
        resume_branch=args.resume_branch,
        retry_unresolved=args.retry_unresolved,
        auto_reconcile=args.auto_reconcile,
        skip_unrecoverable=args.skip_unrecoverable,
        max_doc_size=args.max_doc_size,
        force_large=args.force_large,
    )


def _cmd_branch(args: argparse.Namespace) -> None:
    """Branch staging operations (merge approved candidate repairs)."""
    if args.branch_subcommand == "merge":
        from tools.annotation.merge_branch_to_annotated import main as merge_main
        m_args = []
        if args.branch_dir:
            m_args.extend(["--branch-dir", str(args.branch_dir)])
        if args.target_dir:
            m_args.extend(["--target-dir", str(args.target_dir)])
        if args.report_path:
            m_args.extend(["--report-path", str(args.report_path)])
        if args.pattern:
            m_args.extend(["--pattern", str(args.pattern)])
        if args.min_score is not None:
            m_args.extend(["--min-score", str(args.min_score)])
        if args.dry_run:
            m_args.append("--dry-run")

        old_argv = sys.argv
        try:
            sys.argv = ["merge_branch_to_annotated.py"] + m_args
            merge_main()
        finally:
            sys.argv = old_argv


def _cmd_dataset(args: argparse.Namespace) -> None:
    """Dataset compilation & packaging."""
    if args.dataset_subcommand == "build":
        from tools.dataset.build_training_dataset import main as build_main
        b_args = []
        if args.gold_file:
            b_args.extend(["--gold-file", str(args.gold_file)])
        if args.output_dir:
            b_args.extend(["--output-dir", str(args.output_dir)])
        if args.train_ratio is not None:
            b_args.extend(["--train-ratio", str(args.train_ratio)])
        if args.val_ratio is not None:
            b_args.extend(["--val-ratio", str(args.val_ratio)])
        if args.test_ratio is not None:
            b_args.extend(["--test-ratio", str(args.test_ratio)])
        if args.max_docs:
            b_args.extend(["--max-docs", str(args.max_docs)])
        if args.synthetic_ratio is not None:
            b_args.extend(["--synthetic-ratio", str(args.synthetic_ratio)])
        if args.no_curriculum:
            b_args.append("--no-curriculum")

        old_argv = sys.argv
        try:
            sys.argv = ["build_training_dataset.py"] + b_args
            build_main()
        finally:
            sys.argv = old_argv


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Unified evaluation suite (Gold Benchmark, 500-corpus, or token & DET F1)."""
    if args.eval_target == "gold":
        from tools.evaluation.evaluate_model import main as eval_main
        e_args = []
        if args.gold_manifest:
            e_args.extend(["--gold-manifest", str(args.gold_manifest)])
        if args.checkpoint:
            e_args.extend(["--checkpoint", str(args.checkpoint)])
        old_argv = sys.argv
        try:
            sys.argv = ["evaluate_model.py"] + e_args
            eval_main()
        finally:
            sys.argv = old_argv
    elif args.eval_target == "det-f1":
        from tools.evaluation.eval_tokens_and_det_f1 import main as det_f1_main
        d_args = []
        if args.dataset_dir:
            d_args.extend(["--dataset-dir", str(args.dataset_dir)])
        old_argv = sys.argv
        try:
            sys.argv = ["eval_tokens_and_det_f1.py"] + d_args
            det_f1_main()
        finally:
            sys.argv = old_argv
    elif args.eval_target == "corpus-500":
        from tools.evaluation.evaluate_500_dataset import main as eval_500_main
        old_argv = sys.argv
        try:
            sys.argv = ["evaluate_500_dataset.py"]
            eval_500_main()
        finally:
            sys.argv = old_argv


def _cmd_status(args: argparse.Namespace) -> None:
    """Unified terminal status dashboard across OCR, Annotation, and Repair runs."""
    print("=" * 70)
    print("📊 VSL PIPELINE STATUS DASHBOARD")
    print("=" * 70)

    # 1. OCR Batch Status
    ocr_state_file = WORKSPACE_DIR / "data" / "ocr_batch_500_state.json"
    print("\n🔍 [1. OCR Pipeline]")
    if ocr_state_file.exists():
        try:
            st = json.loads(ocr_state_file.read_text(encoding="utf-8"))
            tot = st.get("total_target_pdfs", 0)
            done = st.get("completed_count", 0)
            fail = st.get("failed_count", 0)
            pages = st.get("total_pages_processed", 0)
            cost = st.get("cumulative_cost_usd", 0.0)
            pct = (done / max(1, tot)) * 100
            print(f"  Target PDFs   : {tot}")
            print(f"  Completed     : {done} ({pct:.1f}%) | Failed: {fail}")
            print(f"  Total Pages   : {pages}")
            print(f"  Est. Cost     : ${cost:.2f} USD")
            print(f"  Last Updated  : {st.get('last_updated', 'N/A')}")
        except Exception as e:
            print(f"  Error reading OCR state: {e}")
    else:
        raw_count = (
            len(list((WORKSPACE_DIR / "data" / "sequence_labelling_input_data_raw").glob("*.pdf")))
            if (WORKSPACE_DIR / "data" / "sequence_labelling_input_data_raw").exists()
            else 0
        )
        md_count = (
            len(list((WORKSPACE_DIR / "data" / "sequence_labelling_input_data").glob("*.md")))
            if (WORKSPACE_DIR / "data" / "sequence_labelling_input_data").exists()
            else 0
        )
        print(f"  Raw PDFs: {raw_count} | Transcribed MDs: {md_count}")

    # 2. Annotation / Parsing Status
    ann_state_file = WORKSPACE_DIR / "data" / "annotation_batch_500_state.json"
    print("\n🏷️  [2. Sequence Labelling Annotation Pipeline]")
    if ann_state_file.exists():
        try:
            st = json.loads(ann_state_file.read_text(encoding="utf-8"))
            tot = st.get("total_target_docs", 0)
            done = st.get("completed_count", 0)
            fail = st.get("failed_count", 0)
            pct = (done / max(1, tot)) * 100
            print(f"  Target Docs   : {tot}")
            print(f"  Completed     : {done} ({pct:.1f}%) | Failed: {fail}")
            print(f"  Last Updated  : {st.get('last_updated', 'N/A')}")
        except Exception as e:
            print(f"  Error reading Annotation state: {e}")
    else:
        annot_count = (
            len(list((WORKSPACE_DIR / "data" / "sequence_labelling_annotated").glob("*/merged.xml")))
            if (WORKSPACE_DIR / "data" / "sequence_labelling_annotated").exists()
            else 0
        )
        print(f"  Canonical Merged XMLs: {annot_count}")

    # 3. Quality Review Status
    rev_report_file = WORKSPACE_DIR / "data" / "review_report.json"
    chunks_report_file = WORKSPACE_DIR / "data" / "review_report_chunks.json"
    target_rev = chunks_report_file if chunks_report_file.exists() else rev_report_file
    print("\n⚖️  [3. Quality Review & Audit Status]")
    if target_rev.exists():
        try:
            rev_data = json.loads(target_rev.read_text(encoding="utf-8"))
            p_cnt = rev_data.get("passed_count", 0)
            n_cnt = rev_data.get("needs_revision_count", 0)
            d_cnt = rev_data.get("discarded_count", 0)
            avg = rev_data.get("average_score", 0.0)
            tot_rev = p_cnt + n_cnt + d_cnt
            print(f"  Source Report : {target_rev.name}")
            print(f"  Total Audited : {tot_rev}")
            print(f"  PASS          : {p_cnt} ({p_cnt/max(1, tot_rev)*100:.1f}%)")
            print(f"  NEEDS_REVISION: {n_cnt} ({n_cnt/max(1, tot_rev)*100:.1f}%)")
            print(f"  DISCARD       : {d_cnt} ({d_cnt/max(1, tot_rev)*100:.1f}%)")
            print(f"  Average Score : {avg:.1f} / 100")
        except Exception as e:
            print(f"  Error reading Review report: {e}")
    else:
        print("  No review_report.json found.")

    # 4. Branch Staging Status
    branch_dir = WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch"
    print("\n🌿 [4. Isolated Branch Staging]")
    if branch_dir.exists():
        branch_docs = list(branch_dir.glob("doc_*")) + list(branch_dir.glob("exam_*"))
        branch_chunks = list(branch_dir.glob("*/chunks/chunk_*.xml"))
        print(f"  Candidate Exam Dirs : {len(branch_docs)}")
        print(f"  Candidate Chunk XMLs: {len(branch_chunks)}")
    else:
        print("  Branch directory empty.")

    # 5. Active TMUX Background Sessions
    res_tmux = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    print("\n🖥️  [5. Active TMUX Background Sessions]")
    if res_tmux.returncode == 0 and res_tmux.stdout.strip():
        for line in res_tmux.stdout.strip().splitlines():
            s_name = line.split(":")[0]
            log_hint = WORKSPACE_DIR / "artifacts" / "ocr_logs" / f"{s_name}.log"
            hint_str = f" (log: {log_hint.name})" if log_hint.exists() else ""
            print(f"  • {line}{hint_str}")
    else:
        print("  No active background tmux sessions running.")

    print("\n" + "=" * 70)


def _cmd_tmux(args: argparse.Namespace) -> None:
    """Manage background tmux sessions for VSL jobs."""
    action = args.tmux_action or "list"
    if action == "list":
        res = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
        if res.returncode != 0 or not res.stdout.strip():
            print("ℹ️  No active TMUX sessions found.")
        else:
            print("🖥️  Active TMUX Sessions:")
            print(res.stdout.strip())
        log_dir = WORKSPACE_DIR / "artifacts" / "ocr_logs"
        if log_dir.exists():
            recent_logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
            if recent_logs:
                print("\n📄 Recent Session Logs:")
                for l in recent_logs:
                    print(f"  • {l.name} ({l.stat().st_size / 1024:.1f} KB) -> tail -f artifacts/ocr_logs/{l.name}")
    elif action == "attach":
        session = args.session or "vsl-ocr"
        print(f"Attaching to '{session}' (Press Ctrl+B D to detach)...")
        os.execvp("tmux", ["tmux", "attach", "-t", session])
    elif action == "kill":
        session = args.session
        if not session:
            print("❌ Error: Specify session name to kill (e.g. vsl tmux kill vsl-ocr)")
            return
        res = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✅ Killed tmux session '{session}'")
        else:
            print(f"❌ Could not kill session '{session}': {res.stderr.strip()}")
    elif action == "logs":
        session = args.session or "vsl-ocr"
        log_path = WORKSPACE_DIR / "artifacts" / "ocr_logs" / f"{session}.log"
        if log_path.exists():
            os.execvp("tail", ["tail", "-f", str(log_path)])
        else:
            print(f"❌ Log file not found: {log_path}")


def _cmd_monitor(args: argparse.Namespace) -> None:
    """Terminal token & budget observability dashboard."""
    from tools.monitoring.token_monitor import main as monitor_main
    mon_args = []
    if args.continuous:
        mon_args.append("--continuous")
    if args.interval:
        mon_args.extend(["--interval", str(args.interval)])
    old_argv = sys.argv
    try:
        sys.argv = ["token_monitor.py"] + mon_args
        monitor_main()
    finally:
        sys.argv = old_argv


def _add_tmux_args(p: argparse.ArgumentParser) -> None:
    """Attach background TMUX execution flags to an operational parser."""
    p.add_argument(
        "--tmux",
        nargs="?",
        const=True,
        default=None,
        metavar="SESSION",
        help="Run command inside a detached background TMUX session (e.g. --tmux or --tmux my-run)",
    )
    p.add_argument(
        "--force-tmux",
        action="store_true",
        default=False,
        help="Force terminate any existing TMUX session with the same name before launching",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsl",
        description="VSL-v2 Unified Sequence Labelling Engine & CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Operational Subcommands")

    # paths
    subparsers.add_parser("paths", help="Print configured dataset and model paths")

    # ocr
    p_ocr = subparsers.add_parser("ocr", help="Run autonomous Vision OCR batch on raw PDFs")
    p_ocr.add_argument("--raw-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data_raw")
    p_ocr.add_argument("--out-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    p_ocr.add_argument("--concurrency", type=int, default=2)
    p_ocr.add_argument("--limit", type=int, default=None)
    p_ocr.add_argument("--overwrite", action="store_true", default=False)
    p_ocr.add_argument("--sleep-on-cap", action="store_true", default=True)
    _add_tmux_args(p_ocr)

    # cartography
    p_carto = subparsers.add_parser("cartography", help="Generate DocumentCartographer manifests")
    p_carto.add_argument("--input-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    p_carto.add_argument("--out-dir", type=Path, default=WORKSPACE_DIR / "data" / "document_cartography")
    p_carto.add_argument("--model", type=str, default="gpt-6-luna")
    p_carto.add_argument("--provider", type=str, default="aixingialaire")
    p_carto.add_argument("--limit", type=int, default=None)
    p_carto.add_argument("--overwrite", action="store_true", default=False)
    _add_tmux_args(p_carto)

    # metadata
    p_meta = subparsers.add_parser("metadata", help="Extract exam metadata catalog (grade, subject, year)")
    p_meta.add_argument("--input-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    p_meta.add_argument("--out-file", type=Path, default=WORKSPACE_DIR / "data" / "dataset_metadata_catalog.json")
    p_meta.add_argument("--model", type=str, default="gpt-6-luna")
    p_meta.add_argument("--provider", type=str, default="aixingialaire")
    p_meta.add_argument("--limit", type=int, default=None)
    _add_tmux_args(p_meta)

    # parse
    p_parse = subparsers.add_parser("parse", help="Run sequence labelling parser pipeline")
    p_parse.add_argument("--raw-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    p_parse.add_argument("--annotated-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated")
    p_parse.add_argument("--model", type=str, default=PARSER_MODEL)
    p_parse.add_argument("--provider", type=str, default=PARSER_PROVIDER)
    p_parse.add_argument("--concurrency", type=int, default=2)
    p_parse.add_argument("--limit", type=int, default=None)
    p_parse.add_argument("--overwrite", action="store_true", default=False)
    p_parse.add_argument("--sleep-on-cap", action="store_true", default=True)
    _add_tmux_args(p_parse)

    # audit
    p_audit = subparsers.add_parser("audit", help="Run deterministic and LLM semantic quality audit")
    p_audit.add_argument("--annotated-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated")
    p_audit.add_argument("--raw-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_input_data")
    p_audit.add_argument("--model", type=str, default=REVIEWER_MODEL)
    p_audit.add_argument("--provider", type=str, default=REVIEWER_PROVIDER)
    p_audit.add_argument("--thinking", type=str, default=REVIEWER_THINKING)
    p_audit.add_argument("--limit", type=int, default=None)
    p_audit.add_argument("--out-report", type=Path, default=WORKSPACE_DIR / "data" / "review_report.json")
    p_audit.add_argument("--overwrite", action="store_true", default=False)
    p_audit.add_argument("--no-llm", action="store_true", default=False)
    p_audit.add_argument("--granularity", type=str, default="chunk", choices=["chunk", "document"])
    _add_tmux_args(p_audit)

    # repair
    p_repair = subparsers.add_parser("repair", help="Run surgical EditorAgent repair loop on failing documents")
    p_repair.add_argument("--report", type=str, default="data/review_report.json")
    p_repair.add_argument("--raw-dir", type=str, default="data/sequence_labelling_input_data")
    p_repair.add_argument("--annotated-dir", type=str, default="data/sequence_labelling_annotated")
    p_repair.add_argument("--branch-dir", type=str, default="data/sequence_labelling_annotated_branch")
    p_repair.add_argument("--filter", dest="filter_decision", type=str, default="needs_revision")
    p_repair.add_argument("--model", type=str, default=EDITOR_MODEL)
    p_repair.add_argument("--provider", type=str, default=EDITOR_PROVIDER)
    p_repair.add_argument("--reviewer-model", type=str, default=REVIEWER_MODEL)
    p_repair.add_argument("--reviewer-provider", type=str, default=REVIEWER_PROVIDER)
    p_repair.add_argument("--editor-thinking", type=str, default=EDITOR_THINKING)
    p_repair.add_argument("--reviewer-thinking", type=str, default=REVIEWER_THINKING)
    p_repair.add_argument("--concurrency", "-c", type=int, default=4)
    p_repair.add_argument("--limit", "-l", type=int, default=None)
    p_repair.add_argument("--max-passes", type=int, default=2)
    p_repair.add_argument("--granularity", type=str, default="chunk", choices=["chunk", "document"])
    p_repair.add_argument("--doc-id", dest="target_doc", type=str, default=None)
    p_repair.add_argument("--auto-save", action="store_true", default=True)
    p_repair.add_argument("--dry-run", action="store_true", default=False)
    p_repair.add_argument("--out-report", type=str, default="artifacts/reports/revision_resolution_report.json")
    p_repair.add_argument("--reuse-prerun-logs", action="store_true", default=True)
    p_repair.add_argument("--resume-branch", action="store_true", default=True)
    p_repair.add_argument("--retry-unresolved", action="store_true", default=False)
    p_repair.add_argument("--auto-reconcile", action="store_true", default=True)
    p_repair.add_argument("--skip-unrecoverable", action="store_true", default=True)
    p_repair.add_argument("--max-doc-size", type=int, default=300000)
    p_repair.add_argument("--force-large", action="store_true", default=False)
    _add_tmux_args(p_repair)

    # branch
    p_branch = subparsers.add_parser("branch", help="Isolated branch staging operations")
    branch_subs = p_branch.add_subparsers(dest="branch_subcommand", help="Branch Subcommands")
    p_bmerge = branch_subs.add_parser("merge", help="Merge approved candidate repairs from branch to canonical annotated dir")
    p_bmerge.add_argument("--branch-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch")
    p_bmerge.add_argument("--target-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated")
    p_bmerge.add_argument("--report-path", type=Path, default=WORKSPACE_DIR / "data" / "review_report.json")
    p_bmerge.add_argument("--pattern", type=str, default=None)
    p_bmerge.add_argument("--min-score", type=float, default=60.0)
    p_bmerge.add_argument("--dry-run", action="store_true", default=False)
    _add_tmux_args(p_bmerge)

    # dataset
    p_data = subparsers.add_parser("dataset", help="Dataset compilation and packaging")
    data_subs = p_data.add_subparsers(dest="dataset_subcommand", help="Dataset Subcommands")
    p_dbuild = data_subs.add_parser("build", help="Build token classification, span, and generative dataset splits")
    p_dbuild.add_argument("--gold-file", type=Path, default=WORKSPACE_DIR / "data" / "gold_benchmark_63_documents.json")
    p_dbuild.add_argument("--output-dir", type=Path, default=WORKSPACE_DIR / "data" / "dataset_training_splits")
    p_dbuild.add_argument("--train-ratio", type=float, default=0.8)
    p_dbuild.add_argument("--val-ratio", type=float, default=0.1)
    p_dbuild.add_argument("--test-ratio", type=float, default=0.1)
    p_dbuild.add_argument("--max-docs", type=int, default=None)
    p_dbuild.add_argument("--synthetic-ratio", type=float, default=0.2)
    p_dbuild.add_argument("--no-curriculum", action="store_true", default=False)
    _add_tmux_args(p_dbuild)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluation and benchmarking suites")
    p_eval.add_argument("eval_target", choices=["gold", "det-f1", "corpus-500"], help="Evaluation target")
    p_eval.add_argument("--gold-manifest", type=Path, default=WORKSPACE_DIR / "data" / "gold_benchmark_63_documents.json")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--dataset-dir", type=Path, default=WORKSPACE_DIR / "data" / "sequence_labelling_annotated")
    _add_tmux_args(p_eval)

    # tmux
    p_tmux = subparsers.add_parser("tmux", help="Inspect and control background TMUX sessions")
    p_tmux.add_argument("tmux_action", choices=["list", "attach", "kill", "logs"], nargs="?", default="list", help="TMUX action")
    p_tmux.add_argument("session", nargs="?", default=None, help="Session name")

    # status
    subparsers.add_parser("status", help="Inspect live status of OCR, annotation, repair batches, and TMUX sessions")

    # monitor
    p_mon = subparsers.add_parser("monitor", help="Terminal token and budget monitor")
    p_mon.add_argument("--continuous", action="store_true", default=False)
    p_mon.add_argument("--interval", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    # Intercept --tmux: if requested, launch in detached background tmux session
    if getattr(args, "tmux", None):
        session_name = (
            args.tmux
            if isinstance(args.tmux, str) and args.tmux is not True
            else f"vsl-{args.subcommand}"
        )
        _launch_in_tmux(
            session_name=session_name,
            subcommand=args.subcommand,
            force=getattr(args, "force_tmux", False),
        )
        return

    handlers = {
        "paths": _cmd_paths,
        "ocr": _cmd_ocr,
        "cartography": _cmd_cartography,
        "metadata": _cmd_metadata,
        "parse": _cmd_parse,
        "audit": _cmd_audit,
        "repair": _cmd_repair,
        "branch": _cmd_branch,
        "dataset": _cmd_dataset,
        "evaluate": _cmd_evaluate,
        "status": _cmd_status,
        "tmux": _cmd_tmux,
        "monitor": _cmd_monitor,
    }

    handler = handlers.get(args.subcommand)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
