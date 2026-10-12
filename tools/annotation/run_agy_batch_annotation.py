#!/usr/bin/env python3
"""
run_agy_batch_annotation.py — Autonomous Multi-Worker Batch Sequence Labelling Orchestrator.

Features:
- Processes raw OCR markdown documents using the Long-Context Parser Pipeline (Greedy Chunker + AnchoredXMLLLMExamParser + Sequence Reconciler).
- Leverages the Antigravity (agy) subscription CLI using Gemini 3.8 Flash High.
- Auto-handles 5-hour rolling cap limits (--sleep-on-cap):
  Catches Agy5HourCapError, saves atomic checkpoint, enters countdown sleep, verifies health via AgyQuotaProbe, and auto-resumes.
- Atomic state tracking (data/annotation_batch_500_state.json) for 100% resilient resumption.
- Pre-flight quota probe before execution.
- Graceful shutdown on SIGINT / SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from sequence_labelling.config import (
    PARSER_MODEL,
    PARSER_PROVIDER,
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_OVERLAP_PAGES,
)
from sequence_labelling.llm.agy_rate_limiter import (
    AGY_INPUT_RATE_PER_M,
    AGY_OUTPUT_RATE_PER_M,
    Agy5HourCapError,
    AgyQuotaProbe,
    RollingUsageMonitor,
)
from tools.annotation.annotate_sequence_labelling_dataset import process_single_document

DEFAULT_STATE_FILE = Path("data/annotation_batch_500_state.json")
DEFAULT_LOG_FILE = Path("artifacts/agy_batch_annotation.log")
Path("artifacts").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("run_agy_batch_annotation")


def setup_logger(log_file: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class AnnotationBatchStateLedger:
    """
    Thread-safe atomic progress ledger for batch sequence labelling annotation.
    """

    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.data: Dict[str, Any] = {
            "total_target_docs": 0,
            "completed_count": 0,
            "failed_count": 0,
            "total_questions_extracted": 0,
            "total_chunks_processed": 0,
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            "completed_docs": {},
            "failed_docs": {},
        }
        self.load()

    def load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file {self.state_file}: {e}")

    def save(self) -> None:
        self.data["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
        temp_file = self.state_file.with_suffix(".tmp")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        temp_file.replace(self.state_file)

    def is_completed(self, filename: str) -> bool:
        return filename in self.data.get("completed_docs", {})

    def record_success(
        self,
        filename: str,
        questions_count: int,
        chunks_count: int,
        duration_sec: float,
        out_dir: str,
    ) -> None:
        self.data.setdefault("completed_docs", {})[filename] = {
            "questions": questions_count,
            "chunks": chunks_count,
            "duration_sec": duration_sec,
            "output_dir": out_dir,
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.data["completed_count"] = len(self.data["completed_docs"])
        self.data["total_questions_extracted"] += questions_count
        self.data["total_chunks_processed"] += chunks_count
        if filename in self.data.get("failed_docs", {}):
            del self.data["failed_docs"][filename]
        self.data["failed_count"] = len(self.data.get("failed_docs", {}))
        self.save()

    def record_failure(self, filename: str, error: str) -> None:
        self.data.setdefault("failed_docs", {})[filename] = {
            "error": error,
            "failed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.data["failed_count"] = len(self.data["failed_docs"])
        self.save()


class AgyBatchAnnotationRunner:
    """
    Autonomous orchestrator for batch sequence labelling annotation.
    """

    def __init__(
        self,
        input_dir: str = "data/batch_500_diverse_raw",
        output_dir: str = "data/sequence_labelling_annotated",
        model: str = PARSER_MODEL,
        provider: str = PARSER_PROVIDER,
        concurrency: int = 2,
        sleep_on_cap: bool = True,
        max_docs: Optional[int] = None,
        pattern: str = "*.md",
        overwrite: bool = False,
        state_file: Path = DEFAULT_STATE_FILE,
        target_tokens: int = CHUNKER_TARGET_TOKENS,
        max_tokens: int = CHUNKER_MAX_TOKENS,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.provider = provider
        self.concurrency = max(1, concurrency)
        self.sleep_on_cap = sleep_on_cap
        self.max_docs = max_docs
        self.pattern = pattern
        self.overwrite = overwrite
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.ledger = AnnotationBatchStateLedger(state_file=state_file)
        self._shutdown_requested = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} received. Requesting clean shutdown...")
        self._shutdown_requested = True

    def _wait_for_cap_reset(self, cap_error: Agy5HourCapError) -> bool:
        """
        Puts runner to sleep during a 5-hour cap limit, displaying countdown
        and probing agy health before auto-resuming.
        """
        logger.warning("=" * 70)
        logger.warning(f"⚠️  [5-HOUR CAP LIMIT DETECTED]")
        logger.warning(f"Cooldown required: {cap_error.formatted_cooldown()}")
        logger.warning(f"Estimated resumption: {cap_error.formatted_resumes_at()}")
        logger.warning(f"Atomic progress checkpoint saved to {self.ledger.state_file}")
        logger.warning("=" * 70)

        sleep_seconds = cap_error.reset_after_sec
        interval = 30

        while sleep_seconds > 0 and not self._shutdown_requested:
            hrs = sleep_seconds // 3600
            mins = (sleep_seconds % 3600) // 60
            secs = sleep_seconds % 60
            logger.info(f"⏳ Sleeping on cap limit... {hrs:02d}h {mins:02d}m {secs:02d}s remaining")
            time.sleep(min(interval, sleep_seconds))
            sleep_seconds -= interval

        if self._shutdown_requested:
            return False

        logger.info("Verifying active quota with AgyQuotaProbe...")
        for attempt in range(5):
            healthy, cooldown, msg = AgyQuotaProbe.probe_health(model=self.model)
            if healthy:
                logger.info("✅ Quota probe verified! Auto-resuming batch annotation execution.")
                return True
            logger.warning(f"Quota probe still pending ({msg}). Waiting 15 minutes before re-check...")
            time.sleep(900)

        return False

    def run(self) -> None:
        logger.info(f"Initializing Batch Sequence Labelling Runner across {self.input_dir}...")
        all_md_files = sorted(list(self.input_dir.glob(self.pattern)))
        self.ledger.data["total_target_docs"] = len(all_md_files)
        self.ledger.save()

        # Identify documents to process
        if not self.overwrite:
            pending_files = [
                f for f in all_md_files
                if not self.ledger.is_completed(f.name)
                and not (self.output_dir / f.stem / "merged.json").exists()
            ]
        else:
            pending_files = all_md_files

        if self.max_docs and self.max_docs > 0:
            pending_files = pending_files[: self.max_docs]

        logger.info(
            f"Total Target Docs: {len(all_md_files)} | "
            f"Already Done: {self.ledger.data['completed_count']} | "
            f"Pending: {len(pending_files)}"
        )

        if not pending_files:
            logger.info("All target documents are already annotated! Nothing to run.")
            return

        # Pre-flight health check if using agy provider
        if self.provider in ["agy", "antigravity"]:
            logger.info("Running pre-flight agy quota health check...")
            healthy, cooldown, msg = AgyQuotaProbe.probe_health(model=self.model)
            if not healthy:
                if cooldown and self.sleep_on_cap:
                    logger.warning(f"Active cap limit detected before run: {msg}")
                    resumed = self._wait_for_cap_reset(
                        Agy5HourCapError(f"Cap active: {msg}", reset_after_sec=cooldown)
                    )
                    if not resumed:
                        logger.info("Shutdown requested during initial cooldown.")
                        return
                else:
                    logger.error(f"Pre-flight agy check failed: {msg}")
                    sys.exit(1)
            else:
                logger.info("✅ Pre-flight quota check passed! Agy provider is healthy and responsive.")

        total_pending = len(pending_files)
        logger.info(f"🚀 Starting batch annotation for {total_pending} documents...")

        for idx, doc_path in enumerate(pending_files, start=1):
            if self._shutdown_requested:
                logger.info("Shutdown requested. Breaking loop safely...")
                break

            doc_name = doc_path.name
            rel_path = doc_path.relative_to(self.input_dir)
            logger.info(f"[{idx}/{total_pending}] Processing '{doc_name}'...")

            doc_processed = False
            while not doc_processed and not self._shutdown_requested:
                try:
                    res = process_single_document(
                        file_path=doc_path,
                        rel_path=rel_path,
                        out_dir=self.output_dir,
                        parser_model=self.model,
                        parser_provider=self.provider,
                        target_tokens=self.target_tokens,
                        max_tokens=self.max_tokens,
                        concurrency=self.concurrency,
                    )
                    self.ledger.record_success(
                        filename=doc_name,
                        questions_count=res.get("questions", 0),
                        chunks_count=res.get("chunks", 1),
                        duration_sec=res.get("duration", 0.0),
                        out_dir=res.get("output_dir", ""),
                    )
                    logger.info(
                        f"✅ Done '{doc_name}': {res.get('questions', 0)} questions, "
                        f"{res.get('chunks', 1)} chunks in {res.get('duration', 0.0):.1f}s | "
                        f"Overall Completed: {self.ledger.data['completed_count']}/{len(all_md_files)}"
                    )
                    doc_processed = True

                except Agy5HourCapError as cap_err:
                    if self.sleep_on_cap:
                        resumed = self._wait_for_cap_reset(cap_err)
                        if not resumed:
                            logger.info("Shutdown requested during cap sleep.")
                            break
                        # After resume, retry this same document
                        logger.info(f"Retrying '{doc_name}' after cap limit reset...")
                    else:
                        logger.error(f"Cap limit reached and sleep_on_cap disabled. Aborting.")
                        self.ledger.record_failure(doc_name, str(cap_err))
                        self._shutdown_requested = True
                        break

                except Exception as exc:
                    logger.error(f"❌ Error processing '{doc_name}': {exc}")
                    self.ledger.record_failure(doc_name, str(exc))
                    doc_processed = True  # Move to next doc on normal processing error

        logger.info("=" * 60)
        logger.info("Batch Annotation Run Summary:")
        logger.info(f"  - Completed: {self.ledger.data['completed_count']} documents")
        logger.info(f"  - Failed: {self.ledger.data['failed_count']} documents")
        logger.info(f"  - Total Questions: {self.ledger.data['total_questions_extracted']:,}")
        logger.info(f"  - Total Chunks: {self.ledger.data['total_chunks_processed']:,}")
        logger.info(f"  - State File: {self.ledger.state_file}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Worker Batch Sequence Labelling Orchestrator for 500 documents."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/batch_500_diverse_raw",
        help="Directory containing raw OCR markdown files (default: data/batch_500_diverse_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sequence_labelling_annotated",
        help="Directory for output annotated XML/JSON (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument("--model", type=str, default=PARSER_MODEL, help=f"Parser model (default: {PARSER_MODEL})")
    parser.add_argument(
        "--provider",
        type=str,
        default=PARSER_PROVIDER,
        choices=["agy", "antigravity", "codex", "xah", "deepseek", "nvidia", "vilao", "commandcode"],
        help=f"LLM Provider (default: {PARSER_PROVIDER})",
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Worker concurrency (default: 2)")
    parser.add_argument("--pattern", type=str, default="*.md", help="File pattern (default: *.md)")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already completed documents")
    parser.add_argument(
        "--state-file",
        type=str,
        default=str(DEFAULT_STATE_FILE),
        help=f"Path to progress state file (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help=f"Path to log file (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=2500,
        help="Target chunk size in tokens (default: 2500)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum chunk size in tokens (default: 4000)",
    )
    parser.add_argument(
        "--no-sleep-on-cap",
        action="store_true",
        help="Disable auto-sleep on 5-hour rolling cap limit",
    )

    args = parser.parse_args()
    setup_logger(Path(args.log_file))

    runner = AgyBatchAnnotationRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        provider=args.provider,
        concurrency=args.concurrency,
        sleep_on_cap=not args.no_sleep_on_cap,
        max_docs=args.max_docs,
        pattern=args.pattern,
        overwrite=args.overwrite,
        state_file=Path(args.state_file),
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )
    runner.run()



if __name__ == "__main__":
    main()
