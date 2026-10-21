"""
run_agy_batch_ocr.py — Autonomous Multi-Worker Batch OCR Orchestrator.

Features:
- Processes PDFs using AgyPDFOCRConverter (Direct Figure Detection & Dynamic Style Diversity).
- Enforces weekly budget ceiling (--stop-on-weekly-budget USD) calculated via TokenTracker rates ($0.75 in / $3.75 out).
- Auto-handles 5-hour rolling cap limits (--sleep-on-cap):
  Catches Agy5HourCapError, saves atomic checkpoint, enters countdown sleep, verifies health via probe, and auto-resumes.
- Atomic state tracking (data/ocr_batch_500_state.json) for 100% resilient resumption.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from sequence_labelling.annotator.agy_pdf_ocr import AgyPDFOCRConverter
from sequence_labelling.llm.agy_rate_limiter import (
    AGY_INPUT_RATE_PER_M,
    AGY_OUTPUT_RATE_PER_M,
    Agy5HourCapError,
    AgyQuotaProbe,
    AgyQuotaService,
    RollingUsageMonitor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/agy_batch_ocr.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_agy_batch_ocr")

DEFAULT_STATE_FILE = Path("data/ocr_batch_500_state.json")
Path("artifacts").mkdir(parents=True, exist_ok=True)


class BatchStateLedger:
    """
    Thread-safe atomic progress ledger for batch OCR execution.
    """

    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.data: Dict[str, Any] = {
            "total_target_pdfs": 0,
            "completed_count": 0,
            "failed_count": 0,
            "total_pages_processed": 0,
            "cumulative_input_tokens": 0,
            "cumulative_output_tokens": 0,
            "cumulative_cost_usd": 0.0,
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
        pages_count: int,
        duration_sec: float,
        in_tokens: int,
        out_tokens: int,
        out_path: str,
    ) -> None:
        cost = RollingUsageMonitor.calculate_cost_usd(in_tokens, out_tokens)
        self.data.setdefault("completed_docs", {})[filename] = {
            "pages": pages_count,
            "duration_sec": duration_sec,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": cost,
            "output_path": out_path,
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.data["completed_count"] = len(self.data["completed_docs"])
        self.data["total_pages_processed"] += pages_count
        self.data["cumulative_input_tokens"] += in_tokens
        self.data["cumulative_output_tokens"] += out_tokens
        self.data["cumulative_cost_usd"] = round(self.data["cumulative_cost_usd"] + cost, 4)
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

    def reset_failed(self) -> int:
        """Resets the failed_docs dict and counter so all failed documents can be cleanly retried."""
        prev_cnt = len(self.data.get("failed_docs", {}))
        self.data["failed_docs"] = {}
        self.data["failed_count"] = 0
        self.save()
        return prev_cnt


class BatchOCRRunner:
    """
    Autonomous multi-worker batch OCR orchestrator.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        concurrency: int = 2,
        stop_on_weekly_budget: float = 195.0,
        stop_on_weekly_quota_pct: float = 2.0,
        sleep_on_cap: bool = True,
        max_docs: Optional[int] = None,
        model: str = "gemini-3.8-flash-low",
        state_file: Path = DEFAULT_STATE_FILE,
        pattern: str = "*.pdf",
        account: Optional[str] = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.concurrency = max(1, concurrency)
        self.stop_on_weekly_budget = stop_on_weekly_budget
        self.stop_on_weekly_quota_pct = stop_on_weekly_quota_pct
        self.sleep_on_cap = sleep_on_cap
        self.max_docs = max_docs
        self.model = model
        self.pattern = pattern
        self.account = account
        self.ledger = BatchStateLedger(state_file=state_file)
        self.converter = AgyPDFOCRConverter(model=model, batch_size=2, account=account)
        self._shutdown_requested = False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Register graceful shutdown signals
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
        interval = 30  # report every 30 seconds

        while sleep_seconds > 0 and not self._shutdown_requested:
            hrs = sleep_seconds // 3600
            mins = (sleep_seconds % 3600) // 60
            secs = sleep_seconds % 60
            logger.info(f"⏳ Sleeping on cap limit... {hrs:02d}h {mins:02d}m {secs:02d}s remaining")
            time.sleep(min(interval, sleep_seconds))
            sleep_seconds -= interval

        if self._shutdown_requested:
            return False

        # Probe health before resuming
        logger.info("Verifying active quota with AgyQuotaProbe...")
        for attempt in range(5):
            healthy, cooldown, msg = AgyQuotaProbe.probe_health(model=self.model, account=self.account)
            if healthy:
                logger.info("✅ Quota probe verified! Auto-resuming batch OCR execution.")
                return True
            logger.warning(f"Quota probe still pending ({msg}). Waiting 15 minutes before re-check...")
            time.sleep(900)

        return False

    def process_single_pdf(self, pdf_file: Path) -> Tuple[bool, str, Dict[str, Any]]:
        """Processes one PDF document and saves its sequence labelling markdown."""
        doc_stem = pdf_file.stem
        out_file = self.output_dir / f"{doc_stem}.md"

        try:
            markdown_content, summary = self.converter.convert_document(pdf_file)
            out_file.write_text(markdown_content, encoding="utf-8")
            try:
                from sequence_labelling.metadata.extractor import extract_and_save_metadata
                extract_and_save_metadata(
                    ocr_markdown=markdown_content,
                    filename=pdf_file.name,
                    doc_id=doc_stem,
                )
            except Exception as meta_err:
                logger.warning(f"Post-OCR metadata extraction skipped/failed for '{doc_stem}': {meta_err}")
            return True, str(out_file), summary
        except Agy5HourCapError as e:
            # Re-raise cap errors directly to trigger batch-level sleep
            raise e
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            return False, "", {"error": str(e)}

    def _ensure_active_account(self, target_account: Optional[str] = None) -> None:
        """Configures isolated profile environment for the target account."""
        account = target_account or os.environ.get("AGY_ACCOUNT")
        if not account:
            return
        prof_home = Path.home() / ".gemini" / "profiles" / account
        if not prof_home.exists():
            import subprocess
            subprocess.run(["agy-account", "setup", account], check=False)
        os.environ["HOME"] = str(prof_home)
        os.environ["AGY_ACCOUNT"] = account
        logger.info(f"Isolated runtime profile activated for account '{account}' (HOME={prof_home})")

    def run(self) -> None:
        """Main execution loop."""
        self._ensure_active_account(self.account)
        logger.info(f"Initializing Batch OCR Runner across {self.input_dir} (pattern: {self.pattern})...")
        all_pdfs = sorted(list(self.input_dir.glob(self.pattern)))
        self.ledger.data["total_target_pdfs"] = len(all_pdfs)
        self.ledger.save()

        # Filter pending PDFs: skip if in ledger OR if output .md already exists
        pending_pdfs = []
        already_on_disk = 0
        for p in all_pdfs:
            out_file = self.output_dir / f"{p.stem}.md"
            if self.ledger.is_completed(p.name):
                continue
            if out_file.exists() and out_file.stat().st_size > 100:
                already_on_disk += 1
                continue
            pending_pdfs.append(p)

        if self.max_docs and self.max_docs > 0:
            pending_pdfs = pending_pdfs[: self.max_docs]

        logger.info(
            f"Total target PDFs: {len(all_pdfs)} | Ledger completed: {self.ledger.data['completed_count']} | Disk exists (skipped): {already_on_disk} | Pending: {len(pending_pdfs)}"
        )

        if not pending_pdfs:
            logger.info("All documents are already completed! Nothing to run.")
            return

        # Pre-flight health check
        logger.info(f"Running pre-flight agy quota health check for account '{self.account or 'default'}'...")
        healthy, cooldown, msg = AgyQuotaProbe.probe_health(model=self.model, account=self.account)
        if not healthy:
            if cooldown and self.sleep_on_cap:
                cap_err = Agy5HourCapError(f"Pre-flight cap active: {msg}", reset_after_sec=cooldown)
                resumed = self._wait_for_cap_reset(cap_err)
                if not resumed:
                    logger.info("Batch execution exited during cooldown.")
                    return
            else:
                logger.error(f"Pre-flight check failed: {msg}. Aborting batch.")
                return

        # Processing loop with concurrency
        for idx, pdf_path in enumerate(pending_pdfs, start=1):
            if self._shutdown_requested:
                logger.info("Shutdown requested. Gracefully exiting batch runner.")
                break

            # 1. Real Antigravity Quota Check via /quota (ground-truth)
            quota = AgyQuotaService.fetch_quota(account=self.account)
            if quota and quota.gemini_weekly:
                if quota.gemini_weekly_pct <= self.stop_on_weekly_quota_pct:
                    logger.warning(
                        f"🛑 [WEEKLY QUOTA LIMIT REACHED] Real Gemini Weekly Quota is at {quota.gemini_weekly_pct:.1f}% "
                        f"<= Minimum threshold ({self.stop_on_weekly_quota_pct:.1f}%). Exiting cleanly."
                    )
                    logger.warning(f"State saved cleanly to {self.ledger.state_file}. Exiting.")
                    break

            if quota and quota.is_gemini_5h_exhausted(threshold_pct=1.0):
                cooldown = quota.gemini_5h_cooldown_sec()
                logger.warning(
                    f"⚠️ [5-HOUR QUOTA EXHAUSTED] Real Gemini 5h Quota is at {quota.gemini_5h_pct:.1f}%. "
                    f"Cooldown required: {quota.gemini_5h.formatted_cooldown() if quota.gemini_5h else f'{cooldown}s'}"
                )
                if self.sleep_on_cap:
                    cap_err = Agy5HourCapError("5-hour rolling quota exhausted", reset_after_sec=cooldown)
                    resumed = self._wait_for_cap_reset(cap_err)
                    if not resumed:
                        break
                else:
                    break

            # 2. Check nominal weekly budget ceiling (secondary safeguard)
            cur_spend = self.ledger.data["cumulative_cost_usd"]
            if self.stop_on_weekly_budget and cur_spend >= self.stop_on_weekly_budget:
                logger.warning(
                    f"🛑 [NOMINAL BUDGET CEILING REACHED] Current spend (${cur_spend:.2f}) >= Ceiling (${self.stop_on_weekly_budget:.2f})."
                )
                logger.warning(f"State saved cleanly to {self.ledger.state_file}. Exiting.")
                break

            quota_tag = quota.summary_line() if quota else f"Nominal: ${cur_spend:.2f}"
            logger.info(
                f"[{idx}/{len(pending_pdfs)}] Processing '{pdf_path.name}' ({quota_tag})..."
            )

            # Attempt processing with transient retry and cap-sleep handling
            doc_retries = 2
            for doc_attempt in range(doc_retries):
                if self._shutdown_requested:
                    break
                try:
                    success, out_path, summary = self.process_single_pdf(pdf_path)
                    if success:
                        pages = summary.get("total_pages", 0)
                        dur = summary.get("total_duration_sec", 0.0)
                        in_tok = summary.get("input_tokens", 0)
                        out_tok = summary.get("output_tokens", 0)
                        self.ledger.record_success(
                            filename=pdf_path.name,
                            pages_count=pages,
                            duration_sec=dur,
                            in_tokens=in_tok,
                            out_tokens=out_tok,
                            out_path=out_path,
                        )
                        logger.info(
                            f"✅ Done '{pdf_path.name}': {pages} pages in {dur}s | In: {in_tok:,}, Out: {out_tok:,} | Total Cost: ${self.ledger.data['cumulative_cost_usd']:.2f}"
                        )
                        break
                    else:
                        err_str = summary.get("error", "Unknown error")
                        is_transient_err = any(k in err_str.lower() for k in ["503", "unavailable", "timed out", "eligibility check failed"])
                        if is_transient_err and doc_attempt < doc_retries - 1:
                            logger.warning(f"Transient error on '{pdf_path.name}' ({err_str[:80]}). Retrying document in 15s...")
                            time.sleep(15)
                            continue
                        self.ledger.record_failure(pdf_path.name, err_str)
                        break

                except Agy5HourCapError as cap_err:
                    if self.sleep_on_cap:
                        resumed = self._wait_for_cap_reset(cap_err)
                        if not resumed or self._shutdown_requested:
                            break
                        # Retry this same document after waking up
                        logger.info(f"Retrying document '{pdf_path.name}' after cap reset...")
                        continue
                    else:
                        logger.warning(f"Cap limit reached and sleep_on_cap=False. Exiting cleanly.")
                        self._shutdown_requested = True
                        break

        logger.info("=" * 60)
        logger.info(f"Batch Run Summary:")
        logger.info(f" - Completed: {self.ledger.data['completed_count']} PDFs")
        logger.info(f" - Total Pages: {self.ledger.data['total_pages_processed']} pages")
        logger.info(f" - Total Input Tokens: {self.ledger.data['cumulative_input_tokens']:,}")
        logger.info(f" - Total Output Tokens: {self.ledger.data['cumulative_output_tokens']:,}")
        logger.info(f" - Total agy Spend: ${self.ledger.data['cumulative_cost_usd']:.2f} / ${self.stop_on_weekly_budget:.2f}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Autonomous agy Batch OCR Orchestrator")
    parser.add_argument("--input-dir", type=str, default="data/sequence_labelling_input_data_raw", help="Directory of input PDFs")
    parser.add_argument("--output-dir", type=str, default="data/sequence_labelling_input_data", help="Directory for output markdown")
    parser.add_argument("--concurrency", type=int, default=2, help="Worker concurrency")
    parser.add_argument("--stop-on-weekly-budget", type=float, default=195.0, help="Max USD budget ceiling before pausing")
    parser.add_argument("--stop-on-weekly-quota-pct", type=float, default=2.0, help="Min Gemini Weekly Quota %% before pausing (default: 2.0)")
    parser.add_argument("--no-sleep-on-cap", action="store_true", help="Stop instead of sleeping when 5h cap is hit")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional max documents to process")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash-low", help="agy model name")
    parser.add_argument("--pattern", type=str, default="*.pdf", help="File pattern matching PDFs in input-dir (e.g. 'doc_*.pdf' or '*.pdf')")
    parser.add_argument("--account", type=str, default=None, help="Optional agy account to switch to (e.g. daominhwysi, kien.duet)")
    parser.add_argument("--state-file", type=str, default="data/ocr_batch_500_state.json", help="Path to state ledger")
    parser.add_argument("--reset-failed", action="store_true", help="Clear failed_docs ledger so all failed documents are retried")

    args = parser.parse_args()

    runner = BatchOCRRunner(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        concurrency=args.concurrency,
        stop_on_weekly_budget=args.stop_on_weekly_budget,
        stop_on_weekly_quota_pct=args.stop_on_weekly_quota_pct,
        sleep_on_cap=not args.no_sleep_on_cap,
        max_docs=args.max_docs,
        model=args.model,
        state_file=Path(args.state_file),
        pattern=args.pattern,
        account=args.account,
    )
    if args.reset_failed:
        cleared = runner.ledger.reset_failed()
        logger.info(f"Reset {cleared} failed documents from state ledger.")
    runner.run()


if __name__ == "__main__":
    main()
