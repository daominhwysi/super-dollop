"""
check_batch_status.py — Live Status Inspector for 500-PDF Batch OCR Run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

import argparse

DEFAULT_STATE_FILE = Path("data/ocr_batch_500_raw_state.json")
FALLBACK_STATE_FILE = Path("data/ocr_batch_500_state.json")
LOG_FILE = Path("artifacts/agy_batch_ocr_tmux.log")


def main():
    parser = argparse.ArgumentParser(description="Check batch OCR status.")
    parser.add_argument("--state-file", type=str, default=None, help="Path to state file")
    args = parser.parse_args()

    if args.state_file:
        state_path = Path(args.state_file)
    elif DEFAULT_STATE_FILE.exists():
        state_path = DEFAULT_STATE_FILE
    else:
        state_path = FALLBACK_STATE_FILE

    console = Console()

    if not state_path.exists():
        console.print(f"[yellow]No batch state file found at {state_path}[/yellow]")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_target = data.get("total_target_pdfs", 500)
    completed_count = data.get("completed_count", 0)
    failed_count = data.get("failed_count", 0)
    total_pages = data.get("total_pages_processed", 0)
    cost_usd = data.get("cumulative_cost_usd", 0.0)
    in_tokens = data.get("cumulative_input_tokens", 0)
    out_tokens = data.get("cumulative_output_tokens", 0)
    last_updated = data.get("last_updated", "")

    pct = (completed_count / max(1, total_target)) * 100

    table = Table(title="🚀 AGY 500 Diverse PDF Batch OCR Status", title_style="bold cyan", border_style="dim")
    table.add_column("Metric", style="bold white", width=25)
    table.add_column("Current Value", style="bold green", width=40)

    table.add_row("Total Target PDFs", f"{total_target:,} documents")
    table.add_row("Completed PDFs", f"{completed_count:,} ({pct:.1f}%)")
    table.add_row("Failed PDFs", f"[red]{failed_count}[/red]")
    table.add_row("Total Pages Converted", f"{total_pages:,} pages")
    table.add_row("Total Input Tokens", f"{in_tokens:,} tokens")
    table.add_row("Total Output Tokens", f"{out_tokens:,} tokens")
    table.add_row("Cumulative agy Spend", f"${cost_usd:.4f} / $195.00 limit")
    table.add_row("Remaining Weekly Quota", f"${max(0.0, 200.0 - cost_usd):.2f}")
    table.add_row("Last Checkpoint Updated", last_updated)

    console.print(table)

    # Show recent completed documents
    completed_docs = data.get("completed_docs", {})
    if completed_docs:
        rec_table = Table(title="Recent Processed Documents", border_style="dim")
        rec_table.add_column("Document Name", style="cyan", max_width=50)
        rec_table.add_column("Pages", justify="right")
        rec_table.add_column("Duration", justify="right")
        rec_table.add_column("Cost", justify="right", style="green")

        for fname, meta in list(completed_docs.items())[-5:]:
            rec_table.add_row(
                fname,
                str(meta.get("pages", 0)),
                f"{meta.get('duration_sec', 0.0):.1f}s",
                f"${meta.get('cost_usd', 0.0):.4f}",
            )
        console.print(rec_table)

    # Useful tmux shortcuts
    console.print(Panel(
        "[bold cyan]Useful Monitoring Commands:[/bold cyan]\n"
        "• View live tmux process: [green]tmux attach -t agy-ocr-500[/green] (detach with Ctrl+B then D)\n"
        "• Peek tmux output:       [green]tmux capture-pane -pt agy-ocr-500 | tail -n 25[/green]\n"
        "• Refresh this status:   [green]uv run python tools/ocr/check_batch_status.py[/green]\n"
        "• Token Observability:   [green]uv run python tools/monitoring/token_monitor.py --project[/green]",
        title="Session Controls",
        border_style="cyan"
    ))


if __name__ == "__main__":
    main()
