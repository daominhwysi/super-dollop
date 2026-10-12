#!/usr/bin/env python3
"""
check_annotation_batch_status.py — Live Status Inspector for 500-Document Batch Annotation Run.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DEFAULT_STATE_FILE = Path("data/annotation_batch_500_state.json")
LOG_FILE = Path("artifacts/agy_batch_annotation_tmux.log")


def main():
    parser = argparse.ArgumentParser(description="Check batch sequence labelling annotation status.")
    parser.add_argument("--state-file", type=str, default=str(DEFAULT_STATE_FILE), help="Path to state file")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    console = Console()

    if not state_path.exists():
        console.print(f"[yellow]No batch annotation state file found at {state_path}[/yellow]")
        console.print("[dim]The batch run may not have initialized its state ledger yet.[/dim]")
        return

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading state file: {e}[/red]")
        return

    total_target = data.get("total_target_docs", 500)
    completed_count = data.get("completed_count", 0)
    failed_count = data.get("failed_count", 0)
    total_questions = data.get("total_questions_extracted", 0)
    total_chunks = data.get("total_chunks_processed", 0)
    last_updated = data.get("last_updated", "")

    pct = (completed_count / max(1, total_target)) * 100

    table = Table(
        title="✨ AGY 500 Document Batch Sequence Labelling Status",
        title_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Metric", style="bold white", width=28)
    table.add_column("Current Value", style="bold green", width=42)

    table.add_row("Total Target Documents", f"{total_target:,} documents")
    table.add_row("Completed Documents", f"{completed_count:,} ({pct:.1f}%)")
    table.add_row("Failed Documents", f"[red]{failed_count}[/red]" if failed_count else "0")
    table.add_row("Total Questions Extracted", f"{total_questions:,} questions")
    table.add_row("Total Chunks Processed", f"{total_chunks:,} chunks")
    table.add_row("Avg Questions / Document", f"{round(total_questions / max(1, completed_count), 1)}")
    table.add_row("Last Checkpoint Updated", last_updated)

    console.print(table)

    # Show recent completed documents
    completed_docs = data.get("completed_docs", {})
    if completed_docs:
        rec_table = Table(title="Recent Annotated Documents", border_style="dim")
        rec_table.add_column("Document Name", style="cyan", max_width=50)
        rec_table.add_column("Questions", justify="right")
        rec_table.add_column("Chunks", justify="right")
        rec_table.add_column("Duration", justify="right")
        rec_table.add_column("Completed At", justify="right", style="dim")

        for fname, meta in list(completed_docs.items())[-5:]:
            rec_table.add_row(
                fname,
                f"{meta.get('questions', 0):,}",
                str(meta.get("chunks", 1)),
                f"{meta.get('duration_sec', 0.0):.1f}s",
                meta.get("completed_at", "")[:19],
            )
        console.print(rec_table)

    # Useful tmux shortcuts
    console.print(
        Panel(
            "[bold cyan]Useful Monitoring Commands:[/bold cyan]\n"
            "• View live tmux session:  [green]tmux attach -t agy-annotate-500[/green] (detach: Ctrl+B then D)\n"
            "• Peek live log output:    [green]tail -f artifacts/agy_batch_annotation_tmux.log[/green]\n"
            "• Refresh this dashboard:  [green]uv run python tools/annotation/check_annotation_batch_status.py[/green]\n"
            "• Token Observability:     [green]uv run python tools/monitoring/token_monitor.py --project[/green]",
            title="Session Controls",
            border_style="magenta",
        )
    )


if __name__ == "__main__":
    main()
