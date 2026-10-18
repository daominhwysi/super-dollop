#!/usr/bin/env python3
"""
token_monitor.py — Unified Antigravity & Pipeline Token Observability Dashboard.

Provides high-resolution token tracking, latency profiling, and cache efficiency
analytics across:
  1. Project Pipeline: Real-time telemetry recorded in artifacts/token_usage/
  2. Global Antigravity: Machine-wide historical and daily usage via safe_ccusage

Usage:
  uv run python tools/token_monitor.py              # Project pipeline today
  uv run python tools/token_monitor.py --global     # Workstation-wide Antigravity
  uv run python tools/token_monitor.py --all        # Both Project & Global
  uv run python tools/token_monitor.py --watch      # Live terminal monitor
  uv run python tools/token_monitor.py --json       # JSON output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sequence_labelling.llm.token_tracker import summarize

console = Console()


def format_number(val: Any) -> str:
    if val is None:
        return "0"
    if isinstance(val, (int, float)):
        return f"{int(val):,d}"
    return str(val)


def render_project_dashboard(date: Optional[str] = None) -> None:
    """Renders the Vietnamese Sequence Labelling pipeline token metrics."""
    s = summarize(date)
    date_str = s.get("date", "Today")
    total_calls = s.get("total_calls", 0)

    # 1. Title Panel
    title = Text.assemble(
        ("📊 Pipeline Token Observability — ", "bold cyan"),
        (f"Date: {date_str} ", "bold white"),
        (f"({total_calls} Calls)", "yellow"),
    )
    console.print(Panel(title, border_style="cyan", padding=(0, 1)))

    # Real-time Antigravity Quota (/quota)
    try:
        from sequence_labelling.llm.agy_rate_limiter import AgyQuotaService
        quota = AgyQuotaService.fetch_quota(timeout=10.0)
        if quota and (quota.gemini_5h or quota.gemini_weekly):
            q_table = Table(box=box.SIMPLE, show_header=True, expand=True)
            q_table.add_column("Gemini 5-Hour Limit", justify="center", style="bold green")
            q_table.add_column("Gemini Weekly Limit", justify="center", style="bold cyan")
            q_table.add_column("Claude/GPT 5-Hour", justify="center", style="bold yellow")
            q_table.add_column("Claude/GPT Weekly", justify="center", style="bold magenta")

            q_table.add_row(
                f"{quota.gemini_5h_pct:.1f}% left ({quota.gemini_5h.formatted_cooldown()})" if quota.gemini_5h else "N/A",
                f"{quota.gemini_weekly_pct:.1f}% left ({quota.gemini_weekly.formatted_reset()})" if quota.gemini_weekly else "N/A",
                f"{quota.claude_5h_pct:.1f}% left" if quota.claude_5h else "N/A",
                f"{quota.claude_weekly_pct:.1f}% left" if quota.claude_weekly else "N/A",
            )
            console.print(Panel(q_table, title="[bold]⚡ Official Antigravity Subscription Quotas (/quota)[/bold]", border_style="green", padding=(0, 1)))
    except Exception:
        pass

    if total_calls == 0:
        console.print("[dim italic]No pipeline token records recorded for this date.[/dim italic]\n")
        return

    # 2. KPI Summary Bar
    inp = s.get("input_tokens", 0)
    cache = s.get("cache_read_tokens", 0)
    hit_rate = s.get("cache_hit_pct", 0.0)
    out = s.get("output_tokens", 0)
    reasoning = s.get("reasoning_tokens", 0)
    tot = s.get("total_tokens", 0)
    dur = s.get("total_duration_sec", 0.0)
    avg_dur = round(dur / total_calls, 2) if total_calls > 0 else 0.0

    kpi_table = Table(box=box.SIMPLE, show_header=True, expand=True)
    kpi_table.add_column("Input Tokens", style="cyan", justify="center")
    kpi_table.add_column("Cache Read (Hit %)", style="green", justify="center")
    kpi_table.add_column("Reasoning / Thinking", style="magenta", justify="center")
    kpi_table.add_column("Output Tokens", style="blue", justify="center")
    kpi_table.add_column("Total Tokens", style="bold white", justify="center")
    kpi_table.add_column("Avg Latency", style="yellow", justify="center")

    kpi_table.add_row(
        format_number(inp),
        f"[bold green]{format_number(cache)}[/bold green] ([bold]{hit_rate}%[/bold])",
        format_number(reasoning),
        format_number(out),
        f"[bold white]{format_number(tot)}[/bold white]",
        f"{avg_dur:.2f}s",
    )
    console.print(kpi_table)

    # 3. Table by Provider & Model
    model_table = Table(
        title="[bold]Breakdown by Model & Provider[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        expand=True,
    )
    model_table.add_column("Provider / Model", style="white", ratio=4)
    model_table.add_column("Calls", justify="right", style="dim", ratio=1)
    model_table.add_column("Input", justify="right", style="cyan", ratio=2)
    model_table.add_column("Cache Read (Hit %)", justify="right", style="green", ratio=3)
    model_table.add_column("Thinking", justify="right", style="magenta", ratio=2)
    model_table.add_column("Output", justify="right", style="blue", ratio=2)
    model_table.add_column("Total Tokens", justify="right", style="bold white", ratio=2)
    model_table.add_column("Avg Sec", justify="right", style="yellow", ratio=2)

    for pm, stats in sorted(s.get("by_provider_model", {}).items(), key=lambda x: -x[1]["total_tokens"]):
        cache_str = f"{format_number(stats['cache_read_tokens'])} ({stats.get('cache_hit_pct', 0.0)}%)"
        model_table.add_row(
            pm,
            str(stats["calls"]),
            format_number(stats["input_tokens"]),
            cache_str,
            format_number(stats["reasoning_tokens"]),
            format_number(stats["output_tokens"]),
            format_number(stats["total_tokens"]),
            f"{stats.get('avg_duration_sec', 0.0):.2f}s",
        )
    console.print(model_table)

    # 4. Table by Pipeline Caller
    callers = s.get("by_caller", {})
    if callers and any(k != "unknown" for k in callers):
        caller_table = Table(
            title="[bold]Breakdown by Pipeline Caller / Component[/bold]",
            box=box.ROUNDED,
            header_style="bold green",
            expand=True,
        )
        caller_table.add_column("Caller / Stage", style="white", ratio=4)
        caller_table.add_column("Calls", justify="right", style="dim", ratio=1)
        caller_table.add_column("Input", justify="right", style="cyan", ratio=2)
        caller_table.add_column("Cache Read (Hit %)", justify="right", style="green", ratio=3)
        caller_table.add_column("Output", justify="right", style="blue", ratio=2)
        caller_table.add_column("Total Tokens", justify="right", style="bold white", ratio=2)
        caller_table.add_column("Avg Sec", justify="right", style="yellow", ratio=2)

        for caller, stats in sorted(callers.items(), key=lambda x: -x[1]["total_tokens"]):
            cache_str = f"{format_number(stats['cache_read_tokens'])} ({stats.get('cache_hit_pct', 0.0)}%)"
            caller_table.add_row(
                caller,
                str(stats["calls"]),
                format_number(stats["input_tokens"]),
                cache_str,
                format_number(stats["output_tokens"]),
                format_number(stats["total_tokens"]),
                f"{stats.get('avg_duration_sec', 0.0):.2f}s",
            )
        console.print(caller_table)

    console.print("")


def render_global_dashboard(limit_days: int = 10) -> None:
    """Renders the machine-wide Antigravity token metrics from safe_ccusage."""
    from tools.monitoring.safe_ccusage import get_safe_antigravity_summary

    try:
        data = get_safe_antigravity_summary()
    except Exception as exc:
        console.print(f"[bold red]Failed to query global Antigravity telemetry:[/bold red] {exc}")
        return

    daily_entries = data.get("daily", [])
    totals = data.get("totals", {})

    title = Text.assemble(
        ("🌐 Global Antigravity Machine Telemetry — ", "bold magenta"),
        ("Source: Antigravity CLI / IDE Databases", "dim white"),
    )
    console.print(Panel(title, border_style="magenta", padding=(0, 1)))

    if not daily_entries:
        console.print("[dim italic]No global Antigravity records found.[/dim italic]\n")
        return

    # Recent daily table
    table = Table(
        title="[bold]Recent Daily Consumption (All Agents & Sessions)[/bold]",
        box=box.ROUNDED,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Date", style="bold white", ratio=2)
    table.add_column("Models Active", style="dim", ratio=4)
    table.add_column("Input", justify="right", style="cyan", ratio=2)
    table.add_column("Cache Read", justify="right", style="green", ratio=3)
    table.add_column("Output", justify="right", style="blue", ratio=2)
    table.add_column("Total Tokens", justify="right", style="bold white", ratio=3)
    table.add_column("Est. Cost", justify="right", style="yellow", ratio=2)

    recent_entries = daily_entries[-limit_days:]
    for d in recent_entries:
        date_val = d.get("date", "")
        models = ", ".join(d.get("modelsUsed", []))
        if len(models) > 35:
            models = models[:32] + "..."
        inp = d.get("inputTokens", 0)
        cache = d.get("cacheReadTokens", 0)
        out = d.get("outputTokens", 0)
        tot = d.get("totalTokens", 0)
        cost = d.get("totalCost", 0.0)

        table.add_row(
            date_val,
            models,
            format_number(inp),
            format_number(cache),
            format_number(out),
            format_number(tot),
            f"${cost:.2f}",
        )

    # Totals row
    tot_inp = totals.get("inputTokens", 0)
    tot_cache = totals.get("cacheReadTokens", 0)
    tot_out = totals.get("outputTokens", 0)
    tot_all = totals.get("totalTokens", 0)
    tot_cost = totals.get("totalCost", 0.0)
    tot_hit = round(100.0 * tot_cache / tot_inp, 1) if tot_inp > 0 else 0.0

    table.add_section()
    table.add_row(
        "[bold white]LIFETIME TOTAL[/bold white]",
        f"[italic green]{tot_hit}% Cache Hit Rate[/italic green]",
        f"[bold cyan]{format_number(tot_inp)}[/bold cyan]",
        f"[bold green]{format_number(tot_cache)}[/bold green]",
        f"[bold blue]{format_number(tot_out)}[/bold blue]",
        f"[bold white]{format_number(tot_all)}[/bold white]",
        f"[bold yellow]${tot_cost:.2f}[/bold yellow]",
    )

    console.print(table)
    console.print("[dim]* Note: Est. Cost reflects commercial API rates. Antigravity CLI usage is included in your active subscription.[/dim]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Antigravity & Pipeline Token Observability Dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", action="store_true", help="Display project pipeline token telemetry")
    parser.add_argument("--global", dest="show_global", action="store_true", help="Display machine-wide Antigravity usage")
    parser.add_argument("--all", action="store_true", help="Display both project and global dashboards")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor and auto-refresh")
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds (default: 3.0)")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--json", action="store_true", help="Output telemetry as JSON")
    args = parser.parse_args()

    # Default to --project if neither was specified
    if not (args.project or args.show_global or args.all):
        args.project = True

    if args.json:
        result = {}
        if args.project or args.all:
            result["project"] = summarize(date=args.date)
        if args.show_global or args.all:
            from tools.monitoring.safe_ccusage import get_safe_antigravity_summary
            result["global"] = get_safe_antigravity_summary()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.watch:
        try:
            while True:
                console.clear()
                if args.project or args.all:
                    render_project_dashboard(date=args.date)
                if args.show_global or args.all:
                    render_global_dashboard()
                console.print(f"[dim]Live Monitor: Refreshing every {args.interval}s. Press Ctrl+C to stop.[/dim]")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Live monitor stopped.[/yellow]")
            return

    if args.project or args.all:
        render_project_dashboard(date=args.date)
    if args.show_global or args.all:
        render_global_dashboard()


if __name__ == "__main__":
    main()
