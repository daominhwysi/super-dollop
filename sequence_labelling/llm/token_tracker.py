"""
token_tracker.py — Unified Multi-Provider Token Usage Logger & Analytics Ledger.

Logs LLM token usage across all providers (deepseek, codex, agy, nvidia, xah, vilao, commandcode)
and models into:
  artifacts/token_usage/token_usage_<YYYY-MM-DD>.jsonl

Each record contains:
  - timestamp         ISO-8601 UTC timestamp
  - provider          Provider name (e.g. 'deepseek', 'codex', 'agy', 'nvidia', etc.)
  - model             Model name used
  - input_tokens      Prompt tokens
  - reasoning_tokens  Reasoning/chain-of-thought tokens (if applicable)
  - output_tokens     Completion tokens (excluding reasoning when separate)
  - total_tokens      Total billed / consumed tokens
  - duration_sec      API execution latency in seconds (optional)
  - caller            Calling component / module (optional)
"""

from __future__ import annotations

import os
import json
import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sequence_labelling.config import TOKEN_USAGE_DIR

# ── thread-safe write lock ────────────────────────────────────────────────────
_write_lock = threading.Lock()


def _get_log_path(date_str: Optional[str] = None) -> Path:
    """Return the JSONL log file path for a date, ensuring directory exists."""
    TOKEN_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not date_str:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return TOKEN_USAGE_DIR / f"token_usage_{date_str}.jsonl"


def _to_int(val: Any) -> int:
    """Safely convert token counts to int."""
    if val is None:
        return 0
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str) and val.strip().isdigit():
        return int(val.strip())
    return 0


def log_token_usage(
    provider: str = "unknown",
    model: str = "unknown",
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cache_read_tokens: int = 0,
    total_tokens: Optional[int] = None,
    duration_sec: Optional[float] = None,
    caller: Optional[str] = None,
) -> Path:
    """
    Directly logs a token usage entry with provider, model, and cache metrics.
    """
    total = total_tokens if total_tokens is not None else (input_tokens + output_tokens + reasoning_tokens)
    
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "provider": provider or "unknown",
        "model": model or "unknown",
        "input_tokens": _to_int(input_tokens),
        "cache_read_tokens": _to_int(cache_read_tokens),
        "reasoning_tokens": _to_int(reasoning_tokens),
        "output_tokens": _to_int(output_tokens),
        "total_tokens": _to_int(total),
    }
    if duration_sec is not None:
        record["duration_sec"] = round(float(duration_sec), 3)
    if caller is not None:
        record["caller"] = str(caller)

    log_path = _get_log_path()
    with _write_lock:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return log_path


def log_response(
    response: Any,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    caller: Optional[str] = None,
    duration_sec: Optional[float] = None,
) -> Path:
    """
    Parses token usage from an LLM response object or dict and records it.

    Supports:
      - OpenAI ChatCompletion / ModelResponse
      - Codex TurnResult (thread.run)
      - AGY usage dict or response
      - Generic dictionary with 'usage' or token keys
    """
    resolved_model = model or getattr(response, "model", None) or "unknown"
    resolved_provider = provider or "unknown"

    # 1. Check for Codex TurnResult
    if hasattr(response, "usage") and hasattr(response, "final_response") and resolved_provider == "unknown":
        resolved_provider = "codex"

    usage = getattr(response, "usage", None)
    if isinstance(response, dict) and usage is None:
        usage = response.get("usage")

    input_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    cache_read_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if usage is not None:
        if isinstance(usage, dict):
            input_tokens = _to_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            completion_tokens = _to_int(usage.get("completion_tokens") or usage.get("output_tokens"))
            total_tokens = _to_int(usage.get("total_tokens"))
            cache_read_tokens = _to_int(
                usage.get("cache_read_tokens")
                or usage.get("cached_tokens")
                or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            )
            
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict):
                reasoning_tokens = _to_int(details.get("reasoning_tokens"))
            else:
                reasoning_tokens = _to_int(getattr(details, "reasoning_tokens", 0))

            if "reasoning_tokens" in usage:
                reasoning_tokens = _to_int(usage.get("reasoning_tokens"))
            elif "thinking_tokens" in usage:
                reasoning_tokens = _to_int(usage.get("thinking_tokens"))

            if "output_tokens" in usage and "completion_tokens" not in usage:
                output_tokens = _to_int(usage.get("output_tokens"))
            else:
                output_tokens = max(0, completion_tokens - reasoning_tokens)

        else:
            input_tokens = _to_int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0))
            completion_tokens = _to_int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0))
            total_tokens = _to_int(getattr(usage, "total_tokens", 0))
            
            cached_val = getattr(usage, "cache_read_tokens", None) or getattr(usage, "cached_tokens", None)
            if cached_val is None:
                p_details = getattr(usage, "prompt_tokens_details", None)
                cached_val = getattr(p_details, "cached_tokens", 0) if p_details else 0
            cache_read_tokens = _to_int(cached_val)

            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = _to_int(
                getattr(details, "reasoning_tokens", 0)
                or getattr(usage, "reasoning_tokens", 0)
                or getattr(usage, "thinking_tokens", 0)
            )

            if getattr(usage, "output_tokens", None) is not None and getattr(usage, "completion_tokens", None) is None:
                output_tokens = _to_int(getattr(usage, "output_tokens", 0))
            else:
                output_tokens = max(0, completion_tokens - reasoning_tokens)

    if total_tokens == 0:
        total_tokens = input_tokens + completion_tokens

    return log_token_usage(
        provider=resolved_provider,
        model=resolved_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        total_tokens=total_tokens,
        duration_sec=duration_sec,
        caller=caller,
    )


# ── summary & analytics helpers ───────────────────────────────────────────────

def load_logs(date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load all records from a token log file.
    date: 'YYYY-MM-DD' string. Defaults to today (UTC).
    """
    log_path = _get_log_path(date)
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def summarize(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute aggregate token statistics overall, by provider, by model, and by caller.
    """
    records = load_logs(date)
    summary: Dict[str, Any] = {
        "date": date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total_calls": len(records),
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration_sec": 0.0,
        "by_provider": {},
        "by_model": {},
        "by_provider_model": {},
        "by_caller": {},
    }

    for r in records:
        inp = r.get("input_tokens", 0)
        cache = r.get("cache_read_tokens", 0)
        res = r.get("reasoning_tokens", 0)
        out = r.get("output_tokens", 0)
        tot = r.get("total_tokens", inp + out + res)
        dur = r.get("duration_sec", 0.0) or 0.0
        prov = r.get("provider", "unknown")
        mod = r.get("model", "unknown")
        caller = r.get("caller") or "unknown"
        prov_mod = f"{prov} / {mod}"

        # Global sums
        summary["input_tokens"] += inp
        summary["cache_read_tokens"] += cache
        summary["reasoning_tokens"] += res
        summary["output_tokens"] += out
        summary["total_tokens"] += tot
        summary["total_duration_sec"] += dur

        def _update_bucket(bucket: Dict[str, Any], key: str):
            if key not in bucket:
                bucket[key] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_duration_sec": 0.0,
                }
            b = bucket[key]
            b["calls"] += 1
            b["input_tokens"] += inp
            b["cache_read_tokens"] += cache
            b["reasoning_tokens"] += res
            b["output_tokens"] += out
            b["total_tokens"] += tot
            b["total_duration_sec"] += dur

        _update_bucket(summary["by_provider"], prov)
        _update_bucket(summary["by_model"], mod)
        _update_bucket(summary["by_provider_model"], prov_mod)
        _update_bucket(summary["by_caller"], caller)

    # Compute cache hit percentages
    def _add_rates(b: Dict[str, Any]):
        inp = b.get("input_tokens", 0)
        cache = b.get("cache_read_tokens", 0)
        calls = b.get("calls", 0)
        dur = b.get("total_duration_sec", 0.0)
        b["cache_hit_pct"] = round(100.0 * cache / inp, 1) if inp > 0 else 0.0
        b["avg_duration_sec"] = round(dur / calls, 2) if calls > 0 else 0.0

    _add_rates(summary)
    for p in summary["by_provider"].values(): _add_rates(p)
    for m in summary["by_model"].values(): _add_rates(m)
    for pm in summary["by_provider_model"].values(): _add_rates(pm)
    for c in summary["by_caller"].values(): _add_rates(c)

    return summary


def print_summary(date: Optional[str] = None) -> None:
    """Print a structured, tabular token usage report for the given date."""
    s = summarize(date)
    date_str = s["date"]
    total_calls = s["total_calls"]

    print(f"\n==========================================================================================================")
    print(f" TOKEN USAGE & OBSERVABILITY REPORT — Date: {date_str} (Total API Calls: {total_calls})")
    print(f"==========================================================================================================")

    if total_calls == 0:
        print(" No token records found for this date.")
        print("==========================================================================================================\n")
        return

    # 1. By Provider Summary
    print("\n--- 🏢 Breakdown by Provider ---")
    header = f"{'Provider':<16} | {'Calls':>6} | {'Input Tokens':>13} | {'Cache Read (Hit %)':>20} | {'Thinking':>10} | {'Output':>10} | {'Total':>13} | {'Avg Sec':>7}"
    print(header)
    print("-" * len(header))
    for prov, stats in sorted(s["by_provider"].items(), key=lambda x: -x[1]["total_tokens"]):
        cache_str = f"{stats['cache_read_tokens']:,d} ({stats['cache_hit_pct']}%)"
        print(f"{prov:<16} | {stats['calls']:>6,d} | {stats['input_tokens']:>13,d} | {cache_str:>20} | {stats['reasoning_tokens']:>10,d} | {stats['output_tokens']:>10,d} | {stats['total_tokens']:>13,d} | {stats['avg_duration_sec']:>6.2f}s")

    # 2. By Model Summary
    print("\n--- 🤖 Breakdown by Model ---")
    header_m = f"{'Model':<26} | {'Calls':>6} | {'Input Tokens':>13} | {'Cache Read (Hit %)':>20} | {'Thinking':>10} | {'Output':>10} | {'Total':>13} | {'Avg Sec':>7}"
    print(header_m)
    print("-" * len(header_m))
    for mod, stats in sorted(s["by_model"].items(), key=lambda x: -x[1]["total_tokens"]):
        display_mod = mod if len(mod) <= 26 else (mod[:23] + "...")
        cache_str = f"{stats['cache_read_tokens']:,d} ({stats['cache_hit_pct']}%)"
        print(f"{display_mod:<26} | {stats['calls']:>6,d} | {stats['input_tokens']:>13,d} | {cache_str:>20} | {stats['reasoning_tokens']:>10,d} | {stats['output_tokens']:>10,d} | {stats['total_tokens']:>13,d} | {stats['avg_duration_sec']:>6.2f}s")

    # 3. By Caller (if present)
    if s["by_caller"] and any(k != "unknown" for k in s["by_caller"]):
        print("\n--- 🎯 Breakdown by Pipeline Caller ---")
        header_c = f"{'Caller / Component':<26} | {'Calls':>6} | {'Input Tokens':>13} | {'Cache Read (Hit %)':>20} | {'Output':>10} | {'Total':>13} | {'Avg Sec':>7}"
        print(header_c)
        print("-" * len(header_c))
        for caller, stats in sorted(s["by_caller"].items(), key=lambda x: -x[1]["total_tokens"]):
            display_c = caller if len(caller) <= 26 else (caller[:23] + "...")
            cache_str = f"{stats['cache_read_tokens']:,d} ({stats['cache_hit_pct']}%)"
            print(f"{display_c:<26} | {stats['calls']:>6,d} | {stats['input_tokens']:>13,d} | {cache_str:>20} | {stats['output_tokens']:>10,d} | {stats['total_tokens']:>13,d} | {stats['avg_duration_sec']:>6.2f}s")

    # 4. Grand Total
    print("\n" + "=" * len(header))
    global_cache = f"{s['cache_read_tokens']:,d} ({s['cache_hit_pct']}%)"
    global_avg_sec = round(s["total_duration_sec"] / s["total_calls"], 2) if s["total_calls"] > 0 else 0.0
    print(f"{'GRAND TOTAL':<16} | {s['total_calls']:>6,d} | {s['input_tokens']:>13,d} | {global_cache:>20} | {s['reasoning_tokens']:>10,d} | {s['output_tokens']:>10,d} | {s['total_tokens']:>13,d} | {global_avg_sec:>6.2f}s")
    print("==========================================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Display LLM token usage breakdown by provider and model.")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh and watch token usage")
    parser.add_argument("--interval", type=float, default=3.0, help="Watch refresh interval in seconds")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(summarize(date=args.date), indent=2, ensure_ascii=False))
        return

    if args.watch:
        import time
        try:
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                print_summary(date=args.date)
                print(f"[Live Monitor] Refreshing every {args.interval}s. Press Ctrl+C to exit.")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting watch mode.")
            return

    print_summary(date=args.date)


if __name__ == "__main__":
    main()
