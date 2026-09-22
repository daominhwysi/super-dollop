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
    total_tokens: Optional[int] = None,
    duration_sec: Optional[float] = None,
    caller: Optional[str] = None,
) -> Path:
    """
    Directly logs a token usage entry with provider and model.
    """
    total = total_tokens if total_tokens is not None else (input_tokens + output_tokens + reasoning_tokens)
    
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "provider": provider or "unknown",
        "model": model or "unknown",
        "input_tokens": _to_int(input_tokens),
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
    output_tokens = 0
    total_tokens = 0

    if usage is not None:
        if isinstance(usage, dict):
            input_tokens = _to_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
            completion_tokens = _to_int(usage.get("completion_tokens") or usage.get("output_tokens"))
            total_tokens = _to_int(usage.get("total_tokens"))
            
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict):
                reasoning_tokens = _to_int(details.get("reasoning_tokens"))
            else:
                reasoning_tokens = _to_int(getattr(details, "reasoning_tokens", 0))

            if "reasoning_tokens" in usage:
                reasoning_tokens = _to_int(usage.get("reasoning_tokens"))

            if "output_tokens" in usage and "completion_tokens" not in usage:
                output_tokens = _to_int(usage.get("output_tokens"))
            else:
                output_tokens = max(0, completion_tokens - reasoning_tokens)

        else:
            input_tokens = _to_int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0))
            completion_tokens = _to_int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0))
            total_tokens = _to_int(getattr(usage, "total_tokens", 0))
            
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = _to_int(getattr(details, "reasoning_tokens", 0) or getattr(usage, "reasoning_tokens", 0))

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
    Compute aggregate token statistics overall, by provider, and by model.
    """
    records = load_logs(date)
    summary: Dict[str, Any] = {
        "date": date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "total_calls": len(records),
        "input_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "by_provider": {},
        "by_model": {},
        "by_provider_model": {},
    }

    for r in records:
        inp = r.get("input_tokens", 0)
        res = r.get("reasoning_tokens", 0)
        out = r.get("output_tokens", 0)
        tot = r.get("total_tokens", inp + out + res)
        prov = r.get("provider", "unknown")
        mod = r.get("model", "unknown")
        prov_mod = f"{prov} / {mod}"

        # Global sums
        summary["input_tokens"] += inp
        summary["reasoning_tokens"] += res
        summary["output_tokens"] += out
        summary["total_tokens"] += tot

        # By Provider
        if prov not in summary["by_provider"]:
            summary["by_provider"][prov] = {
                "calls": 0, "input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0, "total_tokens": 0
            }
        p_dict = summary["by_provider"][prov]
        p_dict["calls"] += 1
        p_dict["input_tokens"] += inp
        p_dict["reasoning_tokens"] += res
        p_dict["output_tokens"] += out
        p_dict["total_tokens"] += tot

        # By Model
        if mod not in summary["by_model"]:
            summary["by_model"][mod] = {
                "calls": 0, "input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0, "total_tokens": 0
            }
        m_dict = summary["by_model"][mod]
        m_dict["calls"] += 1
        m_dict["input_tokens"] += inp
        m_dict["reasoning_tokens"] += res
        m_dict["output_tokens"] += out
        m_dict["total_tokens"] += tot

        # By Provider / Model
        if prov_mod not in summary["by_provider_model"]:
            summary["by_provider_model"][prov_mod] = {
                "calls": 0, "input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0, "total_tokens": 0
            }
        pm_dict = summary["by_provider_model"][prov_mod]
        pm_dict["calls"] += 1
        pm_dict["input_tokens"] += inp
        pm_dict["reasoning_tokens"] += res
        pm_dict["output_tokens"] += out
        pm_dict["total_tokens"] += tot

    return summary


def print_summary(date: Optional[str] = None) -> None:
    """Print a structured, tabular token usage report for the given date."""
    s = summarize(date)
    date_str = s["date"]
    total_calls = s["total_calls"]

    print(f"\n========================================================================================")
    print(f" TOKEN USAGE REPORT — Date: {date_str} (Total API Calls: {total_calls})")
    print(f"========================================================================================")

    if total_calls == 0:
        print(" No token records found for this date.")
        print("========================================================================================\n")
        return

    # 1. By Provider Summary
    print("\n--- 🏢 Breakdown by Provider ---")
    header = f"{'Provider':<18} | {'Calls':>7} | {'Input Tokens':>14} | {'Reasoning':>12} | {'Output':>12} | {'Total':>14}"
    print(header)
    print("-" * len(header))
    for prov, stats in sorted(s["by_provider"].items(), key=lambda x: -x[1]["total_tokens"]):
        print(f"{prov:<18} | {stats['calls']:>7,d} | {stats['input_tokens']:>14,d} | {stats['reasoning_tokens']:>12,d} | {stats['output_tokens']:>12,d} | {stats['total_tokens']:>14,d}")

    # 2. By Model Summary
    print("\n--- 🤖 Breakdown by Model ---")
    header_m = f"{'Model':<30} | {'Calls':>7} | {'Input Tokens':>14} | {'Reasoning':>12} | {'Output':>12} | {'Total':>14}"
    print(header_m)
    print("-" * len(header_m))
    for mod, stats in sorted(s["by_model"].items(), key=lambda x: -x[1]["total_tokens"]):
        display_mod = mod if len(mod) <= 30 else (mod[:27] + "...")
        print(f"{display_mod:<30} | {stats['calls']:>7,d} | {stats['input_tokens']:>14,d} | {stats['reasoning_tokens']:>12,d} | {stats['output_tokens']:>12,d} | {stats['total_tokens']:>14,d}")

    # 3. Combined Provider & Model
    print("\n--- 🔗 Breakdown by Provider & Model ---")
    header_pm = f"{'Provider / Model':<36} | {'Calls':>7} | {'Input':>12} | {'Reasoning':>11} | {'Output':>11} | {'Total Tokens':>14}"
    print(header_pm)
    print("-" * len(header_pm))
    for pm, stats in sorted(s["by_provider_model"].items(), key=lambda x: -x[1]["total_tokens"]):
        display_pm = pm if len(pm) <= 36 else (pm[:33] + "...")
        print(f"{display_pm:<36} | {stats['calls']:>7,d} | {stats['input_tokens']:>12,d} | {stats['reasoning_tokens']:>11,d} | {stats['output_tokens']:>11,d} | {stats['total_tokens']:>14,d}")

    # 4. Grand Total
    print("\n" + "=" * len(header))
    print(f"{'GRAND TOTAL':<18} | {s['total_calls']:>7,d} | {s['input_tokens']:>14,d} | {s['reasoning_tokens']:>12,d} | {s['output_tokens']:>12,d} | {s['total_tokens']:>14,d}")
    print("========================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Display LLM token usage breakdown by provider and model.")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()
    print_summary(date=args.date)


if __name__ == "__main__":
    main()
