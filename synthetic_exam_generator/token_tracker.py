"""
token_tracker.py — Compatibility wrapper delegating to sequence_labelling.llm.token_tracker.
Logs token usage across providers and models into:
  artifacts/token_usage/token_usage_<YYYY-MM-DD>.jsonl
"""

from __future__ import annotations

from typing import Any, Optional
from sequence_labelling.llm.token_tracker import (
    log_response as _canonical_log_response,
    log_token_usage,
    load_logs,
    summarize,
    print_summary,
    main,
    _to_int,
    _get_log_path,
)


def log_response(
    response: Any,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    caller: Optional[str] = "synthetic_exam",
    duration_sec: Optional[float] = None,
) -> None:
    """
    Append a token-usage record for *response* to artifacts/token_usage/.
    """
    _canonical_log_response(
        response=response,
        model=model,
        provider=provider or "deepseek",
        caller=caller,
        duration_sec=duration_sec,
    )


__all__ = [
    "log_response",
    "log_token_usage",
    "load_logs",
    "summarize",
    "print_summary",
    "main",
]

if __name__ == "__main__":
    main()
