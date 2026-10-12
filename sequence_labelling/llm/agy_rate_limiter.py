"""
agy_rate_limiter.py — 5-Hour Cap Limit Detector, Health Probe & Rolling Telemetry.

Provides:
- Agy5HourCapError: Typed exception for quota/rate limit exhaustion with reset duration.
- CapLimitDetector: Detects 429, RESOURCE_EXHAUSTED, and rolling usage caps with cooldown parsing.
- AgyQuotaProbe: Lightweight 1-token health ping probe to verify active quota.
- RollingUsageMonitor: Computes token and budget burn across a sliding 5-hour window.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default cooldown when agy does not specify an explicit reset duration (5 hours)
DEFAULT_CAP_COOLDOWN_SEC = 5 * 3600

# Quota billing rates for agy ($200/week subscription accounting)
AGY_INPUT_RATE_PER_M = 0.75   # $0.75 per 1,000,000 input tokens
AGY_OUTPUT_RATE_PER_M = 3.75  # $3.75 per 1,000,000 output tokens


class Agy5HourCapError(RuntimeError):
    """
    Raised when the agy CLI or underlying Gemini model hits a rolling 5-hour
    usage limit, rate limit, or resource exhaustion error.
    """

    def __init__(self, message: str, reset_after_sec: int, raw_output: str = ""):
        super().__init__(message)
        self.reset_after_sec = max(60, int(reset_after_sec))
        self.raw_output = raw_output
        self.detected_at = time.time()
        self.resumes_at = self.detected_at + self.reset_after_sec

    def formatted_resumes_at(self) -> str:
        resumes_dt = datetime.fromtimestamp(self.resumes_at)
        return resumes_dt.strftime("%Y-%m-%d %H:%M:%S")

    def formatted_cooldown(self) -> str:
        hrs = self.reset_after_sec // 3600
        mins = (self.reset_after_sec % 3600) // 60
        secs = self.reset_after_sec % 60
        return f"{hrs:02d}h {mins:02d}m {secs:02d}s"


def is_quota_or_cap_error(exc: Optional[BaseException]) -> bool:
    """
    Determines if an exception or error represents a quota exhaustion, rate limit,
    or cap limit error across any LLM provider (agy, openai, deepseek, xah, nvidia, etc.).
    """
    if exc is None:
        return False
    if isinstance(exc, Agy5HourCapError):
        return True

    msg = str(exc)
    for p in CapLimitDetector.PATTERNS:
        if re.search(p, msg):
            return True

    extra_patterns = [
        r"(?i)insufficient_quota",
        r"(?i)credit\s*balance",
        r"(?i)cap\s*limit\s*reached",
        r"(?i)quota\s*reached",
        r"(?i)rate\s*limit",
    ]
    return any(re.search(p, msg) for p in extra_patterns)


class CapLimitDetector:
    """
    Regex detection engine for agy / Gemini CLI quota exhaustion and rate limit errors.
    """

    PATTERNS = [
        r"(?i)resource[_\s]*(?:has\s*been\s*)?exhausted",
        r"(?i)rate\s*limit\s*(?:exceeded|reached)",
        r"(?i)quota\s*(?:has\s*been\s*)?(?:exceeded|reached)",
        r"(?i)individual\s*quota\s*reached",
        r"(?i)usage\s*limit\s*(?:reached|exceeded)",
        r"(?i)(?:limit\s*)?resets?\s*(?:in|at)",
        r"(?i)429",
        r"(?i)try\s*again\s*in\s*(\d+)",
        r"(?i)exceeded\s*your\s*current\s*quota",
        r"(?i)check\s*quota",
    ]

    @classmethod
    def is_cap_limit(cls, returncode: int, stdout: str = "", stderr: str = "") -> bool:
        """Determines if the process failed due to a quota cap or rate limit."""
        if returncode == 0:
            return False
        combined = f"{stdout}\n{stderr}".strip()
        if not combined:
            return False
        return any(re.search(p, combined) for p in cls.PATTERNS)

    @classmethod
    def parse_cooldown_seconds(cls, text: str, default_sec: int = DEFAULT_CAP_COOLDOWN_SEC) -> int:
        """
        Extracts remaining cooldown time in seconds from error text.
        Supports patterns like:
          - 'Resets in 1h2m55s' or 'Resets in 11m30s'
          - 'resets in 4 hours 45 minutes'
          - 'try again in 5h'
          - 'try again in 280 minutes'
          - 'retry-after: 3600'
        """
        # 1. Match duration following 'resets in' or 'try again in', e.g. '1h2m55s', '11m30s', '4 hours 45 mins'
        m = re.search(
            r"(?:resets?\s*in|try\s*again\s*in)\s*(?:(\d+)\s*h(?:ours?|rs?)?)?\s*(?:(\d+)\s*m(?:in(?:ute)?s?)?)?\s*(?:(\d+)\s*s(?:ec(?:ond)?s?)?)?",
            text,
            re.IGNORECASE,
        )
        if m and any(m.groups()):
            hrs = int(m.group(1)) if m.group(1) else 0
            mins = int(m.group(2)) if m.group(2) else 0
            secs = int(m.group(3)) if m.group(3) else 0
            total = (hrs * 3600) + (mins * 60) + secs
            if total > 0:
                return total

        # 2. General check for 'X hours Y minutes' or 'Xh Ym' anywhere
        hr_match = re.search(r"(\d+)\s*(?:hours?|hrs?|h\b)", text, re.IGNORECASE)
        min_match = re.search(r"(\d+)\s*(?:minutes?|mins?|m\b)", text, re.IGNORECASE)
        sec_match = re.search(r"(\d+)\s*(?:seconds?|secs?|s\b)", text, re.IGNORECASE)

        if hr_match or min_match:
            hrs = int(hr_match.group(1)) if hr_match else 0
            mins = int(min_match.group(1)) if min_match else 0
            secs = int(sec_match.group(1)) if sec_match else 0
            total = (hrs * 3600) + (mins * 60) + secs
            if total > 0:
                return total

        # 3. Check for retry-after integer seconds
        retry_match = re.search(r"(?i)retry[-_ ]after[:\s]+(\d+)", text)
        if retry_match:
            return max(60, int(retry_match.group(1)))

        return default_sec


class AgyQuotaProbe:
    """
    Lightweight health checker that verifies active agy quota via a 1-token prompt.
    """

    @staticmethod
    def probe_health(
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Probes agy with a tiny prompt.
        Returns:
            (is_healthy, cooldown_seconds_if_capped, status_message)
        """
        target_model = model or "gemini-3.8-flash-low"
        if not any(target_model.endswith(s) for s in ("-low", "-medium", "-high")):
            target_model = f"{target_model}-low"

        cmd = [
            "agy",
            "--disable-slash-commands",
            "-p", "ping",
            "--model", target_model,
            "--output-format", "json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                return True, None, "Healthy (Quota active)"

            raw_err = f"{proc.stdout}\n{proc.stderr}".strip()
            if CapLimitDetector.is_cap_limit(proc.returncode, proc.stdout, proc.stderr):
                cooldown = CapLimitDetector.parse_cooldown_seconds(raw_err)
                return False, cooldown, f"Cap limit active ({cooldown}s cooldown)"

            return False, 60, f"Transient agy error (exit {proc.returncode}): {raw_err[:120]}"

        except subprocess.TimeoutExpired:
            return False, 60, "agy probe timed out after 12s"
        except Exception as exc:
            return False, 60, f"Probe exception: {exc}"


class RollingUsageMonitor:
    """
    Monitors rolling 5-hour sliding window token consumption and cumulative weekly budget.
    """

    @staticmethod
    def calculate_cost_usd(input_tokens: int, output_tokens: int) -> float:
        """Calculates USD cost under $0.75/1M input and $3.75/1M output rates."""
        in_cost = (input_tokens / 1_000_000.0) * AGY_INPUT_RATE_PER_M
        out_cost = (output_tokens / 1_000_000.0) * AGY_OUTPUT_RATE_PER_M
        return round(in_cost + out_cost, 4)

    @staticmethod
    def get_rolling_usage(window_hours: float = 5.0) -> Dict[str, Any]:
        """
        Queries today's (and yesterday's if needed) token logs within the past window_hours.
        """
        try:
            from sequence_labelling.llm.token_tracker import load_logs
            now = datetime.now(tz=timezone.utc)
            cutoff = now - timedelta(hours=window_hours)

            # Load today and yesterday records
            today_str = now.strftime("%Y-%m-%d")
            yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")

            all_records = load_logs(today_str)
            if cutoff.date() < now.date():
                all_records = load_logs(yesterday_str) + all_records

            in_tokens = 0
            out_tokens = 0
            reasoning_tokens = 0
            calls = 0

            for r in all_records:
                ts_str = r.get("timestamp")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff:
                        in_tokens += r.get("input_tokens", 0)
                        out_tokens += r.get("output_tokens", 0)
                        reasoning_tokens += r.get("reasoning_tokens", 0)
                        calls += 1
                except Exception:
                    pass

            total_tokens = in_tokens + out_tokens + reasoning_tokens
            cost_usd = RollingUsageMonitor.calculate_cost_usd(in_tokens, out_tokens)

            return {
                "window_hours": window_hours,
                "calls": calls,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
            }
        except Exception as exc:
            logger.debug(f"Failed to compute rolling usage: {exc}")
            return {
                "window_hours": window_hours,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
