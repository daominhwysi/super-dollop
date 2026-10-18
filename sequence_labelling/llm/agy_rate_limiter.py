"""
agy_rate_limiter.py — 5-Hour Cap Limit Detector, Health Probe & Rolling Telemetry.

Provides:
- Agy5HourCapError: Typed exception for quota/rate limit exhaustion with reset duration.
- CapLimitDetector: Detects 429, RESOURCE_EXHAUSTED, and rolling usage caps with cooldown parsing.
- AgyQuotaProbe: Lightweight 1-token health ping probe to verify active quota.
- RollingUsageMonitor: Computes token and budget burn across a sliding 5-hour window.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
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

        # 4. Try querying real-time agy /quota for exact reset cooldown
        try:
            report = AgyQuotaService.fetch_quota(force_refresh=True, timeout=5.0)
            if report and report.gemini_5h:
                rem_sec = report.gemini_5h.seconds_until_reset()
                if rem_sec > 0:
                    return rem_sec
        except Exception:
            pass

        return default_sec


def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parses ISO timestamp string (e.g. '2026-10-15T06:36:17Z' or '2026-10-15 13:36 +07')."""
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M %z",
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(dt_str, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except Exception:
            continue
    return None


@dataclass
class QuotaBucket:
    """Represents a discrete quota limit window (5-hour or weekly)."""
    id: str
    name: str
    window: str  # "weekly" or "5h"
    remaining_fraction: float  # 0.0 to 1.0 (e.g. 0.3428)
    reset_time: Optional[datetime] = None
    description: str = ""

    @property
    def remaining_pct(self) -> float:
        """Remaining quota as percentage (0.0 to 100.0%)."""
        return round(self.remaining_fraction * 100.0, 2)

    def seconds_until_reset(self) -> int:
        """Seconds until quota window refreshes."""
        if not self.reset_time:
            return 0
        now = datetime.now(timezone.utc)
        target = self.reset_time if self.reset_time.tzinfo else self.reset_time.replace(tzinfo=timezone.utc)
        diff = (target - now).total_seconds()
        return max(0, int(diff))

    def formatted_reset(self) -> str:
        """Formatted reset time string."""
        if not self.reset_time:
            return "N/A"
        try:
            local_dt = self.reset_time.astimezone()
            return local_dt.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return self.reset_time.strftime("%Y-%m-%d %H:%M UTC")

    def formatted_cooldown(self) -> str:
        """Remaining cooldown formatted as 'Xh Ym Zs'."""
        sec = self.seconds_until_reset()
        hrs = sec // 3600
        mins = (sec % 3600) // 60
        secs = sec % 60
        if hrs > 0:
            return f"{hrs}h {mins:02d}m {secs:02d}s"
        return f"{mins}m {secs:02d}s"


@dataclass
class AgyQuotaReport:
    """Comprehensive snapshot of Antigravity CLI quota status across all model groups."""
    gemini_weekly: Optional[QuotaBucket] = None
    gemini_5h: Optional[QuotaBucket] = None
    claude_weekly: Optional[QuotaBucket] = None
    claude_5h: Optional[QuotaBucket] = None
    raw_response: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def gemini_weekly_pct(self) -> float:
        return self.gemini_weekly.remaining_pct if self.gemini_weekly else 100.0

    @property
    def gemini_5h_pct(self) -> float:
        return self.gemini_5h.remaining_pct if self.gemini_5h else 100.0

    @property
    def claude_weekly_pct(self) -> float:
        return self.claude_weekly.remaining_pct if self.claude_weekly else 100.0

    @property
    def claude_5h_pct(self) -> float:
        return self.claude_5h.remaining_pct if self.claude_5h else 100.0

    def is_gemini_5h_exhausted(self, threshold_pct: float = 1.0) -> bool:
        return self.gemini_5h_pct <= threshold_pct

    def is_gemini_weekly_exhausted(self, threshold_pct: float = 1.0) -> bool:
        return self.gemini_weekly_pct <= threshold_pct

    def gemini_5h_cooldown_sec(self) -> int:
        if self.gemini_5h:
            sec = self.gemini_5h.seconds_until_reset()
            if sec > 0:
                return sec
        return DEFAULT_CAP_COOLDOWN_SEC

    def summary_line(self) -> str:
        parts = []
        if self.gemini_5h:
            parts.append(f"Gemini 5h: {self.gemini_5h_pct:.1f}% ({self.gemini_5h.formatted_cooldown()})")
        if self.gemini_weekly:
            parts.append(f"Weekly: {self.gemini_weekly_pct:.1f}%")
        return " | ".join(parts) if parts else "Quota active"


class AgyQuotaService:
    """
    Direct client for querying Google Antigravity / Gemini real-time quota telemetry
    via the agy `/quota` slash command.
    Consumes 0 model tokens.
    """
    _cached_report: Optional[AgyQuotaReport] = None
    _cache_time: float = 0.0
    _cache_account: Optional[str] = None
    CACHE_TTL_SEC: float = 15.0

    @classmethod
    def fetch_quota(
        cls,
        account: Optional[str] = None,
        timeout: float = 15.0,
        force_refresh: bool = False,
    ) -> Optional[AgyQuotaReport]:
        """
        Executes `agy -p /quota --output-format json` to retrieve official quota status.
        Results are cached for CACHE_TTL_SEC unless force_refresh is True.
        """
        now = time.time()
        target_account = account or os.environ.get("AGY_ACCOUNT")
        if (
            not force_refresh
            and cls._cached_report is not None
            and (now - cls._cache_time) < cls.CACHE_TTL_SEC
            and cls._cache_account == target_account
        ):
            return cls._cached_report

        cmd = ["agy"]
        if target_account:
            cmd.extend(["--account", target_account])
        # Note: Do NOT use --disable-slash-commands as /quota is a CLI slash command
        cmd.extend(["-p", "/quota", "--output-format", "json"])

        env = os.environ.copy()
        if target_account:
            prof_home = Path.home() / ".gemini" / "profiles" / target_account
            if prof_home.exists():
                env["HOME"] = str(prof_home)
                env["AGY_ACCOUNT"] = target_account

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            report = cls.parse_quota_output(proc.stdout, proc.stderr)
            if report:
                cls._cached_report = report
                cls._cache_time = now
                cls._cache_account = target_account
            return report
        except Exception as exc:
            logger.debug(f"Failed to query agy quota via CLI: {exc}")
            return cls._cached_report

    @classmethod
    def parse_quota_output(cls, stdout: str, stderr: str = "") -> Optional[AgyQuotaReport]:
        """Parses output from `agy -p /quota` into an AgyQuotaReport."""
        raw = f"{stdout}\n{stderr}".strip()
        if not raw:
            return None

        # 1. Try structured JSON command output
        try:
            data = None
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        cand = json.loads(line)
                        if "command" in cand or "response" in cand:
                            data = cand
                            break
                    except Exception:
                        pass
            if not data and stdout.strip().startswith("{"):
                data = json.loads(stdout.strip())

            if data:
                report = AgyQuotaReport(raw_response=raw)
                cmd_data = data.get("command", {}).get("data", {})
                groups = cmd_data.get("groups", [])
                if groups:
                    for g in groups:
                        g_name = g.get("name", "").lower()
                        for b in g.get("buckets", []):
                            bucket = QuotaBucket(
                                id=b.get("id", ""),
                                name=b.get("name", ""),
                                window=b.get("window", ""),
                                remaining_fraction=float(b.get("remaining_fraction", 1.0)),
                                reset_time=parse_iso_datetime(b.get("reset_time", "")),
                                description=b.get("description", ""),
                            )
                            if "gemini" in g_name or "gemini" in bucket.id:
                                if bucket.window == "weekly":
                                    report.gemini_weekly = bucket
                                elif bucket.window == "5h":
                                    report.gemini_5h = bucket
                            else:
                                if bucket.window == "weekly":
                                    report.claude_weekly = bucket
                                elif bucket.window == "5h":
                                    report.claude_5h = bucket
                    return report

                # Fallback: check text in 'response' field
                resp_text = data.get("response", "")
                if resp_text:
                    parsed = cls._parse_text_table(resp_text)
                    if parsed:
                        parsed.raw_response = raw
                        return parsed
        except Exception as exc:
            logger.debug(f"JSON quota parse error, trying text regex: {exc}")

        # 2. Text table parsing fallback
        return cls._parse_text_table(raw)

    @classmethod
    def _parse_text_table(cls, text: str) -> Optional[AgyQuotaReport]:
        report = AgyQuotaReport(raw_response=text)
        found_any = False

        for line in text.splitlines():
            line = line.strip()
            if not line or "limit remaining" not in line.lower():
                continue

            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if not pct_match:
                continue

            frac = float(pct_match.group(1)) / 100.0

            # Match reset time, e.g. 2026-10-21T06:39:04Z or 2026-10-21 13:39 +07
            time_match = re.search(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?(?:[Z\s\+\-][^\s]*)?)", line)
            reset_dt = parse_iso_datetime(time_match.group(1)) if time_match else None

            is_gemini = "gemini" in line.lower()
            is_weekly = "weekly" in line.lower()
            is_5h = "five hour" in line.lower() or "5h" in line.lower() or "5-hour" in line.lower()

            bucket = QuotaBucket(
                id="gemini-weekly" if (is_gemini and is_weekly) else ("gemini-5h" if is_gemini else ("3p-weekly" if is_weekly else "3p-5h")),
                name="Weekly Limit Remaining" if is_weekly else "Five Hour Limit Remaining",
                window="weekly" if is_weekly else "5h",
                remaining_fraction=frac,
                reset_time=reset_dt,
            )

            if is_gemini:
                if is_weekly:
                    report.gemini_weekly = bucket
                    found_any = True
                elif is_5h:
                    report.gemini_5h = bucket
                    found_any = True
            else:
                if is_weekly:
                    report.claude_weekly = bucket
                    found_any = True
                elif is_5h:
                    report.claude_5h = bucket
                    found_any = True

        return report if found_any else None


class AgyQuotaProbe:
    """
    Health checker that verifies active agy quota.
    Uses AgyQuotaService (0 tokens consumed), falling back to a 1-token prompt if necessary.
    """

    @staticmethod
    def probe_health(
        model: Optional[str] = None,
        timeout: float = 30.0,
        account: Optional[str] = None,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Probes agy quota health.
        Returns:
            (is_healthy, cooldown_seconds_if_capped, status_message)
        """
        # 1. Primary zero-token quota probe via /quota
        try:
            report = AgyQuotaService.fetch_quota(
                account=account,
                timeout=min(timeout, 12.0),
                force_refresh=True,
            )
            if report and (report.gemini_5h or report.gemini_weekly):
                if report.is_gemini_5h_exhausted(threshold_pct=1.0):
                    cooldown = report.gemini_5h_cooldown_sec()
                    return False, cooldown, f"Gemini 5-hour quota exhausted (0% left, resets in {cooldown}s)"
                if report.is_gemini_weekly_exhausted(threshold_pct=1.0):
                    return False, None, f"Gemini weekly quota exhausted ({report.gemini_weekly_pct:.1f}% left, resets {report.gemini_weekly.formatted_reset()})"
                return True, None, f"Healthy (Gemini 5h: {report.gemini_5h_pct:.1f}%, Weekly: {report.gemini_weekly_pct:.1f}%)"
        except Exception as exc:
            logger.debug(f"AgyQuotaService probe error: {exc}. Trying fallback ping...")

        # 2. Fallback 1-token prompt ping
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

        env = os.environ.copy()
        target_acc = account or os.environ.get("AGY_ACCOUNT")
        if target_acc:
            prof_home = Path.home() / ".gemini" / "profiles" / target_acc
            if prof_home.exists():
                env["HOME"] = str(prof_home)
                env["AGY_ACCOUNT"] = target_acc

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if proc.returncode == 0:
                return True, None, "Healthy (Quota active)"

            raw_err = f"{proc.stdout}\n{proc.stderr}".strip()
            if CapLimitDetector.is_cap_limit(proc.returncode, proc.stdout, proc.stderr):
                cooldown = CapLimitDetector.parse_cooldown_seconds(raw_err)
                return False, cooldown, f"Cap limit active ({cooldown}s cooldown)"

            return False, 60, f"Transient agy error (exit {proc.returncode}): {raw_err[:120]}"

        except subprocess.TimeoutExpired:
            return False, 60, f"agy probe timed out after {int(timeout)}s"
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
