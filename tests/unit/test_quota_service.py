"""
Unit tests for AgyQuotaService, AgyQuotaReport, and QuotaBucket parsing.
"""

from datetime import datetime, timezone
from sequence_labelling.llm.agy_rate_limiter import (
    AgyQuotaReport,
    AgyQuotaService,
    QuotaBucket,
    parse_iso_datetime,
)

SAMPLE_JSON_OUTPUT = """
{
  "conversation_id": "",
  "status": "SUCCESS",
  "response": "Gemini Models\\tWeekly Limit Remaining\\t34%\\t2026-10-21T06:39:04Z\\nGemini Models\\tFive Hour Limit Remaining\\t62%\\t2026-10-15T06:36:17Z\\nClaude and GPT models\\tWeekly Limit Remaining\\t45%\\t2026-10-21T07:06:10Z\\nClaude and GPT models\\tFive Hour Limit Remaining\\t100%\\t2026-10-15T09:46:55Z\\n",
  "duration_seconds": 0,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  },
  "command": {
    "name": "usage",
    "data": {
      "groups": [
        {
          "name": "Gemini Models",
          "buckets": [
            {
              "id": "gemini-weekly",
              "name": "Weekly Limit Remaining",
              "window": "weekly",
              "remaining_fraction": 0.3375,
              "reset_time": "2026-10-21T06:39:04Z"
            },
            {
              "id": "gemini-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 0.6196,
              "reset_time": "2026-10-15T06:36:17Z"
            }
          ]
        },
        {
          "name": "Claude and GPT models",
          "buckets": [
            {
              "id": "3p-weekly",
              "name": "Weekly Limit Remaining",
              "window": "weekly",
              "remaining_fraction": 0.4540,
              "reset_time": "2026-10-21T07:06:10Z"
            },
            {
              "id": "3p-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 1.0,
              "reset_time": "2026-10-15T09:46:55Z"
            }
          ]
        }
      ]
    }
  }
}
"""

SAMPLE_TEXT_OUTPUT = """
Quota:
Gemini Models          Weekly Limit Remaining     34%   2026-10-21 13:39 +07
Gemini Models          Five Hour Limit Remaining  62%   2026-10-15 13:36 +07
Claude and GPT models  Weekly Limit Remaining     45%   2026-10-21 14:06 +07
Claude and GPT models  Five Hour Limit Remaining  100%  2026-10-15 16:46 +07
"""


def test_parse_json_quota_output():
    report = AgyQuotaService.parse_quota_output(SAMPLE_JSON_OUTPUT)
    assert report is not None
    assert report.gemini_weekly is not None
    assert report.gemini_5h is not None
    assert report.claude_weekly is not None
    assert report.claude_5h is not None

    assert report.gemini_weekly_pct == 33.75
    assert report.gemini_5h_pct == 61.96
    assert report.claude_weekly_pct == 45.40
    assert report.claude_5h_pct == 100.0

    assert not report.is_gemini_5h_exhausted(threshold_pct=1.0)
    assert not report.is_gemini_weekly_exhausted(threshold_pct=1.0)


def test_parse_text_table_quota_output():
    report = AgyQuotaService.parse_quota_output(SAMPLE_TEXT_OUTPUT)
    assert report is not None
    assert report.gemini_weekly is not None
    assert report.gemini_5h is not None

    assert report.gemini_weekly_pct == 34.0
    assert report.gemini_5h_pct == 62.0
    assert report.claude_weekly_pct == 45.0
    assert report.claude_5h_pct == 100.0


def test_quota_exhaustion_detection():
    exhausted_report = AgyQuotaReport(
        gemini_weekly=QuotaBucket(
            id="gemini-weekly",
            name="Weekly Limit Remaining",
            window="weekly",
            remaining_fraction=0.005,
        ),
        gemini_5h=QuotaBucket(
            id="gemini-5h",
            name="Five Hour Limit Remaining",
            window="5h",
            remaining_fraction=0.0,
            reset_time=datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
        ),
    )
    assert exhausted_report.is_gemini_5h_exhausted(threshold_pct=1.0)
    assert exhausted_report.is_gemini_weekly_exhausted(threshold_pct=1.0)
    assert exhausted_report.gemini_5h_pct == 0.0
    assert exhausted_report.gemini_weekly_pct == 0.5


def test_parse_iso_datetime():
    dt1 = parse_iso_datetime("2026-10-15T06:36:17Z")
    assert dt1 is not None
    assert dt1.year == 2026
    assert dt1.month == 10
    assert dt1.day == 15
    assert dt1.hour == 6

    dt2 = parse_iso_datetime("2026-10-15 13:36 +0700")
    assert dt2 is not None
    assert dt2.year == 2026

    assert parse_iso_datetime("") is None
    assert parse_iso_datetime(None) is None
