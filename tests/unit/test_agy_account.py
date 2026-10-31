"""
Unit tests for agy-account quota checker command and parsing logic.
"""

from datetime import datetime, timezone
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

# Load agy-account module directly from /home/daominhwysi/.local/bin/agy-account
script_path = Path("/home/daominhwysi/.local/bin/agy-account")
loader = SourceFileLoader("agy_account", str(script_path))
spec = importlib.util.spec_from_loader("agy_account", loader)
agy_account = importlib.util.module_from_spec(spec)
loader.exec_module(agy_account)

SAMPLE_JSON = """{
  "conversation_id": "",
  "status": "SUCCESS",
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
              "remaining_fraction": 0.65,
              "reset_time": "2026-11-01T05:08:06Z"
            },
            {
              "id": "gemini-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 0.85,
              "reset_time": "2026-10-30T16:06:24Z"
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
              "remaining_fraction": 1.0,
              "reset_time": "2026-11-06T13:18:48Z"
            },
            {
              "id": "3p-5h",
              "name": "Five Hour Limit Remaining",
              "window": "5h",
              "remaining_fraction": 1.0,
              "reset_time": "2026-10-30T18:18:48Z"
            }
          ]
        }
      ]
    }
  }
}"""

SAMPLE_TSV = """Gemini Models\tWeekly Limit Remaining\t65%\t2026-11-01T05:08:06Z
Gemini Models\tFive Hour Limit Remaining\t85%\t2026-10-30T16:06:24Z
Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-06T13:18:48Z
Claude and GPT models\tFive Hour Limit Remaining\t100%\t2026-10-30T18:18:48Z
"""


def test_parse_quota_data_json():
    data = agy_account.parse_quota_data("test_acc", "test@gmail.com", SAMPLE_JSON)
    assert data["account"] == "test_acc"
    assert data["email"] == "test@gmail.com"
    assert len(data["groups"]) == 2

    gemini_group = data["groups"][0]
    assert gemini_group["name"] == "Gemini Models"
    assert len(gemini_group["buckets"]) == 2

    b_weekly = gemini_group["buckets"][0]
    assert b_weekly["remaining_pct"] == 65.0
    assert b_weekly["window"] == "weekly"

    b_5h = gemini_group["buckets"][1]
    assert b_5h["remaining_pct"] == 85.0
    assert b_5h["window"] == "5h"


def test_parse_quota_data_tsv_fallback():
    data = agy_account.parse_quota_data("test_acc", "test@gmail.com", SAMPLE_TSV)
    assert data["account"] == "test_acc"
    assert len(data["groups"]) >= 2

    # Verify Gemini Models
    gemini_buckets = None
    for g in data["groups"]:
        if g["name"] == "Gemini Models":
            gemini_buckets = g["buckets"]
            break
    assert gemini_buckets is not None
    assert len(gemini_buckets) == 2


def test_sort_buckets_places_5h_first():
    buckets = [
        {"id": "gemini-weekly", "name": "Weekly Limit Remaining", "window": "weekly"},
        {"id": "gemini-5h", "name": "Five Hour Limit Remaining", "window": "5h"},
    ]
    sorted_b = agy_account.sort_buckets(buckets)
    assert sorted_b[0]["window"] == "5h"
    assert sorted_b[1]["window"] == "weekly"


def test_render_meter():
    m1 = agy_account.render_meter(1.0, width=10)
    assert m1 == "[██████████]"
    m2 = agy_account.render_meter(0.5, width=10)
    assert m2 == "[█████░░░░░]"
    m3 = agy_account.render_meter(0.0, width=10)
    assert m3 == "[░░░░░░░░░░]"


def test_format_cooldown_and_local_time():
    future_dt = datetime.now(timezone.utc)
    # Passed/current returns Full
    assert agy_account.format_cooldown(future_dt) == "Full"

    # None returns N/A
    assert agy_account.format_cooldown(None) == "N/A"
    assert agy_account.format_local_time(None) == ""


def test_cmd_quota_single_and_json(capsys, tmp_path):
    mock_quota_res = {
        "account": "test_acc",
        "email": "test@gmail.com",
        "groups": [
            {
                "name": "Gemini Models",
                "buckets": [
                    {
                        "id": "gemini-5h",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.9,
                        "remaining_pct": 90.0,
                        "cooldown_str": "in 1h 00m",
                        "local_time_str": "10-30 22:00 +07",
                    }
                ],
            }
        ],
    }

    test_token = tmp_path / "test_acc.token"
    test_token.write_text("{}", encoding="utf-8")

    with patch.object(agy_account, "fetch_account_quota", return_value=mock_quota_res), \
         patch.object(agy_account, "get_active_profile_name", return_value="test_acc"), \
         patch.object(agy_account, "ACCOUNTS_DIR", tmp_path):
        # Test JSON output
        rc_json = agy_account.cmd_quota("test_acc", as_json=True)
        assert rc_json == 0
        captured_json = capsys.readouterr().out
        assert '"account": "test_acc"' in captured_json
        assert '"remaining_pct": 90.0' in captured_json

        # Test Text table output
        rc_text = agy_account.cmd_quota("test_acc", as_json=False)
        assert rc_text == 0
        captured_text = capsys.readouterr().out
        assert "Profile 'test_acc'" in captured_text
        assert "90.0%" in captured_text


def test_cmd_quota_all_accounts(capsys, tmp_path):
    mock_res_1 = {
        "account": "acc1",
        "email": "acc1@gmail.com",
        "groups": [
            {
                "name": "Gemini Models",
                "buckets": [
                    {
                        "id": "gemini-5h",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.8,
                        "remaining_pct": 80.0,
                        "cooldown_str": "in 1h",
                    },
                    {
                        "id": "gemini-weekly",
                        "name": "Weekly Limit Remaining",
                        "window": "weekly",
                        "remaining_fraction": 0.6,
                        "remaining_pct": 60.0,
                    },
                ],
            }
        ],
    }
    mock_res_2 = {
        "account": "acc2",
        "email": "acc2@gmail.com",
        "groups": [
            {
                "name": "Gemini Models",
                "buckets": [
                    {
                        "id": "gemini-5h",
                        "name": "Five Hour Limit Remaining",
                        "window": "5h",
                        "remaining_fraction": 0.5,
                        "remaining_pct": 50.0,
                        "cooldown_str": "in 2h",
                    },
                    {
                        "id": "gemini-weekly",
                        "name": "Weekly Limit Remaining",
                        "window": "weekly",
                        "remaining_fraction": 0.4,
                        "remaining_pct": 40.0,
                    },
                ],
            }
        ],
    }

    (tmp_path / "acc1.token").write_text("{}", encoding="utf-8")
    (tmp_path / "acc2.token").write_text("{}", encoding="utf-8")

    def mock_fetch(acc, **kwargs):
        return mock_res_1 if acc == "acc1" else mock_res_2

    with patch.object(agy_account, "fetch_account_quota", side_effect=mock_fetch), \
         patch.object(agy_account, "get_active_profile_name", return_value="acc1"), \
         patch.object(agy_account, "ACCOUNTS_DIR", tmp_path):
        rc = agy_account.cmd_quota(all_accounts=True, as_json=False)
        assert rc == 0
        captured = capsys.readouterr().out
        assert "SAVED ACCOUNT PROFILES QUOTA OVERVIEW" in captured
        assert "acc1" in captured
        assert "acc2" in captured
        assert "80.0%" in captured
        assert "50.0%" in captured
