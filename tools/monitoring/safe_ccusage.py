#!/usr/bin/env python3
"""
safe_ccusage.py — Lock-Safe Antigravity CLI Telemetry Bridge.

Mitigates SQLite database lock errors (SQLITE_BUSY / code 5) encountered by
ccusage when Antigravity CLI or the IDE has active write locks on conversation databases.

Creates an incremental, lock-free byte snapshot in /tmp/antigravity_safe_telemetry/
and invokes ccusage with ANTIGRAVITY_DATA_DIR pointing to the safe shadow mirror.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_DIR = Path(os.path.expanduser("~/.gemini/antigravity-cli/conversations"))
SHADOW_BASE_DIR = Path("/tmp/antigravity_safe_telemetry")
SHADOW_CONV_DIR = SHADOW_BASE_DIR / "conversations"


def sync_shadow_conversations(verbose: bool = False) -> int:
    """
    Incrementally syncs SQLite databases from ~/.gemini/antigravity-cli/conversations
    into /tmp/antigravity_safe_telemetry/conversations using direct binary reads,
    bypassing SQLite POSIX advisory file locks.
    Returns the number of files synced/updated.
    """
    if not SOURCE_DIR.exists():
        return 0

    SHADOW_CONV_DIR.mkdir(parents=True, exist_ok=True)
    synced = 0

    source_files = list(SOURCE_DIR.glob("*.db"))
    for src_path in source_files:
        dst_path = SHADOW_CONV_DIR / src_path.name
        needs_copy = False

        try:
            src_stat = src_path.stat()
            if not dst_path.exists():
                needs_copy = True
            else:
                dst_stat = dst_path.stat()
                if (
                    src_stat.st_mtime > dst_stat.st_mtime
                    or src_stat.st_size != dst_stat.st_size
                ):
                    needs_copy = True

            if needs_copy:
                tmp_dst = dst_path.with_suffix(f".tmp_{os.getpid()}")
                with open(src_path, "rb") as f_in, open(tmp_dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.replace(tmp_dst, dst_path)
                synced += 1
        except Exception as exc:
            if verbose:
                print(f"[safe_ccusage] Warning: Failed to mirror {src_path.name}: {exc}", file=sys.stderr)

    return synced


def find_ccusage_executable() -> List[str]:
    """Locate the ccusage executable binary or runner."""
    # 1. Check bun global bin
    bun_bin = Path(os.path.expanduser("~/.bun/bin/ccusage"))
    if bun_bin.exists() and os.access(bun_bin, os.X_OK):
        return [str(bun_bin)]

    # 2. Check system PATH
    system_path = shutil.which("ccusage")
    if system_path:
        return [system_path]

    # 3. Fallback to npx
    npx_path = shutil.which("npx")
    if npx_path:
        return [npx_path, "ccusage"]

    raise FileNotFoundError("Could not find ccusage executable or npx in PATH.")


def run_safe_ccusage(args: Optional[List[str]] = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """
    Runs ccusage antigravity with given args against the safe shadow directory.
    """
    sync_shadow_conversations()
    ccusage_cmd = find_ccusage_executable()

    env = dict(os.environ)
    env["ANTIGRAVITY_DATA_DIR"] = str(SHADOW_BASE_DIR)

    full_cmd = ccusage_cmd + ["antigravity"] + (args or ["daily"])

    return subprocess.run(
        full_cmd,
        env=env,
        capture_output=capture_output,
        text=True,
    )


def get_safe_antigravity_summary(since: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches JSON daily summary from ccusage and returns parsed dictionary.
    """
    import json
    cmd_args = ["daily", "--json"]
    if since:
        cmd_args.extend(["--since", since])

    proc = run_safe_ccusage(args=cmd_args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ccusage failed (exit {proc.returncode}): {proc.stderr}")

    stdout = proc.stdout.strip()
    # Find JSON payload
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    if json_start != -1 and json_end > json_start:
        return json.loads(stdout[json_start : json_end + 1])
    return {}


def main():
    args = sys.argv[1:]
    if not args:
        args = ["daily"]

    try:
        proc = run_safe_ccusage(args=args, capture_output=False)
        sys.exit(proc.returncode)
    except Exception as exc:
        print(f"Error running safe_ccusage: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
