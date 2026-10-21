#!/usr/bin/env python3
"""
check_cartography_reparse_status.py — Real-time Status Inspector for Cartography Reparse.

Inspects tmux sessions, log files, and output directory to report progress across shards.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = WORKSPACE_DIR / "artifacts" / "ocr_logs"
ANNOTATED_DIR = WORKSPACE_DIR / "data" / "sequence_labelling_annotated_branch"
AUDIT_FILE = WORKSPACE_DIR / "artifacts" / "reports" / "chunk_seam_severing_audit.json"


def get_tmux_sessions() -> Set[str]:
    try:
        out = subprocess.check_output(["tmux", "ls"], stderr=subprocess.DEVNULL, text=True)
        return {line.split(":")[0].strip() for line in out.strip().splitlines()}
    except Exception:
        return set()


def main() -> None:
    print("=" * 65)
    print("📊 CARTOGRAPHY-LOCKED REPARSE REAL-TIME STATUS")
    print("=" * 65)

    tmux_sessions = get_tmux_sessions()
    shard0_active = "luna_carto_shard0" in tmux_sessions
    shard1_active = "luna_carto_shard1" in tmux_sessions

    print(f"🖥️  Tmux Shard 0 (luna_carto_shard0): {'🟢 ACTIVE' if shard0_active else '🔴 STOPPED'}")
    print(f"🖥️  Tmux Shard 1 (luna_carto_shard1): {'🟢 ACTIVE' if shard1_active else '🔴 STOPPED'}")
    print("-" * 65)

    if not AUDIT_FILE.exists():
        print(f"Audit file not found: {AUDIT_FILE}")
        return

    audit_data = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    targets = audit_data.get("results", [])
    total_targets = len(targets)

    completed = []
    in_progress = []
    pending = []
    pass_count = 0
    needs_rev_count = 0
    discard_count = 0
    scores = []

    for item in targets:
        doc_id = item["doc_id"]
        doc_dir = ANNOTATED_DIR / doc_id
        merged_xml = doc_dir / "merged.xml"
        merged_json = doc_dir / "merged.json"
        audit_file = doc_dir / "audit_report.json"

        if merged_xml.exists() and merged_json.exists():
            try:
                jdata = json.loads(merged_json.read_text(encoding="utf-8"))
                if jdata.get("cartography_locked"):
                    completed.append((doc_id, item.get("subject", "Unknown"), jdata))
                    dec = jdata.get("review_decision")
                    if audit_file.exists():
                        try:
                            adata = json.loads(audit_file.read_text(encoding="utf-8"))
                            dec = adata.get("decision", dec)
                            if adata.get("overall_score") is not None:
                                scores.append(float(adata["overall_score"]))
                        except Exception:
                            pass
                    if dec == "PASS":
                        pass_count += 1
                    elif dec == "NEEDS_REVISION":
                        needs_rev_count += 1
                    elif dec == "DISCARD":
                        discard_count += 1
                    continue
            except Exception:
                pass

        chunks_dir = doc_dir / "chunks"
        if chunks_dir.exists() and list(chunks_dir.glob("chunk_*.json")):
            in_progress.append((doc_id, item.get("subject", "Unknown")))
        else:
            pending.append((doc_id, item.get("subject", "Unknown")))

    avg_score = (sum(scores) / len(scores)) if scores else 0.0

    print(f"Total Target Documents: {total_targets}")
    print(f"  ✅ Completed: {len(completed)} (🟢 PASS: {pass_count} | ⚠️ NEEDS_REV: {needs_rev_count} | ❌ DISCARD: {discard_count} | Avg Score: {avg_score:.1f})")
    print(f"  ⏳ In Progress (Active Chunks):   {len(in_progress)}")
    print(f"  💤 Pending:                      {len(pending)}")
    print("-" * 65)

    # Subject breakdown
    subj_total: Dict[str, int] = {}
    subj_done: Dict[str, int] = {}
    for item in targets:
        s = item.get("subject", "Unknown")
        subj_total[s] = subj_total.get(s, 0) + 1
    for doc_id, s, _ in completed:
        subj_done[s] = subj_done.get(s, 0) + 1

    print("Breakdown by Subject:")
    for s, tot in sorted(subj_total.items(), key=lambda x: (x[0] == "Toán", -x[1])):
        done = subj_done.get(s, 0)
        pct = (done / tot) * 100 if tot else 0
        status_symbol = "✅" if done == tot else ("⏳" if done > 0 else "▫️")
        print(f"  {status_symbol} {s:12s}: {done:2d}/{tot:2d} ({pct:5.1f}%)")

    print("-" * 65)

    # Check recent logs
    for shard_idx in (0, 1):
        log_file = LOG_DIR / f"reparse_carto_shard{shard_idx}.log"
        if log_file.exists():
            lines = [l.strip() for l in log_file.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            last_lines = lines[-4:] if len(lines) >= 4 else lines
            print(f"Recent Shard {shard_idx} Activity:")
            for l in last_lines:
                print(f"  [{shard_idx}] {l[:100]}")
    print("=" * 65)


if __name__ == "__main__":
    main()
