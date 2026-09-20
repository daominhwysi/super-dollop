import json
import re
import sys
from pathlib import Path
from typing import List, Tuple
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sequence_labelling.annotator.editor import SearchReplacePatcher, SearchReplaceBlock
from sequence_labelling.annotator.xml_cleaner import XMLCleaner

class BeforePatcher:
    """Baseline sequential patcher (Level 1-3 only, top-to-bottom, no redundancy guard)."""
    @classmethod
    def apply_blocks(cls, original_text: str, blocks: List[SearchReplaceBlock]) -> Tuple[str, int, List[str]]:
        current_text = original_text
        applied_count = 0
        failed_reasons = []

        for idx, block in enumerate(blocks, start=1):
            search_str = block.search_text
            replace_str = block.replace_text

            # Level 1: Exact
            if search_str in current_text:
                current_text = current_text.replace(search_str, replace_str, 1)
                applied_count += 1
                continue

            # Level 2: Line-trimmed
            search_lines = [line.rstrip() for line in search_str.splitlines()]
            normalized_search = "\n".join(search_lines)
            doc_lines = [line.rstrip() for line in current_text.splitlines()]
            normalized_doc = "\n".join(doc_lines)

            if normalized_search in normalized_doc:
                start_line_idx = -1
                for i in range(len(doc_lines) - len(search_lines) + 1):
                    if doc_lines[i : i + len(search_lines)] == search_lines:
                        start_line_idx = i
                        break
                if start_line_idx != -1:
                    orig_raw_lines = current_text.splitlines(keepends=True)
                    end_line_idx = start_line_idx + len(search_lines)
                    rep_lines = [line + "\n" for line in replace_str.splitlines()]
                    if rep_lines and not replace_str.endswith("\n"):
                        rep_lines[-1] = rep_lines[-1].rstrip("\n")
                    patched_lines = (
                        orig_raw_lines[:start_line_idx]
                        + rep_lines
                        + orig_raw_lines[end_line_idx:]
                    )
                    current_text = "".join(patched_lines)
                    applied_count += 1
                    continue

            # Level 3: Fuzzy
            fuzzy_patched = SearchReplacePatcher._apply_fuzzy_whitespace_replace(
                current_text, search_str, replace_str
            )
            if fuzzy_patched is not None:
                current_text = fuzzy_patched
                applied_count += 1
                continue

            snippet = search_str[:80].replace("\n", " ")
            failed_reasons.append(
                f"Block #{idx} SEARCH block not found in target XML: '{snippet}...'"
            )

        return current_text, applied_count, failed_reasons


def main():
    report_path = Path("backend/logs/revision_resolution_report.json")
    if not report_path.exists():
        print(f"Report {report_path} not found.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ann_dir = Path("data/sequence_labelling_annotated")

    print(f"{'Doc ID':<26} | {'Blocks':<6} | {'Before App/Fail':<16} | {'After App/Fail':<16} | {'Delta':<8} | XML Valid (B/A)")
    print("-" * 90)

    total_blocks = 0
    before_applied_total = 0
    before_failed_total = 0
    after_applied_total = 0
    after_failed_total = 0
    improved_docs_count = 0

    results = data.get("results", [])
    for r in results:
        doc_id = r.get("doc_id")
        rounds = r.get("rounds", [])
        if not rounds:
            continue
        rnd = rounds[0]
        editor_output = rnd.get("editor_output")
        if not editor_output:
            continue

        ann_files = list(ann_dir.glob(f"**/{doc_id}/merged.xml"))
        if not ann_files:
            continue
        orig_xml = ann_files[0].read_text(encoding="utf-8")

        blocks = SearchReplacePatcher.parse_blocks(editor_output)
        if not blocks:
            continue

        b_text, b_app, b_fail = BeforePatcher.apply_blocks(orig_xml, blocks)
        a_text, a_app, a_fail = SearchReplacePatcher.apply_blocks(orig_xml, blocks)

        # Check XML validity
        b_res = XMLCleaner.clean(b_text)
        a_res = XMLCleaner.clean(a_text)
        b_valid = b_res.is_valid_after
        a_valid = a_res.is_valid_after

        total_blocks += len(blocks)
        before_applied_total += b_app
        before_failed_total += len(b_fail)
        after_applied_total += a_app
        after_failed_total += len(a_fail)

        diff = a_app - b_app
        if diff > 0:
            improved_docs_count += 1

        if diff != 0 or len(b_fail) > 0:
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            valid_str = f"{b_valid}/{a_valid}"
            print(f"{doc_id:<26} | {len(blocks):<6} | {b_app:>6} / {len(b_fail):<7} | {a_app:>6} / {len(a_fail):<7} | {diff_str:<8} | {valid_str}")

    print("-" * 90)
    print(f"SUMMARY ({len(results)} candidate docs, {total_blocks} total diff blocks):")
    print(f"  Before Patcher: {before_applied_total} applied, {before_failed_total} failed ({before_applied_total/total_blocks*100:.1f}% success)")
    print(f"  After Patcher:  {after_applied_total} applied, {after_failed_total} failed ({after_applied_total/total_blocks*100:.1f}% success)")
    print(f"  Net Improvement: +{after_applied_total - before_applied_total} additional patches successfully applied")
    print(f"  Documents with increased patch yield: {improved_docs_count}")

if __name__ == "__main__":
    main()
