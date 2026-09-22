#!/usr/bin/env python3
"""
Batch Section Classification & Branch Repair with OpenAI Codex (GPT-5.6 Luna).

Extracts all <section> tags from annotated documents, deduplicates unique section strings,
batches them up to a target token budget (Xk tokens), and uses Codex to classify whether
each section is VALID (EXAM_SECTION) or INVALID (junk: metadata titles, footers, reading directions,
textbook theory, answer keys, absorbed question content).

For documents with invalid or mislabelled sections, surgically unwraps or adjusts the tags
and writes the repaired merged.xml and merged.json to the branch folder
(data/sequence_labelling_annotated_branch), strictly preserving the canonical dataset.

Usage:
  # Dry-run inspection of batches and token counts:
  uv run python scripts/classify_sections_codex.py --dry-run --target-tokens 4000

  # Run test with limit:
  uv run python scripts/classify_sections_codex.py --limit 10 --thinking medium

  # Full run with 4k token batches and medium thinking:
  uv run python scripts/classify_sections_codex.py --target-tokens 4000 --thinking medium --save-branch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_labelling.llm.deepseek_client import chat


# ── Token Counting ─────────────────────────────────────────────────────────────

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _TIKTOKEN_ENC = None


def count_tokens(text: str) -> int:
    """Counts tokens using o200k_base if available, else accurate fallback."""
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        try:
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            pass
    # Fallback heuristic: ~3.2 characters per token for Vietnamese/Markdown
    return max(1, int(len(text) / 3.2))


# ── Classification Prompt ──────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are an expert NLP sequence labeling auditor and educational exam structure analyst.
Your task is to classify whether candidate <section> tag contents extracted from exam OCR text are VALID exam section headers or INVALID junk/mislabelled elements based on the project criteria.

## Project Criteria:

1. VALID (EXAM_SECTION):
   - A genuine structural section or part division within an examination paper that partitions questions into distinct blocks.
   - Examples:
     * "PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn"
     * "PHẦN II. Câu trắc nghiệm đúng sai"
     * "PHẦN III. Câu trắc nghiệm trả lời ngắn"
     * "PART 5: INCOMPLETE SENTENCES"
     * "PHẦN TỰ LUẬN (3,0 điểm)"
     * "Chủ đề 1: Động lực học chất điểm có 12 câu hỏi từ 1 đến 12"
     * "Section A: Listening Comprehension"
   - General instructions directing the candidate for that entire section (e.g., "Thí sinh trả lời từ câu 1 đến câu 12. Mỗi câu hỏi thí sinh chỉ chọn một phương án.") may accompany the section header and are VALID.

2. INVALID Categories:
   - EXAM_HEADER_METADATA: Exam title, ministry / school name, grade, subject, exam code, time allowed, general examination cover text.
     (e.g., "SỞ GD&ĐT HÀ NỘI", "TRƯỜNG THPT CHUYÊN...", "ĐỀ THI THỬ TỐT NGHIỆP THPT NĂM 2025", "Môn: Toán", "Thời gian làm bài: 90 phút", "Mã đề: 101").
   - DOCUMENT_FOOTER_END: End markers or exam footers.
     (e.g., "——— Hết ———", "——— Hết phần thi Tiếng Anh ———", "**-------- HẾT --------**", "Thí sinh không được sử dụng tài liệu. Cán bộ coi thi không giải thích gì thêm.").
   - STIMULUS_READING_DIRECTION: Instructions tied to a specific reading passage, cloze blank, or question subset.
     (e.g., "Read the following passage and mark the letter A, B, C or D...", "Read the passage below and choose A, B, C or D to fill in each blank from 631 to 635.", "*Read the passage carefully.*").
   - THEORY_TEXTBOOK_HEADING: Textbook chapters, theory lessons, exercise classification tags, or study guide headings.
     (e.g., "## 1.2. Tiểu sử và tiến trình hoạt động cách mạng...", "## 2.2. Kịch", "**Dạng 7: Kịch**", "### *Tiểu thuyết chương hồi*").
   - SOLUTION_ANSWER_KEY_HEADING: Answer key, explanation, or solution section titles.
     (e.g., "# HƯỚNG DẪN GIẢI CHI TIẾT", "# BẢNG ĐÁP ÁN", "## A. ĐÁP ÁN").
   - ABSORBED_QUESTION_CONTENT: Stems, question numbers, options, sub-questions, or solutions mistakenly enclosed inside <section>.
   - FRAGMENT_NOISE: Isolated markdown horizontal rules (---, ***), page numbers, or random punctuation noise.

## Output Format:
You MUST respond ONLY with a valid JSON array of objects. Do NOT include markdown text outside the JSON array.
Each object in the array must strictly have these fields:
- "id": string matching the input item id (e.g. "SEC_0001")
- "decision": either "VALID" or "INVALID"
- "category": one of ["EXAM_SECTION", "EXAM_HEADER_METADATA", "DOCUMENT_FOOTER_END", "STIMULUS_READING_DIRECTION", "THEORY_TEXTBOOK_HEADING", "SOLUTION_ANSWER_KEY_HEADING", "ABSORBED_QUESTION_CONTENT", "FRAGMENT_NOISE"]
- "reason": concise explanation (1-2 sentences) of why it is valid or invalid
- "clean_header": if the section text contains BOTH metadata and a real section header (e.g. school name followed by "PHẦN I. Câu trắc nghiệm..."), extract the exact verbatim substring corresponding to the genuine section header. Otherwise, set to null.
"""


def _clean_json_response(raw_resp: str) -> str:
    """Strips Markdown fences or surrounding conversational wrappers."""
    text = raw_resp.strip()
    # Find ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Find outer array bracket
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


# ── Extraction & Occurrence Tracking ──────────────────────────────────────────

def compute_text_hash(text: str) -> str:
    """Normalized MD5 hash of section text."""
    norm = " ".join(text.strip().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


class SectionOccurrence:
    def __init__(self, doc_id: str, raw_xml: str, start: int, end: int, inner_text: str):
        self.doc_id = doc_id
        self.raw_xml = raw_xml
        self.start = start
        self.end = end
        self.inner_text = inner_text
        self.text_hash = compute_text_hash(inner_text)


def collect_all_sections(annotated_dir: Path) -> Tuple[List[SectionOccurrence], Dict[str, Dict[str, Any]]]:
    """
    Scans all merged.xml files and returns:
    1. List of all SectionOccurrence objects.
    2. Dictionary of unique section entries keyed by text_hash:
       {
           text_hash: {
               "text_hash": str,
               "inner_text": str,
               "tokens": int,
               "doc_ids": list,
               "occurrence_count": int,
           }
       }
    """
    occurrences: List[SectionOccurrence] = []
    unique_sections: Dict[str, Dict[str, Any]] = {}

    section_re = re.compile(r"<section>(.*?)</section>", re.DOTALL)

    xml_files = sorted(annotated_dir.glob("*/merged.xml"))
    for xml_path in xml_files:
        doc_id = xml_path.parent.name
        try:
            content = xml_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"⚠️ Error reading {xml_path}: {exc}")
            continue

        for match in section_re.finditer(content):
            start, end = match.span()
            inner = match.group(1).strip()
            if not inner:
                continue

            occ = SectionOccurrence(
                doc_id=doc_id,
                raw_xml=match.group(0),
                start=start,
                end=end,
                inner_text=inner,
            )
            occurrences.append(occ)

            h = occ.text_hash
            if h not in unique_sections:
                unique_sections[h] = {
                    "text_hash": h,
                    "inner_text": inner,
                    "tokens": count_tokens(inner),
                    "doc_ids": [doc_id],
                    "occurrence_count": 1,
                }
            else:
                unique_sections[h]["occurrence_count"] += 1
                if doc_id not in unique_sections[h]["doc_ids"]:
                    unique_sections[h]["doc_ids"].append(doc_id)

    return occurrences, unique_sections


# ── Batch Packing ─────────────────────────────────────────────────────────────

def create_batches(
    items_to_classify: List[Dict[str, Any]],
    target_tokens: int = 4000,
) -> List[List[Dict[str, Any]]]:
    """
    Packs items into batches where the estimated prompt token count
    does not exceed target_tokens (with reasonable overhead buffer).
    """
    base_prompt_tokens = count_tokens(CLASSIFIER_SYSTEM_PROMPT) + 150
    batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_tokens = base_prompt_tokens

    for item in items_to_classify:
        # Approximate formatted XML representation tokens
        item_text = item["inner_text"]
        item_tokens = count_tokens(item_text) + 20

        # If single item is huge, give it its own batch
        if item_tokens >= target_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = base_prompt_tokens
            batches.append([item])
            continue

        if current_tokens + item_tokens > target_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [item]
            current_tokens = base_prompt_tokens + item_tokens
        else:
            current_batch.append(item)
            current_tokens += item_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def format_batch_user_prompt(batch: List[Dict[str, Any]]) -> str:
    """Formats a list of candidate items into an XML user prompt."""
    lines = ["Classify the following candidate <section> tag contents:\n"]
    for idx, item in enumerate(batch, 1):
        item_id = item.get("assigned_id") or f"SEC_{idx:04d}"
        item["assigned_id"] = item_id
        lines.append(f'<item id="{item_id}">\n{item["inner_text"]}\n</item>\n')
    return "\n".join(lines)


# ── Codex Classifier Engine ───────────────────────────────────────────────────

def run_codex_batch(
    batch: List[Dict[str, Any]],
    model: str = "gpt-5.6-luna",
    provider: str = "codex",
    thinking: Any = "medium",
    max_retries: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Sends a batch of sections to Codex and parses the JSON response.
    Returns a dictionary mapping assigned_id -> classification dict.
    """
    user_prompt = format_batch_user_prompt(batch)
    expected_ids = {item["assigned_id"] for item in batch}

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            raw_response = chat(
                prompt=user_prompt,
                system=CLASSIFIER_SYSTEM_PROMPT,
                model=model,
                provider=provider,
                thinking=thinking,
            )
            dur = time.time() - t0

            cleaned = _clean_json_response(raw_response)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed)}")

            results: Dict[str, Dict[str, Any]] = {}
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                item_id = entry.get("id")
                if item_id in expected_ids:
                    results[item_id] = {
                        "decision": str(entry.get("decision", "INVALID")).upper(),
                        "category": str(entry.get("category", "EXAM_HEADER_METADATA")),
                        "reason": str(entry.get("reason", "")),
                        "clean_header": entry.get("clean_header"),
                        "latency_sec": round(dur, 2),
                    }

            # Check if all items were classified
            missing_ids = expected_ids - set(results.keys())
            if missing_ids and attempt < max_retries:
                print(f"   [Retry {attempt}/{max_retries}] Missing {len(missing_ids)} items in response. Retrying...")
                time.sleep(2)
                continue

            # Fill any remaining missing with defensive defaults
            for mid in missing_ids:
                results[mid] = {
                    "decision": "INVALID",
                    "category": "FRAGMENT_NOISE",
                    "reason": "Missing from model response output; marked invalid defensively.",
                    "clean_header": None,
                    "latency_sec": round(dur, 2),
                }

            return results

        except Exception as exc:
            print(f"   [Attempt {attempt}/{max_retries} Failed]: {exc}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"Codex classification failed after {max_retries} attempts: {exc}")

    return {}


# ── Branch Surgical Editor ────────────────────────────────────────────────────

def apply_section_edits_to_xml(
    original_xml: str,
    classifications_by_hash: Dict[str, Dict[str, Any]],
) -> Tuple[str, int, int]:
    """
    Surgically unwraps or adjusts <section> tags in original_xml based on classifications.
    Returns: (edited_xml, invalid_unwrapped_count, adjusted_headers_count)
    """
    section_pattern = re.compile(r"<section>(.*?)</section>", re.DOTALL)
    unwrapped_count = 0
    adjusted_count = 0

    def replace_section(m: re.Match) -> str:
        nonlocal unwrapped_count, adjusted_count
        raw_inner = m.group(1)
        clean_inner = raw_inner.strip()
        h = compute_text_hash(clean_inner)
        cls_info = classifications_by_hash.get(h)

        if not cls_info:
            # Not classified, keep untouched
            return m.group(0)

        decision = cls_info.get("decision", "VALID")
        clean_header = cls_info.get("clean_header")

        if decision == "VALID":
            # Genuine section: keep untouched
            return m.group(0)

        # Handle INVALID
        # Case A: Partial/mixed header where clean_header is an exact substring
        if clean_header and isinstance(clean_header, str):
            clean_hdr_str = clean_header.strip()
            if clean_hdr_str and clean_hdr_str in raw_inner:
                idx = raw_inner.find(clean_hdr_str)
                prefix = raw_inner[:idx]
                suffix = raw_inner[idx + len(clean_hdr_str):]
                adjusted_count += 1
                return f"{prefix}<section>{clean_hdr_str}</section>{suffix}"

        # Case B: Pure junk -> unwrap <section> tag completely, keep verbatim text
        unwrapped_count += 1
        return raw_inner

    edited_xml = section_pattern.sub(replace_section, original_xml)
    return edited_xml, unwrapped_count, adjusted_count


def save_document_to_branch(
    doc_id: str,
    original_xml_path: Path,
    edited_xml: str,
    raw_ocr_path: Optional[Path],
    branch_dir: Path,
    unwrapped_count: int,
    adjusted_count: int,
) -> Dict[str, Any]:
    """
    Saves edited merged.xml and re-parsed merged.json to the branch folder.
    """
    doc_branch_dir = branch_dir / doc_id
    doc_branch_dir.mkdir(parents=True, exist_ok=True)

    branch_xml_path = doc_branch_dir / "merged.xml"
    branch_xml_path.write_text(edited_xml, encoding="utf-8")

    # Re-generate structured merged.json
    raw_ocr_text = ""
    if raw_ocr_path and raw_ocr_path.exists():
        try:
            raw_ocr_text = raw_ocr_path.read_text(encoding="utf-8")
        except Exception:
            pass

    branch_json_path = doc_branch_dir / "merged.json"
    json_error = None
    questions_count = 0
    stimuli_count = 0

    try:
        from sequence_labelling.annotator.annotate_ocr import parse_xml_annotations
        from sequence_labelling.parser.long_parser.anchored_xml_llm_parser import parse_xml_with_anchors
        spans, stimuli, questions = parse_xml_with_anchors(raw_ocr_text, edited_xml)
        questions_count = len(questions)
        stimuli_count = len(stimuli)
        json_data = {
            "document_id": doc_id,
            "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "questions_count": questions_count,
            "questions": questions,
            "stimuli": stimuli,
        }
        branch_json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        json_error = str(exc)
        print(f"   ⚠️ Could not generate {branch_json_path}: {exc}")

    # Copy audit_report.json if present in canonical directory
    orig_audit = original_xml_path.parent / "audit_report.json"
    if orig_audit.exists() and not (doc_branch_dir / "audit_report.json").exists():
        try:
            shutil.copy2(orig_audit, doc_branch_dir / "audit_report.json")
        except Exception:
            pass

    # Save cleanup metadata
    state = {
        "doc_id": doc_id,
        "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unwrapped_junk_sections": unwrapped_count,
        "adjusted_headers": adjusted_count,
        "branch_xml_path": str(branch_xml_path),
        "branch_json_path": str(branch_json_path),
        "merged_json_error": json_error,
        "questions_count": questions_count,
        "stimuli_count": stimuli_count,
    }
    (doc_branch_dir / "section_cleanup_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return state


# ── Main Runner ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Section Classification and Branch Repair using OpenAI Codex (GPT-5.6 Luna)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sequence_labelling_annotated",
        help="Path to canonical annotated directory (default: data/sequence_labelling_annotated)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sequence_labelling_input_data",
        help="Path to raw markdown directory (default: data/sequence_labelling_input_data)",
    )
    parser.add_argument(
        "--branch-dir",
        type=Path,
        default=REPO_ROOT / "data" / "sequence_labelling_annotated_branch",
        help="Path to branch output directory (default: data/sequence_labelling_annotated_branch)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "section_classification_report.json",
        help="Path to global classification report JSON (default: data/section_classification_report.json)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=4000,
        help="Target token budget per batch (default: 4000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.6-luna",
        help="Codex model name (default: gpt-5.6-luna)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="codex",
        help="LLM provider (default: codex)",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="medium",
        help="Codex reasoning effort: none, low, medium, high (default: medium)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of unique section items to classify",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=None,
        help="Limit number of batches to process",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume; re-classify everything from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and batch sections without invoking Codex or modifying disk",
    )
    parser.add_argument(
        "--save-branch",
        action="store_true",
        default=True,
        help="Surgically repair documents with invalid sections and save to branch directory",
    )
    parser.add_argument(
        "--no-save-branch",
        dest="save_branch",
        action="store_false",
        help="Only produce the classification report without writing repaired files to branch",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BATCH SECTION CLASSIFIER & BRANCH REPAIR (CODEX GPT-5.6 LUNA)")
    print(f"  Input Directory   : {args.input_dir}")
    print(f"  Branch Directory  : {args.branch_dir}")
    print(f"  Report Output     : {args.output}")
    print(f"  Target Tokens     : {args.target_tokens} tokens/batch")
    print(f"  Codex Model       : {args.model} (thinking: {args.thinking})")
    print(f"  Provider          : {args.provider}")
    print(f"  Save to Branch    : {'YES' if args.save_branch and not args.dry_run else 'NO (Dry Run / Report Only)'}")
    print("=" * 80)

    # 1. Collect all sections from annotated directory
    print("\n[Step 1/4] Extracting <section> tags from annotated documents...")
    occurrences, unique_sections = collect_all_sections(args.input_dir)
    total_occ = len(occurrences)
    total_uniq = len(unique_sections)
    total_tokens = sum(item["tokens"] for item in unique_sections.values())

    print(f"   -> Found {total_occ:,} total section occurrences across dataset.")
    print(f"   -> Deduplicated into {total_uniq:,} unique section strings.")
    print(f"   -> Total estimated section tokens: ~{total_tokens:,} tokens.")

    # 2. Check existing report for resume
    cached_classifications: Dict[str, Dict[str, Any]] = {}
    if not args.no_resume and args.output.exists():
        try:
            old_report = json.loads(args.output.read_text(encoding="utf-8"))
            cached_classifications = old_report.get("unique_classifications", {})
            print(f"   -> Loaded {len(cached_classifications):,} existing classifications from {args.output.name}.")
        except Exception as exc:
            print(f"   ⚠️ Could not read existing report {args.output}: {exc}")

    # Determine pending items to classify
    pending_items: List[Dict[str, Any]] = []
    for h, item in unique_sections.items():
        if h not in cached_classifications:
            pending_items.append(item)

    print(f"   -> {len(cached_classifications):,} already classified, {len(pending_items):,} pending.")

    if args.limit is not None and args.limit > 0:
        pending_items = pending_items[: args.limit]
        print(f"   -> Applying --limit: processing {len(pending_items)} items.")

    # 3. Create token-targeted batches
    batches = create_batches(pending_items, target_tokens=args.target_tokens)
    if args.batch_limit is not None and args.batch_limit > 0:
        batches = batches[: args.batch_limit]
        print(f"   -> Applying --batch-limit: processing {len(batches)} batches.")

    print(f"\n[Step 2/4] Formed {len(batches)} batches (Target: ~{args.target_tokens} tokens/batch).")
    for i, b in enumerate(batches[:5], 1):
        b_tokens = sum(it["tokens"] for it in b)
        print(f"   Batch {i:02d}: {len(b)} sections (~{b_tokens} tokens)")
    if len(batches) > 5:
        print(f"   ... and {len(batches) - 5} more batches.")

    if args.dry_run:
        print("\n✅ [Dry Run Complete] No LLM calls made and no files modified.")
        sys.exit(0)

    # 4. Process batches with Codex
    print(f"\n[Step 3/4] Running Codex classification on {len(batches)} batches...")
    new_classifications: Dict[str, Dict[str, Any]] = dict(cached_classifications)
    global_item_counter = len(cached_classifications)

    for b_idx, batch in enumerate(batches, 1):
        # Assign unique IDs
        for it in batch:
            global_item_counter += 1
            it["assigned_id"] = f"SEC_{global_item_counter:04d}"

        b_tokens = sum(it["tokens"] for it in batch)
        print(f"\n>> Batch [{b_idx}/{len(batches)}] ({len(batch)} items, ~{b_tokens} tokens)...", end="", flush=True)

        batch_res = run_codex_batch(
            batch=batch,
            model=args.model,
            provider=args.provider,
            thinking=args.thinking,
        )

        valid_in_batch = sum(1 for r in batch_res.values() if r.get("decision") == "VALID")
        invalid_in_batch = len(batch_res) - valid_in_batch
        print(f" Done => {valid_in_batch} VALID, {invalid_in_batch} INVALID")

        # Map batch results back by text_hash
        for it in batch:
            assigned_id = it["assigned_id"]
            h = it["text_hash"]
            res = batch_res.get(assigned_id, {
                "decision": "INVALID",
                "category": "FRAGMENT_NOISE",
                "reason": "Missing output",
                "clean_header": None,
            })
            new_classifications[h] = {
                "text_hash": h,
                "inner_text": it["inner_text"],
                "decision": res["decision"],
                "category": res["category"],
                "reason": res["reason"],
                "clean_header": res.get("clean_header"),
                "occurrence_count": it["occurrence_count"],
                "doc_ids": it["doc_ids"],
            }

        # Incremental checkpoint save
        summary_obj = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "provider": args.provider,
            "thinking": args.thinking,
            "target_tokens": args.target_tokens,
            "total_occurrences_found": total_occ,
            "unique_sections_count": total_uniq,
            "classified_count": len(new_classifications),
            "unique_classifications": new_classifications,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ All batches completed. Total classified: {len(new_classifications):,}/{total_uniq:,}.")

    # 5. Surgical Branch Repair
    if args.save_branch and not args.dry_run:
        print(f"\n[Step 4/4] Surgically repairing documents with invalid sections -> {args.branch_dir}...")
        args.branch_dir.mkdir(parents=True, exist_ok=True)

        # Identify documents that have at least one classified invalid or adjusted section
        affected_doc_ids: Set[str] = set()
        for h, cls_info in new_classifications.items():
            if cls_info.get("decision") == "INVALID" or cls_info.get("clean_header"):
                for d in cls_info.get("doc_ids", []):
                    affected_doc_ids.add(d)

        print(f"   -> Found {len(affected_doc_ids)} documents containing invalid/modified section tags.")

        repaired_docs_count = 0
        total_unwrapped = 0
        total_adjusted = 0

        for doc_id in sorted(affected_doc_ids):
            xml_path = args.input_dir / doc_id / "merged.xml"
            if not xml_path.exists():
                continue

            try:
                orig_xml = xml_path.read_text(encoding="utf-8")
                edited_xml, unwrapped, adjusted = apply_section_edits_to_xml(orig_xml, new_classifications)

                if unwrapped > 0 or adjusted > 0:
                    raw_ocr_path = args.raw_dir / f"{doc_id}.md"
                    save_document_to_branch(
                        doc_id=doc_id,
                        original_xml_path=xml_path,
                        edited_xml=edited_xml,
                        raw_ocr_path=raw_ocr_path,
                        branch_dir=args.branch_dir,
                        unwrapped_count=unwrapped,
                        adjusted_count=adjusted,
                    )
                    repaired_docs_count += 1
                    total_unwrapped += unwrapped
                    total_adjusted += adjusted
            except Exception as exc:
                print(f"   ⚠️ Error processing doc {doc_id}: {exc}")

        print(f"   -> Repaired {repaired_docs_count} documents.")
        print(f"   -> Unwrapped {total_unwrapped} junk <section> tags.")
        print(f"   -> Adjusted {total_adjusted} partial section headers.")
        print(f"   -> All repaired files saved to: {args.branch_dir}")

    # Summary Statistics
    valid_count = sum(1 for c in new_classifications.values() if c.get("decision") == "VALID")
    invalid_count = sum(1 for c in new_classifications.values() if c.get("decision") == "INVALID")

    category_counts: Dict[str, int] = {}
    for c in new_classifications.values():
        cat = c.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\n" + "=" * 80)
    print("FINAL CLASSIFICATION SUMMARY")
    print(f"  Total Unique Sections Classified : {len(new_classifications):,}")
    print(f"  VALID (EXAM_SECTION)             : {valid_count:,} ({valid_count / max(1, len(new_classifications)) * 100:.1f}%)")
    print(f"  INVALID (Junk & Mislabelled)     : {invalid_count:,} ({invalid_count / max(1, len(new_classifications)) * 100:.1f}%)")
    print("\n  Breakdown by Category:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    - {cat:<30}: {cnt:4d} ({cnt / max(1, len(new_classifications)) * 100:.1f}%)")
    print(f"\n  Report saved to                  : {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
