from typing import List, Dict, Any


PAGE_SEPARATOR = "\n\n"


def build_chunk_plan(chunk_pages: List[Dict[str, Any]], chunk_index: int) -> Dict[str, Any]:
    parts: List[str] = []
    ranges: List[Dict[str, Any]] = []
    local_offset = 0

    for page_index, page in enumerate(chunk_pages):
        if page_index:
            parts.append(PAGE_SEPARATOR)
            local_offset += len(PAGE_SEPARATOR)
        page_text = str(page.get("text", ""))
        local_start = local_offset
        parts.append(page_text)
        local_offset += len(page_text)
        ranges.append({
            "page_id": int(page.get("p", page_index + 1)),
            "local_start": local_start,
            "local_end": local_offset,
            "global_start": int(page.get("global_start", 0)),
            "global_end": int(page.get("global_end", len(page_text))),
            "is_overlap": bool(page.get("is_overlap_context")),
        })

    return {
        "chunk_index": chunk_index,
        "raw_chunk_text": "".join(parts),
        "page_ids": [item["page_id"] for item in ranges],
        "overlap_page_ids": [item["page_id"] for item in ranges if item["is_overlap"]],
        "page_offset_ranges": ranges,
    }

def greedy_oversize_chunker(
    page_list: List[Dict[str, Any]],
    target_tokens: int = 12000,
    max_tokens: int = 16000,
    overlap_pages: int = 0
) -> List[List[Dict[str, Any]]]:
    """
    Partitions pages into target-sized chunks (~25k tokens) with configurable safety-net page overlap.
    NEVER cuts inside an active passage or question group unless forced by max_tokens.
    """
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_tokens = 0
    active_context_header = None

    for page in page_list:
        if active_context_header and not current_chunk:
            page["injected_context_header"] = active_context_header

        current_chunk.append(page)
        page_tokens = page.get("estimated_tokens") or max(50, len(page.get("text", "").split()))
        current_tokens += page_tokens

        if current_tokens >= target_tokens:
            is_in_group = page.get("tail") in ("OPEN_GROUP", "OPEN_THEORY", "OPEN_STIM", "OPEN_STEM", "OPEN_OPT", "CONT_THEORY")

            if not is_in_group:
                chunks.append(list(current_chunk))
                if overlap_pages > 0 and len(current_chunk) > overlap_pages:
                    overlap_slice = [dict(p) for p in current_chunk[-overlap_pages:]]
                    for p in overlap_slice:
                        p["is_overlap_context"] = True
                    current_chunk = overlap_slice
                    current_tokens = sum(p.get("estimated_tokens") or max(50, len(p.get("text", "").split())) for p in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0
                active_context_header = None
            elif current_tokens >= max_tokens:
                active_context_header = page.get("active_stimulus_id") or page.get("stimulus_id")
                chunks.append(list(current_chunk))
                if overlap_pages > 0 and len(current_chunk) > overlap_pages:
                    overlap_slice = [dict(p) for p in current_chunk[-overlap_pages:]]
                    for p in overlap_slice:
                        p["is_overlap_context"] = True
                    current_chunk = overlap_slice
                    current_tokens = sum(p.get("estimated_tokens") or max(50, len(p.get("text", "").split())) for p in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0

    if current_chunk and any(not p.get("is_overlap_context") for p in current_chunk):
        chunks.append(current_chunk)

    return chunks
