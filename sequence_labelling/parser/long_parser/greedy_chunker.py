from typing import List, Dict, Any, Optional
from sequence_labelling.parser.long_parser.document_cartographer import CartographyManifest


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

def split_unpaged_document(
    page: Dict[str, Any],
    target_tokens: int = 7000,
    max_tokens: int = 10000,
    overlap_fallback_tokens: int = 750,
) -> List[Dict[str, Any]]:
    """Splits a single unpaged document block into pseudo-pages targeting target_tokens with overlap_fallback_tokens."""
    text = page.get("text", "")
    global_start = page.get("global_start", 0)
    paragraphs = text.split("\n\n")
    para_offsets: List[int] = []
    pos = 0
    for para in paragraphs:
        para_offsets.append(pos)
        pos += len(para) + 2

    sub_pages: List[Dict[str, Any]] = []
    first_idx = 0  # index of the first paragraph in the current window (may be carried-over overlap)
    curr_tokens = 0
    new_paras = 0  # paragraphs in the window that were not already emitted
    p_num = 1

    def emit(last_idx: int) -> None:
        body = "\n\n".join(paragraphs[first_idx:last_idx + 1])
        start = global_start + para_offsets[first_idx]
        sub_pages.append({
            "p": p_num,
            "text": body,
            "estimated_tokens": curr_tokens,
            "global_start": start,
            "global_end": start + len(body),
            "head": "CLEAN",
            "tail": "CLEAN",
        })

    for idx, para in enumerate(paragraphs):
        curr_tokens += max(1, len(para.split()))
        new_paras += 1

        if curr_tokens >= target_tokens:
            emit(idx)
            p_num += 1

            overlap_first = idx + 1
            overlap_toks = 0
            while overlap_first > first_idx + 1 and overlap_toks < overlap_fallback_tokens:
                overlap_first -= 1
                overlap_toks += max(1, len(paragraphs[overlap_first].split()))
            first_idx = overlap_first
            curr_tokens = overlap_toks
            new_paras = 0

    if new_paras:
        emit(len(paragraphs) - 1)
    return sub_pages


def greedy_oversize_chunker(
    page_list: List[Dict[str, Any]],
    target_tokens: int = 7000,
    max_tokens: int = 10000,
    overlap_pages: int = 1,
    cartography: Optional[CartographyManifest] = None,
    overlap_fallback_tokens: int = 750,
) -> List[List[Dict[str, Any]]]:
    """
    Partitions pages into target-sized chunks (~7k target, 10k max ceiling) with configurable safety-net page overlap.
    Uses CartographyManifest clean-cut seams to GUARANTEE that cuts never sever active stimuli
    or cross-page questions unless forced by max_tokens.
    Falls back to overlap_fallback_tokens (750 tokens) when a page is not split.
    """
    if not page_list:
        return []

    # Pre-processing: Split oversized individual pages (> max_tokens) into pseudo-pages
    expanded_pages: List[Dict[str, Any]] = []
    for p in page_list:
        p_toks = p.get("estimated_tokens") or max(50, len(p.get("text", "").split()))
        if p_toks > max_tokens:
            split_pages = split_unpaged_document(p, target_tokens, max_tokens, overlap_fallback_tokens)
            if len(page_list) > 1:
                for sp in split_pages:
                    sp["p"] = p.get("p", 1)
            expanded_pages.extend(split_pages)
        else:
            expanded_pages.append(p)

    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_tokens = 0
    active_context_header = None

    # Determine terminal exam page from cartography (end of all questions and stimuli)
    max_manifest_page = 0
    if cartography is not None:
        for obj in cartography.objects:
            if obj.page_span and obj.type in ("question", "stimulus"):
                max_manifest_page = max(max_manifest_page, obj.page_span[1])

    for page in expanded_pages:
        p_num = page.get("p", 1)
        if active_context_header and not current_chunk:
            page["injected_context_header"] = active_context_header

        current_chunk.append(page)
        page_tokens = page.get("estimated_tokens") or max(50, len(page.get("text", "").split()))
        current_tokens += page_tokens

        # Determine whether to cut chunk at this page boundary
        should_cut = False
        if cartography is not None:
            # Topological boundary snapping: snap cleanly if seam is clean once target budget reached
            is_clean = cartography.is_seam_clean(p_num)
            if is_clean and max_manifest_page > 0 and p_num == max_manifest_page and p_num < len(expanded_pages):
                # Terminal exam boundary cut: cleanly isolate the exam from post-exam appendices / solutions
                should_cut = True
            elif is_clean and current_tokens >= target_tokens:
                should_cut = True
            elif current_tokens >= max_tokens:
                should_cut = True
        else:
            is_in_group = page.get("tail") in ("OPEN_GROUP", "OPEN_THEORY", "OPEN_STIM", "OPEN_STEM", "OPEN_OPT", "CONT_THEORY")
            if current_tokens >= target_tokens and not is_in_group:
                should_cut = True
            elif current_tokens >= max_tokens:
                should_cut = True

        if should_cut:
            if cartography is not None and not cartography.is_seam_clean(p_num):
                # Cut forced at max_tokens across an active stimulus
                straddling_stims = [
                    s.id for s in cartography.stimuli.values()
                    if s.page_span[0] <= p_num <= s.page_span[1]
                ]
                active_context_header = straddling_stims[0] if straddling_stims else None
            elif cartography is None and page.get("tail") in ("OPEN_GROUP", "OPEN_THEORY", "OPEN_STIM", "OPEN_STEM", "OPEN_OPT", "CONT_THEORY"):
                active_context_header = page.get("active_stimulus_id") or page.get("stimulus_id")
            else:
                active_context_header = None

            chunks.append(list(current_chunk))
            if overlap_pages > 0 and len(current_chunk) > overlap_pages:
                overlap_slice = [dict(p) for p in current_chunk[-overlap_pages:]]
                for p in overlap_slice:
                    p["is_overlap_context"] = True
                current_chunk = overlap_slice
                current_tokens = sum(p.get("estimated_tokens") or max(50, len(p.get("text", "").split())) for p in current_chunk)
            elif overlap_pages > 0 and len(current_chunk) >= 1:
                # Page is not split into multiple pages: fallback to overlap_fallback_tokens (750 tokens)
                last_p = dict(current_chunk[-1])
                text = last_p.get("text", "")
                words = text.split()
                if len(words) > overlap_fallback_tokens:
                    trailing_words = words[-overlap_fallback_tokens:]
                    overlap_text = " ".join(trailing_words)
                    probe = " ".join(trailing_words[:4])
                    pos = text.rfind(probe)
                    char_start = pos if pos != -1 else max(0, len(text) - len(overlap_text))
                    g_start = last_p.get("global_start", 0) + char_start
                    g_end = last_p.get("global_end", len(text))
                    last_p["text"] = overlap_text
                    last_p["estimated_tokens"] = overlap_fallback_tokens
                    last_p["is_overlap_context"] = True
                    last_p["global_start"] = g_start
                    last_p["global_end"] = g_end
                    current_chunk = [last_p]
                    current_tokens = overlap_fallback_tokens
                else:
                    last_p["is_overlap_context"] = True
                    current_chunk = [last_p]
                    current_tokens = last_p.get("estimated_tokens", len(words))
            else:
                current_chunk = []
                current_tokens = 0

    if current_chunk and any(not p.get("is_overlap_context") for p in current_chunk):
        chunks.append(current_chunk)

    return chunks
