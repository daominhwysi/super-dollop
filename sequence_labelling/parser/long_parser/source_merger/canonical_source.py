import re
import unicodedata
import difflib
from typing import List, Tuple, Dict, Any, Optional
from sequence_labelling.parser.long_parser.source_merger.types import ChunkInput


class PiecewiseMap:
    """
    Maps chunk-local character offsets to global canonical document character offsets.
    Handles overlapping intervals monotonically.
    """
    def __init__(self, chunk_index: int, local_length: int):
        self.chunk_index = chunk_index
        self.local_length = local_length
        # List of (local_start, local_end, global_start, global_end)
        self.segments: List[Tuple[int, int, int, int]] = []

    def add_segment(self, local_start: int, local_end: int, global_start: int, global_end: int):
        self.segments.append((local_start, local_end, global_start, global_end))
        self.segments.sort(key=lambda item: (item[0], item[1]))

    def map_local_to_global(self, local_offset: int) -> int:
        if local_offset < 0:
            return 0
        if not self.segments:
            return local_offset

        for l_start, l_end, g_start, g_end in self.segments:
            if l_start <= local_offset <= l_end:
                l_span = l_end - l_start
                g_span = g_end - g_start
                if l_span == 0:
                    return g_start
                fraction = (local_offset - l_start) / l_span
                return int(round(g_start + fraction * g_span))

        # Fallback extrapolation
        last_l_start, last_l_end, last_g_start, last_g_end = self.segments[-1]
        if local_offset > last_l_end:
            return last_g_end + (local_offset - last_l_end)
        
        first_l_start, first_l_end, first_g_start, first_g_end = self.segments[0]
        if local_offset < first_l_start:
            return max(0, first_g_start - (first_l_start - local_offset))

        return local_offset

    def map_span_to_global(
        self,
        local_start: int,
        local_end: int,
        source_text: str,
    ) -> Optional[Tuple[int, int]]:
        """Map a local span without bridging unrelated global segments."""
        if local_end < local_start:
            return None

        start = max(0, min(self.local_length, local_start))
        end = max(start, min(self.local_length, local_end))
        touched = [
            segment
            for segment in self.segments
            if segment[0] < end and segment[1] > start
        ]
        if not touched:
            return None

        first = touched[0]
        last = touched[-1]
        if len(touched) > 1:
            for index, (left, right) in enumerate(zip(touched, touched[1:])):
                local_gap = source_text[left[1]:right[0]]
                global_gap = right[2] - left[3]
                global_contiguous = global_gap == len(local_gap)
                local_contiguous = left[1] == right[0] or not local_gap.strip()
                if global_contiguous and local_contiguous:
                    continue

                seam_text = source_text[left[1]:end]
                if start < left[1] <= end and not seam_text.strip():
                    end = left[1]
                    touched = touched[:index + 1]
                    last = left
                    break
                return None

        def map_in(segment: Tuple[int, int, int, int], offset: int) -> int:
            l_start, l_end, g_start, g_end = segment
            if l_end == l_start:
                return g_start
            ratio = (offset - l_start) / (l_end - l_start)
            return int(round(g_start + ratio * (g_end - g_start)))

        return map_in(first, start), map_in(last, end)

def _normalize_string(text: str) -> Tuple[str, List[int]]:
    norm_chars = []
    index_map = []
    
    i = 0
    n = len(text)
    in_ws = False

    while i < n:
        char = text[i]

        if char == '\r':
            if i + 1 < n and text[i + 1] == '\n':
                i += 1
            char = '\n'

        char = unicodedata.normalize('NFC', char)

        if char.isspace():
            if not in_ws:
                norm_chars.append(' ')
                index_map.append(i)
                in_ws = True
        else:
            in_ws = False
            norm_chars.append(char)
            index_map.append(i)

        i += 1

    norm_str = "".join(norm_chars)
    return norm_str, index_map


def _strip_xml(text: str) -> str:
    """Strip all XML tags to get pure source text for matching."""
    return re.sub(r"<[^>]+>", "", text or "")


def align_original_boundaries(
    prev_chunk: ChunkInput,
    curr_chunk: ChunkInput,
    max_window: int = 5000
) -> Tuple[int, int, int, int, float]:
    """
    Boundary alignment ladder between prev_chunk and curr_chunk:
    1. Exact suffix-prefix / containment match
    2. Normalized suffix-prefix / containment match
    3. Token-level shingle anchor alignment
    """
    prev_raw = prev_chunk.original_text
    curr_raw = curr_chunk.original_text

    prev_clean = _strip_xml(prev_raw)
    curr_clean = _strip_xml(curr_raw)

    if not prev_clean.strip() or not curr_clean.strip():
        return len(prev_raw), len(prev_raw), 0, 0, 0.0

    # 1. Exact match (100% overlap)
    if prev_clean == curr_clean:
        return 0, len(prev_raw), 0, len(curr_raw), 1.0

    # 2. Check if curr is fully contained in prev
    if curr_clean in prev_clean:
        c_idx = prev_clean.find(curr_clean)
        return c_idx, c_idx + len(curr_clean), 0, len(curr_raw), 1.0

    # 3. Check tail of prev matching substring in curr
    prev_tail_len = min(len(prev_clean), max_window)
    prev_tail = prev_clean[-prev_tail_len:]
    curr_head = curr_clean[:max_window]

    # Search for longest suffix of prev_tail that appears in curr_head (min 20 non-ws chars)
    norm_tail, map_tail = _normalize_string(prev_tail)
    norm_head, map_head = _normalize_string(curr_head)

    # Ladder 1: Exact string search of tail suffix in head
    for k in range(len(prev_tail), 19, -1):
        sub = prev_tail[-k:]
        if len(re.sub(r"\s+", "", sub)) < 15:
            continue
        if sub in curr_head:
            c_idx = curr_head.find(sub)
            p_start = len(prev_raw) - k
            p_end = len(prev_raw)
            c_start = c_idx
            c_end = c_idx + k
            if c_start == 0 and p_end >= len(prev_clean):
                # Check if curr_raw has no new question labels beyond prev_raw
                curr_q_nums = set(re.findall(r"\b\d{1,4}\b", curr_clean))
                prev_q_nums = set(re.findall(r"\b\d{1,4}\b", prev_clean))
                if curr_q_nums.issubset(prev_q_nums):
                    c_end = len(curr_raw)
            return p_start, p_end, c_start, c_end, 1.0

    # Ladder 2: Normalized string search of tail suffix in head
    for k in range(len(norm_tail), 14, -1):
        norm_sub = norm_tail[-k:]
        if len(norm_sub.strip()) < 15:
            continue
        if norm_sub in norm_head:
            c_n_idx = norm_head.find(norm_sub)
            n_p_start = len(norm_tail) - k
            
            p_start_local = map_tail[n_p_start] if n_p_start < len(map_tail) else len(prev_tail) - k
            p_start = (len(prev_raw) - len(prev_tail)) + p_start_local
            p_end = len(prev_raw)

            c_start = map_head[c_n_idx] if c_n_idx < len(map_head) else c_n_idx
            c_end_idx = c_n_idx + k - 1
            c_end = map_head[c_end_idx] + 1 if c_end_idx < len(map_head) else min(len(curr_raw), c_start + k)
            while c_end < len(curr_raw) and curr_raw[c_end] in ".\n\r ":
                c_end += 1

            return p_start, p_end, c_start, c_end, 0.98

    # 4. Token shingle alignment
    prev_tokens = [m.group(0) for m in re.finditer(r"\S+", norm_tail)]
    curr_tokens = [m.group(0) for m in re.finditer(r"\S+", norm_head)]

    if len(prev_tokens) < 3 or len(curr_tokens) < 3:
        return len(prev_raw), len(prev_raw), 0, 0, 0.0

    shingle_size = 3
    curr_shingle_map = {}
    for idx in range(len(curr_tokens) - shingle_size + 1):
        shingle = " ".join(curr_tokens[idx:idx + shingle_size])
        if shingle not in curr_shingle_map:
            curr_shingle_map[shingle] = idx

    matched_pairs = []
    for idx in range(len(prev_tokens) - shingle_size + 1):
        shingle = " ".join(prev_tokens[idx:idx + shingle_size])
        if shingle in curr_shingle_map:
            matched_pairs.append((idx, curr_shingle_map[shingle]))

    if not matched_pairs:
        # Fallback: search for matching question label tokens (e.g. **2.**, **179.**, etc.)
        for p_i, p_tok in enumerate(prev_tokens):
            if re.match(r"^\*{0,2}\d{1,4}\.\*{0,2}$", p_tok):
                for c_i, c_tok in enumerate(curr_tokens):
                    if p_tok == c_tok:
                        matched_pairs.append((p_i, c_i))
                        break

    if not matched_pairs:
        return len(prev_raw), len(prev_raw), 0, 0, 0.0

    p_t_start = matched_pairs[0][0]
    p_t_end = matched_pairs[-1][0] + shingle_size

    c_t_start = matched_pairs[0][1]
    c_t_end = matched_pairs[-1][1] + shingle_size

    p_matches = list(re.finditer(r"\S+", norm_tail))
    c_matches = list(re.finditer(r"\S+", norm_head))

    n_p_start = p_matches[p_t_start].start()
    n_p_end = p_matches[min(p_t_end - 1, len(p_matches) - 1)].end()

    n_c_start = c_matches[c_t_start].start()
    n_c_end = c_matches[min(c_t_end - 1, len(c_matches) - 1)].end()

    prev_offset_base = len(prev_raw) - len(prev_tail)
    
    orig_p_start = map_tail[n_p_start] if n_p_start < len(map_tail) else len(prev_tail)
    orig_p_end = map_tail[min(n_p_end - 1, len(map_tail) - 1)] + 1 if n_p_end > 0 else len(prev_tail)

    orig_c_start = map_head[n_c_start] if n_c_start < len(map_head) else 0
    orig_c_end = map_head[min(n_c_end - 1, len(map_head) - 1)] + 1 if n_c_end > 0 else len(curr_head)

    prev_overlap_start = prev_offset_base + orig_p_start
    prev_overlap_end = prev_offset_base + orig_p_end

    curr_overlap_start = orig_c_start
    curr_overlap_end = orig_c_end



    confidence = min(1.0, 0.85 + len(matched_pairs) * 0.05)

    return prev_overlap_start, prev_overlap_end, curr_overlap_start, curr_overlap_end, float(confidence)


def build_canonical_source(chunks: List[ChunkInput]) -> Tuple[str, List[PiecewiseMap], List[Dict[str, Any]]]:
    """
    Build unified canonical original text and piecewise local-to-global maps for all chunks.
    """
    if not chunks:
        return "", [], []

    if all(chunk.page_offset_ranges for chunk in chunks):
        global_pages: Dict[Tuple[int, int], str] = {}
        for chunk in chunks:
            for page_range in chunk.page_offset_ranges or []:
                key = (page_range.global_start, page_range.global_end)
                page_text = chunk.original_text[
                    page_range.local_start:page_range.local_end
                ]
                global_pages.setdefault(key, page_text)

        canonical_parts: List[str] = []
        global_to_canonical: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for key, page_text in sorted(global_pages.items()):
            if canonical_parts:
                canonical_parts.append("\n\n")
            start = sum(len(part) for part in canonical_parts)
            canonical_parts.append(page_text)
            global_to_canonical[key] = (start, start + len(page_text))
        canonical = "".join(canonical_parts)

        maps: List[PiecewiseMap] = []
        for chunk in chunks:
            piecewise = PiecewiseMap(chunk.index, len(chunk.original_text))
            for page_range in chunk.page_offset_ranges or []:
                g_start, g_end = global_to_canonical[
                    (page_range.global_start, page_range.global_end)
                ]
                piecewise.add_segment(
                    page_range.local_start,
                    page_range.local_end,
                    g_start,
                    g_end,
                )
            maps.append(piecewise)

        reports = [
            {
                "pair": (index - 1, index),
                "score": 1.0,
                "status": "PROVENANCE",
            }
            for index in range(1, len(chunks))
        ]
        return canonical, maps, reports

    cleaned_texts = []
    for c in chunks:
        c_text = c.original_text
        if "<" in c_text and ">" in c_text:
            cleaned_c = _strip_xml(c_text)
            cleaned_texts.append(cleaned_c if cleaned_c.strip() else c_text)
        else:
            cleaned_texts.append(c_text)

    canonical = cleaned_texts[0]
    map0 = PiecewiseMap(0, len(chunks[0].original_text))
    map0.add_segment(0, len(chunks[0].original_text), 0, len(canonical))

    chunk_maps = [map0]
    overlap_reports = []

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]

        p_start, p_end, c_start, c_end, score = align_original_boundaries(prev_chunk, curr_chunk)
        prev_map = chunk_maps[i - 1]

        curr_map = PiecewiseMap(i, len(curr_chunk.original_text))

        if score >= 0.80 and c_end > c_start:
            # Prefix before overlap (e.g. repeated header) maps to corresponding prev global offset
            if c_start > 0:
                header_prefix = cleaned_texts[i][:c_start]
                if canonical.startswith(header_prefix):
                    curr_map.add_segment(0, c_start, 0, len(header_prefix))
                else:
                    g_p_start = prev_map.map_local_to_global(max(0, p_start - c_start))
                    g_p_mid = prev_map.map_local_to_global(p_start)
                    curr_map.add_segment(0, c_start, g_p_start, g_p_mid)

            # Main overlap range maps to existing global range
            g_overlap_start = prev_map.map_local_to_global(p_start)
            g_overlap_end = prev_map.map_local_to_global(p_end)
            curr_map.add_segment(c_start, c_end, g_overlap_start, g_overlap_end)

            # Non-overlapping suffix is appended to canonical
            suffix_start = c_end
            suffix_end = len(curr_chunk.original_text)
            g_suffix_start = len(canonical)

            appended_text = cleaned_texts[i][min(suffix_start, len(cleaned_texts[i])):]
            canonical += appended_text
            g_suffix_end = len(canonical)

            if suffix_start < suffix_end:
                curr_map.add_segment(suffix_start, suffix_end, g_suffix_start, g_suffix_end)

            overlap_reports.append({
                "pair": (i - 1, i),
                "score": score,
                "prev_overlap": (p_start, p_end),
                "curr_overlap": (c_start, c_end),
                "status": "ACCEPTED",
            })
        else:
            # No overlap - concatenate entire current original text
            g_start = len(canonical)
            canonical += cleaned_texts[i]
            g_end = len(canonical)

            curr_map.add_segment(0, len(curr_chunk.original_text), g_start, g_end)

            overlap_reports.append({
                "pair": (i - 1, i),
                "score": score,
                "status": "NO_OVERLAP",
            })

        chunk_maps.append(curr_map)

    return canonical, chunk_maps, overlap_reports
