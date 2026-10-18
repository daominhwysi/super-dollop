import re
import unicodedata
import difflib
from typing import List, Tuple, Dict, Any, Optional
from sequence_labelling.parser.long_parser.source_merger.types import ChunkInput

# Overlaps below this score are treated as "no overlap" and the chunk is appended whole.
OVERLAP_ACCEPT_SCORE = 0.80
# Token-shingle alignment needs at least this many matched anchors; a single shared
# token (e.g. a question number such as "3.") is not evidence of an overlap.
MIN_OVERLAP_ANCHORS = 3

# Only annotation markup and sentinel/page wrappers are removed. Ordinary "<" and ">"
# in exam text (inequalities, comparisons) are source content and must be preserved.
ANNOTATION_MARKUP_RE = re.compile(
    r"</?(?:section|stimulus|question|question_label|stem|option_label|option_text|explanation)(?:\s[^<>]*)?/?>"
    r"|<\|[^<>|]*\|>"
    r"|</?pages?>|</?page_metadata>",
    flags=re.IGNORECASE,
)


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

    def set_segments(self, segments: List[Tuple[int, int, int, int]]):
        self.segments = sorted(segments, key=lambda item: (item[0], item[1]))

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


def strip_canonical_markup(text: str) -> Tuple[str, List[int]]:
    """
    Remove annotation markup from raw chunk text.
    Returns (clean_text, keep) where keep[k] is the raw index of clean_text[k].
    """
    if not text:
        return "", []
    keep: List[int] = []
    last = 0
    for match in ANNOTATION_MARKUP_RE.finditer(text):
        keep.extend(range(last, match.start()))
        last = match.end()
    keep.extend(range(last, len(text)))
    clean = "".join(text[i] for i in keep)
    if not clean.strip():
        # Markup-only chunk: fall back to the raw text so no content is lost
        return text, list(range(len(text)))
    return clean, keep


def _project_block(prev_g: List[int], a: int, b: int, count: int) -> List[int]:
    """Global offsets for `count` chars that correspond to prev clean range [a, b)."""
    if count <= 0 or b <= a:
        return []
    span = b - a
    if span == count:
        return [prev_g[a + k] for k in range(count)]
    return [prev_g[a + min(span - 1, (k * span) // count)] for k in range(count)]


def _map_repeated_prefix(prev_clean: str, prev_g: List[int], prefix: str) -> Optional[List[int]]:
    """
    If `prefix` (text at the start of the current chunk before the overlap) already
    occurs in the previous chunk, return its global offsets; otherwise None.
    """
    if not prefix.strip():
        return None
    q = prev_clean.rfind(prefix)
    if q >= 0:
        return prev_g[q:q + len(prefix)]

    norm_prefix, _ = _normalize_string(prefix)
    norm_prefix = norm_prefix.strip()
    if len(norm_prefix) < 15:
        return None
    norm_prev, map_prev = _normalize_string(prev_clean)
    q = norm_prev.rfind(norm_prefix)
    if q < 0:
        return None
    start = map_prev[q]
    end = map_prev[min(q + len(norm_prefix) - 1, len(map_prev) - 1)] + 1
    return _project_block(prev_g, start, end, len(prefix))


def _runs_to_segments(keep: List[int], global_of: List[int]) -> List[Tuple[int, int, int, int]]:
    """Split a chunk into runs that are contiguous in both raw and global coordinates."""
    segments = []
    k = 0
    n = len(keep)
    while k < n:
        end = k + 1
        while end < n and keep[end] == keep[end - 1] + 1 and global_of[end] == global_of[end - 1] + 1:
            end += 1
        segments.append((keep[k], keep[end - 1] + 1, global_of[k], global_of[end - 1] + 1))
        k = end
    return segments


def _align_clean(prev_clean: str, curr_clean: str, max_window: int = 5000) -> Tuple[int, int, int, int, float]:
    """
    Boundary alignment ladder between two markup-free chunk texts. All offsets are
    clean-text offsets:
    1. Exact suffix-prefix / containment match
    2. Normalized suffix-prefix match
    3. Token-level shingle anchor alignment (requires several anchors)
    """
    if not prev_clean.strip() or not curr_clean.strip():
        return len(prev_clean), len(prev_clean), 0, 0, 0.0

    # 1. Exact match (100% overlap)
    if prev_clean == curr_clean:
        return 0, len(prev_clean), 0, len(curr_clean), 1.0

    # 2. Check if curr is fully contained in prev
    if curr_clean in prev_clean:
        c_idx = prev_clean.find(curr_clean)
        return c_idx, c_idx + len(curr_clean), 0, len(curr_clean), 1.0

    # 3. Check tail of prev matching substring in curr
    prev_tail_len = min(len(prev_clean), max_window)
    prev_tail = prev_clean[-prev_tail_len:]
    curr_head = curr_clean[:max_window]

    norm_tail, map_tail = _normalize_string(prev_tail)
    norm_head, map_head = _normalize_string(curr_head)

    # Ladder 1: Exact string search of tail suffix in head
    for k in range(len(prev_tail), 19, -1):
        sub = prev_tail[-k:]
        if len(re.sub(r"\s+", "", sub)) < 15:
            continue
        if sub in curr_head:
            c_idx = curr_head.find(sub)
            p_start = len(prev_clean) - k
            p_end = len(prev_clean)
            c_start = c_idx
            c_end = c_idx + k
            # Text in curr after the matched suffix is new source content; it is appended
            # by the caller rather than assumed to be covered by prev.
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
            p_start = (len(prev_clean) - len(prev_tail)) + p_start_local
            p_end = len(prev_clean)

            c_start = map_head[c_n_idx] if c_n_idx < len(map_head) else c_n_idx
            c_end_idx = c_n_idx + k - 1
            # The overlap ends exactly at the matched text; any trailing whitespace or
            # punctuation after it is new content and is appended by the caller.
            c_end = map_head[c_end_idx] + 1 if c_end_idx < len(map_head) else min(len(curr_clean), c_start + k)

            return p_start, p_end, c_start, c_end, 0.98

    # Ladder 3: Token shingle alignment
    prev_tokens = [m.group(0) for m in re.finditer(r"\S+", norm_tail)]
    curr_tokens = [m.group(0) for m in re.finditer(r"\S+", norm_head)]

    if len(prev_tokens) < 3 or len(curr_tokens) < 3:
        return len(prev_clean), len(prev_clean), 0, 0, 0.0

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
        # Fallback: question-number tokens (e.g. **2.**, **179.**). Only used as an
        # anchor set, never as a single-token overlap.
        for p_i, p_tok in enumerate(prev_tokens):
            if re.match(r"^\*{0,2}\d{1,4}\.\*{0,2}$", p_tok):
                for c_i, c_tok in enumerate(curr_tokens):
                    if p_tok == c_tok:
                        matched_pairs.append((p_i, c_i))
                        break

    if len(matched_pairs) < MIN_OVERLAP_ANCHORS:
        return len(prev_clean), len(prev_clean), 0, 0, 0.0

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

    prev_offset_base = len(prev_clean) - len(prev_tail)

    orig_p_start = map_tail[n_p_start] if n_p_start < len(map_tail) else len(prev_tail)
    orig_p_end = map_tail[min(n_p_end - 1, len(map_tail) - 1)] + 1 if n_p_end > 0 else len(prev_tail)

    orig_c_start = map_head[n_c_start] if n_c_start < len(map_head) else 0
    orig_c_end = map_head[min(n_c_end - 1, len(map_head) - 1)] + 1 if n_c_end > 0 else len(curr_head)

    prev_overlap_start = prev_offset_base + orig_p_start
    prev_overlap_end = prev_offset_base + orig_p_end

    confidence = min(0.95, 0.75 + len(matched_pairs) * 0.05)

    return prev_overlap_start, prev_overlap_end, orig_c_start, orig_c_end, float(confidence)


def _build_page_canonical(chunks: List[ChunkInput]) -> Tuple[str, List[PiecewiseMap], List[Dict[str, Any]]]:
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


def build_canonical_source(chunks: List[ChunkInput]) -> Tuple[str, List[PiecewiseMap], List[Dict[str, Any]]]:
    """
    Build unified canonical original text and piecewise local-to-global maps for all chunks.

    All alignment happens on markup-free chunk text. Every markup-free character of every
    chunk is assigned exactly one canonical offset (overlap characters reuse the previous
    chunk's offsets; all other characters are appended), so the canonical text cannot
    silently drop source content. Per-chunk maps convert raw chunk offsets exactly.
    """
    if not chunks:
        return "", [], []

    if all(chunk.page_offset_ranges for chunk in chunks):
        return _build_page_canonical(chunks)

    cleans: List[str] = []
    keeps: List[List[int]] = []
    for chunk in chunks:
        clean, keep = strip_canonical_markup(chunk.original_text)
        cleans.append(clean)
        keeps.append(keep)

    canonical = cleans[0]
    global_of: List[List[int]] = [list(range(len(cleans[0])))]
    overlap_reports: List[Dict[str, Any]] = []

    for i in range(1, len(chunks)):
        prev_clean = cleans[i - 1]
        curr_clean = cleans[i]
        prev_g = global_of[i - 1]

        p_start, p_end, c_start, c_end, score = _align_clean(prev_clean, curr_clean)
        curr_g: List[int] = [0] * len(curr_clean)
        pending: List[Tuple[int, int]] = []
        report: Dict[str, Any] = {"pair": (i - 1, i), "score": score}

        if score >= OVERLAP_ACCEPT_SCORE and c_end > c_start:
            curr_g[c_start:c_end] = _project_block(prev_g, p_start, p_end, c_end - c_start)
            report.update(
                status="ACCEPTED",
                prev_overlap=(p_start, p_end),
                curr_overlap=(c_start, c_end),
            )

            prefix_g = _map_repeated_prefix(prev_clean, prev_g, curr_clean[:c_start])
            if prefix_g is not None:
                curr_g[:c_start] = prefix_g
                report["prefix"] = "REPEATED"
            else:
                # Text before the overlap that the previous chunk does not contain is
                # new source content: append it instead of mapping it onto other text.
                if c_start > 0:
                    pending.append((0, c_start))
                    report["prefix"] = "APPENDED"
            pending.append((c_end, len(curr_clean)))
        else:
            report["status"] = "NO_OVERLAP"
            pending.append((0, len(curr_clean)))

        for a, b in pending:
            if b <= a:
                continue
            base = len(canonical)
            canonical += curr_clean[a:b]
            curr_g[a:b] = list(range(base, base + (b - a)))

        overlap_reports.append(report)
        global_of.append(curr_g)

    # Source-fidelity check against the raw-derived chunk text (not the canonical text itself)
    for i, (clean, g) in enumerate(zip(cleans, global_of)):
        mismatched = 0
        for k, ch in enumerate(clean):
            out = canonical[g[k]]
            if out != ch and not (out.isspace() and ch.isspace()):
                mismatched += 1
        if mismatched:
            overlap_reports.append({
                "status": "FIDELITY_MISMATCH",
                "chunk_index": chunks[i].index,
                "mismatched_chars": mismatched,
            })

    maps: List[PiecewiseMap] = []
    for chunk, keep, g in zip(chunks, keeps, global_of):
        piecewise = PiecewiseMap(chunk.index, len(chunk.original_text))
        piecewise.set_segments(_runs_to_segments(keep, g))
        maps.append(piecewise)

    return canonical, maps, overlap_reports
