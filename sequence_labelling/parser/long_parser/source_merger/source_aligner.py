import re
import difflib
from typing import List, Tuple, Dict, Any
from sequence_labelling.parser.long_parser.source_merger.canonical_source import _normalize_string


class SourceAligner:
    """
    Aligns parsed text P (de-tagged XML) back to original chunk text S.
    Uses robust character & token sequence matching to map positions in P monotonically to S.
    """
    def __init__(self, parsed_text: str, original_text: str):
        self.parsed_text = parsed_text
        self.original_text = original_text
        self.alignment_kind = "exact"
        self.p_to_s_map: List[int] = []
        self._build_alignment_map()

    def _build_alignment_map(self):
        p_len = len(self.parsed_text)
        s_len = len(self.original_text)

        if p_len == 0:
            self.p_to_s_map = [0]
            return

        if s_len == 0:
            self.p_to_s_map = [0] * (p_len + 1)
            return

        # 1. Exact match
        if self.parsed_text == self.original_text:
            self.alignment_kind = "exact"
            self.p_to_s_map = list(range(p_len + 1))
            return

        # 2. Normalized match
        norm_p, map_p = _normalize_string(self.parsed_text)
        norm_s, map_s = _normalize_string(self.original_text)

        if norm_p == norm_s:
            self.alignment_kind = "normalized"
            self.p_to_s_map = [-1] * (p_len + 1)
            len_norm = len(norm_p)

            for k in range(len_norm):
                orig_p_start = map_p[k]
                orig_p_end = map_p[k + 1] if k + 1 < len(map_p) else p_len
                orig_s_start = map_s[k]
                orig_s_end = map_s[k + 1] if k + 1 < len(map_s) else s_len

                self.p_to_s_map[orig_p_start] = orig_s_start
                self.p_to_s_map[orig_p_end] = orig_s_end

            # Fill unmapped positions
            last_s = 0
            for i in range(p_len + 1):
                if self.p_to_s_map[i] != -1:
                    last_s = self.p_to_s_map[i]
                else:
                    self.p_to_s_map[i] = last_s
            return

        # 3. Token-anchored monotonic character mapping
        self.alignment_kind = "fuzzy"
        self.p_to_s_map = [-1] * (p_len + 1)

        # Match non-whitespace character sequence between norm_p and norm_s
        map_clean_p = [i for i, c in enumerate(self.parsed_text) if not c.isspace()]
        map_clean_s = [i for i, c in enumerate(self.original_text) if not c.isspace()]

        clean_p = "".join([self.parsed_text[i] for i in map_clean_p])
        clean_s = "".join([self.original_text[i] for i in map_clean_s])

        matcher = difflib.SequenceMatcher(None, clean_p, clean_s)
        blocks = matcher.get_matching_blocks()

        for p_start, s_start, length in blocks:
            for k in range(length):
                cp_idx = p_start + k
                cs_idx = s_start + k
                if cp_idx < len(map_clean_p) and cs_idx < len(map_clean_s):
                    orig_p_start = map_clean_p[cp_idx]
                    orig_p_end = orig_p_start + 1
                    orig_s_start = map_clean_s[cs_idx]
                    orig_s_end = orig_s_start + 1

                    if self.p_to_s_map[orig_p_start] == -1:
                        self.p_to_s_map[orig_p_start] = orig_s_start
                    self.p_to_s_map[orig_p_end] = orig_s_end

        # Monotonic fill for unmapped positions
        last_s = 0
        for i in range(p_len + 1):
            if self.p_to_s_map[i] != -1:
                last_s = self.p_to_s_map[i]
            else:
                self.p_to_s_map[i] = min(s_len, last_s)

        self.p_to_s_map[p_len] = s_len

    def map_span_to_source(self, p_start: int, p_end: int) -> Tuple[int, int, float, str]:
        p_len = len(self.parsed_text)
        s_len = len(self.original_text)

        p_start_clamped = max(0, min(p_len, p_start))
        p_end_clamped = max(p_start_clamped, min(p_len, p_end))

        if p_start_clamped == p_end_clamped:
            s_pos = self.p_to_s_map[p_start_clamped] if p_start_clamped < len(self.p_to_s_map) else s_len
            return s_pos, s_pos, 1.0, self.alignment_kind

        s_start = self.p_to_s_map[p_start_clamped]
        s_end = self.p_to_s_map[p_end_clamped]

        if s_end < s_start:
            s_end = s_start

        if s_start == s_end and p_start_clamped < p_end_clamped:
            return s_start, s_end, 0.0, "rejected"

        p_sub = self.parsed_text[p_start_clamped:p_end_clamped]
        s_sub = self.original_text[s_start:s_end]

        if p_sub and not p_sub[0].isspace():
            while s_start < s_end and self.original_text[s_start].isspace():
                s_start += 1
        if p_sub and not p_sub[-1].isspace():
            while s_end > s_start and self.original_text[s_end - 1].isspace():
                s_end -= 1

        s_sub = self.original_text[s_start:s_end]

        norm_p_sub = re.sub(r"\s+", "", p_sub)
        norm_s_sub = re.sub(r"\s+", "", s_sub)

        if norm_p_sub == norm_s_sub or not norm_p_sub:
            quality = 1.0
        else:
            matcher = difflib.SequenceMatcher(None, norm_p_sub, norm_s_sub)
            aligned_chars = sum(b.size for b in matcher.get_matching_blocks())
            s_quality = aligned_chars / max(1, len(norm_s_sub))
            p_quality = aligned_chars / max(1, len(norm_p_sub))
            
            p_words = set(re.findall(r"\w+", p_sub))
            s_words = set(re.findall(r"\w+", s_sub))
            w_quality = (len(p_words.intersection(s_words)) / max(1, len(p_words))) if p_words else 1.0

            quality = max(s_quality, p_quality, w_quality, matcher.ratio())

        kind = self.alignment_kind
        if quality < 0.90:
            kind = "rejected"

        return s_start, s_end, float(min(1.0, quality)), kind
