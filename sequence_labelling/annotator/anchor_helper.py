"""
Anchor Helper utility for Sequence Labelled Exam Documents.

Automatically extracts and validates stimulus start_anchor and end_anchor attributes,
converts paired <stimulus>...</stimulus> tags into standard self-closing anchors,
and cleans up orphaned </stimulus> tags.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


class AnchorHelper:
    """Helper to detect and fix stimulus anchor structures."""

    @staticmethod
    def extract_anchor_words(text: str, num_words: int = 5) -> str:
        """Extract first num_words from text, stripping XML tags and excess whitespace."""
        # Strip XML tags
        clean = re.sub(r"<[^>]+>", " ", text)
        words = clean.split()
        if not words:
            return ""
        selected = words[:num_words]
        return " ".join(selected)

    @staticmethod
    def extract_end_anchor_words(text: str, num_words: int = 5) -> str:
        """Extract last num_words from text, stripping XML tags and excess whitespace."""
        clean = re.sub(r"<[^>]+>", " ", text)
        words = clean.split()
        if not words:
            return ""
        selected = words[-num_words:]
        return " ".join(selected)

    @classmethod
    def auto_anchor_stimuli(
        cls,
        xml_content: str,
        raw_ocr_text: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """
        Converts paired <stimulus> tags into valid self-closing anchor tags,
        populating start_anchor and end_anchor from the passage text.
        """
        fixes: List[str] = []
        if not xml_content:
            return xml_content, fixes

        updated_xml = xml_content
        norm_raw = " ".join(raw_ocr_text.split()) if raw_ocr_text else None

        # 1. First, handle paired <stimulus ...>...</stimulus>
        paired_pattern = re.compile(
            r"<stimulus\b(?![^>]*/>)([^>]*)>(.*?)</stimulus>",
            re.DOTALL | re.IGNORECASE,
        )

        def replace_paired_stimulus(match: re.Match) -> str:
            attr_str = match.group(1).strip()
            inner_content = match.group(2)

            # Determine ID
            id_m = re.search(r'id="([^"]*)"', attr_str)
            stim_id = id_m.group(1) if id_m else "stim_1"

            # Check existing anchors
            start_m = re.search(r'start_anchor="([^"]*)"', attr_str)
            end_m = re.search(r'end_anchor="([^"]*)"', attr_str)

            start_anchor = start_m.group(1) if start_m else ""
            end_anchor = end_m.group(1) if end_m else ""

            # If missing, extract from inner_content
            if not start_anchor:
                start_anchor = cls.extract_anchor_words(inner_content, 5)
            if not end_anchor:
                end_anchor = cls.extract_end_anchor_words(inner_content, 5)

            # Refine against raw_ocr_text if available
            if norm_raw and start_anchor:
                norm_start = " ".join(start_anchor.split())
                if norm_start not in norm_raw:
                    # Try 4 words or 3 words
                    words = start_anchor.split()
                    for n in [4, 3, 2]:
                        cand = " ".join(words[:n])
                        if cand in norm_raw:
                            start_anchor = cand
                            break

            if norm_raw and end_anchor:
                norm_end = " ".join(end_anchor.split())
                if norm_end not in norm_raw:
                    words = end_anchor.split()
                    for n in [4, 3, 2]:
                        cand = " ".join(words[-n:])
                        if cand in norm_raw:
                            end_anchor = cand
                            break

            # Escape quotes in anchors
            start_anchor_esc = start_anchor.replace('"', '&quot;')
            end_anchor_esc = end_anchor.replace('"', '&quot;')

            anchor_tag = f'<stimulus id="{stim_id}" start_anchor="{start_anchor_esc}" end_anchor="{end_anchor_esc}" />'
            fixes.append(
                f"Converted paired <stimulus id=\"{stim_id}\"> to self-closing anchor"
            )

            # In ground truth, the passage text follows the anchor tag
            return f"{anchor_tag}\n{inner_content}"

        updated_xml = paired_pattern.sub(replace_paired_stimulus, updated_xml)

        # 2. Handle self-closing <stimulus ... /> that are missing start_anchor or end_anchor
        self_closing_pattern = re.compile(
            r"<stimulus\b([^>]*?)(?<!/)\s*/>",
            re.IGNORECASE,
        )

        def fix_self_closing(match: re.Match) -> str:
            attr_str = match.group(1).strip()
            id_m = re.search(r'id="([^"]*)"', attr_str)
            stim_id = id_m.group(1) if id_m else "stim_1"

            start_m = re.search(r'start_anchor="([^"]*)"', attr_str)
            end_m = re.search(r'end_anchor="([^"]*)"', attr_str)

            if start_m and end_m:
                return match.group(0)  # already has both

            # Look ahead in text up to next question_label or section
            pos = match.end()
            remaining = updated_xml[pos:pos + 3000]
            next_boundary = re.search(r"<(question_label|section)\b", remaining, re.IGNORECASE)
            passage_snippet = remaining[:next_boundary.start()] if next_boundary else remaining[:500]

            start_anchor = start_m.group(1) if start_m else cls.extract_anchor_words(passage_snippet, 5)
            end_anchor = end_m.group(1) if end_m else cls.extract_end_anchor_words(passage_snippet, 5)

            start_anchor_esc = start_anchor.replace('"', '&quot;')
            end_anchor_esc = end_anchor.replace('"', '&quot;')

            fixes.append(
                f"Added missing start/end anchors to <stimulus id=\"{stim_id}\" />"
            )
            return f'<stimulus id="{stim_id}" start_anchor="{start_anchor_esc}" end_anchor="{end_anchor_esc}" />'

        updated_xml = self_closing_pattern.sub(fix_self_closing, updated_xml)

        # 3. Clean any orphaned </stimulus> tags
        orphaned_closing = re.findall(r"</stimulus>", updated_xml, re.IGNORECASE)
        if orphaned_closing:
            updated_xml = re.sub(r"\s*</stimulus>", "", updated_xml, flags=re.IGNORECASE)
            fixes.append(f"Removed {len(orphaned_closing)} orphaned </stimulus> tag(s)")

        # 4. Renumber stimulus IDs if needed (stim_1, stim_2, ...)
        stim_id_pattern = re.compile(r'<stimulus\b([^>]*?)id="([^"]*)"([^>]*?)/>')
        counter = 1
        def renumber_stim(m: re.Match) -> str:
            nonlocal counter
            p1 = m.group(1)
            p2 = m.group(3)
            res = f'<stimulus{p1}id="stim_{counter}"{p2}/>'
            counter += 1
            return res

        # Check if IDs are numbered
        updated_xml = stim_id_pattern.sub(renumber_stim, updated_xml)

        return updated_xml, fixes
