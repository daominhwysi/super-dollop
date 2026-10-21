"""
synthetic_exam_generator/hard_negative_injector.py

Safe Hard-Negative Module Injector for Synthetic Sequence-Labelling Exams.
Splices modular expository pedagogical text and outlines into synthetic exams
with 100% character-exact span preservation and strict structural safety guarantees.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hard_negative_injector")

# Subject alignment map (Exam subject -> Hard-negative subjects)
SUBJECT_ALIGNMENT: Dict[str, List[str]] = {
    "math": ["math_algebra", "math_geometry"],
    "physics": ["physics"],
    "english": ["toeic"],
    "chemistry": ["physics", "math_algebra"],
    "biology": ["physics", "math_algebra"],
    "literature": ["toeic"],
}


@dataclass
class HardNegativeModule:
    module_id: str
    module_type: str
    subject: str
    raw_text: str
    spans: List[Dict[str, Any]]  # Local spans relative to module start (0)


@dataclass
class InjectionConfig:
    enabled: bool = True
    injection_prob: float = 0.35          # 35% of synthetic exams receive an injected module
    max_modules_per_exam: int = 1         # Prevents class-balance skew towards 'O'
    max_module_chars: int = 1500          # Bounded token length (~300-350 tokens)
    site_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "inter_section": 0.40,
            "inter_question": 0.30,
            "pre_exam": 0.15,
            "post_exam": 0.15,
        }
    )


class HardNegativeBank:
    """Loads, indexes, and samples modular blocks from data/synthetic_hard_negative/."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.modules_by_subject: Dict[str, List[HardNegativeModule]] = {}
        self._load_bank()

    def _load_bank(self) -> None:
        if not self.data_dir.exists():
            logger.warning(f"Hard-negative bank directory not found: {self.data_dir}")
            return

        xml_files = sorted(self.data_dir.glob("*.xml"))
        module_pattern = re.compile(
            r'<!--\s*MODULE:\s*id="([^"]+)"\s*type="([^"]+)"\s*subject="([^"]+)"\s*-->\s*(.*?)\s*<!--\s*/MODULE\s*-->',
            re.DOTALL,
        )

        for xml_file in xml_files:
            try:
                content = xml_file.read_text(encoding="utf-8")
                matches = module_pattern.findall(content)
                for m_id, m_type, m_subj, m_body in matches:
                    raw_text, spans = self._parse_module_xml(m_body.strip())
                    if not raw_text.strip():
                        continue
                    mod = HardNegativeModule(
                        module_id=m_id,
                        module_type=m_type,
                        subject=m_subj.lower(),
                        raw_text=raw_text,
                        spans=spans,
                    )
                    self.modules_by_subject.setdefault(mod.subject, []).append(mod)
            except Exception as e:
                logger.warning(f"Failed to load modules from {xml_file.name}: {e}")

        total_mods = sum(len(v) for v in self.modules_by_subject.values())
        logger.info(f"Loaded {total_mods} hard-negative modules across {len(self.modules_by_subject)} subjects.")

    @staticmethod
    def _parse_module_xml(xml_body: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parses module XML body into clean raw_text as 100% background 'O' content.
        Guarantees:
        - Strips any nested or duplicate generator comments (<!-- ... -->).
        - Strips all sequence tags (<section>, etc.) so theory is 100% untagged ('O' tokens).
          Textbook headings/notes are NOT exam sections and must be purely background 'O'.
        - Normalizes <br> / <br/> into newlines.
        - Preserves math inequalities (e.g. $x < 0$).
        - Returns empty spans list ([]) because pedagogical theory has zero sequence entities.
        """
        cleaned = re.sub(r"<!--.*?-->", "", xml_body, flags=re.DOTALL)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        # Strip sequence tags so theory text is 100% untagged background 'O'
        raw_text = re.sub(
            r"</?(?:section|question_label|stem|option_label|option_text|explanation|stimulus|figure)\b[^>]*>",
            "",
            cleaned,
        )
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
        return raw_text, []

    def sample_module(
        self, exam_subject: str, max_chars: int, rng: random.Random
    ) -> Optional[HardNegativeModule]:
        target_subjs = SUBJECT_ALIGNMENT.get(exam_subject.lower(), ["math_algebra", "physics"])
        candidates = []
        for s in target_subjs:
            candidates.extend(self.modules_by_subject.get(s, []))
        if not candidates:
            for pool in self.modules_by_subject.values():
                candidates.extend(pool)
        if not candidates:
            return None

        bounded = [m for m in candidates if len(m.raw_text) <= max_chars]
        if not bounded:
            return rng.choice(candidates)
        return rng.choice(bounded)


class SyntheticHardNegativeInjector:
    """Safe insertion engine executing verified span-offset transformations."""

    def __init__(self, bank: HardNegativeBank, config: Optional[InjectionConfig] = None):
        self.bank = bank
        self.config = config or InjectionConfig()

    def inject_into_reconstructed_exam(
        self,
        raw_text: str,
        spans: List[Dict[str, Any]],
        subject: str,
        rng: Optional[random.Random] = None,
        include_span_text: Optional[bool] = None,
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Safely injects hard-negative module(s) into an exam document.
        Returns: (augmented_raw_text, updated_spans, was_injected)
        """
        rng = rng or random.Random()
        if not self.config.enabled or rng.random() > self.config.injection_prob:
            return raw_text, spans, False

        curr_text = raw_text
        curr_spans = [dict(s) for s in spans]
        injected = False

        include_text = (
            include_span_text
            if include_span_text is not None
            else any("text" in s for s in spans)
        )

        num_injections = min(self.config.max_modules_per_exam, 2)
        for _ in range(num_injections):
            safe_points = self._discover_safe_points(curr_text, curr_spans)
            if not safe_points:
                break

            module = self.bank.sample_module(subject, self.config.max_module_chars, rng)
            if not module:
                break

            insert_pos, site_type = self._choose_site(safe_points, rng)

            prefix_sep = "\n\n" if not curr_text[:insert_pos].endswith("\n\n") else ""
            suffix_sep = "\n\n" if not curr_text[insert_pos:].startswith("\n\n") else ""
            payload = f"{prefix_sep}{module.raw_text}{suffix_sep}"

            payload_start = insert_pos + len(prefix_sep)
            delta_len = len(payload)

            updated_spans: List[Dict[str, Any]] = []
            for s in curr_spans:
                if s["end"] <= insert_pos:
                    updated_spans.append(dict(s))
                elif s["start"] >= insert_pos:
                    shifted = dict(s)
                    shifted["start"] += delta_len
                    shifted["end"] += delta_len
                    updated_spans.append(shifted)
                else:
                    logger.error(f"Safety violation: insertion at {insert_pos} split span {s}!")
                    return raw_text, spans, False

            for ms in module.spans:
                mod_span = dict(ms)
                mod_span["start"] += payload_start
                mod_span["end"] += payload_start
                updated_spans.append(mod_span)

            curr_text = curr_text[:insert_pos] + payload + curr_text[insert_pos:]
            updated_spans.sort(key=lambda s: s["start"])

            for s in updated_spans:
                if include_text:
                    s["text"] = curr_text[s["start"] : s["end"]]
                else:
                    s.pop("text", None)

            curr_spans = updated_spans
            injected = True

        return curr_text, curr_spans, injected

    def _discover_safe_points(
        self, raw_text: str, spans: List[Dict[str, Any]]
    ) -> List[Tuple[int, str]]:
        safe_points: List[Tuple[int, str]] = []
        if not spans:
            return safe_points

        safe_points.append((0, "pre_exam"))
        safe_points.append((len(raw_text), "post_exam"))

        sorted_spans = sorted(spans, key=lambda s: s["start"])
        stimulus_active = False

        for i in range(len(sorted_spans) - 1):
            curr_s = sorted_spans[i]
            next_s = sorted_spans[i + 1]

            if curr_s["label"] == "stimulus":
                stimulus_active = True

            if next_s["label"] == "section":
                stimulus_active = False
                if curr_s["end"] <= next_s["start"]:
                    safe_points.append((next_s["start"], "inter_section"))

            if (
                not stimulus_active
                and curr_s["label"] in {"option_text", "explanation", "stem"}
                and next_s["label"] == "question_label"
                and curr_s["end"] <= next_s["start"]
            ):
                safe_points.append((next_s["start"], "inter_question"))

        return safe_points

    def _choose_site(
        self, safe_points: List[Tuple[int, str]], rng: random.Random
    ) -> Tuple[int, str]:
        by_site: Dict[str, List[int]] = {}
        for pos, st in safe_points:
            by_site.setdefault(st, []).append(pos)

        available_types = [st for st in by_site.keys() if st in self.config.site_weights]
        weights = [self.config.site_weights[st] for st in available_types]

        selected_type = rng.choices(available_types, weights=weights, k=1)[0]
        selected_pos = rng.choice(by_site[selected_type])
        return selected_pos, selected_type
