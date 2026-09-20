import re
from typing import Dict, Any, List, Optional, Tuple, Set


class SequenceValidatorAgent:
    """
    Sequence Validator & Repair Agent for Azozo Long-Context Parser Engine v2.0.
    
    Responsibilities:
    1. Audits LLM sequence output against raw OCR chunk input.
    2. Detects missing/un-tagged question labels (e.g. **186.** or Question 186).
    3. Recovers untagged question stems, choices (A-D), and linked stimuli via deterministic fallback regex parsing.
    4. Audits question sequence continuity and option completeness across the entire document.
    """

    @staticmethod
    def extract_raw_question_numbers(raw_text: str) -> List[Tuple[str, int]]:
        """
        Finds all question numbers in raw text with their character offsets.
        Matches patterns like: **186.**, 186., Question 186.
        """
        pattern = re.compile(r"(?:^|\n)\s*(?:<question_label>)?\s*(?:\*\*)?(?:Question\s+)?(\d{1,4})\.\s*(?:\*\*)?\s*(?:</question_label>)?", re.IGNORECASE)
        matches = []
        for match in pattern.finditer(raw_text):
            q_num = match.group(1)
            matches.append((q_num, match.start()))
        return matches

    @staticmethod
    def _parse_untagged_question_block(block_text: str, q_num: str) -> Optional[Dict[str, Any]]:
        """
        Parses a raw un-tagged text block into a structured question object.
        """
        lines = [line.strip() for line in block_text.splitlines() if line.strip()]
        if not lines:
            return None

        # Clean stem header
        stem_raw = lines[0]
        stem_clean = re.sub(r"^(?:<question_label>)?\s*(?:\*\*)?(?:Question\s+)?\d{1,4}\.\s*(?:\*\*)?\s*(?:</question_label>)?\s*", "", stem_raw, flags=re.IGNORECASE).strip()
        
        # Gather stem and options
        stem_parts = [stem_clean]
        options: List[Dict[str, str]] = []
        
        opt_pattern = re.compile(r"\(([A-E])\)\s*(.*?)(?=\s*\([A-E]\)|$)")

        for line in lines[1:]:
            opt_matches = list(opt_pattern.finditer(line))
            if opt_matches:
                for m in opt_matches:
                    options.append({"label": m.group(1), "text": m.group(2).strip()})
            else:
                if not options:
                    stem_parts.append(line)

        full_stem = " ".join(part for part in stem_parts if part).strip()
        
        if not full_stem and not options:
            return None

        return {
            "question_number": q_num,
            "stem": full_stem,
            "options": options,
            "correct_answer": None,
            "explanation": None,
            "recovered_by_validator": True
        }

    def repair_chunk_questions(
        self,
        raw_chunk_text: str,
        parsed_questions: List[Dict[str, Any]],
        chunk_index: int = 0,
        active_stimulus_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Audits parsed questions against raw chunk text and recovers any missing/un-tagged questions.
        """
        raw_matches = self.extract_raw_question_numbers(raw_chunk_text)
        if not raw_matches:
            return parsed_questions, 0

        # Existing extracted question numbers
        existing_nums: Set[str] = set()
        for q in parsed_questions:
            num = str(q.get("question_number") or q.get("num") or "").strip()
            num_match = re.search(r"\d+", num)
            if num_match:
                existing_nums.add(num_match.group(0))

        repaired_questions = list(parsed_questions)
        recovered_count = 0

        for i, (q_num, start_idx) in enumerate(raw_matches):
            if q_num in existing_nums:
                continue

            # Determine boundary for this un-tagged question block
            end_idx = raw_matches[i + 1][1] if i + 1 < len(raw_matches) else len(raw_chunk_text)
            block_text = raw_chunk_text[start_idx:end_idx].strip()

            recovered_q = self._parse_untagged_question_block(block_text, q_num)
            if recovered_q:
                recovered_q["id"] = f"chunk_{chunk_index}_q_{q_num}"
                recovered_q["chunk_index"] = chunk_index
                if active_stimulus_id:
                    recovered_q["stimulus_id"] = active_stimulus_id

                repaired_questions.append(recovered_q)
                existing_nums.add(q_num)
                recovered_count += 1

        # Sort repaired questions by question number
        def sort_key(q):
            num_str = str(q.get("question_number") or "0")
            digits = re.search(r"\d+", num_str)
            return int(digits.group(0)) if digits else 9999

        repaired_questions.sort(key=sort_key)
        return repaired_questions, recovered_count

    def audit_document(
        self,
        questions: List[Dict[str, Any]],
        stimuli: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Performs a full audit of document sequence continuity and field completeness.
        """
        question_numbers = []
        issues = []
        incomplete_options = 0

        for q in questions:
            num_str = str(q.get("question_number") or "")
            m = re.search(r"\d+", num_str)
            if m:
                question_numbers.append(int(m.group(0)))

            opts = q.get("options") or []
            if len(opts) < 2:
                incomplete_options += 1
                issues.append(f"Question {num_str} has fewer than 2 options ({len(opts)} options)")

            stim_id = q.get("stimulus_id")
            if stim_id and stim_id not in stimuli:
                issues.append(f"Question {num_str} references missing stimulus ID '{stim_id}'")

        missing_gaps = []
        if question_numbers:
            question_numbers.sort()
            first_num = question_numbers[0]
            last_num = question_numbers[-1]
            full_range = set(range(first_num, last_num + 1))
            missing_gaps = sorted(list(full_range - set(question_numbers)))

        return {
            "total_questions": len(questions),
            "question_range": f"{question_numbers[0]} - {question_numbers[-1]}" if question_numbers else "N/A",
            "missing_question_gaps": missing_gaps,
            "incomplete_options_count": incomplete_options,
            "total_stimuli": len(stimuli),
            "is_continuous": len(missing_gaps) == 0,
            "issues": issues
        }
