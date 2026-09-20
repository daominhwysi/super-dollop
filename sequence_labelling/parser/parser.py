import re
import uuid
from typing import List, Dict, Any, Tuple

def parse_spans_into_structured_questions(raw_text: str, spans: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Converts sequence labelling character spans into structured question & options objects.
    Returns (questions, stimuli) where stimuli maps stimulus_id -> passage text.
    """
    questions: List[Dict[str, Any]] = []
    current_q = None
    active_stimulus_id = None
    stimuli: Dict[str, str] = {}
    current_section = ""

    sorted_spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))

    for span in sorted_spans:
        label = span["label"]
        text = span["text"].strip()
        params = span.get("params", {})

        if label == "section":
            current_section = text
            active_stimulus_id = None

        elif label == "stimulus":
            stim_id = params.get("id", f"stim_{len(stimuli) + 1}")
            active_stimulus_id = stim_id
            stimuli[stim_id] = text

        elif label == "question":
            if current_q and (current_q["stem"] or current_q["options"]):
                questions.append(current_q)

            stim_id = (
                params.get("stimulus_id")
                or params.get("refer_to_stimulus")
                or params.get("stimulus_ref")
                or active_stimulus_id
            )
            current_q = {
                "id": params.get("id") or f"q_{len(questions) + 1}_{uuid.uuid4().hex[:6]}",
                "question_number": params.get("question_number") or f"Câu {len(questions) + 1}",
                "stem": "",
                "options": [],
                "correct_answer": params.get("correct_answer", ""),
                "explanation": "",
                "stimulus_id": stim_id,
                "stimulus_text": stimuli.get(stim_id, "") if stim_id else "",
                "section": current_section,
            }

        elif label == "question_label":
            stim_id = (
                params.get("stimulus_id")
                or params.get("refer_to_stimulus")
                or params.get("stimulus_ref")
                or active_stimulus_id
            )

            if current_q and not current_q["stem"] and not current_q["options"]:
                current_q["question_number"] = text
                if stim_id:
                    current_q["stimulus_id"] = stim_id
                    current_q["stimulus_text"] = stimuli.get(stim_id, "")
            else:
                if current_q and (current_q["stem"] or current_q["options"]):
                    questions.append(current_q)

                current_q = {
                    "id": params.get("id") or f"q_{len(questions) + 1}_{uuid.uuid4().hex[:6]}",
                    "question_number": text,
                    "stem": "",
                    "options": [],
                    "correct_answer": params.get("correct_answer", ""),
                    "explanation": "",
                    "stimulus_id": stim_id,
                    "stimulus_text": stimuli.get(stim_id, "") if stim_id else "",
                    "section": current_section,
                }

        elif label == "stem":
            if not current_q:
                stim_id = params.get("stimulus_id") or active_stimulus_id
                current_q = {
                    "id": params.get("id") or f"q_1_{uuid.uuid4().hex[:6]}",
                    "question_number": "Câu 1",
                    "stem": text,
                    "options": [],
                    "correct_answer": "",
                    "explanation": "",
                    "stimulus_id": stim_id,
                    "stimulus_text": stimuli.get(stim_id, "") if stim_id else "",
                    "section": current_section,
                }
            else:
                current_q["stem"] = (current_q["stem"] + " " + text).strip()
                stim_id = params.get("stimulus_id") or active_stimulus_id
                if stim_id:
                    current_q["stimulus_id"] = stim_id
                    current_q["stimulus_text"] = stimuli.get(stim_id, "")

        elif label == "option_label":
            if current_q:
                clean_lbl = re.sub(r"[\*\.\:\s]", "", text)
                current_q["_last_option_label"] = clean_lbl if clean_lbl else "A"

        elif label == "option_text":
            if current_q:
                opt_lbl = current_q.pop("_last_option_label", f"Option {len(current_q['options']) + 1}")
                current_q["options"].append({"label": opt_lbl, "text": text})

        elif label == "explanation":
            if current_q:
                current_q["explanation"] = text

    if current_q and (current_q["stem"] or current_q["options"]):
        questions.append(current_q)

    for idx, q in enumerate(questions):
        if not q["options"]:
            q["options"] = [
                {"label": "A", "text": "Phương án A"},
                {"label": "B", "text": "Phương án B"},
                {"label": "C", "text": "Phương án C"},
                {"label": "D", "text": "Phương án D"},
            ]

    return questions, stimuli

def regex_parse_questions(raw_text: str) -> List[Dict[str, Any]]:
    """
    Fallback regex parser to structure question stems and options from raw OCR text.
    """
    questions = []
    lines = raw_text.splitlines()
    current_q = None

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        line_stim_id = None
        stim_match = re.search(r'(?:stimulus_id|refer_to_stimulus)\s*=\s*["\']?([a-zA-Z0-9_\-]+)["\']?', line_s, re.IGNORECASE)
        if stim_match:
            line_stim_id = stim_match.group(1)

        q_match = re.match(r"^(?:\*\*)?(?:(?:Câu|Question)\s*)?(\d{1,4})[\.\:\)]*(?:\*\*)?\s*(.*)", line_s, re.IGNORECASE)
        if q_match and len(line_s) > 3 and not re.match(r"^\d{1,4}$", line_s):
            if current_q and (current_q["stem"] or current_q["options"]):
                questions.append(current_q)
            q_num = q_match.group(1).strip()
            q_stem = q_match.group(2).strip()
            current_q = {
                "id": f"q_{q_num}_{uuid.uuid4().hex[:6]}",
                "question_number": q_num,
                "stem": q_stem,
                "options": [],
                "correct_answer": "",
                "explanation": "",
                "stimulus_id": line_stim_id,
            }
        elif current_q:
            if line_stim_id and not current_q.get("stimulus_id"):
                current_q["stimulus_id"] = line_stim_id
            opt_match = re.match(r"^(?:[\-\*]\s*)?(\*\*|\b)?([A-D])[\.\:\)](\*\*)?\s*(.*)", line_s)
            if opt_match:
                opt_lbl = opt_match.group(2).upper()
                opt_txt = opt_match.group(4).strip()
                current_q["options"].append({"label": opt_lbl, "text": opt_txt})
            elif line_s.lower().startswith("hướng dẫn") or line_s.lower().startswith("lời giải") or line_s.lower().startswith("giải thích"):
                current_q["explanation"] = line_s
            else:
                if not current_q["options"]:
                    current_q["stem"] = (current_q["stem"] + " " + line_s).strip()
                else:
                    current_q["options"][-1]["text"] = (current_q["options"][-1]["text"] + " " + line_s).strip()

    if current_q and (current_q["stem"] or current_q["options"]):
        questions.append(current_q)

    for q in questions:
        inline_opts = re.findall(r'\(?([A-D])[\.\:\)]\s*([^(\n]+)', q["stem"])
        if len(inline_opts) >= 2:
            stem_parts = re.split(r'\(?[A-D][\.\:\)]', q["stem"], maxsplit=1)
            if stem_parts:
                q["stem"] = stem_parts[0].strip()
            q["options"] = [{"label": label, "text": text.strip()} for label, text in inline_opts]
        elif not q["options"]:
            q["options"] = [
                {"label": "A", "text": "Phương án A"},
                {"label": "B", "text": "Phương án B"},
                {"label": "C", "text": "Phương án C"},
                {"label": "D", "text": "Phương án D"},
            ]

    return questions
