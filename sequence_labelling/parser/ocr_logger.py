import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from sequence_labelling.config import LOGS_DIR


def create_request_log_folder(request_id: Optional[str] = None) -> Path:
    """
    Creates a new log directory for an OCR/annotate request.
    Folder structure: <LOGS_DIR>/req_YYYYMMDD_HHMMSS_<id>/
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_id = request_id or uuid.uuid4().hex[:8]
    folder_name = f"req_{timestamp}_{req_id}"
    log_folder = LOGS_DIR / folder_name
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder


def log_ocr_annotate_request(
    file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    raw_text_input: Optional[str] = None,
    ocr_text: Optional[str] = None,
    annotation_res: Optional[Dict[str, Any]] = None,
    questions: Optional[List[Dict[str, Any]]] = None,
    duration: float = 0.0,
    status: str = "success",
    error_message: Optional[str] = None,
    ocr_model: Optional[str] = None,
    annotator_model: Optional[str] = None,
    source: str = "api",
) -> Path:
    """
    Logs an OCR and/or Annotate request to a dedicated folder inside LOGS_DIR.

    Saved files in the request folder:
    - metadata.json: Request details, models, status, timestamps, statistics
    - input.pdf or input_raw.txt: Original input provided by the user
    - ocr_output.txt: OCR extracted raw text
    - annotation_result.json: LLM annotation result (spans, tokens, BIO tags)
    - annotation_result.xml: Ground truth tagged XML string
    - parsed_questions.json: Structured question list extracted from annotation/regex
    """
    log_folder = create_request_log_folder()

    # 1. Save input file or raw text input
    if file_bytes:
        ext = Path(file_name).suffix if file_name and Path(file_name).suffix else ".pdf"
        input_file_path = log_folder / f"input{ext}"
        with open(input_file_path, "wb") as f:
            f.write(file_bytes)
    elif raw_text_input:
        with open(log_folder / "input_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_text_input)

    # 2. Save OCR output text
    if ocr_text:
        with open(log_folder / "ocr_output.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)

    # 3. Save annotation outputs (JSON & XML)
    raw_xml = ""
    spans_count = 0
    tokens_count = 0
    if annotation_res:
        raw_xml = annotation_res.get("raw_xml", "")
        spans_count = len(annotation_res.get("spans", []))
        tokens_count = len(annotation_res.get("tokens", []))

        with open(log_folder / "annotation_result.json", "w", encoding="utf-8") as f:
            json.dump(annotation_res, f, ensure_ascii=False, indent=2)

        if raw_xml:
            with open(log_folder / "annotation_result.xml", "w", encoding="utf-8") as f:
                f.write(raw_xml)

    # 4. Save structured parsed questions
    if questions is not None:
        with open(log_folder / "parsed_questions.json", "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

    # 5. Metadata json
    meta = {
        "timestamp": datetime.now().isoformat(),
        "log_folder": log_folder.name,
        "source": source,
        "filename": file_name,
        "status": status,
        "duration_seconds": round(duration, 4),
        "ocr_model": ocr_model,
        "annotator_model": annotator_model,
        "spans_count": spans_count,
        "tokens_count": tokens_count,
        "questions_count": len(questions) if questions else 0,
        "error": error_message,
    }
    with open(log_folder / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return log_folder
