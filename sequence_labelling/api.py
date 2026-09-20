"""HTTP worker API for OCR and sequence labelling jobs."""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .annotator.annotate_ocr import OCRAnnotator
from .annotator.pdf_converter import PDFOCRConverter
from .config import OCR_BATCH_SIZE, OCR_CONCURRENCY, OCR_MODEL, OCR_PROVIDER, PARSER_MODEL, PARSER_PROVIDER
from .llm.deepseek_client import chat
from .parser.parser import parse_spans_into_structured_questions, regex_parse_questions

app = FastAPI(title="Vietnamese Sequence Labelling v2")
jobs: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set(job_id: str, **values: Any) -> None:
    jobs[job_id].update(values)


def _run_job(job_id: str, payload: bytes | None, filename: str | None, raw_text: str | None) -> None:
    _set(job_id, status="processing", progress=5, message="Starting OCR", started_at=_now())
    temp_path: Path | None = None
    try:
        if payload is not None:
            ext = (filename.split(".")[-1].lower() if filename and "." in filename else "pdf")
            if ext in ["png", "jpg", "jpeg", "webp", "bmp"]:
                import pymupdf as fitz
                img_doc = fitz.open(stream=payload, filetype=ext)
                payload = img_doc.convert_to_pdf()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            converter = PDFOCRConverter(model=OCR_MODEL, provider=OCR_PROVIDER, batch_size=OCR_BATCH_SIZE, concurrency=OCR_CONCURRENCY)
            text = converter.convert_pdf(temp_path)
        elif raw_text:
            text = raw_text
        else:
            raise ValueError("Provide either a PDF/image file or raw_text")
        _set(job_id, progress=50, message="Annotating source text")
        try:
            result = OCRAnnotator(model=PARSER_MODEL, provider=PARSER_PROVIDER).annotate_text(text, request_id=job_id)
            questions, stimuli = parse_spans_into_structured_questions(result.get("raw_text", text), result.get("spans", []))
            raw_xml = result.get("raw_xml", "")
        except Exception:
            result = None
            questions, stimuli, raw_xml = regex_parse_questions(text), {}, ""
        _set(job_id, status="completed", progress=100, message="Completed", completed_at=_now(), result={
            "success": True,
            "filename": filename,
            "raw_text": text,
            "raw_xml": raw_xml,
            "spans_count": len(result.get("spans", [])) if result else 0,
            "tokens_count": len(result.get("tokens", [])) if result else 0,
            "questions": questions,
            "stimuli": stimuli,
        })
    except Exception as exc:
        _set(job_id, status="failed", progress=100, message="Failed", error=str(exc), completed_at=_now())
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sequence-labelling-worker"}


@app.post("/v1/jobs")
async def create_job(file: UploadFile | None = File(None), raw_text: str | None = Form(None)) -> dict[str, Any]:
    if not file and not raw_text:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or raw_text")
    payload = await file.read() if file else None
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    jobs[job_id] = {"id": job_id, "status": "pending", "progress": 0, "message": "Queued", "filename": file.filename if file else None, "created_at": _now()}
    asyncio.create_task(asyncio.to_thread(_run_job, job_id, payload, file.filename if file else None, raw_text))
    return jobs[job_id]


@app.get("/v1/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return list(jobs.values())


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.delete("/v1/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, bool]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    jobs.pop(job_id)
    return {"success": True}


@app.post("/v1/answer-keys/map")
async def map_answer_key(file: UploadFile | None = File(None), raw_text: str | None = Form(None), questions: str = Form(...)) -> dict[str, Any]:
    if not file and not raw_text:
        raise HTTPException(status_code=400, detail="Provide either a file or raw_text")
    if file:
        payload = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(payload)
            path = Path(handle.name)
        try:
            extracted = await asyncio.to_thread(PDFOCRConverter(model=OCR_MODEL, provider=OCR_PROVIDER, batch_size=OCR_BATCH_SIZE, concurrency=OCR_CONCURRENCY).convert_pdf, path)
        finally:
            path.unlink(missing_ok=True)
    else:
        extracted = raw_text or ""
    prompt = f"Answer key text:\n{extracted}\n\nQuestions JSON:\n{questions}"
    reply = await asyncio.to_thread(chat, prompt=prompt, system="Return only a JSON object mapping question IDs to answer labels.", model=PARSER_MODEL, provider=PARSER_PROVIDER)
    try:
        start, end = reply.find("{"), reply.rfind("}")
        mapping = json.loads(reply[start:end + 1] if start >= 0 and end > start else reply)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Worker returned invalid answer mapping: {exc}") from exc
    return {"success": True, "extracted_text": extracted, "mapping": mapping}


class ChatCompletionRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None
    system: str | None = None
    model: str | None = None
    provider: str | None = "agy"
    thinking: Any | None = None
    conversation_id: str | None = None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    """
    General OpenAI-compatible chat completion endpoint.
    Routes to agy (or configured provider) under the local subscription ($0 API token cost).
    """
    try:
        reply = await asyncio.to_thread(
            chat,
            prompt=request.prompt,
            system=request.system or "You are a helpful assistant",
            model=request.model or PARSER_MODEL,
            provider=request.provider or "agy",
            thinking=request.thinking,
            messages=request.messages if request.messages else None,
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": reply,
                    }
                }
            ],
            "model": request.model or PARSER_MODEL,
            "provider": request.provider or "agy",
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat execution failed: {exc}") from exc

