import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sequence_labelling.config import LLM_LOGS_DIR


def _to_int(value: Any) -> Optional[int]:
    """Convert token-like values to int when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    """
    Normalize usage-like payloads across OpenAI-like SDK/object or plain dict formats.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump") and callable(usage.model_dump):
        try:
            dumped = usage.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(usage, "dict") and callable(usage.dict):
        try:
            dumped = usage.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "completion_tokens_details": getattr(usage, "completion_tokens_details", None),
    }


def _extract_usage_tokens(usage: Any) -> Dict[str, Optional[int]]:
    """Extract prompt/completion/total/reasoning token counts from usage payload."""
    usage_dict = _usage_to_dict(usage)

    prompt_tokens = _to_int(usage_dict.get("prompt_tokens"))
    completion_tokens = _to_int(usage_dict.get("completion_tokens"))
    total_tokens = _to_int(usage_dict.get("total_tokens"))

    details = usage_dict.get("completion_tokens_details")
    if details is not None and not isinstance(details, dict):
        details = _usage_to_dict(details)
    reasoning_tokens = _to_int(details.get("reasoning_tokens")) if isinstance(details, dict) else None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def _max_or_default(current: int, candidate: Optional[int]) -> int:
    if candidate is None:
        return current
    if current == 0:
        return candidate
    return max(current, candidate)


def _extract_response_components(response: Any) -> Tuple[str, str, str]:
    """
    Extracts (output_text, reasoning_content, reasoning_summary) from LLM response objects:
    - OpenAI `client.responses.create` format (response.output array with type="reasoning" and summary)
    - OpenAI / DeepSeek / NVIDIA `chat.completions.create` format (reasoning_content, model_extra)
    """
    output_parts: List[str] = []
    reasoning_parts: List[str] = []
    summary_parts: List[str] = []

    if response is None:
        return "", "", ""

    if isinstance(response, str):
        return response, "", ""

    if hasattr(response, "final_response") and getattr(response, "final_response") is not None:
        return str(getattr(response, "final_response")), "", ""

    # 1. Handle OpenAI `client.responses.create` format
    outputs = getattr(response, "output", None)
    if outputs is None and isinstance(response, dict):
        outputs = response.get("output")

    if isinstance(outputs, list):
        for item in outputs:
            item_dict = item.model_dump() if hasattr(item, "model_dump") and callable(item.model_dump) else (item if isinstance(item, dict) else {})
            item_type = item_dict.get("type") or getattr(item, "type", None)

            if item_type == "reasoning":
                summary_data = item_dict.get("summary") or getattr(item, "summary", None)
                if isinstance(summary_data, list):
                    for sum_item in summary_data:
                        sum_dict = sum_item.model_dump() if hasattr(sum_item, "model_dump") and callable(sum_item.model_dump) else (sum_item if isinstance(sum_item, dict) else {})
                        text = sum_dict.get("text") or getattr(sum_item, "text", "")
                        if text:
                            summary_parts.append(str(text))
                elif isinstance(summary_data, str) and summary_data:
                    summary_parts.append(summary_data)
            elif item_type == "message":
                content_data = item_dict.get("content") or getattr(item, "content", None)
                if isinstance(content_data, list):
                    for c_item in content_data:
                        c_dict = c_item.model_dump() if hasattr(c_item, "model_dump") and callable(c_item.model_dump) else (c_item if isinstance(c_item, dict) else {})
                        text = c_dict.get("text") or getattr(c_item, "text", "")
                        if text:
                            output_parts.append(str(text))

    # 2. Handle Chat Completions format (choices[0].message)
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if isinstance(choices, list) and choices:
        choice = choices[0]
        msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if msg:
            if isinstance(msg, dict):
                c_text = msg.get("content") or ""
                r_text = msg.get("reasoning_content") or ""
                r_obj = msg.get("reasoning") or {}
                m_extra = msg.get("model_extra") or {}
            else:
                c_text = getattr(msg, "content", None) or ""
                r_text = getattr(msg, "reasoning_content", None) or ""
                r_obj = getattr(msg, "reasoning", None) or {}
                m_extra = getattr(msg, "model_extra", None) or {}

            if c_text:
                output_parts.append(str(c_text))
            if r_text:
                reasoning_parts.append(str(r_text))

            if isinstance(r_obj, str) and r_obj:
                reasoning_parts.append(r_obj)
            elif isinstance(r_obj, dict):
                s_text = r_obj.get("summary") or r_obj.get("text")
                if s_text:
                    summary_parts.append(str(s_text))

            if isinstance(m_extra, dict):
                extra_r = m_extra.get("reasoning_content") or m_extra.get("reasoning")
                if extra_r and str(extra_r) not in reasoning_parts:
                    reasoning_parts.append(str(extra_r))
                extra_s = m_extra.get("reasoning_summary") or m_extra.get("summary")
                if extra_s and str(extra_s) not in summary_parts:
                    summary_parts.append(str(extra_s))

    output_text = "\n".join(output_parts).strip()
    reasoning_content = "\n\n".join(reasoning_parts).strip()
    reasoning_summary = "\n\n".join(summary_parts).strip()
    return output_text, reasoning_content, reasoning_summary


class StreamingLLMLogger:
    """
    Live/Streaming LLM Logger that writes and periodically updates (flushes every 5s)
    logs/llm_logs/<YYYY-MM-DD>/<request>.md as stream chunks arrive.
    """
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        model: str = "",
        provider: str = "",
        request_id: Optional[str] = None,
        flush_interval_sec: float = 5.0,
        start_time: Optional[float] = None,
    ):
        self.messages = messages
        self.model = model
        self.provider = provider
        self.flush_interval_sec = flush_interval_sec
        self.start_time = start_time if start_time is not None else time.time()

        today_str = datetime.now().strftime("%Y-%m-%d")
        self.day_dir = LLM_LOGS_DIR / today_str
        self.day_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(self.day_dir.glob("*.md")))
        next_idx = existing_count + 1

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        req_uuid = uuid.uuid4().hex[:8]
        if request_id:
            safe_req_id = "".join(
                c if c.isalnum() or c in ("-", "_", ".") else "_"
                for c in str(request_id)
            )
            self.req_id = safe_req_id
            self.req_filename = f"{next_idx:03d}_req_{self.req_id}.md"
        else:
            self.req_id = req_uuid
            self.req_filename = f"{next_idx:03d}_req_{timestamp_str}_{self.req_id}.md"
        self.log_file = self.day_dir / self.req_filename

        self.output_text_chunks: List[str] = []
        self.reasoning_chunks: List[str] = []
        self.reasoning_summary_chunks: List[str] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.reasoning_tokens = 0
        self.last_flush_time = self.start_time
        self._estimated_prompt_tokens = sum(
            len(str(msg.get("content", "")).split())
            for msg in messages
            if isinstance(msg, dict) and msg.get("content") is not None
        )

        # Initial flush to create log file immediately
        self.flush(is_final=False)

    def append_chunk(
        self,
        content: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        usage: Optional[Any] = None
    ):
        if content:
            self.output_text_chunks.append(str(content))
        if reasoning:
            self.reasoning_chunks.append(str(reasoning))
        if reasoning_summary:
            self.reasoning_summary_chunks.append(str(reasoning_summary))

        if usage:
            usage_counts = _extract_usage_tokens(usage)
            self.prompt_tokens = _max_or_default(self.prompt_tokens, usage_counts["prompt_tokens"])
            self.completion_tokens = _max_or_default(self.completion_tokens, usage_counts["completion_tokens"])
            self.total_tokens = _max_or_default(self.total_tokens, usage_counts["total_tokens"])
            if usage_counts["reasoning_tokens"] is not None:
                self.reasoning_tokens = _max_or_default(self.reasoning_tokens, usage_counts["reasoning_tokens"])

        now = time.time()
        if now - self.last_flush_time >= self.flush_interval_sec:
            self.flush(is_final=False)

    def flush(self, is_final: bool = False):
        now = time.time()
        self.last_flush_time = now
        duration_sec = max(0.001, now - self.start_time)
        full_output = "".join(self.output_text_chunks)
        full_reasoning = "".join(self.reasoning_chunks)
        full_summary = "".join(self.reasoning_summary_chunks)
        resolved_prompt_tokens = self.prompt_tokens or self._estimated_prompt_tokens
        resolved_completion_tokens = self.completion_tokens or (len(full_output.split()) if full_output else 0)
        resolved_total_tokens = self.total_tokens or (resolved_prompt_tokens + resolved_completion_tokens)
        resolved_reasoning_tokens = (
            self.reasoning_tokens
            if self.reasoning_tokens
            else (len(full_reasoning.split()) if full_reasoning else 0)
        )

        md = []
        status_suffix = " (Completed)" if is_final else " (Streaming... ⏳)"
        md.append(f"# 🤖 LLM Request Log: `{self.req_filename[:-3]}`{status_suffix}")
        md.append(f"- **Timestamp:** `{datetime.now().isoformat()}`")
        md.append(f"- **Model:** `{self.model}`")
        if self.provider:
            md.append(f"- **Provider:** `{self.provider}`")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 📥 Input")
        md.append("")
        for idx, msg in enumerate(self.messages):
            if isinstance(msg, dict):
                role = str(msg.get("role") or f"message_{idx+1}").capitalize()
                content = str(msg.get("content") or "")
            else:
                role = f"Message {idx+1}"
                content = str(msg)

            md.append("<details>")
            md.append(f"<summary>Role: {role}</summary>")
            md.append("")
            md.append("```markdown")
            md.append(content)
            md.append("```")
            md.append("</details>")
            md.append("")

        md.append("---")
        md.append("")
        md.append("## 📤 Output")
        md.append("")
        md.append("<details>")
        md.append("<summary>Completion Output</summary>")
        md.append("")
        md.append("```markdown")
        md.append(full_output if full_output else "*(No completion output emitted)*")
        md.append("```")
        md.append("</details>")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 🧠 Reasoning & Summary")
        md.append(f"- **Reasoning Tokens Count:** `{resolved_reasoning_tokens}`")
        md.append("")
        if full_summary:
            md.append("<details>")
            md.append("<summary>Reasoning Summary</summary>")
            md.append("")
            md.append("```markdown")
            md.append(full_summary)
            md.append("```")
            md.append("</details>")
            md.append("")
        if full_reasoning:
            md.append("<details>")
            md.append("<summary>Full Reasoning Content</summary>")
            md.append("")
            md.append("```markdown")
            md.append(full_reasoning)
            md.append("```")
            md.append("</details>")
            md.append("")
        if not full_summary and not full_reasoning:
            md.append("*(No separate reasoning summary or content emitted)*")
            md.append("")
        md.append("---")
        md.append("")
        md.append("## 📊 Stats")
        md.append(f"- **Execution Time:** `{duration_sec:.3f}s`")
        md.append(f"- **Prompt Tokens:** `{resolved_prompt_tokens:,}`")
        md.append(f"- **Completion Tokens:** `{resolved_completion_tokens:,}`")
        md.append(f"- **Reasoning Tokens:** `{resolved_reasoning_tokens:,}`")
        md.append(f"- **Total Tokens:** `{resolved_total_tokens if resolved_total_tokens else (resolved_prompt_tokens + resolved_completion_tokens + resolved_reasoning_tokens):,}`")
        md.append("")

        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

    def finalize(self):
        self.flush(is_final=True)
        try:
            from sequence_labelling.llm.token_tracker import log_token_usage
            tot_tokens = self.total_tokens if self.total_tokens else (self.prompt_tokens + self.completion_tokens)
            duration = max(0.0, time.time() - self.start_time)
            log_token_usage(
                provider=self.provider or "unknown",
                model=self.model or "unknown",
                input_tokens=self.prompt_tokens,
                output_tokens=max(0, self.completion_tokens - self.reasoning_tokens),
                reasoning_tokens=self.reasoning_tokens,
                total_tokens=tot_tokens,
                duration_sec=duration,
                caller="llm_logger"
            )
        except Exception:
            pass


def log_llm_call(
    messages: List[Dict[str, Any]],
    response: Any,
    model: str = "",
    provider: str = "",
    duration_sec: float = 0.0,
    request_id: Optional[str] = None,
    usage: Optional[Any] = None,
) -> Path:
    """
    Synchronous/One-shot fallback helper for non-streamed LLM responses.
    Extracts output content, reasoning content, and reasoning summary.
    """
    start_t = time.time() - duration_sec
    logger = StreamingLLMLogger(
        messages=messages,
        model=model,
        provider=provider,
        request_id=request_id,
        start_time=start_t
    )

    output_text, reasoning_content, reasoning_summary = _extract_response_components(response)
    resp_usage = usage if usage is not None else getattr(response, "usage", None)
    if resp_usage is None and isinstance(response, dict):
        resp_usage = response.get("usage")

    logger.append_chunk(
        content=output_text,
        reasoning=reasoning_content,
        reasoning_summary=reasoning_summary,
        usage=resp_usage
    )
    logger.finalize()
    return logger.log_file

