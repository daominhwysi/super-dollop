"""
Antigravity (agy) Provider Adapter — Proxied via unified-llm-proxy (port 8765).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

from sequence_labelling.llm.agy_rate_limiter import Agy5HourCapError, CapLimitDetector

PROXY_BASE_URL = os.environ.get("UNIFIED_LLM_PROXY_URL", "http://127.0.0.1:8765/v1")
_client = OpenAI(base_url=PROXY_BASE_URL, api_key="sk-proxy")

DEFAULT_AGY_MODEL = "gemini-3.8-flash-high"


def chat_with_agy(
    prompt: Optional[str] = None,
    system: str = "You are a helpful assistant",
    model: Optional[str] = None,
    thinking: Optional[Any] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    images: Optional[List[Tuple[str, bytes]]] = None,
    conversation_id: Optional[str] = None,
    timeout: Optional[float] = None,
    caller: Optional[str] = None,
    auto_log: bool = True,
    account: Optional[str] = None,
    agent: Optional[str] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    target_model = model or DEFAULT_AGY_MODEL
    if messages is not None:
        chat_messages = list(messages)
    else:
        chat_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt or ""},
        ]

    kwargs: Dict[str, Any] = {
        "model": target_model,
        "messages": chat_messages,
    }
    if thinking is not None:
        kwargs["extra_body"] = {"thinking": thinking, "account": account, "agent": agent, "caller": caller or "sequence_labelling"}
    else:
        kwargs["extra_body"] = {"account": account, "agent": agent, "caller": caller or "sequence_labelling"}

    if timeout:
        kwargs["timeout"] = float(timeout)

    resp = _client.chat.completions.create(**kwargs)
    reply_text = resp.choices[0].message.content or ""
    usage = {}
    if hasattr(resp, "usage") and resp.usage:
        usage = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(resp.usage, "total_tokens", 0),
        }

    return reply_text, resp.id, usage
