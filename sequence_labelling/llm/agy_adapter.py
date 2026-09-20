"""
Antigravity (agy) Provider Adapter.

Integrates the local Antigravity (agy) subscription CLI as a first-class LLM provider:
- Fast Text Mode: --disable-slash-commands with JSON output (~5s)
- Sandboxed Vision Mode: Ephemeral isolated temp folder with --agent research --sandbox
- Conversation Memory: Tracks and resumes context via conversation_id
- Concurrency Management: Threading semaphore to protect local session DB
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Protect local CLI session storage by limiting concurrent executions
_AGY_LOCK = threading.BoundedSemaphore(2)

DEFAULT_AGY_MODEL = "gemini-3.8-flash-high"


def format_messages_to_prompt(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Formats messages list or prompt+system into a unified instruction prompt.
    Returns (system_instruction, combined_user_prompt).
    """
    system_text = system or "You are a helpful assistant"
    user_parts: List[str] = []

    if messages:
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role == "system":
                system_text = content
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")
        if not user_parts and prompt:
            user_parts.append(prompt.strip())
    elif prompt:
        user_parts.append(prompt.strip())

    combined_user_text = "\n\n".join(user_parts)
    return system_text, combined_user_text


def _clean_response(text: str) -> str:
    """Strips standard delimiters and thinking tags if present."""
    # Strip <think>...</think> if model output includes reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<<<TARGET_TEXT_START>>>\s*", "", cleaned)
    cleaned = re.sub(r"\s*<<<TARGET_TEXT_END>>>$", "", cleaned)
    cleaned = re.sub(r"^<<<START>>>\s*", "", cleaned)
    cleaned = re.sub(r"\s*<<<END>>>$", "", cleaned)
    return cleaned.strip()


def _resolve_model_and_effort(model: Optional[str], thinking: Optional[Any]) -> Tuple[str, Optional[str]]:
    """
    Resolves the target model and reasoning effort flag for agy CLI.
    Supports both model names with effort suffixes (e.g. 'gemini-3.8-flash-high')
    and base model names with separate thinking/effort parameter (e.g. model='gemini-3.8-flash', thinking='medium').
    """
    raw_model = (model or DEFAULT_AGY_MODEL).strip()

    effort = None
    if thinking is not None:
        t_str = str(thinking).strip().lower()
        if t_str in ("high", "xhigh", "max", "true"):
            effort = "high"
        elif t_str in ("low", "min", "minimal"):
            effort = "low"
        elif t_str in ("medium", "med", "default", "false"):
            effort = "medium"

    for suffix in ("high", "medium", "low"):
        if raw_model.endswith(f"-{suffix}"):
            if effort is None:
                effort = suffix
            base_model = raw_model[: -(len(suffix) + 1)]
            return base_model, effort

    if effort is None and ("flash" in raw_model or "gemini" in raw_model or "pro" in raw_model):
        effort = "high"

    return raw_model, effort


def chat_with_agy(
    prompt: Optional[str] = None,
    system: str = "You are a helpful assistant",
    model: Optional[str] = None,
    thinking: Optional[Any] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None,
    conversation_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Synchronously calls agy CLI under the active Antigravity subscription.
    Returns (response_text, new_conversation_id, usage_metadata).
    timeout: Optional timeout in seconds. None means no timeout.
    """
    target_model, effort = _resolve_model_and_effort(model=model, thinking=thinking)
    system_text, user_text = format_messages_to_prompt(messages=messages, prompt=prompt, system=system)
    print_timeout_arg = f"{int(timeout)}s" if timeout is not None else "24h"

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode 1: Sandboxed Vision Mode (If Image is Attached)
    # ─────────────────────────────────────────────────────────────────────────────
    if image_bytes:
        filename = image_filename or "target_image.png"
        with tempfile.TemporaryDirectory(prefix="agy_vision_sandbox_") as sandbox_dir:
            image_path = Path(sandbox_dir) / filename
            image_path.write_bytes(image_bytes)

            vision_prompt = (
                f"System instruction: {system_text}\n\n"
                f"Inspect the image '{filename}' located in your current workspace directory. "
                f"{user_text}\n"
                "Output strictly the final response. Do NOT output conversational greetings or schedule tasks."
            )

            cmd = [
                "agy",
                "--sandbox",
                "--agent", "research",
                "--model", target_model,
                "--print-timeout", print_timeout_arg,
                "--dangerously-skip-permissions",
                "--output-format", "json",
            ]
            if effort:
                cmd.extend(["--effort", effort])
            cmd.extend(["-p", vision_prompt])
            if conversation_id:
                cmd.extend(["--conversation", conversation_id])

            with _AGY_LOCK:
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=sandbox_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    raise TimeoutError(f"Sandboxed agy vision execution timed out after {timeout}s")

            if proc.returncode != 0:
                raise RuntimeError(f"agy vision failed (exit {proc.returncode}): {proc.stderr.strip()}")

            raw_stdout = proc.stdout.strip()
            return _parse_agy_json_output(raw_stdout)

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode 2: Fast Pure-Text Mode (Zero Tools, ~5s Turnaround)
    # ─────────────────────────────────────────────────────────────────────────────
    full_prompt = (
        f"System instruction: {system_text}\n\n"
        f"{user_text}"
    )

    cmd = [
        "agy",
        "--disable-slash-commands",
        "--model", target_model,
        "--print-timeout", print_timeout_arg,
        "--output-format", "json",
    ]
    if effort:
        cmd.extend(["--effort", effort])
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])

    with _AGY_LOCK:
        try:
            proc = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"agy text execution timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(f"agy execution failed (exit {proc.returncode}): {proc.stderr.strip()}")

    raw_stdout = proc.stdout.strip()
    return _parse_agy_json_output(raw_stdout)


def _parse_agy_json_output(raw_stdout: str) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Parses JSON output returned by agy --output-format json."""
    if not raw_stdout:
        return "", None, {}

    try:
        # If agy prints warnings or progress before the JSON line, find the last JSON object
        json_start = raw_stdout.rfind("{\n  \"conversation_id\":")
        if json_start == -1:
            json_start = raw_stdout.find("{")
        json_end = raw_stdout.rfind("}")

        if json_start != -1 and json_end > json_start:
            payload_str = raw_stdout[json_start : json_end + 1]
            data = json.loads(payload_str)
            raw_response = data.get("response", "")
            conv_id = data.get("conversation_id")
            usage = data.get("usage", {})
            return _clean_response(raw_response), conv_id, usage
    except Exception as exc:
        logger.warning(f"Failed to decode agy JSON output: {exc}. Falling back to raw text.")

    # Fallback to raw stdout if JSON parse fails
    return _clean_response(raw_stdout), None, {}
