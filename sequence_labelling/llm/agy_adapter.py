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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sequence_labelling.llm.agy_rate_limiter import Agy5HourCapError, CapLimitDetector

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
    system_text = (system or "You are a helpful assistant").replace("\x00", "")
    user_parts: List[str] = []

    if messages:
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).replace("\x00", "").strip()
            if role == "system":
                system_text = content
            elif role == "user":
                user_parts.append(content)
            elif role == "assistant":
                user_parts.append(f"Assistant: {content}")
        if not user_parts and prompt:
            user_parts.append(prompt.replace("\x00", "").strip())
    elif prompt:
        user_parts.append(prompt.replace("\x00", "").strip())

    combined_user_text = "\n\n".join(user_parts).replace("\x00", "")
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
            # Model name already has effort suffix built-in (e.g. 'gemini-3.8-flash-low')
            return raw_model, None

    if effort is None and ("flash" in raw_model or "gemini" in raw_model or "pro" in raw_model):
        effort = "high"

    # For gemini models in agy CLI, hyphenated model names are canonical (e.g. gemini-3.8-flash-high)
    if "gemini" in raw_model and effort:
        return f"{raw_model}-{effort}", None

    return raw_model, effort


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
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Synchronously calls agy CLI under the active Antigravity subscription.
    Returns (response_text, new_conversation_id, usage_metadata).
    timeout: Optional timeout in seconds. None means no timeout.
    caller: Optional calling component name for telemetry attribution.
    auto_log: Whether to automatically persist token usage to token_tracker ledger.
    """
    target_model, effort = _resolve_model_and_effort(model=model, thinking=thinking)
    system_text, user_text = format_messages_to_prompt(messages=messages, prompt=prompt, system=system)
    print_timeout_arg = f"{int(timeout)}s" if timeout is not None else "24h"
    start_time = time.time()

    # Normalize attached images
    resolved_images: List[Tuple[str, bytes]] = []
    if images:
        resolved_images.extend(images)
    elif image_bytes:
        resolved_images.append((image_filename or "target_image.png", image_bytes))

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode 1: Sandboxed Vision Mode (If Image(s) Attached)
    # ─────────────────────────────────────────────────────────────────────────────
    if resolved_images:
        with tempfile.TemporaryDirectory(prefix="agy_vision_sandbox_") as sandbox_dir:
            sandbox_path = Path(sandbox_dir)
            filenames = []
            for fname, fbytes in resolved_images:
                img_path = sandbox_path / fname
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(fbytes)
                filenames.append(fname)

            if len(filenames) == 1:
                inspection_target = f"the image '{filenames[0]}'"
            else:
                formatted_list = ", ".join(f"'{f}'" for f in filenames)
                inspection_target = f"the document images: {formatted_list}"

            vision_prompt = (
                f"System instruction: {system_text}\n\n"
                f"CRITICAL WORKFLOW INSTRUCTION:\n"
                f"1. You MUST call `view_file` on {inspection_target} to inspect the image.\n"
                f"2. You are strictly FORBIDDEN from calling `run_command` or any other tool.\n"
                f"3. Immediately after viewing the file, produce the required response.\n\n"
                f"{user_text}\n\n"
                "Output strictly the final response. Do NOT output conversational greetings or schedule tasks."
            ).replace("\x00", "")

            cmd = [
                "agy",
                "--sandbox",
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
                raw_err = f"{proc.stdout}\n{proc.stderr}".strip()
                if CapLimitDetector.is_cap_limit(proc.returncode, proc.stdout, proc.stderr):
                    cooldown = CapLimitDetector.parse_cooldown_seconds(raw_err)
                    raise Agy5HourCapError(
                        f"agy vision cap limit reached ({cooldown}s cooldown): {raw_err[:200]}",
                        reset_after_sec=cooldown,
                        raw_output=raw_err,
                    )
                raise RuntimeError(f"agy vision failed (exit {proc.returncode}): {proc.stderr.strip()}")

            raw_stdout = proc.stdout.strip()
            resp_text, conv_id, usage = _parse_agy_json_output(raw_stdout)
            duration_sec = time.time() - start_time
            if auto_log and usage:
                _auto_log_usage(
                    model=target_model,
                    usage=usage,
                    duration_sec=duration_sec,
                    caller=caller or "agy_adapter_vision",
                )
            return resp_text, conv_id, usage

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode 2: Fast Pure-Text Mode (Zero Tools, Isolated CWD)
    # ─────────────────────────────────────────────────────────────────────────────
    full_prompt = (
        f"System instruction: {system_text}\n\n"
        f"CRITICAL WORKFLOW INSTRUCTION:\n"
        f"You are strictly FORBIDDEN from calling any tools (such as run_command, view_file, write_to_file, manage_task, etc.).\n"
        f"Immediately output ONLY the final text response directly without conversational filler or tool invocations.\n\n"
        f"{user_text}"
    ).replace("\x00", "")

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

    max_retries = 3
    proc = None
    for attempt in range(max_retries + 1):
        with tempfile.TemporaryDirectory(prefix="agy_text_sandbox_") as sandbox_dir:
            with _AGY_LOCK:
                try:
                    proc = subprocess.run(
                        cmd,
                        input=full_prompt,
                        cwd=sandbox_dir,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    raise TimeoutError(f"agy text execution timed out after {timeout}s")

        if proc.returncode == 0:
            break

        raw_err = f"{proc.stdout}\n{proc.stderr}".strip()
        if CapLimitDetector.is_cap_limit(proc.returncode, proc.stdout, proc.stderr):
            if attempt < max_retries:
                backoff_sec = (attempt + 1) * 20
                logger.warning(
                    f"[agy Rate Limit] Transient rate limit / 429 detected on attempt {attempt + 1}/{max_retries + 1}. "
                    f"Backing off {backoff_sec}s before retrying..."
                )
                time.sleep(backoff_sec)
                continue
            cooldown = CapLimitDetector.parse_cooldown_seconds(raw_err)
            raise Agy5HourCapError(
                f"agy text cap limit reached ({cooldown}s cooldown): {raw_err[:200]}",
                reset_after_sec=cooldown,
                raw_output=raw_err,
            )
        raise RuntimeError(f"agy execution failed (exit {proc.returncode}): {proc.stderr.strip()}")

    raw_stdout = proc.stdout.strip()
    resp_text, conv_id, usage = _parse_agy_json_output(raw_stdout)
    duration_sec = time.time() - start_time
    if auto_log and usage:
        _auto_log_usage(
            model=target_model,
            usage=usage,
            duration_sec=duration_sec,
            caller=caller or "agy_adapter_text",
        )
    return resp_text, conv_id, usage


def _auto_log_usage(
    model: str,
    usage: Dict[str, Any],
    duration_sec: float,
    caller: str,
) -> None:
    """Helper to safely record token metrics to token_tracker without raising."""
    try:
        from sequence_labelling.llm.token_tracker import log_token_usage
        log_token_usage(
            provider="agy",
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("thinking_tokens", 0) or usage.get("reasoning_tokens", 0),
            cache_read_tokens=usage.get("cache_read_tokens", 0),
            total_tokens=usage.get("total_tokens"),
            duration_sec=duration_sec,
            caller=caller,
        )
    except Exception as exc:
        logger.debug(f"Failed to auto-log agy token usage: {exc}")


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
