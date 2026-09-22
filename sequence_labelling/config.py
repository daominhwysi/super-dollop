"""Configuration for the standalone Vietnamese sequence-labelling worker."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR
load_dotenv(PROJECT_DIR / ".env")
CONFIG_FILE = PROJECT_DIR / "configs" / "parser_config.yaml"
if not CONFIG_FILE.exists():
    if (PROJECT_DIR / "parser_config.yaml").exists():
        CONFIG_FILE = PROJECT_DIR / "parser_config.yaml"
    elif (PROJECT_DIR / "config.yaml").exists():
        CONFIG_FILE = PROJECT_DIR / "config.yaml"
DATA_DIR = Path(os.environ.get("SEQUENCE_LABEL_DATA_DIR", PROJECT_DIR / "data")).expanduser().resolve()
MODEL_DIR = Path(os.environ.get("SEQUENCE_LABEL_MODEL_DIR", PROJECT_DIR / "models")).expanduser().resolve()
ARTIFACTS_DIR = Path(os.environ.get("SEQUENCE_LABEL_ARTIFACTS_DIR", PROJECT_DIR / "artifacts")).expanduser().resolve()
LOGS_DIR = ARTIFACTS_DIR / "ocr_logs"
LLM_LOGS_DIR = ARTIFACTS_DIR / "llm_logs"
TOKEN_USAGE_DIR = ARTIFACTS_DIR / "token_usage"
TOKEN_USAGE_DIR.mkdir(parents=True, exist_ok=True)

config_data: dict[str, Any] = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
providers = config_data.get("providers", {})
models = config_data.get("models", {})
ocr_cfg = models.get("ocr", {})
parser_cfg = models.get("parser", {})
linker_cfg = models.get("linker", {})
reviewer_cfg = models.get("reviewer", {})
chunker_cfg = models.get("chunker", {})

OCR_MODEL = ocr_cfg.get("model_name")
OCR_PROVIDER = ocr_cfg.get("provider")
OCR_BATCH_SIZE = int(ocr_cfg.get("batch_size", 5))
OCR_CONCURRENCY = int(ocr_cfg.get("concurrency", 1))
PARSER_MODEL = parser_cfg.get("model_name")
PARSER_PROVIDER = parser_cfg.get("provider")
PARSER_THINKING = parser_cfg.get("thinking")
PARSER_MAX_TOKENS = parser_cfg.get("max_tokens")
LINKER_MODEL = linker_cfg.get("model_name")
LINKER_PROVIDER = linker_cfg.get("provider")
CHUNKER_TARGET_TOKENS = int(chunker_cfg.get("target_tokens", 12000))
CHUNKER_MAX_TOKENS = int(chunker_cfg.get("max_tokens", 16000))
CHUNKER_OVERLAP_PAGES = int(chunker_cfg.get("overlap_pages", 0))
REVIEWER_MODEL = reviewer_cfg.get("model_name") or PARSER_MODEL
REVIEWER_PROVIDER = reviewer_cfg.get("provider") or "deepseek"
REVIEWER_THINKING = reviewer_cfg.get("thinking") or "medium"
REVIEWER_MIN_SCORE = int(reviewer_cfg.get("min_score_threshold", 75))
editor_cfg = models.get("editor", {})
EDITOR_MODEL = editor_cfg.get("model_name") or PARSER_MODEL
EDITOR_PROVIDER = editor_cfg.get("provider") or PARSER_PROVIDER
EDITOR_THINKING = editor_cfg.get("thinking") or "low"
FIGURE_CFG = ocr_cfg.get("figure_detection", {})
FIGURE_DETECTION_ENABLED = bool(FIGURE_CFG.get("enabled", True))
FIGURE_MODEL_PATH = MODEL_DIR / Path(FIGURE_CFG.get("model_path", "iter1-haswell-int8.onnx")).name
FIGURE_CONFIDENCE_THRESHOLD = float(FIGURE_CFG.get("confidence_threshold", 0.3))
FIGURE_CLASS_NAMES = tuple(FIGURE_CFG.get("class_names", ["bangbienthien", "class_1"]))
FIGURE_INCLUDED_CLASS_IDS = tuple(FIGURE_CFG.get("included_class_ids", [1]))


def get_provider_base_url(provider_name: str) -> str:
    return str(providers.get(provider_name.lower(), {}).get("base_url", ""))


def get_provider_api_key(provider_name: str) -> str:
    aliases = {
        "commandcode": ("CMD_API_KEY", "COMMANDCODE_API_KEY", "DEEPSEEK_API_KEY"),
        "xah": ("XAH_API_KEY", "LLM_API_KEY"),
        "nvidia": ("NVIDIA_API_KEY", "DEEPSEEK_API_KEY"),
        "vilao": ("LLM_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "codex": ("CODEX_API_KEY", "OPENAI_API_KEY"),
    }
    configured = providers.get(provider_name.lower(), {}).get("api_key_env")
    names = (configured,) if configured else aliases.get(provider_name.lower(), ())
    for name in names:
        if name and os.environ.get(name):
            return os.environ[name]
    return ""
