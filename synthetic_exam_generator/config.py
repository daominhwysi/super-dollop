import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "parser_config.yaml"
if not DEFAULT_CONFIG_PATH.exists():
    if (PROJECT_ROOT / "parser_config.yaml").exists():
        DEFAULT_CONFIG_PATH = PROJECT_ROOT / "parser_config.yaml"
    elif (PROJECT_ROOT / "config.yaml").exists():
        DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Default configuration fallback
DEFAULT_CONFIG: Dict[str, Any] = {
    "providers": {
        "codex": {
            "base_url": "",
            "api_key_env": "OPENAI_API_KEY",
        },
        "xah": {
            "base_url": "https://api.xah.io/v1",
            "api_key_env": "XAH_API_KEY",
        },
        "vilao": {
            "base_url": "https://api.vilao.ai/v1",
            "api_key_env": "LLM_API_KEY",
        },
        "nvidia": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "commandcode": {
            "base_url": "http://127.0.0.1:3050/v1",
            "api_key_env": "CMD_API_KEY",
        },
        "agy": {
            "base_url": "local://agy-cli",
            "api_key_env": "",
        },
    },
    "generation": {
        "model": "gpt-5.6-luna",
        "provider": "codex",
        "thinking": "low",
        "concurrency": 2,
        "num_exams": 50,
        "output_dir": "data/synthetic_exams",
        "curriculum_dir": "data/curriculum",
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merges dictionary update into base."""
    merged = dict(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Loads project YAML configuration file with fallbacks.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return DEFAULT_CONFIG

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        return deep_merge(DEFAULT_CONFIG, user_config)
    except Exception as e:
        print(f"Warning: Failed to parse '{path}': {e}. Using default config.")
        return DEFAULT_CONFIG
