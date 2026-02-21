# Copyright 2026 ThisIsHwang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_int(env_var: str, default: int) -> int:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_languages(env_var: str, default: str) -> List[str]:
    raw = os.getenv(env_var, default)
    langs = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if langs:
        return langs
    fallback = [part.strip().lower() for part in default.split(",") if part.strip()]
    return fallback or ["en"]


def _get_float(env_var: str, default: float) -> float:
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    hf_base_url: str = os.getenv("HF_BASE_URL", "https://huggingface.co")
    top_k: int = _get_int("TOP_K", 10)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    openai_tts_model: str = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    openai_tts_voice: str = os.getenv("OPENAI_TTS_VOICE", "alloy")
    tts_style_instruction: Optional[str] = os.getenv("TTS_STYLE_INSTRUCTION")
    tts_speed: float = _get_float("TTS_SPEED", 1.2)
    youtube_client_secrets: Optional[str] = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE")
    youtube_token_file: Optional[str] = os.getenv("YOUTUBE_TOKEN_FILE")
    output_base_dir: str = os.getenv("OUTPUT_BASE_DIR", "./outputs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    languages: List[str] = field(default_factory=lambda: _get_languages("LANGUAGES", "en,ko"))


def load_config() -> Config:
    return Config()
