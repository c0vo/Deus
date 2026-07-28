"""
Project Scrooge V2 — Gemini LLM Configuration

Configures the google.genai SDK with the API key from settings.
Provides a shared client and helper methods for generating structured responses.
"""

from __future__ import annotations

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from config.logging_config import get_logger
from config.settings import settings

log = get_logger(__name__)

# Default safety settings (we want market news, sometimes involves crime/hacks/etc)
DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

def get_client() -> genai.Client | None:
    """Get a configured GenAI Client if API key exists."""
    if not settings.gemini_api_key:
        log.warning("gemini.missing_key", reason="No API key found in settings")
        return None
        
    return genai.Client(api_key=settings.gemini_api_key)

def is_configured() -> bool:
    """Check if the LLM is configured and ready to use."""
    return bool(settings.gemini_api_key)

def get_deepseek_client() -> AsyncOpenAI | None:
    """Get a configured DeepSeek Client if API key exists."""
    if not settings.deepseek_api_key:
        log.warning("deepseek.missing_key", reason="No API key found in settings")
        return None
        
    return AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

def is_deepseek_configured() -> bool:
    """Check if the DeepSeek LLM is configured and ready to use."""
    return bool(settings.deepseek_api_key)
