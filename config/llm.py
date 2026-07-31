"""
Deus — Gemini LLM Configuration

Configures the google.genai SDK with the API key from settings.
Provides a shared client and helper methods for generating structured responses.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from google import genai
from google.genai import types
from openai import AsyncOpenAI
from pydantic import TypeAdapter

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

T = TypeVar("T")


def strip_code_fence(text: str) -> str:
    """Models occasionally wrap JSON in markdown despite being told not to."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_structured(text: str, schema: type[T] | Any) -> T:
    """
    Validate raw LLM text against `schema` — a Pydantic model, or a container
    of one such as `list[TickerNote]`.

    This is the *second* line of defence, for providers that cannot constrain
    decoding to a schema (DeepSeek) and for the rare Gemini response that
    arrives without `.parsed` populated. The first line is `response_schema`
    on the request itself; prefer that wherever the provider supports it.

    `strict=False` is the load-bearing argument: models writing multi-paragraph
    prose into a string field routinely emit real newlines instead of `\\n`,
    and a raw control character inside a JSON string is a hard parse error for
    the default decoder. Tolerating them here turns the single most common
    malformed response into a successful parse rather than a fallback that
    leaks raw JSON to the caller.
    """
    payload = json.loads(strip_code_fence(text), strict=False)
    return TypeAdapter(schema).validate_python(payload)


def salvage_json_field(text: str, field: str) -> str | None:
    """
    Last resort: pull one string field out of a JSON blob that will not parse
    at all (truncated mid-object, doubled closing brace, etc).

    Exists so a malformed response degrades to *the prose we wanted* instead of
    dumping a raw `{"executive_summary": ...}` blob into the UI.
    """
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"(.*?)"\s*[,}}]', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"', strict=False)
    except json.JSONDecodeError:
        return match.group(1).replace("\\n", "\n").replace('\\"', '"').strip() or None


def get_deepseek_client() -> AsyncOpenAI | None:
    """Get a configured DeepSeek Client if API key exists."""
    if not settings.deepseek_api_key:
        log.warning("deepseek.missing_key", reason="No API key found in settings")
        return None
        
    return AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

def is_deepseek_configured() -> bool:
    """Check if the DeepSeek LLM is configured and ready to use."""
    return bool(settings.deepseek_api_key)


# Status codes worth another attempt: rate limits, timeouts, and the 5xx family.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    """
    True if `exc` is worth retrying, False if a retry would just re-buy the
    same failure.

    The distinction matters because every call in this pipeline runs at
    temperature=0.0: a response that failed to parse will fail to parse again,
    byte for byte, so retrying it costs three times the tokens for the same
    outcome. Only genuine infrastructure faults get a second attempt.
    """
    import asyncio as _asyncio

    import httpx
    import openai

    if isinstance(exc, (
        _asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.InternalServerError,
    )):
        return True

    # Deterministic client errors — bad request, auth, malformed payload.
    if isinstance(exc, (openai.BadRequestError, openai.AuthenticationError,
                        openai.PermissionDeniedError, openai.NotFoundError)):
        return False

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS

    # google-genai surfaces HTTP failures as text; fall back to sniffing it.
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("timeout", "timed out", "unavailable", "rate limit",
                       "429", "500", "502", "503", "504", "overloaded")
    )
