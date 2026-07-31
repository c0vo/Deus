"""
Deus — LLM usage instrumentation

One helper so that recording what a call cost is a two-line change at the call
site rather than a thirty-line block of start_time / is_error / try / except /
finally. That verbosity is why roughly a dozen call sites were never
instrumented at all, which in turn is why the cost dashboard under-reported:
the operations missing from `llm_usage_log` included the embedder, both
extraction scanners, the trend forecaster, and the user-facing streaming chat.

Usage:

    with track_llm(self.db, model, "ipo_extract", prompt_text=prompt) as u:
        u.response = await client.chat.completions.create(...)

This is a *sync* context manager on purpose — it contains no awaits itself, so
it wraps async and sync call sites alike.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

from config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class UsageRecord:
    """Caller assigns `.response`; everything else is derived from it."""
    response: Any = None
    # Set directly when a provider reports usage in a shape this module cannot
    # introspect — notably a stream, where usage arrives on a trailing chunk.
    prompt_tokens: Optional[int] = None
    candidate_tokens: Optional[int] = None
    response_text: Optional[str] = None


def _extract_usage(rec: UsageRecord) -> tuple[int, int, Optional[str]]:
    """
    Normalises the two SDK shapes in use into (prompt, candidate, text).

    Gemini exposes `usage_metadata.prompt_token_count` / `.candidates_token_count`;
    the OpenAI-compatible DeepSeek client exposes `usage.prompt_tokens` /
    `.completion_tokens`. Explicit values set on the record always win.
    """
    prompt = rec.prompt_tokens
    candidate = rec.candidate_tokens
    text = rec.response_text
    response = rec.response

    if response is not None and (prompt is None or candidate is None):
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = getattr(response, "usage", None)

        if usage_metadata is not None:
            prompt = prompt if prompt is not None else getattr(usage_metadata, "prompt_token_count", None)
            candidate = candidate if candidate is not None else getattr(usage_metadata, "candidates_token_count", None)
        elif usage is not None:
            prompt = prompt if prompt is not None else getattr(usage, "prompt_tokens", None)
            candidate = candidate if candidate is not None else getattr(usage, "completion_tokens", None)

    if text is None and response is not None:
        text = getattr(response, "text", None)
        if text is None:
            try:
                text = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                text = None

    return int(prompt or 0), int(candidate or 0), text


@contextmanager
def track_llm(
    db,
    model_name: str,
    operation: str,
    prompt_text: Optional[str] = None,
    store_text: bool = False,
):
    """
    Records one LLM call — tokens, latency, and success or failure.

    `store_text` is off by default: `llm_usage_log` keeps the full prompt and
    response when asked to, and at classification volume that turns the table
    into a prompt archive. Turn it on for the low-volume, high-value calls
    worth being able to inspect after the fact.

    Never swallows the underlying exception — it logs the failure and re-raises,
    so retry and fallback logic upstream still sees it.
    """
    rec = UsageRecord()
    start = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        try:
            db.log_llm_usage(
                model_name=model_name,
                operation=operation,
                prompt_tokens=0,
                candidate_tokens=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                is_error=True,
                error_message=str(e),
                prompt_text=prompt_text if store_text else None,
            )
        except Exception as log_error:
            log.warning("usage.log_failed", operation=operation, error=str(log_error))
        raise
    else:
        prompt_tokens, candidate_tokens, response_text = _extract_usage(rec)
        try:
            db.log_llm_usage(
                model_name=model_name,
                operation=operation,
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                latency_ms=int((time.perf_counter() - start) * 1000),
                is_error=False,
                prompt_text=prompt_text if store_text else None,
                response_text=response_text if store_text else None,
            )
        except Exception as log_error:
            log.warning("usage.log_failed", operation=operation, error=str(log_error))
