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
class UsageTally:
    """
    Process-wide running totals for every tracked LLM call.

    `llm_usage_log` already holds the per-call truth, but a caller that wants
    "how many calls did *this* pipeline cycle make" would have to query the
    table by timestamp and hope no other lane wrote in the same window. A
    monotonic counter makes it a subtraction instead: snapshot before, snapshot
    after, diff.

    That is what `pipeline_metrics.llm_calls_count` needed — the orchestrator
    initialised the field and nothing ever incremented it, so every cycle
    recorded 0 calls and $0.00 and the dashboard's LLM-calls cell was a
    permanent zero.

    Deliberately not thread-safe beyond CPython's per-bytecode atomicity: these
    are telemetry counters, and a lost increment under a race costs one call in
    a cycle count, not correctness anywhere.
    """

    calls: int = 0
    errors: int = 0
    cost: float = 0.0

    def snapshot(self) -> "UsageTally":
        """A detached copy, for diffing against a later state."""
        return UsageTally(calls=self.calls, errors=self.errors, cost=self.cost)


# The module-level instance every `track_llm` increments. Imported as
# `from config.usage import tally` — rebinding the name in a caller's module
# would detach it from the one track_llm writes to, so read it through the
# module (`config.usage.tally`) anywhere the value matters at call time.
tally = UsageTally()


@dataclass
class UsageRecord:
    """Caller assigns `.response`; everything else is derived from it."""
    response: Any = None
    # Set directly when a provider reports usage in a shape this module cannot
    # introspect — notably a stream, where usage arrives on a trailing chunk.
    prompt_tokens: Optional[int] = None
    candidate_tokens: Optional[int] = None
    response_text: Optional[str] = None
    # What the call actually cost, as reported by the provider. This replaces
    # a hardcoded per-model price table that had to be hand-updated every time
    # a provider repriced — and was silently 3x wrong when it wasn't.
    cost: Optional[float] = None


def _extract_usage(rec: UsageRecord) -> tuple[int, int, Optional[str], Optional[float]]:
    """
    Normalises a response into (prompt, candidate, text, cost).

    Reads `usage.prompt_tokens` / `.completion_tokens` — the OpenAI-compatible
    shape every call now speaks — and the `cost` OpenRouter reports alongside
    them. Explicit values set on the record always win, which is how streams
    report: their usage arrives on a trailing chunk, after the response object
    the caller already handed over.
    """
    prompt = rec.prompt_tokens
    candidate = rec.candidate_tokens
    text = rec.response_text
    cost = rec.cost
    response = rec.response

    if response is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            if prompt is None:
                prompt = getattr(usage, "prompt_tokens", None)
            if candidate is None:
                candidate = getattr(usage, "completion_tokens", None)
        if cost is None:
            cost = getattr(response, "cost", None)
            if cost is None and usage is not None:
                from config.llm import response_cost
                cost = response_cost(usage)

    if text is None and response is not None:
        text = getattr(response, "text", None)
        if text is None:
            try:
                text = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                text = None

    return int(prompt or 0), int(candidate or 0), text, cost


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

    It is the per-call-site opt-in only. `settings.log_llm_payloads` is the
    global kill switch and takes precedence: with it off, payloads from
    successful calls are dropped in `Database.log_llm_usage` no matter what
    `store_text` says. Failed calls always keep theirs.

    Never swallows the underlying exception — it logs the failure and re-raises,
    so retry and fallback logic upstream still sees it.
    """
    rec = UsageRecord()
    start = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        tally.calls += 1
        tally.errors += 1
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
        prompt_tokens, candidate_tokens, response_text, cost = _extract_usage(rec)
        # Counted here rather than in Database.log_llm_usage so that a failure to
        # write the row cannot also lose the count.
        tally.calls += 1
        tally.cost += float(cost or 0.0)
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
                cost_usd=cost,
            )
        except Exception as log_error:
            log.warning("usage.log_failed", operation=operation, error=str(log_error))
