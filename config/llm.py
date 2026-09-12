"""
Deus — LLM access, provider-neutral

Every model call in this project goes through OpenRouter, over one API key and
one OpenAI-compatible client. Which *provider* serves a given function is not a
code decision any more: it is the prefix of the slug in that function's
`MODEL_<FUNCTION>` setting, so moving the ranker from Google to DeepSeek is an
.env edit and nothing else.

That is the whole point of this module. It used to hold two clients — a
`genai.Client` for Gemini and an `AsyncOpenAI` pointed at api.deepseek.com — and
each call site was written against one SDK's request shape. A function could not
change provider without a rewrite, which is exactly the manual work this
removes.

The three provider dialects that used to live at call sites are normalised here:

  * thinking budgets   — `ThinkingLevel.LOW` / `reasoning_effort="xhigh"` /
                         `extra_body={"thinking": {"type": "disabled"}}`
                         all become `reasoning="low" | "xhigh" | "none"`.
  * structured output  — a Pydantic model (or `list[Model]`) becomes a
                         `response_format` json_schema, and comes back parsed.
  * usage accounting   — OpenRouter reports the real per-request cost, so
                         nothing here estimates it from a price table.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional, TypeVar, get_args, get_origin

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from config.logging_config import get_logger
from config.settings import settings

log = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sent on every request. OpenRouter uses these only for its public model-usage
# rankings; they are optional and carry nothing about the account.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/deus-scrooge",
    "X-Title": "Deus",
}

# Cached per event loop rather than globally. AsyncOpenAI owns an httpx client
# whose connections belong to the loop that first used them, so a single
# module-level instance raises "Event loop is closed" the moment a second loop
# appears — which is every pytest-asyncio test and any second asyncio.run().
_CLIENTS: dict[int, AsyncOpenAI] = {}


def get_llm_client() -> AsyncOpenAI | None:
    """
    The shared OpenRouter client for the running loop.

    Cached because the previous per-provider factories built a fresh client —
    and so a fresh httpx connection pool — on every single call, including once
    per article in the classifier loop.
    """
    if not settings.openrouter_api_key:
        log.warning("llm.missing_key", reason="OPENROUTER_API_KEY is not set")
        return None

    try:
        key = id(asyncio.get_running_loop())
    except RuntimeError:
        key = 0  # built outside a loop; whoever awaits it first owns it

    client = _CLIENTS.get(key)
    if client is None:
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=_ATTRIBUTION_HEADERS,
            timeout=settings.llm_timeout_seconds,
            # Retries are the caller's decision. Every lane already has its own
            # bounded loop keyed on `is_transient`, and SDK-level retries would
            # silently multiply that budget.
            max_retries=0,
        )
        _CLIENTS[key] = client
    return client


def reset_client_cache() -> None:
    """Drop cached clients — for tests, and for a key rotation at runtime."""
    _CLIENTS.clear()


def is_llm_configured() -> bool:
    """Whether any model call can be made at all."""
    return bool(settings.openrouter_api_key)


def config_error(setting_name: str) -> str:
    """
    Why an LLM-backed feature is off, phrased so the reader edits the right line.

    Callers gate on `is_llm_configured() and settings.<model>`, and reporting
    that pair as one "MODEL_X is not configured" is actively misleading: the
    common case on a fresh deploy is a correct MODEL_X and a missing key, which
    sends the reader to re-check a setting that was never the problem.

    The `.env` note matters more than it looks. `env_file` in Settings is a
    relative path, so a worker started from anywhere but the repo root reads no
    file at all and *every* setting silently falls back to its empty default —
    a failure that otherwise presents as "I set that, and it says I didn't".
    """
    if settings.openrouter_api_key:
        return f"{setting_name} is not set in .env"

    env_path = Path(settings.model_config.get("env_file", ".env")).resolve()
    if not env_path.exists():
        return (
            f"{setting_name} is unusable: no .env was found at {env_path}, so "
            f"OPENROUTER_API_KEY and every MODEL_* setting are empty. The path "
            f"is relative to the working directory - start the process from the "
            f"repo root."
        )
    return (
        f"{setting_name} is unusable: OPENROUTER_API_KEY is empty in {env_path}. "
        f"The model slug itself is fine; every LLM call is off until the key is set."
    )


def response_cost(usage: Any) -> Optional[float]:
    """
    Pull OpenRouter's reported cost off a usage object.

    `cost` is an extra field the OpenAI SDK does not declare, so depending on
    SDK version it arrives as a plain attribute or inside `model_extra`.
    """
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


T = TypeVar("T")


# ── Response shapes ──────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """
    One completion.

    `usage` deliberately holds the raw OpenAI usage object rather than a
    normalised copy: `config.usage._extract_usage` already reads
    `.usage.prompt_tokens` / `.completion_tokens` and `.text` off whatever it is
    handed, so instrumentation keeps working without a translation layer.
    """
    text: str = ""
    parsed: Any = None
    usage: Any = None
    finish_reason: Optional[str] = None
    cost: Optional[float] = None
    reasoning: Optional[str] = None
    raw: Any = None


@dataclass
class StreamChunk:
    """
    One delta off a stream.

    `reasoning` is separate from `text` because the thesis engine streams the
    chain of thought to the UI on its own channel while the answer accumulates.
    Usage arrives on the trailing chunk, after the last content delta.
    """
    text: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Any = None


@dataclass
class EmbeddingResult:
    """Vectors aligned 1:1 with the input list, plus what the call cost."""
    vectors: list[Optional[list[float]]] = field(default_factory=list)
    usage: Any = None
    cost: Optional[float] = None


# ── Parsing helpers (provider-neutral, unchanged) ────────────────────────


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


def parse_json_list(text: str) -> list:
    """
    Parse a response that should be a JSON array, tolerating the object the
    json_mode contract actually forces.

    `response_format={"type": "json_object"}` constrains the root to an object,
    so a prompt asking for a bare array gets one back wrapped under a key the
    model picks for itself — `{"events": [...]}`, `{"rankings": [...]}`.
    Pinning one key would just move the breakage to the next model, so a lone
    list value is unwrapped whatever it happens to be called. The schema path
    has the same constraint and answers it with a fixed `{"items": [...]}`
    envelope (see `_unwrap_envelope`); json_mode callers cannot, because
    nothing on the request tells the model which key to use.

    Raises ValueError when there is no single unambiguous array to return —
    two list values are as unusable as none, and guessing between them would
    attach the wrong scores to the wrong articles.
    """
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # A model told "one result per item" sometimes reads that as one JSON
        # object per line. Concatenated objects are a decode error at the start
        # of the second one, which loses a response that is entirely usable.
        stream = _decode_json_stream(cleaned)
        if len(stream) > 1:
            return stream
        raise

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    raise ValueError(f"Expected a JSON array, got {type(payload).__name__}")


def _decode_json_stream(text: str) -> list:
    """Read as many whole JSON values as sit back to back in `text`.

    Returns what it managed to read; a truncated tail is dropped rather than
    failing the values that arrived intact before it.
    """
    decoder = json.JSONDecoder(strict=False)
    values: list = []
    idx, end = 0, len(text)
    while idx < end:
        while idx < end and text[idx] in " \t\r\n":
            idx += 1
        if idx >= end:
            break
        try:
            value, idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        values.append(value)
    return values


def salvage_json_array(text: str, *, key: str | None = None) -> list:
    """
    Recover the whole elements of a JSON array whose tail is truncated.

    Written for one observed failure: a model asked to echo short ids degenerated
    into repeating one (`"id": "a1a1a1a1a1…`) until it hit the output cap, so the
    response ended mid-string and `json.loads` rejected all of it. Every complete
    object before the runaway one was perfectly good, paid-for work, and
    discarding the lot cost the whole batch.

    `key` names the envelope field holding the array (`items` for the schema
    path); without it the first `[` in the text is taken as the start. Returns
    `[]` when nothing whole can be read, which callers treat exactly as a parse
    failure.
    """
    cleaned = strip_code_fence(text)

    start = -1
    if key:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', cleaned)
        if match:
            start = match.end() - 1
    if start < 0:
        start = cleaned.find("[")
    if start < 0:
        return []

    decoder = json.JSONDecoder(strict=False)
    values: list = []
    idx, end = start + 1, len(cleaned)
    while idx < end:
        while idx < end and cleaned[idx] in " \t\r\n,":
            idx += 1
        if idx >= end or cleaned[idx] == "]":
            break
        try:
            value, idx = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            break  # the truncated tail; everything before it stands
        values.append(value)
    return values


def parse_structured(text: str, schema: type[T] | Any) -> T:
    """
    Validate raw LLM text against `schema` — a Pydantic model, or a container
    of one such as `list[TickerNote]`.

    This is the *second* line of defence, behind `response_format` on the
    request itself. It still matters: `strict` mode is off by default (see
    `_build_response_format`), so the schema guides decoding rather than
    constraining it, and a model can still return prose-shaped JSON.

    `strict=False` is the load-bearing argument: models writing multi-paragraph
    prose into a string field routinely emit real newlines instead of `\\n`,
    and a raw control character inside a JSON string is a hard parse error for
    the default decoder. Tolerating them here turns the single most common
    malformed response into a successful parse rather than a fallback that
    leaks raw JSON to the caller.
    """
    payload = json.loads(strip_code_fence(text), strict=False)
    return TypeAdapter(schema).validate_python(payload)


def salvage_json_field(text: str, field_name: str) -> str | None:
    """
    Last resort: pull one string field out of a JSON blob that will not parse
    at all (truncated mid-object, doubled closing brace, etc).

    Exists so a malformed response degrades to *the prose we wanted* instead of
    dumping a raw `{"executive_summary": ...}` blob into the UI.
    """
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"(.*?)"\s*[,}}]', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"', strict=False)
    except json.JSONDecodeError:
        return match.group(1).replace("\\n", "\n").replace('\\"', '"').strip() or None


# ── Request translation ──────────────────────────────────────────────────

# The one place the project's thinking-budget vocabulary is defined. Call sites
# pass a plain string; OpenRouter maps `effort` onto each provider's own dial
# (Google thinkingLevel, OpenAI-style effort, Anthropic token budgets).
_REASONING_LEVELS = {
    "none": {"enabled": False},
    "minimal": {"effort": "minimal"},
    "low": {"effort": "low"},
    "medium": {"effort": "medium"},
    "high": {"effort": "high"},
    "xhigh": {"effort": "xhigh"},
    "max": {"effort": "max"},
}

# JSON Schema roots must be objects, so `list[Model]` is wrapped in one and
# unwrapped again after parsing. Callers never see the envelope.
_LIST_ENVELOPE_KEY = "items"


def _build_reasoning(reasoning: Optional[str]) -> Optional[dict]:
    """`None` means 'say nothing', which leaves the model's own default alone."""
    if reasoning is None:
        return None
    try:
        return dict(_REASONING_LEVELS[reasoning])
    except KeyError:
        log.warning(
            "llm.unknown_reasoning_level",
            level=reasoning,
            allowed=sorted(_REASONING_LEVELS),
            action="omitted",
        )
        return None


def _is_list_schema(schema: Any) -> bool:
    return get_origin(schema) in (list, set, tuple)


def _inline_defs(schema: dict) -> dict:
    """
    Flatten `$defs` / `$ref` into one self-contained schema tree.

    Pydantic emits a `$ref` for every nested model — `CandidateList.companies`
    refers to `CandidateCompany` that way. Some structured-output backends
    behind OpenRouter resolve refs and some reject them with an opaque 400, so
    inlining removes the question entirely.

    Assumes no recursive models. None of this project's schemas are recursive;
    one added later would not terminate here.
    """
    defs = {k: v for k, v in (schema.get("$defs") or {}).items()}
    if not defs:
        return schema

    schema = {k: v for k, v in schema.items() if k != "$defs"}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.rsplit("/", 1)[-1])
                if target is not None:
                    merged = walk(json.loads(json.dumps(target)))
                    merged.update({k: v for k, v in node.items() if k != "$ref"})
                    return merged
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _json_schema_for(schema: Any, exact_items: Optional[int] = None) -> tuple[dict, bool]:
    """
    Returns (json_schema_body, is_enveloped).

    A bare Pydantic model maps straight across. `list[Model]` cannot: a JSON
    Schema root has to be an object, so it goes inside `{"items": [...]}` and
    `_unwrap_envelope` takes it back out.

    `exact_items` pins that array to a known length with `minItems`/`maxItems`,
    for the callers that send N inputs and need N results back. A Python type
    cannot express it — `list[Model]` says "an array of these", never "exactly
    ten of these" — so it is a request-level argument rather than part of the
    schema object. It is the only thing that stopped the batch classifier being
    answered with a single item: a length the decoder can count is enforceable
    in a way that "one object per input article" in prose is not.
    """
    if _is_list_schema(schema):
        (inner,) = get_args(schema)
        items_schema: dict[str, Any] = {
            "type": "array",
            "items": _inline_defs(TypeAdapter(inner).json_schema()),
        }
        if exact_items is not None and exact_items > 0:
            items_schema["minItems"] = int(exact_items)
            items_schema["maxItems"] = int(exact_items)
        return (
            {
                "type": "object",
                "properties": {_LIST_ENVELOPE_KEY: items_schema},
                "required": [_LIST_ENVELOPE_KEY],
            },
            True,
        )

    if exact_items is not None:
        # Only the list envelope has an array to constrain. Silently ignoring
        # this would make a caller think it had asked for a length.
        log.warning(
            "llm.exact_items_ignored",
            schema=_schema_name(schema),
            reason="exact_items applies to list[Model] schemas only",
        )

    return _inline_defs(TypeAdapter(schema).json_schema()), False


def _schema_name(schema: Any) -> str:
    if _is_list_schema(schema):
        (inner,) = get_args(schema)
        return f"{getattr(inner, '__name__', 'item')}_list"
    return getattr(schema, "__name__", "response")


def _build_response_format(
    schema: Any, json_mode: bool, exact_items: Optional[int] = None
) -> tuple[Optional[dict], bool]:
    """
    Returns (response_format, is_enveloped).

    `strict` is left False on purpose. Strict mode additionally demands
    `additionalProperties: false` on every object and every property listed in
    `required` — which Pydantic models with default values do not satisfy — and
    it narrows routing to endpoints that implement it natively. Every schema
    call site in this project already falls back to `parse_structured`, so the
    schema is worth more as strong guidance across all providers than as a hard
    constraint across few.

    `exact_items` is passed through to `_json_schema_for`; see it for why a
    length lives on the request rather than in the schema type.
    """
    if schema is not None:
        body, enveloped = _json_schema_for(schema, exact_items)
        return (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(schema),
                    "strict": False,
                    "schema": body,
                },
            },
            enveloped,
        )

    if json_mode:
        return {"type": "json_object"}, False

    return None, False


def _unwrap_envelope(payload: Any) -> Any:
    """Take a `list[Model]` result back out of its `{"items": [...]}` wrapper."""
    if isinstance(payload, dict) and _LIST_ENVELOPE_KEY in payload:
        return payload[_LIST_ENVELOPE_KEY]
    return payload


def _parse_into(text: str, schema: Any, enveloped: bool) -> Any:
    """
    Populate `LLMResponse.parsed`, or leave it None and let the caller fall
    back. Never raises: a schema miss must not lose the text that came with it.
    """
    if schema is None or not text.strip():
        return None
    try:
        payload = json.loads(strip_code_fence(text), strict=False)
        if enveloped:
            payload = _unwrap_envelope(payload)
        return TypeAdapter(schema).validate_python(payload)
    except Exception as e:
        log.warning("llm.schema_parse_failed", schema=_schema_name(schema), error=str(e))
        return None


def _build_messages(
    prompt: Optional[str],
    messages: Optional[list[dict]],
    system: Optional[str],
) -> list[dict]:
    if messages is not None:
        if system and not any(m.get("role") == "system" for m in messages):
            return [{"role": "system", "content": system}, *messages]
        return list(messages)

    built: list[dict] = []
    if system:
        built.append({"role": "system", "content": system})
    built.append({"role": "user", "content": prompt or ""})
    return built


def _build_kwargs(
    *,
    model: str,
    prompt: Optional[str],
    messages: Optional[list[dict]],
    system: Optional[str],
    schema: Any,
    json_mode: bool,
    reasoning: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    exact_items: Optional[int] = None,
) -> tuple[dict, bool]:
    response_format, enveloped = _build_response_format(schema, json_mode, exact_items)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(prompt, messages, system),
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    reasoning_body = _build_reasoning(reasoning)
    if reasoning_body is not None:
        kwargs["extra_body"] = {"reasoning": reasoning_body}

    return kwargs, enveloped


def _note_finish(model: str, finish_reason: Optional[str]) -> None:
    """
    Surface a stop that is neither a finished answer nor a budget overrun.

    This is the refusal canary. Gemini's `DEFAULT_SAFETY_SETTINGS` used to
    relax blocking for market news that involves crime, hacks and sanctions,
    and a block came back as an explicit BLOCK_* reason. OpenRouter has no
    equivalent request-side knob, so a refusal now arrives only as a
    finish_reason nobody would otherwise be watching.
    """
    if finish_reason and finish_reason not in ("stop", "length", "tool_calls"):
        log.warning("llm.unexpected_finish", model=model, finish_reason=finish_reason)


def _extract_reasoning(message_or_delta: Any) -> str:
    """
    OpenRouter returns reasoning as plaintext `reasoning` plus a structured
    `reasoning_details` array. Prefer the plaintext; fall back to concatenating
    the text-bearing details, skipping encrypted ones.
    """
    plain = getattr(message_or_delta, "reasoning", None)
    if plain:
        return plain

    details = getattr(message_or_delta, "reasoning_details", None) or []
    parts = []
    for detail in details:
        if isinstance(detail, dict):
            text = detail.get("text") or detail.get("summary")
        else:
            text = getattr(detail, "text", None) or getattr(detail, "summary", None)
        if text:
            parts.append(text)
    return "".join(parts)


# ── The three entry points ───────────────────────────────────────────────


async def complete(
    *,
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    system: Optional[str] = None,
    schema: Any = None,
    json_mode: bool = False,
    reasoning: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    exact_items: Optional[int] = None,
) -> LLMResponse:
    """
    One non-streaming completion.

    Raises if the client is unconfigured or the model name is empty, rather
    than returning a sentinel — an unset `MODEL_<FUNCTION>` is a configuration
    error, and the retry loops upstream classify it as non-transient and stop.

    `exact_items` requires a `list[Model]` schema and pins the returned array to
    that many entries. Use it whenever the answer has one entry per input.
    """
    client = get_llm_client()
    if client is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    if not model:
        raise RuntimeError("No model configured for this call — see MODEL_* settings")

    kwargs, enveloped = _build_kwargs(
        model=model, prompt=prompt, messages=messages, system=system,
        schema=schema, json_mode=json_mode, reasoning=reasoning,
        temperature=temperature, max_tokens=max_tokens,
        exact_items=exact_items,
    )

    response = await client.chat.completions.create(**kwargs)

    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    text = (getattr(message, "content", None) or "") if message else ""
    usage = getattr(response, "usage", None)
    finish_reason = getattr(choice, "finish_reason", None) if choice else None
    _note_finish(model, finish_reason)

    return LLMResponse(
        text=text,
        parsed=_parse_into(text, schema, enveloped),
        usage=usage,
        finish_reason=finish_reason,
        cost=response_cost(usage),
        reasoning=_extract_reasoning(message) if message else None,
        raw=response,
    )


async def stream_complete(
    *,
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    system: Optional[str] = None,
    schema: Any = None,
    json_mode: bool = False,
    reasoning: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[StreamChunk]:
    """
    The streaming counterpart. Yields content and reasoning deltas as they
    arrive, then one final chunk carrying usage.

    `stream_options={"include_usage": True}` is not optional: without it the
    stream reports no usage at all, which is how the most expensive operation
    in the system once logged $0.00.
    """
    client = get_llm_client()
    if client is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    if not model:
        raise RuntimeError("No model configured for this call — see MODEL_* settings")

    kwargs, _ = _build_kwargs(
        model=model, prompt=prompt, messages=messages, system=system,
        schema=schema, json_mode=json_mode, reasoning=reasoning,
        temperature=temperature, max_tokens=max_tokens,
    )

    stream = await client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True},
    )

    async for chunk in stream:
        # Usage is read off every chunk, not just choices-less ones. Providers
        # disagree about where it goes: DeepSeek-direct sent a final chunk with
        # an empty `choices` list, while OpenRouter attaches it to the last
        # chunk that still carries a choice. Checking only the former shape
        # silently drops usage for the whole stream — which is how the debate,
        # the single most expensive operation here, once logged $0.00.
        usage = getattr(chunk, "usage", None)

        if not chunk.choices:
            if usage is not None:
                yield StreamChunk(usage=usage)
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            _note_finish(model, choice.finish_reason)
        yield StreamChunk(
            text=getattr(delta, "content", None) or "",
            reasoning=_extract_reasoning(delta),
            # Rides on the final content chunk rather than arriving on its own.
            finish_reason=choice.finish_reason,
            usage=usage,
        )


async def embed(texts: list[str], *, model: str) -> EmbeddingResult:
    """
    Embed a batch. Returns vectors aligned 1:1 with `texts`.

    Results are placed by the response's own `index` rather than by arrival
    order — the OpenAI embeddings shape carries one, and relying on order is an
    assumption that costs a silently mismatched corpus if it ever breaks.
    """
    client = get_llm_client()
    if client is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    if not model:
        raise RuntimeError("No embedding model configured — see MODEL_EMBEDDING")
    if not texts:
        return EmbeddingResult(vectors=[])

    response = await client.embeddings.create(
        model=model, input=texts, encoding_format="float",
    )

    vectors: list[Optional[list[float]]] = [None] * len(texts)
    for item in response.data:
        index = getattr(item, "index", None)
        if index is None or not (0 <= index < len(vectors)):
            log.warning("llm.embed_index_out_of_range", index=index, count=len(texts))
            continue
        vectors[index] = item.embedding

    usage = getattr(response, "usage", None)
    return EmbeddingResult(vectors=vectors, usage=usage, cost=response_cost(usage))


# ── Retry classification ─────────────────────────────────────────────────

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

    # Deterministic client errors — bad request, auth, malformed payload. An
    # unset MODEL_* setting surfaces here as a RuntimeError and is likewise
    # never worth a retry.
    if isinstance(exc, (openai.BadRequestError, openai.AuthenticationError,
                        openai.PermissionDeniedError, openai.NotFoundError,
                        RuntimeError)):
        return False

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS

    return False
