"""
Small in-process TTL cache for hot API responses.

Lives in config/ alongside settings, logging_config, llm and usage — the other
process-wide cross-cutting infrastructure. There is no utils package and this
is not worth inventing one for.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Hashable

# Distinguishes "not cached" from "cached the value None", which a plain
# `is not None` check would conflate.
_MISS = object()


class TTLCache:
    """
    Bounded time-to-live cache with single-flight rebuilds.

    asyncio-only — every caller runs on the API event loop, so the dicts need
    no lock. What they do need is the in-flight map: without it, three
    dashboards hitting a cold entry each pay the full build cost at the same
    moment, which is precisely the thundering herd the cache exists to stop.
    """

    def __init__(self, ttl_seconds: float, max_entries: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._values: dict[Hashable, tuple[float, Any]] = {}
        self._inflight: dict[Hashable, asyncio.Future] = {}

    def get(self, key: Hashable) -> Any:
        """Return the cached value, or _MISS if absent or expired."""
        entry = self._values.get(key)
        if entry is None:
            return _MISS
        expires_at, value = entry
        if expires_at < time.monotonic():
            self._values.pop(key, None)
            return _MISS
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if key not in self._values and len(self._values) >= self.max_entries:
            # Cheapest sane eviction: drop the oldest insertion. dicts preserve
            # insertion order and these caches hold a handful of keys.
            oldest = next(iter(self._values), None)
            if oldest is not None:
                self._values.pop(oldest, None)
        self._values[key] = (time.monotonic() + self.ttl_seconds, value)

    def clear(self) -> None:
        self._values.clear()

    async def get_or_build(
        self, key: Hashable, factory: Callable[[], Awaitable[Any]]
    ) -> Any:
        """
        Return the cached value, invoking `factory` at most once across
        concurrent callers for the same key.
        """
        value = self.get(key)
        if value is not _MISS:
            return value

        inflight = self._inflight.get(key)
        if inflight is not None:
            # Shielded so one caller disconnecting cannot cancel the build
            # that the other waiters are still depending on.
            return await asyncio.shield(inflight)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        # Waiters may all go away before the future resolves; retrieving the
        # exception here keeps asyncio from logging it as never-retrieved.
        future.add_done_callback(
            lambda f: None if f.cancelled() else f.exception()
        )
        self._inflight[key] = future
        try:
            built = await factory()
        except Exception as e:
            future.set_exception(e)
            raise
        else:
            self.set(key, built)
            future.set_result(built)
            return built
        finally:
            self._inflight.pop(key, None)
