"""
SSE Event Manager for THE_BRAIN real-time dashboard.

Provides a central pub/sub event bus that pipeline components publish events
into, and SSE endpoints subscribe to. Multiple subscribers receive the same
events.

Backed by the `sse_events` table rather than an in-memory queue, because the
ingest pipeline runs in its own process (see worker.py) while the SSE endpoints
live in the API process. An in-memory bus cannot cross that boundary: the
publishers would all be on one side and the only subscriber on the other, and
the dashboard would silently go static.

The publish/subscribe API is unchanged, so every existing call site keeps
working. Two behavioural notes:

  * Events arrive with up to `poll_interval` of latency (default 1s) instead of
    immediately. This is a status dashboard, not a trading feed.
  * SSE now survives an API restart, which the in-memory bus did not.

Usage:
    from api.sse_manager import event_bus

    # Subscribe (in an SSE endpoint):
    sub = event_bus.subscribe(["pipeline_status", "new_articles"])

    # Publish (in a pipeline component):
    await event_bus.publish("new_articles", {"count": 5, "articles": [...]})

    # Unsubscribe (when client disconnects):
    event_bus.unsubscribe(sub.id)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from config.logging_config import get_logger
from data.database import Database

log = get_logger(__name__)

# Shared Database handle. Both processes call configure() during startup; the
# lazy fallback keeps ad-hoc imports (scripts, tests) working.
_db: Optional[Database] = None


def configure(db: Database) -> None:
    """Point the bus at the process-wide Database instance."""
    global _db
    _db = db


def _get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


class SSESubscriber:
    """A single SSE subscriber with its own async queue."""

    def __init__(self, subscriber_id: str, topics: list[str]):
        self.id = subscriber_id
        self.topics = set(topics)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=256)


class SSEEventBus:
    """
    Singleton event bus for real-time dashboard events.

    Event types:
    - pipeline_status: Pipeline cycle timing and counts
    - new_articles: Batch of recently ingested articles
    - sector_heatmap: Full sector sentiment snapshot
    - rotation_signal: New sector rotation detected
    - ipo_alert: New IPO event detected
    - trend_forecast: New trend forecast generated
    - hot_tickers: Updated hot tickers list
    - market_ticker: Scrolling market prices
    - sentiment_distribution: Overall sentiment breakdown
    - embedding_status: Vector DB telemetry
    """

    _instance: Optional["SSEEventBus"] = None

    def __new__(cls) -> "SSEEventBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers: dict[str, SSESubscriber] = {}
            cls._instance._counter = 0
            cls._instance._tailer_task: Optional[asyncio.Task] = None
            cls._instance._last_id = 0
        return cls._instance

    # ── Publish side (worker process) ────────────────────────────────────

    async def publish(self, event_type: str, data: Any) -> None:
        """
        Append an event to the outbox for delivery to every subscriber
        listening on this topic, in whichever process they live.
        """
        payload = json.dumps(data, default=str)
        try:
            await asyncio.to_thread(_get_db().insert_sse_event, event_type, payload)
        except Exception as e:
            # A dropped status event must never take down a pipeline cycle.
            log.warning("sse.publish_failed", topic=event_type, error=str(e))

    # ── Subscribe side (API process) ─────────────────────────────────────

    def subscribe(self, topics: list[str]) -> SSESubscriber:
        """Create a new subscriber that listens to the given topic events."""
        self._counter += 1
        sub = SSESubscriber(f"sub_{self._counter}", topics)
        self._subscribers[sub.id] = sub
        return sub

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber by ID."""
        self._subscribers.pop(subscriber_id, None)

    async def _dispatch(self, event_type: str, data: Any) -> None:
        """Push one event to every subscriber listening on its topic."""
        to_remove: list[str] = []
        for sub_id, sub in list(self._subscribers.items()):
            if event_type in sub.topics or "*" in sub.topics:
                try:
                    sub.queue.put_nowait((event_type, data))
                except asyncio.QueueFull:
                    # A client too slow to drain 256 events is gone or wedged.
                    to_remove.append(sub_id)
        for sub_id in to_remove:
            self.unsubscribe(sub_id)

    # ── Tailer (API process) ─────────────────────────────────────────────

    async def _tail(self, poll_interval: float) -> None:
        """Poll the outbox and fan new rows out to in-process subscribers."""
        db = _get_db()
        try:
            self._last_id = await asyncio.to_thread(db.get_max_sse_event_id)
        except Exception as e:
            log.warning("sse.tailer_start_offset_failed", error=str(e))
            self._last_id = 0

        log.info("sse.tailer_started", from_id=self._last_id,
                 poll_interval=poll_interval)

        while True:
            try:
                await asyncio.sleep(poll_interval)
                # Nobody is listening — skip the query entirely. The landing
                # page holds an SSE connection open, so this idles to nothing
                # only when no dashboard is actually open.
                if not self._subscribers:
                    continue

                rows = await asyncio.to_thread(
                    db.get_sse_events_since, self._last_id
                )
                for row in rows:
                    self._last_id = max(self._last_id, row["id"])
                    try:
                        data = json.loads(row["payload"])
                    except (TypeError, ValueError):
                        continue
                    await self._dispatch(row["topic"], data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("sse.tailer_poll_failed", error=str(e))

    def start_tailer(self, poll_interval: float = 1.0) -> None:
        """
        Begin relaying outbox rows to subscribers. Called once from the API
        lifespan; a no-op if already running.
        """
        if self._tailer_task and not self._tailer_task.done():
            return
        self._tailer_task = asyncio.create_task(self._tail(poll_interval))

    async def stop_tailer(self) -> None:
        """Cancel the tailer on shutdown."""
        task = self._tailer_task
        self._tailer_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# Module-level singleton
event_bus = SSEEventBus()
