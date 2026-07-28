"""
SSE Event Manager for THE_BRAIN real-time dashboard.

Provides a central pub/sub event bus that pipeline components can publish
events into, and SSE endpoints can subscribe to. Multiple subscribers
receive the same events.

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
from typing import Any, Optional


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
        return cls._instance

    def subscribe(self, topics: list[str]) -> SSESubscriber:
        """Create a new subscriber that listens to the given topic events."""
        self._counter += 1
        sub = SSESubscriber(f"sub_{self._counter}", topics)
        self._subscribers[sub.id] = sub
        return sub

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber by ID."""
        self._subscribers.pop(subscriber_id, None)

    async def publish(self, event_type: str, data: Any) -> None:
        """
        Push an event to all subscribers who listen to this topic.
        Dropped subscribers (full queue) are automatically removed.
        """
        to_remove: list[str] = []
        for sub_id, sub in self._subscribers.items():
            if event_type in sub.topics or "*" in sub.topics:
                try:
                    await asyncio.wait_for(sub.queue.put((event_type, data)), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.QueueFull):
                    to_remove.append(sub_id)
        for sub_id in to_remove:
            self.unsubscribe(sub_id)


# Module-level singleton
event_bus = SSEEventBus()
