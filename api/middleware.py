"""
ASGI middleware for the static frontend.

Kept as raw ASGI rather than BaseHTTPMiddleware (`@app.middleware("http")`):
that base class wraps every response in a streaming pump, which adds latency to
all requests and interacts badly with FileResponse and long-lived SSE streams.
A send-wrapper that mutates the response-start headers is smaller and free.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Next.js content-hashes everything under /_next/static, so those URLs are
# genuinely immutable — a rebuild emits new filenames rather than new bytes.
_IMMUTABLE_PREFIXES = ("/_next/static/",)
_LONG_LIVED_PREFIXES = ("/favicon.ico",)

CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_LONG_LIVED = "public, max-age=604800"
CACHE_REVALIDATE = "no-cache"


def cache_policy_for(path: str) -> str:
    """Pick a Cache-Control value for a request path."""
    if path.startswith(_IMMUTABLE_PREFIXES):
        return CACHE_IMMUTABLE
    if path.startswith(_LONG_LIVED_PREFIXES):
        return CACHE_LONG_LIVED
    # HTML documents and API responses must revalidate. A cached index.html
    # would keep referencing chunk hashes that no longer exist after a deploy.
    return CACHE_REVALIDATE


class CacheControlMiddleware:
    """
    Attach Cache-Control headers by path.

    Nothing set these before, so every reload re-downloaded all ~600 KB of
    JS even though the filenames are content-hashed and could have been
    cached forever.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        policy = cache_policy_for(scope.get("path", ""))

        async def send_with_cache_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # Never override a handler that set its own policy.
                if "cache-control" not in headers:
                    headers["cache-control"] = policy
            await send(message)

        await self.app(scope, receive, send_with_cache_header)
