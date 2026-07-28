import os
import pytest
import httpx
import json
from tests.e2e.mock_backend import app

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def api_client():
    """
    Returns an httpx.AsyncClient targeting the E2E_SERVER_URL if set,
    or wrapping the mock_backend app directly.
    """
    server_url = os.environ.get("E2E_SERVER_URL")
    if server_url:
        async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:
            yield client
    else:
        # Wrap the FastAPI app directly using ASGITransport
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
            yield client

@pytest.fixture
def parse_sse_stream():
    """
    Asynchronously parses lines from an HTTPX streaming response into events.
    """
    async def _parse(response):
        events = []
        current_event = None
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    data = json.loads(data_str)
                except Exception:
                    data = data_str
                events.append({"event": current_event, "data": data})
                current_event = None
        return events
    return _parse
