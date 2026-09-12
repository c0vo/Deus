"""
Deus - Main Entry Point

Refactored into a FastAPI application.
Provides WebSocket endpoints for real-time market data, news, and chat.
"""

import asyncio
import logging
import mimetypes
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.server import router as api_router
from api.auth import AccessControlMiddleware, auth_router
from api.middleware import CacheControlMiddleware
from api import sse_manager
from api.sse_manager import event_bus

from config.logging_config import get_logger, setup_logging
from config.settings import settings, preflight_models
from data.database import Database
from bot.telegram_bot import DeusBot
from orchestrator.scheduler import PipelineOrchestrator
# Startup/shutdown helpers are shared with the worker so the in-process
# development path behaves identically to the deployed one.
import worker

log = get_logger(__name__)

# Global instances for app lifecycle
db = Database()
bot = None
orchestrator = None

# ── Frontend paths ────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
FRONTEND_OUT = os.path.join(FRONTEND_DIR, "out")
FRONTEND_BUILT = os.path.isdir(FRONTEND_OUT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.

    Everything here must be local and fast: uvicorn does not create the
    listening socket until this function reaches its yield, so any network
    call made before that point delays — or prevents — the dashboard from
    being reachable at all. The Telegram bot and the ingest pipeline live in
    worker.py for exactly this reason.
    """
    global bot, orchestrator
    telegram_running = False

    setup_logging()
    log.info("system.starting", version="2.0.0")

    # 1. Init Database
    db.initialize()
    app.state.db = db

    # 2. Relay pipeline events published by the worker process out to SSE
    #    clients. Without this the Brain dashboard receives its initial
    #    snapshot and then never updates.
    sse_manager.configure(db)
    event_bus.start_tailer()

    # 3. The pipeline and bot normally run in worker.py so they can never
    #    stall this event loop. ENABLE_IN_PROCESS_WORKER puts them back here
    #    for single-process local development.
    if settings.enable_in_process_worker:
        log.warning("worker.in_process",
                    reason="ENABLE_IN_PROCESS_WORKER is set; pipeline shares the API event loop")
        bot = DeusBot(db=db)
        bot.initialize()
        app.state.bot = bot
        orchestrator = PipelineOrchestrator(db=db, alert_manager=bot.alert_manager)
        app.state.orchestrator = orchestrator
        telegram_running = await worker._start_telegram(bot)
        orchestrator.start(interval_minutes=settings.pipeline_interval_minutes)
    else:
        log.info("worker.external",
                 hint="run `python worker.py` for the pipeline and Telegram bot")

    try:
        log.info("system.ready")
        yield  # Let the FastAPI app run

    finally:
        log.info("system.shutdown_initiated")
        await event_bus.stop_tailer()
        if orchestrator or bot:
            await worker._shutdown(bot, orchestrator, telegram_running)
        log.info("system.shutdown_complete")

# The SPA catch-all serves unknown files via FileResponse with no explicit
# media_type, so Starlette falls back to mimetypes.guess_type. .webmanifest is
# not in Python's table, so Next's out/manifest.webmanifest would go out as
# text/plain and the PWA manifest would be ignored.
mimetypes.add_type("application/manifest+json", ".webmanifest")

app = FastAPI(title="Deus", lifespan=lifespan)

# Same-origin by default. In the Termux deployment FastAPI serves the static
# export and the API from the same port, so no cross-origin request is ever
# legitimate and CORS_ORIGINS stays empty. It used to be allow_origins=["*"],
# which let any site the phone's browser visited read every endpoint.
# allow_credentials is on because the session cookie is what authenticates, and
# it is only safe here because the origin list is explicit — the browser refuses
# credentialed requests against a wildcard.
_cors_origins = settings.cors_origin_list
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Starlette builds the stack so the LAST registered middleware ends up
# OUTERMOST: add_middleware inserts at index 0, and build_middleware_stack then
# wraps in reverse. The resulting order is:
#
#     AccessControl -> CacheControl -> GZip -> CORS -> router
#
# All of them wrap the whole router, so the /_next mount and the catch-all below
# are covered without any ordering relationship to the routes themselves.
#
# GZip excludes text/event-stream by default (starlette/middleware/gzip.py),
# so /api/brain/stream and the chat/debate streams keep flowing unbuffered.
# compresslevel is 6 rather than the library default of 9 because compression
# runs synchronously on the event loop — level 9 over a 370 KB chunk is a
# multi-hundred-millisecond stall on a phone CPU for ~3% more ratio.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.add_middleware(CacheControlMiddleware)
# Registered last so it is outermost: an untrusted peer or a missing session is
# refused before the gzip and cache-header work happens, and before the /_next
# StaticFiles mount below can serve a single byte of the dashboard.
app.add_middleware(AccessControlMiddleware)

# ── Mount API router (must come BEFORE the catch-all) ─────────────────
# auth_router first: /login and /healthz have to win against the SPA catch-all.
app.include_router(auth_router)
app.include_router(api_router)


# ── Static frontend serving (production / Termux mode) ────────────────
# If frontend/out/ exists (from `npm run build`), serve the static export
# directly from FastAPI on port 8000. No port 3000 needed.
if FRONTEND_BUILT:
    _next_dir = os.path.join(FRONTEND_OUT, "_next")
    if os.path.isdir(_next_dir):
        app.mount("/_next", StaticFiles(directory=_next_dir), name="next_static")

    # ── Explicit root route ──────────────────────────────────────────
    # The catch-all /{page_name:path} can fail to match bare "/" in some
    # Starlette versions (path converter regex .* vs empty segment).
    # Register an explicit GET+HEAD for / so the dashboard always loads.

    @app.api_route("/", methods=["GET", "HEAD"])
    async def serve_root(request: Request):
        root_index = os.path.join(os.path.abspath(FRONTEND_OUT), "index.html")
        if os.path.isfile(root_index):
            log.info("frontend.serve", path="/", file=root_index)
            return FileResponse(root_index, media_type="text/html")
        raise HTTPException(status_code=404, detail="index.html not found")

    @app.api_route("/{page_name:path}", methods=["GET", "HEAD"])
    async def serve_frontend(request: Request, page_name: str):
        """Catch-all: serve the static Next.js export for any non-API route."""
        # Don't intercept /api/* or /ws/* — those are handled by the routers above
        if page_name.startswith("api/") or page_name.startswith("ws/"):
            raise HTTPException(status_code=404, detail="Not Found")

        # ── PATH TRAVERSAL FIX ──
        # Ensure that any requested path strictly resides within FRONTEND_OUT
        safe_base = os.path.abspath(FRONTEND_OUT)

        def get_safe_path(target_subpath: str) -> str:
            resolved_path = os.path.abspath(os.path.join(safe_base, target_subpath))
            # Compare against base + separator, not a bare prefix: a plain
            # startswith() would also accept sibling dirs like "out.bak".
            if resolved_path != safe_base and not resolved_path.startswith(safe_base + os.sep):
                raise HTTPException(status_code=403, detail="Forbidden")
            return resolved_path

        try:
            # Try exact page HTML (e.g. /predict -> /predict.html)
            page_html = get_safe_path(f"{page_name}.html")
            if os.path.isfile(page_html):
                log.info("frontend.serve", path=request.url.path, file=page_html)
                return FileResponse(page_html, media_type="text/html")

            # Try directory index (e.g. /predict -> /predict/index.html)
            index_html = get_safe_path(os.path.join(page_name, "index.html"))
            if os.path.isfile(index_html):
                log.info("frontend.serve", path=request.url.path, file=index_html)
                return FileResponse(index_html, media_type="text/html")

            # Root path
            root_index = get_safe_path("index.html")
            if page_name == "" and os.path.isfile(root_index):
                log.info("frontend.serve", path=request.url.path, file=root_index)
                return FileResponse(root_index, media_type="text/html")

            # Try serving as a static file (favicon.ico, images, RSC .txt payloads, etc.)
            static_file = get_safe_path(page_name)
            if os.path.isfile(static_file):
                log.info("frontend.serve_static", path=request.url.path, file=static_file)
                return FileResponse(static_file)

            # Fallback to index.html for client-side routing
            if os.path.isfile(root_index):
                log.info("frontend.serve_fallback", path=request.url.path, fallback=root_index)
                return FileResponse(root_index, media_type="text/html")

            log.warning("frontend.not_found", path=request.url.path)
            raise HTTPException(status_code=404, detail="Not Found")
        except HTTPException:
            raise
        except Exception as exc:
            log.error("frontend.serve_error", path=request.url.path, error=str(exc))
            raise HTTPException(status_code=500, detail="Internal Server Error")

    log.info("frontend.mode", mode="static", path=FRONTEND_OUT)

else:
    # No static build — show a helpful root message
    @app.get("/")
    async def root():
        return {
            "message": "Deus API is running",
            "frontend": "Use 'npm run dev' in frontend/ for dev mode, or 'npm run build' for static serving"
        }

    log.info("frontend.mode", mode="dev_server",
             hint="Run 'npm run dev' in frontend/ or 'cd frontend && npm run build' for static mode")


# --- WebSocket Endpoints (legacy, kept for compatibility) ---
@app.websocket("/ws/markets")
async def ws_markets(websocket: WebSocket):
    """Real-time market updates using yfinance."""
    await websocket.accept()
    try:
        import yfinance as yf

        while True:
            tracked = db.get_tracked_tickers()
            if not tracked:
                tracked = ["AAPL", "MSFT", "GOOGL"]

            tick_data = {}
            for ticker in tracked:
                try:
                    t = yf.Ticker(ticker)
                    price = t.fast_info.get("lastPrice", 0.0)
                    if price == 0.0:
                        df = t.history(period="1d")
                        if not df.empty:
                            price = float(df["Close"].iloc[-1])
                    tick_data[ticker] = price
                except Exception:
                    pass

            await websocket.send_json({"type": "tick", "data": tick_data})
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        log.info("websocket.markets.disconnected")

@app.websocket("/ws/news")
async def ws_news(websocket: WebSocket):
    """Real-time news stream from the database."""
    await websocket.accept()
    try:
        while True:
            recent = db.get_recent_articles(limit=10)
            await websocket.send_json({"type": "news", "data": recent})
            await asyncio.sleep(10)

    except WebSocketDisconnect:
        log.info("websocket.news.disconnected")

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """Interactive chat with the ChatOrchestrator (RAG-powered)."""
    from pipeline.chat_orchestrator import ChatOrchestrator

    await websocket.accept()
    try:
        while True:
            user_msg = await websocket.receive_text()

            # Was a hand-rolled copy of the router -> rag -> stream sequence,
            # the third one in the codebase. It drifted: this path never graded
            # its context and never searched the web at all, so the legacy
            # client got a different answer from the dashboard for the same
            # question. iter_events is now the only implementation.
            orchestrator = ChatOrchestrator(db)
            async for event, data in orchestrator.iter_events(user_msg):
                if event == "token":
                    await websocket.send_json(
                        {"type": "token", "text": str(data.get("text", ""))}
                    )
                elif event == "step":
                    await websocket.send_json({"type": "status", **data})
                elif event == "error":
                    await websocket.send_json({
                        "type": "error",
                        "message": data if isinstance(data, str) else str(data),
                    })

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("websocket.chat.disconnected")


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import subprocess

    # Same check the worker runs. The API process serves chat and the thesis
    # stream, so an unset MODEL_* is user-visible here too.
    preflight_models()

    # If no static build exists, auto-build OR start dev server
    if os.path.isdir(FRONTEND_DIR) and not FRONTEND_BUILT:
        # Check if node_modules exist (npm install has been run)
        node_modules = os.path.join(FRONTEND_DIR, "node_modules")
        if os.path.isdir(node_modules):
            import atexit
            log.info("frontend.dev_server.starting")
            frontend_proc = subprocess.Popen(
                "npm run dev",
                cwd=FRONTEND_DIR,
                shell=True,
            )
            atexit.register(frontend_proc.terminate)
            log.info("frontend.dev_server.started", pid=frontend_proc.pid,
                     url="http://localhost:3000")
        else:
            log.warning("frontend.not_installed",
                        hint="Run 'cd frontend && npm install' first")

    elif FRONTEND_BUILT:
        log.info("frontend.static.ready",
                 url=f"http://0.0.0.0:{settings.api_port}")

    # State the access posture at boot. An instance that is listening on every
    # interface with no passphrase looks completely healthy in the logs
    # otherwise, which is exactly the failure worth being loud about.
    log.info(
        "api.access_control",
        bind=settings.api_host,
        login_required=settings.auth_enabled,
        trusted_networks=settings.trusted_network_list or ["<any>"],
        cors_origins=settings.cors_origin_list or ["<same-origin only>"],
    )
    if settings.api_host not in ("127.0.0.1", "localhost", "::1") and not settings.auth_enabled:
        log.warning(
            "api.open_instance",
            hint="Listening on all interfaces with no DASHBOARD_PASSPHRASE set. "
                 "Anyone who can reach this port has full access, including the "
                 "endpoints that spend API credits.",
        )

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=False)
