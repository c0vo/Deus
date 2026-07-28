"""
Project Scrooge V2 - Main Entry Point

Refactored into a FastAPI application.
Provides WebSocket endpoints for real-time market data, news, and chat.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.server import router as api_router

from config.logging_config import get_logger, setup_logging
from config.settings import settings
from data.database import Database
from bot.telegram_bot import ScroogeBot
from orchestrator.scheduler import PipelineOrchestrator

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
    """Lifecycle manager for the FastAPI application."""
    global bot, orchestrator
    
    setup_logging()
    log.info("system.starting", version="2.0.0")
    
    # 1. Init Database
    db.initialize()
    app.state.db = db
    
    # 2. Init Bot
    bot = ScroogeBot(db=db)
    bot.initialize()
    app.state.bot = bot
    
    # 3. Init Orchestrator
    orchestrator = PipelineOrchestrator(db=db, alert_manager=bot.alert_manager)
    app.state.orchestrator = orchestrator
    
    # Start Bot and Orchestrator
    try:
        if bot.application:
            log.info("telegram.starting_polling")
            await bot.application.initialize()
            await bot.application.start()
            await bot.application.updater.start_polling()
            
        orchestrator.start(interval_minutes=5)
        log.info("system.ready")
        
        yield  # Let the FastAPI app run
        
    finally:
        log.info("system.shutdown_initiated")
        if orchestrator:
            orchestrator.stop()
        if bot and bot.application:
            await bot.application.updater.stop()
            await bot.application.stop()
            await bot.application.shutdown()
        log.info("system.shutdown_complete")

app = FastAPI(title="Project Scrooge V2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Removed allow_credentials=True since it's incompatible with allow_origins=["*"] and not needed here
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount API router (must come BEFORE the catch-all) ─────────────────
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
            "message": "Project Scrooge API is running",
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
    from pipeline.chat_orchestrator import ChatOrchestrator, build_chat_prompt
    from google.genai import types
    from config.llm import get_client, DEFAULT_SAFETY_SETTINGS

    await websocket.accept()
    try:
        while True:
            user_msg = await websocket.receive_text()

            orchestrator = ChatOrchestrator(db)
            state = {"query": user_msg, "context": "", "routing_decision": "", "final_answer": ""}

            # Router
            router_res = await orchestrator.router_node(state)
            decision = router_res.get("routing_decision", "shallow")
            state["routing_decision"] = decision

            await websocket.send_json({
                "type": "status",
                "intent": decision,
                "reasoning": f"Routed to {decision} agent"
            })

            # RAG
            rag_res = await orchestrator.rag_node(state)
            context = rag_res.get("context", "")
            state["context"] = context

            # Generation
            client = get_client()
            model = (
                settings.gemini_model_chat_shallow if decision == "shallow"
                else settings.gemini_model_chat_complex
            )

            prompt = build_chat_prompt(user_msg, context)

            response = client.models.generate_content_stream(
                model=model or "gemini-3.1-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=DEFAULT_SAFETY_SETTINGS,
                    max_output_tokens=2048,
                ),
            )

            for chunk in response:
                if chunk.text:
                    await websocket.send_json({"type": "token", "text": chunk.text})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        log.info("websocket.chat.disconnected")


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import subprocess

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

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=False)
