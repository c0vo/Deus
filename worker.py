"""
Deus - Background Worker Entry Point

Runs the ingest pipeline and the Telegram bot in their own process, separate
from the API served by main.py.

Why the split: APScheduler here is an AsyncIOScheduler, and the pipeline does
substantial synchronous work — SQLite writes, numpy dedup over embeddings,
sklearn training, blocking LLM round-trips. Sharing an event loop with the API
meant a single ingest cycle could stall every HTTP request for as long as it
ran, which on a phone is minutes. The API process is now read-mostly and never
competes with ingest for the loop.

The two processes share state through the SQLite database (WAL mode, so
concurrent readers never block) and through the `sse_events` outbox table,
which relays real-time dashboard events across the process boundary.

Usage:
    python worker.py

deploy.sh starts this alongside main.py and supervises both.
"""

import asyncio
import signal

from api import sse_manager
from bot.telegram_bot import DeusBot
from config.logging_config import get_logger, setup_logging
from config.settings import settings, preflight_models
from data.database import Database
from orchestrator.scheduler import PipelineOrchestrator

log = get_logger(__name__)


async def _start_telegram(bot: DeusBot) -> bool:
    """
    Bring Telegram polling up, retrying a slow or unreachable network.

    Returns True once polling is running. A phone often has no usable
    connection in the first seconds after boot, and Telegram being down must
    never stop the pipeline from running.
    """
    if not bot.application:
        log.info("telegram.disabled", reason="no bot token configured")
        return False

    for attempt, delay in enumerate((0, 15, 60, 300), start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            await asyncio.wait_for(bot.application.initialize(), timeout=30)
            await bot.application.start()
            await bot.application.updater.start_polling()
            log.info("telegram.polling_started", attempt=attempt)
            return True
        except Exception as e:
            log.error("telegram.start_failed", attempt=attempt, error=str(e))
            try:
                await bot.application.shutdown()
            except Exception:
                pass

    log.error("telegram.giving_up", reason="all connection attempts failed")
    return False


async def _shutdown(bot: DeusBot, orchestrator: PipelineOrchestrator,
                    telegram_running: bool) -> None:
    """Stop everything that actually started."""
    log.info("worker.shutdown_initiated")
    if orchestrator:
        orchestrator.stop()
    if telegram_running and bot and bot.application:
        try:
            if bot.application.updater and bot.application.updater.running:
                await bot.application.updater.stop()
            if bot.application.running:
                await bot.application.stop()
            await bot.application.shutdown()
        except Exception as e:
            log.error("telegram.shutdown_failed", error=str(e))
    log.info("worker.shutdown_complete")


async def main() -> None:
    setup_logging()
    log.info("worker.starting", version="2.0.0")

    # Report LLM misconfiguration once, here, rather than as a wall of
    # identical failures on the first pipeline cycle 90 seconds from now.
    preflight_models()

    db = Database()
    db.initialize()

    # Publishers in this process write events to the shared outbox; the API
    # process tails it. Without this they would fall back to their own
    # Database instance, which works but opens a second handle needlessly.
    sse_manager.configure(db)

    bot = DeusBot(db=db)
    bot.initialize()

    orchestrator = PipelineOrchestrator(db=db, alert_manager=bot.alert_manager)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows has no add_signal_handler; SIGINT still raises
            # KeyboardInterrupt, which asyncio.run surfaces below.
            signal.signal(sig, lambda *_: stop_event.set())

    telegram_running = await _start_telegram(bot)
    orchestrator.start(interval_minutes=settings.pipeline_interval_minutes)
    log.info("worker.ready", telegram=telegram_running,
             pipeline_interval_minutes=settings.pipeline_interval_minutes)

    try:
        await stop_event.wait()
    finally:
        await _shutdown(bot, orchestrator, telegram_running)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
