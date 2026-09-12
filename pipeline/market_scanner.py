"""
Market Scanner — price and volume alerts that say something specific.

Three things were wrong with the version this replaces:

1. **It fetched Yahoo itself**, once per watchlist symbol every ten minutes,
   duplicating the `price_feed` job that already writes `latest_prices` every
   sixty seconds. Two fetchers against the same endpoint is how throttling
   starts. `run_scan` now makes zero HTTP calls: it reads the stored quote and
   skips anything the feed has not refreshed recently.
2. **The threshold was a symmetric hardcoded 5% on daily closes**, so a tracked
   position could fall 4.9% without a word. A drop in something you hold is not
   the same event as a 5% pop in a name off the default list, and the two now
   have different thresholds — and a drop that deepens re-alerts instead of
   being suppressed by "already alerted today".
3. **The reasoning was generic by construction.** The prompt handed the model a
   two-day news query and, when it came back empty, told it to reason from
   general knowledge. `pipeline/grounded_answer.py` is the replacement: grade
   the evidence, search the web when it is thin for a tracked drop, and say
   plainly when nothing explains the move.

Alerts are also no longer Telegram-only. Every one is persisted to `alerts` and
published on the `alert` SSE topic, so the dashboard card shows what went out
while the phone was asleep.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Optional

import httpx

from api.sse_manager import event_bus
from bot.alerts import AlertManager
from bot.formatters import escape_html, render_price_alert
from config.llm import complete, is_llm_configured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from data.watchlist import DEFAULT_WATCHLIST, INDEX_TICKERS
from pipeline.grounded_answer import explain_move
from pipeline.macro_calendar import MacroCalendar, today_et
from pipeline.technical_rating import TechnicalRatingTracker

log = get_logger(__name__)

# Re-exported: DEFAULT_WATCHLIST lived here before price_feed needed it too, and
# importing it from this module drags in bot.alerts for no reason. Existing
# importers keep working.
__all__ = ["MarketScanner", "DEFAULT_WATCHLIST"]

# Kinds the escalation rule measures "has the move deepened?" across. A volume
# alert is not a move, so it must not raise the bar for a later price alert.
MOVE_ALERT_KINDS = ("price_drop", "price_move")

# Sessions in the anomalous-volume baseline.
VOLUME_BASELINE_SESSIONS = 20


def _quote_age_seconds(quote: dict) -> Optional[float]:
    """How old a stored quote is, or None when that cannot be determined."""
    raw = quote.get("updated_at")
    if not raw:
        return None
    try:
        stamp = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        # CURRENT_TIMESTAMP is UTC and stored without a zone.
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()


class MarketScanner:
    """Watches stored quotes for significant moves and explains them."""

    def __init__(self, db: Database, alert_manager: Optional[AlertManager] = None):
        self.db = db
        self.alert_manager = alert_manager
        self.macro = MacroCalendar(db)
        self.ratings = TechnicalRatingTracker(db)

    # ── Scan ─────────────────────────────────────────────────────────────

    async def run_scan(self) -> None:
        """One pass over the stored quotes. Makes no outbound HTTP calls."""
        if not self.alert_manager:
            log.warning("scanner.no_alert_manager")
            return

        tracked = set(await asyncio.to_thread(self.db.get_tracked_tickers) or [])
        universe = sorted(tracked | set(DEFAULT_WATCHLIST))
        quotes = await asyncio.to_thread(self.db.get_latest_prices, universe)

        # A quote the feed has not touched in three refresh intervals is not a
        # live price. Alerting off one means announcing yesterday's close as
        # today's move, which is worse than saying nothing.
        max_age = max(settings.price_refresh_seconds * 3, 60)
        fresh: dict[str, dict] = {}
        for ticker, quote in quotes.items():
            age = _quote_age_seconds(quote)
            if age is None or age > max_age:
                log.debug("scanner.stale_quote", ticker=ticker, age_seconds=age)
                continue
            fresh[ticker] = quote

        index_ctx = {
            t: fresh[t].get("daily_change_pct")
            for t in INDEX_TICKERS
            if t in fresh and fresh[t].get("daily_change_pct") is not None
        }
        macro_today = await asyncio.to_thread(self.macro.context_lines)

        log.info("scanner.starting", universe=len(universe), fresh=len(fresh),
                 missing=len(universe) - len(quotes), index=len(index_ctx))

        for ticker, quote in fresh.items():
            try:
                await self._check_ticker(
                    ticker, quote, tracked=tracked,
                    index_ctx=index_ctx, macro_today=macro_today,
                )
            except Exception as e:
                log.error("scanner.ticker_failed", ticker=ticker, error=str(e))

    def _threshold(self, ticker: str, pct: float, tracked: set[str]) -> float:
        """
        The move that counts as an alert for this ticker, in this direction.

        Deliberately asymmetric. A tracked position falling is the thing the
        user asked to hear about early; everything else — a tracked name rising,
        or any default-watchlist move — keeps the original 5% either-way rule so
        the channel stays readable.
        """
        if ticker in tracked and pct < 0:
            return settings.alert_drop_pct_tracked
        return settings.alert_move_pct

    async def _should_alert(self, ticker: str, pct: float) -> bool:
        """
        First alert of the day, or one that has deepened past the step.

        The old rule was "once per day, full stop", so a position that opened
        down 3% and closed down 11% produced exactly one message, sent at the
        shallowest point of the day.
        """
        last = await asyncio.to_thread(
            self.db.get_last_alert_abs_pct_today, ticker, MOVE_ALERT_KINDS
        )
        if last is None:
            return True
        return abs(pct) >= last + settings.alert_escalation_step_pct

    async def _check_ticker(
        self,
        ticker: str,
        quote: dict,
        *,
        tracked: set[str],
        index_ctx: dict[str, float],
        macro_today: list[str],
    ) -> None:
        """Evaluate one stored quote for a move and for a volume spike."""
        pct = quote.get("daily_change_pct")
        price = quote.get("price")
        if pct is None or price is None:
            return
        pct = float(pct)
        price = float(price)

        threshold = self._threshold(ticker, pct, tracked)
        if abs(pct) >= threshold:
            is_tracked_drop = ticker in tracked and pct < 0
            kind = "price_drop" if is_tracked_drop else "price_move"
            if await self._should_alert(ticker, pct):
                log.info("scanner.trigger", ticker=ticker, pct=round(pct, 2),
                         threshold=threshold, kind=kind)
                await self._send_move_alert(
                    ticker, pct, price, kind=kind,
                    # Only a tracked drop is worth a Tavily search. Letting every
                    # 5% wobble on the default list search the web is how an
                    # unattended job runs up a bill on a volatile day.
                    allow_web=is_tracked_drop,
                    index_ctx=index_ctx, macro_today=macro_today,
                )
            else:
                log.debug("scanner.suppressed", ticker=ticker, pct=round(pct, 2))

        await self._check_volume(ticker, quote, pct, price, index_ctx, macro_today)

    # ── Alerts ───────────────────────────────────────────────────────────

    async def _send_move_alert(
        self,
        ticker: str,
        pct: float,
        price: float,
        *,
        kind: str,
        allow_web: bool,
        index_ctx: dict[str, float],
        macro_today: list[str],
        vol_multiple: Optional[float] = None,
    ) -> None:
        """Explain, render, persist, publish, send — in that order."""
        answer = await explain_move(
            self.db, ticker, pct, price,
            index_context=index_ctx,
            macro_today=macro_today,
            model=settings.model_market_scanner,
            allow_web=allow_web,
        )

        text = render_price_alert(
            ticker=ticker, pct=pct, price=price, kind=kind, answer=answer,
            index_context=index_ctx, macro_lines=macro_today,
            technical_line=await self._technical_line(ticker),
            vol_multiple=vol_multiple,
        )

        direction = "up" if pct >= 0 else "down"
        title = (
            f"{ticker} volume {vol_multiple:.1f}x average"
            if kind == "volume" and vol_multiple
            else f"{ticker} {direction} {abs(pct):.2f}%"
        )
        explanation = answer.explanation
        await self._persist_and_publish({
            "ticker": ticker,
            "kind": kind,
            "pct": pct,
            "price": price,
            "severity": self._severity(pct),
            "title": title,
            "summary": (explanation.cause if explanation else answer.text) or None,
            "body_html": text,
            "sources_json": answer.sources,
            "grounded_by": answer.grounded_by,
        })

        await self.alert_manager.send_html(text)
        log.info("scanner.alert_sent", ticker=ticker, kind=kind,
                 grounded_by=answer.grounded_by, sources=len(answer.sources))

    @staticmethod
    def _severity(pct: float) -> str:
        """Coarse band, for colouring the dashboard row."""
        magnitude = abs(pct)
        if magnitude >= 10.0:
            return "critical"
        if magnitude >= 5.0:
            return "high"
        return "medium"

    async def _technical_line(self, ticker: str) -> str:
        """The first line of the technical rating report, or nothing."""
        try:
            report = await asyncio.to_thread(self.ratings.get_report, ticker)
        except Exception as e:
            log.warning("scanner.rating_failed", ticker=ticker, error=str(e))
            return ""
        first = (report or "").strip().splitlines()
        if not first:
            return ""
        head = first[0].strip()
        # get_report names the absence in its first line when there is no
        # rating; that sentence is not worth a section in the message.
        return "" if head.lower().startswith("no technical rating") else head

    async def _persist_and_publish(self, row: dict) -> None:
        """
        Store the alert, then push the stored row onto the SSE bus.

        The stored row rather than the one built here: the dashboard card
        prepends live events into the list it fetched from /api/alerts, so the
        two have to be the same shape down to `id` and `created_at`.
        """
        try:
            alert_id = await asyncio.to_thread(self.db.insert_alert, row)
            stored = await asyncio.to_thread(self.db.get_alert, alert_id)
        except Exception as e:
            log.error("scanner.alert_persist_failed",
                      ticker=row.get("ticker"), error=str(e))
            return
        try:
            await event_bus.publish("alert", stored or row)
        except Exception as e:
            # The push is a nicety; the Telegram message and the stored row are
            # the product. A bus failure must not swallow either.
            log.warning("scanner.alert_publish_failed",
                        ticker=row.get("ticker"), error=str(e))

    # ── Volume ───────────────────────────────────────────────────────────

    async def _check_volume(
        self,
        ticker: str,
        quote: dict,
        pct: float,
        price: float,
        index_ctx: dict[str, float],
        macro_today: list[str],
    ) -> None:
        """
        Alert when today's volume dwarfs the 20-session average.

        Two bugs fixed at once. The old check read volumes out of a Yahoo
        `range=2d` response — at most two bars — behind a `len(volumes) >= 20`
        guard, so it could never fire at all. And its dedup key was the bare
        ticker in `sent_alerts`, which has no date column, so the first time it
        ever did fire would have been the last.
        """
        volume = quote.get("volume")
        if not volume:
            return
        average = await asyncio.to_thread(
            self.db.get_avg_volume, ticker, VOLUME_BASELINE_SESSIONS
        )
        if not average or average <= 0:
            return

        multiple = float(volume) / average
        if multiple < settings.alert_volume_multiple:
            return

        key = f"{ticker}_{today_et().isoformat()}"
        if await asyncio.to_thread(self.db.was_alert_sent, key, "anomalous_volume"):
            return

        log.info("scanner.volume_trigger", ticker=ticker,
                 multiple=round(multiple, 2))
        await self._send_move_alert(
            ticker, pct, price, kind="volume", allow_web=False,
            index_ctx=index_ctx, macro_today=macro_today,
            vol_multiple=multiple,
        )
        await asyncio.to_thread(self.db.record_alert, key, "anomalous_volume")

    # ── Earnings whisper ─────────────────────────────────────────────────

    async def check_earnings(self) -> None:
        """Query Finnhub for tracked tickers reporting on the next trading day."""
        # settings reads .env through pydantic-settings, which does NOT export
        # into os.environ — os.getenv here returned None on every run, so this
        # alert had never fired.
        api_key = settings.finnhub_api_key
        if not api_key or not self.alert_manager:
            return

        tracked = await asyncio.to_thread(self.db.get_tracked_tickers)
        combined_watchlist = sorted(set(DEFAULT_WATCHLIST) | set(tracked or []))

        # We want earnings for tomorrow (or Monday if it's Friday)
        today = dt.datetime.now()
        tomorrow = today + dt.timedelta(days=1)
        if today.weekday() == 4:  # Friday
            tomorrow = today + dt.timedelta(days=3)

        date_str = tomorrow.strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=10) as http_client:
            for ticker in combined_watchlist:
                try:
                    url = (
                        f"https://finnhub.io/api/v1/calendar/earnings"
                        f"?from={date_str}&to={date_str}&symbol={ticker}&token={api_key}"
                    )
                    resp = await http_client.get(url)
                    resp.raise_for_status()
                    data = resp.json()

                    if not data.get("earningsCalendar", []):
                        continue
                    key = f"{ticker}_{date_str}"
                    if await asyncio.to_thread(
                        self.db.was_alert_sent, key, "earnings_whisper"
                    ):
                        continue
                    await self._send_earnings_whisper(ticker, date_str)
                    await asyncio.to_thread(
                        self.db.record_alert, key, "earnings_whisper"
                    )
                except Exception as e:
                    log.error("scanner.earnings_failed", ticker=ticker, error=str(e))

    async def _send_earnings_whisper(self, ticker: str, date_str: str) -> None:
        """Sends an earnings whisper alert generated by the LLM."""
        if not is_llm_configured() or not settings.model_market_scanner:
            return

        rows = await asyncio.to_thread(
            self.db.get_recent_articles_for_ticker, ticker, 24 * 14, 15
        )
        if rows:
            context = "\n".join(
                f"- {r.get('headline')}: "
                f"{r.get('classification_summary') or r.get('summary') or ''}"
                for r in rows
            )
        else:
            context = "No specific news found in the database over the last two weeks."

        prompt = (
            f"You are a professional, precise Wall Street analyst writing an earnings preview.\n"
            f"The stock {ticker} reports earnings tomorrow ({date_str}).\n\n"
            f"Recent news context (last 14 days) for {ticker}:\n{context}\n\n"
            f"Write a concise 'Whisper Alert' covering:\n"
            f"1. Overall market sentiment heading into this print (bullish/bearish/anxious/mixed) — with a verbatim quote from the context if available.\n"
            f"2. The 1-2 key metrics or themes the market will be watching most closely.\n"
            f"3. Whether expectations seem realistic or stretched based on the news flow.\n"
            f"If the context says there is no news, say so plainly and do not invent a whisper number.\n"
            f"Use HTML for formatting: <b>bold</b> for key terms, <i>italic</i> for nuance. No emojis, no Markdown."
        )
        try:
            with track_llm(self.db, settings.model_market_scanner, "earnings_whisper",
                           prompt_text=prompt, store_text=True) as u:
                u.response = response = await complete(
                    model=settings.model_market_scanner,
                    prompt=prompt,
                    reasoning="low",
                )
            reasoning = escape_html(response.text.strip())
            reasoning = reasoning.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            reasoning = reasoning.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")

            safe_ticker = escape_html(ticker)
            message = f"🗣️ <b>EARNINGS WHISPER: {safe_ticker}</b>\n\n"
            message += (
                f"<b>{safe_ticker}</b> reports earnings tomorrow ({escape_html(date_str)}). "
                f"Here is the read on the room:\n\n"
            )
            message += f"<i>{reasoning}</i>"

            sources = [
                {
                    "title": r.get("headline"),
                    "url": r.get("url") or "",
                    "source": r.get("source_name") or "in-house",
                    "published_at": str(r.get("published_at") or "")[:10],
                    "kind": "db",
                }
                for r in rows[:3]
            ]
            await self._persist_and_publish({
                "ticker": ticker,
                "kind": "earnings_whisper",
                "pct": None,
                "price": None,
                "severity": "medium",
                "title": f"{ticker} reports earnings {date_str}",
                "summary": response.text.strip()[:400],
                "body_html": message,
                "sources_json": sources,
                "grounded_by": "db" if rows else "none",
            })

            await self.alert_manager.send_html(message)
            log.info("scanner.earnings_whisper_sent", ticker=ticker)

        except Exception as e:
            log.error("scanner.whisper_failed", ticker=ticker, error=str(e))
