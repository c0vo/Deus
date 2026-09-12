"""
Daily stance — one evidence-based call per tracked ticker, every morning.

Replaces the news-only advisor note that printed HOLD for every ticker. That
note had three separate paths to the same word: a prompt offering only HOLD or
SELL and instructing the model to fall back to the former, a hardcoded
"HOLD - No significant news today." whenever a ticker had no classified article
in 24h (the normal case for an ETF), and "HOLD - Unable to generate advice."
when `MODEL_DAILY_ADVISOR` was unset. Absence of evidence, absence of news and
absence of a model all rendered as a considered decision to hold.

Three things change here:

- **The evidence is the whole fact sheet**, not the last 24h of headlines:
  returns against SPY, RSI, distance from the 52-week high, the stored technical
  rating, sell-side consensus and target, the live ML probability, insider and
  off-exchange positioning, the standing 5-day debate, and what is on the macro
  calendar this week. A ticker with no news still has all of it.
- **Failure is named, never rendered as HOLD.** A ticker missing from the
  model's response is `NO CALL`; an unset model makes every row `UNAVAILABLE`.
  Both read as "the pipeline did not produce a call", which is what happened.
- **`changed_since_yesterday` is computed from the stored record**, never asked
  of the model — it has no way to know what it said yesterday, and asking
  invites it to invent an answer.

The expensive Bull/Bear debate is not run here. `material_changes` selects the
few tickers whose facts actually moved, and the scheduler runs those debates
*after* the message has been sent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field

from config.llm import complete, is_llm_configured, parse_structured
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from pipeline.darkpool import DarkPoolTracker
from pipeline.insider_tracker import InsiderTracker
from pipeline.macro_calendar import MacroCalendar, today_et

log = get_logger(__name__)

# The four calls the model may make. BUY/ADD and TRIM exist because the old
# prompt's HOLD-or-SELL menu could not express "this is working, add to it" or
# "take some off" — so every constructive read collapsed into HOLD.
StanceAction = Literal["BUY/ADD", "HOLD", "TRIM", "SELL"]

# Written by the code, never by the model. Neither is a stance: they say the
# pipeline produced no call, which is the distinction the old advisor erased.
NO_CALL = "NO CALL"
UNAVAILABLE = "UNAVAILABLE"

ARROW_NEW = "🆕"
ARROW_UNCHANGED = "↔"
ARROW_UPGRADED = "⬆"
ARROW_DOWNGRADED = "⬇"

# Ranked weakest → strongest so the arrow is a comparison of positions.
# NO CALL and UNAVAILABLE are deliberately absent: a failure is neither an
# upgrade nor a downgrade, and rows carrying one get no arrow at all.
_ACTION_RANK: dict[str, int] = {"SELL": 0, "TRIM": 1, "HOLD": 2, "BUY/ADD": 3}

# Insider and dark-pool reports are written for the Bull/Bear debate and run to
# several hundred words each. At one block per ticker they would dominate the
# prompt, so they are trimmed to their opening summary lines.
REPORT_CHARS = 300

# Daily bars pulled per ticker: enough for a 52-week high with room for
# holidays, and no more — this runs once per ticker per morning.
HISTORY_BARS = 280

# Sessions in a trading year, for the 52-week high window.
YEAR_SESSIONS = 252


class Stance(BaseModel):
    """One morning call on one ticker.

    `changed_since_yesterday` is NOT a field here by design: it is derived in
    `DailyStanceEngine._arrow` from the previous stored stance. A model asked
    whether it changed its mind has no record to check against and will answer
    anyway.
    """

    ticker: str = Field(description="Ticker symbol, exactly as given in the FACTS heading.")
    action: StanceAction = Field(
        description="The call for today. All four are available; HOLD is not a default."
    )
    conviction: Literal["Low", "Medium", "High"] = Field(
        description="Conviction in the call. Pick the nearest of the three."
    )
    thesis: str = Field(
        description="One or two sentences naming the specific facts behind the call."
    )
    key_risk: str = Field(
        description="The single fact from this ticker's block that most threatens the call."
    )
    evidence_used: list[str] = Field(
        default_factory=list,
        description="Short tags for the facts actually used, e.g. rsi, technical_rating, "
                    "analyst_target, news, darkpool, insider, ml, macro, debate.",
    )
    what_would_change_my_mind: str = Field(
        description="A concrete, checkable trigger that would flip the call."
    )


DAILY_STANCE_PROMPT = """You are the portfolio manager writing the morning stance note on a client's tracked positions.

For EACH ticker in the FACTS section below, choose exactly one action and justify it from that ticker's own facts.

ACTIONS — all four are available. This is not a hold-or-sell question.
- BUY/ADD: the evidence supports increasing the position now.
- HOLD: keep the position as it is.
- TRIM: reduce the position but stay in it.
- SELL: exit the position.

RULES
1. HOLD is not a default — justify it with named evidence like any other call. "Nothing happened" is not a justification; "RSI 54 with the daily technical rating at Neutral and no news in 48h" is.
2. If the only evidence is "no news", say so explicitly and state that the call rests on technicals and positioning — then quote the indicator values you used (RSI, the technical rating label, the analyst target and implied upside, the off-exchange short ratio, the ML probability).
3. Every claim must name a number, a label or a headline that appears in that ticker's FACTS block. Do not use outside knowledge and do not invent figures, dates or price levels.
4. BUY/ADD and TRIM are expected whenever the facts support them. A note where every ticker is HOLD is a note that did not read the facts.
5. A fact block that says data is absent ("no analyst coverage", "no news in 48h") is information, not a gap to fill. Reason from what is there.
6. evidence_used lists short tags for the facts you actually used — for example rsi, technical_rating, analyst_target, news, darkpool, insider, ml, macro, debate. Name only what you used.
7. key_risk is the one fact that most threatens the call, taken from the same block.
8. what_would_change_my_mind is a concrete, checkable trigger — "a daily close back above $412", "CPI above 3.1% on Thursday" — never a sentiment like "if conditions worsen".
9. Do NOT mention yesterday's stance, whether you changed your mind, or how long you have held a view. That is computed from the stored record, not from you.
10. Plain text only: no markdown, no HTML, no bullet characters, no currency symbols other than $.
11. Return exactly one entry per ticker, with `ticker` spelled exactly as in its FACTS heading, and no entries for tickers that are not listed.
"""


@dataclass
class StanceRow:
    """What actually gets stored, rendered and re-read: a call plus its history.

    Deliberately NOT a `Stance`. `Stance.action` is a four-value Literal because
    that Literal becomes the JSON schema the model is handed — a model cannot be
    allowed to answer NO CALL, and one that tries fails validation. But the row
    the code writes has to be able to say NO CALL and UNAVAILABLE, which are
    statements about the pipeline rather than about the position. Keeping them in
    two types is what stops "the model did not answer" from being spellable as a
    stance and vice versa.

    `arrow`, `prev_action` and `cached_debate` come from the stored record, never
    from the model.
    """

    ticker: str
    action: str
    conviction: str = "Low"
    thesis: str = ""
    key_risk: str = ""
    evidence_used: list[str] = field(default_factory=list)
    what_would_change_my_mind: str = ""
    arrow: str = ARROW_NEW
    prev_action: Optional[str] = None
    cached_debate: Optional[dict] = None

    @classmethod
    def from_stance(cls, stance: Stance, **extra) -> "StanceRow":
        return cls(
            ticker=stance.ticker.upper(),
            action=stance.action,
            conviction=stance.conviction,
            thesis=stance.thesis,
            key_risk=stance.key_risk,
            evidence_used=list(stance.evidence_used),
            what_would_change_my_mind=stance.what_would_change_my_mind,
            **extra,
        )


@dataclass
class StanceBatch:
    """One morning's worth of stances, with the facts they were built from.

    The facts travel with the rows because `material_changes` re-reads them to
    decide which debates to re-run, and rebuilding the sheets would repeat every
    query for no new information.
    """

    rows: list[StanceRow] = field(default_factory=list)
    facts: dict[str, dict] = field(default_factory=dict)
    model: str = ""
    date: str = ""


def compute_rsi14(closes: list[float], period: int = 14) -> Optional[float]:
    """Wilder-window RSI over a list of closes, or None below the warm-up.

    Deliberately local rather than reused from `StockPredictor._compute_rsi`:
    that is an instance method on a class whose import pulls in scikit-learn and
    joblib and whose constructor creates `storage/models/`. Eight lines of
    arithmetic is not worth loading the ML stack into the morning note.
    """
    if len(closes) < period + 1:
        return None
    diffs = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    window = diffs[-period:]
    gains = [d for d in window if d > 0]
    losses = [-d for d in window if d < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _pct_change(closes: list[float], sessions: int) -> Optional[float]:
    """Percent change over the last `sessions` bars, or None without the history."""
    if len(closes) < sessions + 1:
        return None
    past = closes[-(sessions + 1)]
    if not past:
        return None
    return (closes[-1] / past - 1) * 100


def _fmt(value: Optional[float], suffix: str = "", digits: int = 2) -> str:
    """Render a number, or the word 'n/a' — never a bare 'None' in a prompt."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}{suffix}" if suffix == "%" else f"{value:.{digits}f}{suffix}"


class DailyStanceEngine:
    """Builds the fact sheets, asks for one batched set of stances, stores them."""

    def __init__(self, db):
        self.db = db
        # Benchmark and calendar are market-wide, so they are fetched once per
        # batch rather than once per ticker. Populated on first use inside the
        # single executor hop that builds every fact sheet.
        self._spy_closes: Optional[list[float]] = None
        self._macro_lines: Optional[list[str]] = None

    # ── Fact sheet ───────────────────────────────────────────────────────

    def build_fact_sheet(self, ticker: str) -> dict:
        """Everything known about one ticker this morning. Synchronous.

        Hits SQLite directly, so every caller wraps it in `asyncio.to_thread`
        (`compose` builds the whole batch in one hop). Sections are gathered
        independently — the `add` pattern from `pipeline/agents.py` — so one
        empty or failing source degrades to its own named absence instead of
        blanking the sheet.
        """
        ticker = ticker.upper()
        facts: dict[str, Any] = {"ticker": ticker}

        def add(label: str, build: Callable[[], Any]) -> None:
            try:
                value = build()
            except Exception as e:
                log.warning("daily_stance.fact_section_failed", ticker=ticker,
                            section=label, error=str(e) or repr(e))
                return
            if value is not None:
                facts[label] = value

        add("price_block", lambda: self._price_block(ticker))
        add("technical_ratings",
            lambda: self.db.get_latest_technical_ratings(ticker) or None)
        add("analyst", lambda: self._analyst_block(ticker))
        add("ml", lambda: self._ml_block(ticker))
        add("insider_report",
            lambda: self._trim(InsiderTracker(self.db).get_report(ticker, days=90)))
        add("darkpool_report",
            lambda: self._trim(DarkPoolTracker(self.db).get_report(ticker, days=20)))
        add("news", lambda: self.db.get_recent_articles_for_ticker(
            ticker, hours=48, limit=3) or None)
        add("debate", lambda: self._debate_block(ticker))
        add("macro", self._macro)
        add("yesterday", lambda: self._yesterday(ticker))

        # Flattened for the consumers that need to compare numbers rather than
        # read prose: material_changes reads ret_1d and top_news_importance.
        price = facts.get("price_block") or {}
        facts["price"] = price.get("price")
        facts["ret_1d"] = price.get("ret_1d")
        news = facts.get("news") or []
        importances = [n.get("importance_score") for n in news
                       if n.get("importance_score") is not None]
        facts["top_news_importance"] = max(importances) if importances else None
        return facts

    def _price_block(self, ticker: str) -> Optional[dict]:
        rows = self.db.get_price_history(ticker, limit=HISTORY_BARS)
        closes = [float(r["close"]) for r in rows if r.get("close") is not None]
        if not closes:
            return None

        spy = self._spy()
        block: dict[str, Any] = {
            "price": closes[-1],
            "as_of": rows[-1].get("date"),
            "bars": len(closes),
            "ret_1d": _pct_change(closes, 1),
            "ret_5d": _pct_change(closes, 5),
            "ret_20d": _pct_change(closes, 20),
            "rsi14": compute_rsi14(closes),
        }

        window = closes[-YEAR_SESSIONS:]
        high = max(window)
        if high:
            block["pct_from_high"] = (closes[-1] / high - 1) * 100
            block["high_52w"] = high
            block["high_window_sessions"] = len(window)

        # Relative performance only when the benchmark is actually stored and
        # the ticker is not itself SPY — "SPY vs SPY: +0.00%" is noise.
        if spy and ticker != "SPY":
            for key, sessions in (("1d", 1), ("5d", 5), ("20d", 20)):
                own = block.get(f"ret_{key}")
                bench = _pct_change(spy, sessions)
                if own is not None and bench is not None:
                    block[f"vs_spy_{key}"] = own - bench
        return block

    def _spy(self) -> Optional[list[float]]:
        if self._spy_closes is None:
            try:
                rows = self.db.get_price_history("SPY", limit=HISTORY_BARS)
                self._spy_closes = [float(r["close"]) for r in rows
                                    if r.get("close") is not None]
            except Exception as e:
                log.warning("daily_stance.spy_history_failed", error=str(e))
                self._spy_closes = []
        return self._spy_closes or None

    def _analyst_block(self, ticker: str) -> Optional[dict]:
        row = self.db.get_latest_analyst_consensus(ticker)
        if not row:
            return None
        block = dict(row)
        # Upside against the stored spot rather than today's close: that is the
        # price the target was captured against, so the ratio stays reproducible.
        target = row.get("target_mean")
        spot = row.get("spot_price")
        if target and spot:
            block["target_upside_pct"] = (float(target) / float(spot) - 1) * 100
        return block

    def _ml_block(self, ticker: str) -> Optional[list[dict]]:
        """Live model probabilities only — never a fit.

        `active_only` keeps a call while its horizon has not elapsed. Training
        and prediction belong to the worker's own jobs; the morning note reads
        what is already there.
        """
        preds = self.db.get_recent_predictions(ticker, limit=20, active_only=True)
        seen: dict[int, dict] = {}
        for p in preds:
            horizon = p.get("horizon_days")
            if horizon is None or horizon in seen:
                continue
            seen[horizon] = {
                "horizon_days": horizon,
                "direction": p.get("predicted_direction"),
                "confidence": p.get("confidence"),
                "model_type": p.get("model_type"),
            }
        return [seen[h] for h in sorted(seen)] or None

    @staticmethod
    def _trim(text: Optional[str]) -> Optional[str]:
        """Cut a debate-length tracker report down to its opening summary."""
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > REPORT_CHARS:
            text = text[:REPORT_CHARS].rstrip() + "…"
        return text

    def _debate_block(self, ticker: str) -> Optional[dict]:
        cached = self.db.get_cached_advisory(ticker, days=5)
        if not cached:
            return None
        return {
            "direction": cached.get("trader_direction") or "UNKNOWN",
            "conviction": cached.get("trader_conviction") or "",
            "date": cached.get("_cache_date") or "",
            "executive_summary": (cached.get("executive_summary") or "")[:REPORT_CHARS],
        }

    def _macro(self) -> Optional[list[str]]:
        if self._macro_lines is None:
            try:
                rows = MacroCalendar(self.db).upcoming(days=7)
            except Exception as e:
                log.warning("daily_stance.macro_failed", error=str(e))
                rows = []
            lines = []
            for row in rows:
                line = f"{row.get('date')} {row.get('name')}"
                if row.get("time_et"):
                    line += f" {row['time_et']} ET"
                lines.append(line)
            self._macro_lines = lines
        return self._macro_lines or None

    def _yesterday(self, ticker: str) -> Optional[dict]:
        """The most recent stance strictly before today.

        `before` matters on a re-run: without it the engine would read the row
        it just wrote and report every ticker as unchanged.
        """
        row = self.db.get_latest_stance(ticker, before=self._today())
        if not row:
            return None
        return {"action": row.get("action"), "conviction": row.get("conviction"),
                "date": row.get("date"), "thesis": row.get("thesis")}

    def _today(self) -> str:
        """The Eastern-time session this note is about.

        Not UTC: the job fires at 05:05 KST, which is the afternoon of the US
        session being reviewed, and the stance belongs to that session.
        """
        return today_et().isoformat()

    # ── Rendering the facts for the prompt ───────────────────────────────

    def render_fact_block(self, facts: dict) -> str:
        """One ticker's facts as prompt text, naming every absence explicitly.

        An omitted line reads to a model as an oversight it should fill in. A
        line saying "no analyst coverage stored" is evidence, and rule 5 of the
        prompt tells it to treat it that way.
        """
        ticker = facts.get("ticker", "?")
        lines = [f"FACTS for {ticker}"]

        price = facts.get("price_block")
        if price:
            lines.append(
                f"- Price {_fmt(price.get('price'))} (close {price.get('as_of')}); "
                f"1d {_fmt(price.get('ret_1d'), '%')}, "
                f"5d {_fmt(price.get('ret_5d'), '%')}, "
                f"20d {_fmt(price.get('ret_20d'), '%')}"
            )
            rel = [f"{k} {_fmt(price.get(f'vs_spy_{k}'), '%')}"
                   for k in ("1d", "5d", "20d") if price.get(f"vs_spy_{k}") is not None]
            lines.append("- vs SPY: " + ", ".join(rel) if rel
                         else "- vs SPY: no benchmark history stored")
            lines.append(f"- RSI14 {_fmt(price.get('rsi14'), digits=1)}")
            if price.get("pct_from_high") is not None:
                lines.append(
                    f"- {_fmt(price.get('pct_from_high'), '%')} from the "
                    f"{price.get('high_window_sessions')}-session high of "
                    f"{_fmt(price.get('high_52w'))}"
                )
            else:
                lines.append("- 52-week high: not enough stored history")
        else:
            lines.append("- Price: NO price history stored for this ticker")

        ratings = facts.get("technical_ratings")
        if ratings:
            parts = []
            for r in ratings:
                parts.append(
                    f"{r.get('timeframe')}={r.get('summary_label')} "
                    f"({_fmt(r.get('summary_score'), digits=2)}; "
                    f"{r.get('buy_votes')}B/{r.get('neutral_votes')}N/"
                    f"{r.get('sell_votes')}S)"
                )
            lines.append("- Technical rating: " + "; ".join(parts))
        else:
            lines.append("- Technical rating: none stored")

        analyst = facts.get("analyst")
        if analyst:
            lines.append(
                f"- Analysts: {analyst.get('recommendation_key') or 'n/a'} "
                f"(mean {_fmt(analyst.get('recommendation_mean'), digits=2)} on "
                f"{analyst.get('analyst_count') or 0} opinions), target "
                f"{_fmt(analyst.get('target_mean'))}, implied upside "
                f"{_fmt(analyst.get('target_upside_pct'), '%')} "
                f"as of {analyst.get('session_date')}"
            )
        else:
            lines.append("- Analysts: no analyst coverage stored")

        ml = facts.get("ml")
        if ml:
            parts = [f"{m['horizon_days']}d {m.get('direction')} "
                     f"{(float(m.get('confidence') or 0) * 100):.0f}% "
                     f"({m.get('model_type')})" for m in ml]
            lines.append("- ML probability: " + "; ".join(parts))
        else:
            lines.append("- ML probability: no live model prediction")

        lines.append("- Insider/stakes: " +
                     (facts.get("insider_report") or "no disclosures on file"))
        lines.append("- Off-exchange volume: " +
                     (facts.get("darkpool_report") or "no off-exchange data on file"))

        news = facts.get("news")
        if news:
            lines.append("- News (48h, top 3):")
            for n in news:
                score = n.get("importance_score")
                summary = (n.get("classification_summary") or "")[:180]
                lines.append(
                    f"  * [{_fmt(score, digits=1)}] {n.get('published_at', '')[:10]} "
                    f"{n.get('source_name')}: {n.get('headline')}"
                    + (f" — {summary}" if summary else "")
                )
        else:
            lines.append("- News: no news in 48h")

        debate = facts.get("debate")
        if debate:
            lines.append(
                f"- Standing 5-day debate: {debate.get('direction')} "
                f"({debate.get('conviction')}, {debate.get('date')})"
                + (f" — {debate.get('executive_summary')}"
                   if debate.get("executive_summary") else "")
            )
        else:
            lines.append("- Standing 5-day debate: none in the last 5 days")

        macro = facts.get("macro")
        lines.append("- Macro this week: " + ("; ".join(macro) if macro
                                              else "nothing scheduled in the next 7 days"))

        yday = facts.get("yesterday")
        if yday:
            lines.append(
                f"- Previous stance on record: {yday.get('action')} "
                f"({yday.get('conviction')}) on {yday.get('date')}"
            )
        else:
            lines.append("- Previous stance on record: none")

        return "\n".join(lines)

    def build_prompt(self, facts_by_ticker: dict[str, dict]) -> str:
        """The full batched prompt: the rules once, then one block per ticker."""
        blocks = [self.render_fact_block(facts_by_ticker[t])
                  for t in facts_by_ticker]
        tickers = ", ".join(facts_by_ticker)
        return (
            DAILY_STANCE_PROMPT
            + f"\nReturn exactly {len(facts_by_ticker)} entries, one for each of: "
            + tickers
            + "\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )

    # ── Composition ──────────────────────────────────────────────────────

    async def compose(self, tickers: list[str]) -> StanceBatch:
        """Fact sheets for every ticker, then one batched call for the stances.

        One request for the whole watchlist rather than one per ticker: the
        facts are independent but the note is read as a whole, and a single
        request is one cost line and one failure mode instead of N.
        """
        tickers = [t.upper() for t in tickers]
        today = self._today()
        if not tickers:
            return StanceBatch(rows=[], facts={}, model=settings.model_daily_advisor,
                               date=today)

        # One executor hop for the whole batch. Per-ticker hops would each open
        # their own SQLite connection, and the two processes share the file.
        facts = await asyncio.to_thread(
            lambda: {t: self.build_fact_sheet(t) for t in tickers}
        )

        model = settings.model_daily_advisor
        if not model or not is_llm_configured():
            # Not a stance, and not HOLD. The note still goes out carrying the
            # fact sheets, because a silent morning and a broken morning have to
            # look different in the chat.
            log.warning("daily_stance.model_unconfigured",
                        model_set=bool(model), llm_configured=is_llm_configured())
            rows = [self._no_call(
                facts[t], t, UNAVAILABLE,
                thesis="MODEL_DAILY_ADVISOR is not configured — no stance was generated.",
                trigger="Set MODEL_DAILY_ADVISOR and restart the worker.",
            ) for t in tickers]
            return StanceBatch(rows=rows, facts=facts, model="", date=today)

        prompt = self.build_prompt(facts)
        parsed: list[Stance] = []
        try:
            with track_llm(self.db, model, "daily_stance_batch",
                           prompt_text=prompt, store_text=True) as u:
                u.response = response = await complete(
                    model=model, prompt=prompt, schema=list[Stance], reasoning="low",
                )
            if isinstance(response.parsed, list):
                parsed = response.parsed
            else:
                # `strict` is off on every schema call site, so the fallback is
                # load-bearing rather than defensive.
                parsed = parse_structured(response.text, list[Stance])
        except Exception as e:
            log.error("daily_stance.compose_failed", error=str(e) or repr(e),
                      tickers=len(tickers))

        by_ticker = {s.ticker.upper(): s for s in parsed if s.ticker}
        rows: list[StanceRow] = []
        missing: list[str] = []
        for t in tickers:
            stance = by_ticker.get(t)
            if stance is None:
                missing.append(t)
                rows.append(self._no_call(
                    facts[t], t, NO_CALL,
                    thesis="The model returned no stance for this ticker.",
                    trigger="A successful re-run of the daily stance job.",
                ))
                continue
            # Normalise the spelling so persist and the grid key on the same
            # symbol the watchlist uses.
            prev = (facts[t].get("yesterday") or {}).get("action")
            rows.append(StanceRow.from_stance(
                stance.model_copy(update={"ticker": t}),
                arrow=self._arrow(prev, stance.action),
                prev_action=prev,
                cached_debate=facts[t].get("debate"),
            ))

        if missing:
            log.warning("daily_stance.tickers_missing_from_response",
                        tickers=missing, returned=len(by_ticker))
        log.info("daily_stance.composed", tickers=len(tickers),
                 stances=len(by_ticker), missing=len(missing), model=model)
        return StanceBatch(rows=rows, facts=facts, model=model, date=today)

    def _no_call(self, facts: dict, ticker: str, action: str, *,
                 thesis: str, trigger: str) -> StanceRow:
        """A row saying the pipeline produced no call — never a HOLD.

        `prev_action` is still recorded, so the stored history shows what the
        last real stance was, but the arrow is empty: a failure is neither an
        upgrade, a downgrade, nor a considered decision to stand pat.
        """
        prev = (facts.get("yesterday") or {}).get("action")
        return StanceRow(
            ticker=ticker,
            action=action,
            conviction="Low",
            thesis=thesis,
            key_risk="No call was made; the position is unreviewed today.",
            evidence_used=[],
            what_would_change_my_mind=trigger,
            arrow=self._arrow(prev, action),
            prev_action=prev,
            cached_debate=facts.get("debate"),
        )

    @staticmethod
    def _arrow(prev_action: Optional[str], action: str) -> str:
        """Compare today's call against the stored one. Never asked of the model.

        A row with no real action (NO CALL / UNAVAILABLE) gets no arrow: it was
        not an upgrade, a downgrade, or a considered decision to stand pat.
        """
        if action not in _ACTION_RANK:
            return ""
        if not prev_action or prev_action not in _ACTION_RANK:
            return ARROW_NEW
        if _ACTION_RANK[action] > _ACTION_RANK[prev_action]:
            return ARROW_UPGRADED
        if _ACTION_RANK[action] < _ACTION_RANK[prev_action]:
            return ARROW_DOWNGRADED
        return ARROW_UNCHANGED

    # ── Persistence ──────────────────────────────────────────────────────

    def persist(self, batch: StanceBatch) -> int:
        """Write the batch to `stances`. Synchronous; wrap in `asyncio.to_thread`.

        Stores the fact sheet alongside the call for the same reason
        `digests.facts_json` exists: a stance whose numbers cannot be traced
        back to its inputs is indistinguishable from an invented one.
        """
        written = 0
        for row in batch.rows:
            try:
                self.db.upsert_stance(
                    row.ticker, batch.date, row.action,
                    conviction=row.conviction,
                    thesis=row.thesis,
                    key_risk=row.key_risk,
                    what_would_change=row.what_would_change_my_mind,
                    evidence_json=row.evidence_used,
                    facts_json=batch.facts.get(row.ticker),
                    prev_action=row.prev_action,
                    model=batch.model or None,
                )
                written += 1
            except Exception as e:
                log.error("daily_stance.persist_failed", ticker=row.ticker,
                          error=str(e) or repr(e))
        return written

    # ── Selective debate re-runs ─────────────────────────────────────────

    def material_changes(self, rows: list[StanceRow],
                         facts: dict[str, dict]) -> list[str]:
        """Tickers whose facts moved enough to justify a fresh Bull/Bear debate.

        The debate costs four reasoning calls per ticker, so it is rationed:
        only a real price move, a changed call, or genuinely important news
        earns one, a ticker already debated today is skipped so the 08:30
        `run_daily_predictions` job and this one cannot both pay for it, and the
        whole list is capped per day.

        Ordered flip-first, then by news importance, then by move size, so a cap
        of two spends on the two most informative re-runs rather than the first
        two tickers in the watchlist.
        """
        today = self._today()
        flagged: list[tuple[int, float, float, str]] = []

        for row in rows:
            ticker = row.ticker
            f = facts.get(ticker) or {}

            debate = f.get("debate") or {}
            debate_date = str(debate.get("date") or "")
            if debate_date and debate_date >= today:
                # Already debated today. The cache key is a UTC date while
                # `today` is the ET session, and UTC never lags ET, so `>=`
                # catches a debate run this morning under either clock.
                continue

            ret_1d = f.get("ret_1d")
            moved = (abs(ret_1d) >= settings.alert_drop_pct_tracked
                     if ret_1d is not None else False)
            flipped = row.arrow in (ARROW_UPGRADED, ARROW_DOWNGRADED)
            importance = f.get("top_news_importance") or 0.0
            big_news = importance >= settings.advisor_rerun_min_importance

            if moved or flipped or big_news:
                flagged.append((1 if flipped else 0, float(importance),
                                abs(ret_1d) if ret_1d is not None else 0.0, ticker))

        flagged.sort(reverse=True)
        picked = [t for *_, t in flagged[:max(settings.advisor_rerun_max_per_day, 0)]]
        if flagged:
            log.info("daily_stance.material_changes", flagged=len(flagged),
                     picked=picked, cap=settings.advisor_rerun_max_per_day)
        return picked
