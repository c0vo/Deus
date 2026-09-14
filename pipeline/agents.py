import json
import asyncio
import math
from typing import TypedDict, Optional, Dict, Callable, Awaitable, Literal
import yfinance as yf
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from config.llm import (
    complete,
    response_cost,
    is_llm_configured,
    parse_structured,
    salvage_json_field,
    stream_complete,
)
from config.settings import settings
from config.logging_config import get_logger
from config.usage import track_llm
from data.tickers import KR, US, classify_market
from pipeline.insider_tracker import InsiderTracker
from pipeline.kr_flows import KrFlowTracker
from pipeline.darkpool import DarkPoolTracker
from pipeline.market_regime import MarketRegimeTracker
from pipeline.analyst_ratings import AnalystRatingsTracker
from pipeline.technical_rating import TechnicalRatingTracker

log = get_logger(__name__)


def _finite(value) -> Optional[float]:
    """`value` as a float, or None when it is missing, non-numeric, NaN or infinite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _or_na(value: Optional[float], spec: str) -> str:
    return "N/A" if value is None else format(value, spec)


def _feature_snapshot(raw) -> dict:
    """The ML baseline's feature snapshot as a dict, whatever form it arrived in.

    The predictor sends a JSON string with NaN written as null. Advisories
    cached by older code can hold a dict, or a string encoded twice.
    """
    value = raw
    for _ in range(2):
        if not isinstance(value, (str, bytes, bytearray)):
            break
        try:
            value = json.loads(value or "{}")
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


# Passed to Gemini as `response_schema`, which constrains decoding so the long
# markdown body lands in `full_advisory` correctly escaped. Describing the shape
# in the prompt alone was not enough: the model reliably emitted literal
# newlines inside the string, which the JSON decoder rejects as control
# characters — and the parse failure then leaked the raw blob to the UI.
#
# The docstring is sent to the model as the schema `description`, so it stays short.
class TraderAdvisory(BaseModel):
    """The trade advisory: the call itself, then the reasoning behind it."""

    # Declared before the prose so a response that runs out of tokens loses the
    # tail of the write-up rather than the decision. The panel headline reads
    # these two directly — before they existed the UI had nothing but the ML
    # baseline to show, and printed UNKNOWN whenever no model was trained.
    direction: Literal["BUY", "SELL", "HOLD"] = Field(
        description="The trade call for the stated horizon."
    )
    conviction: Literal["Low", "Medium", "High"] = Field(
        description="Conviction in the call. Pick the nearest of the three."
    )
    executive_summary: str = Field(
        description="TLDR: [BUY/SELL/HOLD] — 1-2 sentence actionable reason, no markdown."
    )
    full_advisory: str = Field(
        description="Complete markdown analysis with ### section headers."
    )


# ── Differentiated system messages for debate agents ──────────────────────
# Previously both Bull and Bear shared a generic "top-tier financial researcher"
# message. Differentiated roles produce sharper, more adversarial debate.

_BULL_SYSTEM_MESSAGE = (
    "You are a bullish equity researcher. Your role is to build the strongest "
    "possible long thesis using concrete catalysts from the provided context. "
    "You MUST: (1) anchor every claim in a specific news item, date, or figure "
    "from the context; (2) estimate magnitude, not just direction (e.g., "
    "'could add 5-8%' not just 'will go up'); (3) acknowledge the bear case's "
    "valid points before countering them — steelman, don't strawman; "
    "(4) flag when your thesis depends on an unverified assumption. "
    "Be direct, concise, and fact-driven. No fluff, rhetoric, or sassy language."
)

_BEAR_SYSTEM_MESSAGE = (
    "You are a skeptical equity researcher. Your role is to stress-test bullish "
    "narratives by finding what could go wrong. You MUST: (1) identify specific "
    "risks the bull case ignores or downplays; (2) quantify downside when "
    "possible (e.g., 'if X misses, expect 10-15% drawdown'); (3) distinguish "
    "between temporary headwinds and structural problems; (4) concede when the "
    "bull case has genuinely strong evidence you cannot refute — a credible "
    "skeptic knows when to stand down. "
    "Be direct, concise, and fact-driven. No fluff, rhetoric, or sassy language."
)

_TRADER_SYSTEM_MESSAGE = (
    "You are the Head Trader and Risk Manager at a quantitative hedge fund. "
    "You synthesize conflicting analyst reports into actionable trade decisions. "
    "Your decision weights: (1) news sentiment and specific catalysts as primary "
    "drivers, (2) fundamental data as structural context, (3) ML predictions as "
    "a minor confirmatory signal only. An ML baseline reporting no measurable edge "
    "is the historical base rate, not a forecast: never count it, or an argument "
    "resting on it, as directional evidence. For every recommendation, explicitly state "
    "your conviction level, time horizon, and the key risk that would invalidate "
    "your thesis. Format output as clean Markdown with ### headers."
)

# Shared debate rules — extracted to avoid duplication
_COMMON_RULES = (
    "CRITICAL RULES:\n"
    "1. Be extremely direct, concise, and fact-driven. No fluff, rhetoric, or sassy language.\n"
    "2. Ground your arguments primarily in the RECENT news context. You MUST explicitly call out and analyze any specific upcoming catalysts, exact dates (e.g., IPOs, earnings), and figures mentioned in the news.\n"
    "3. If the opposing side makes a valid point, intelligently acknowledge it. Meaningful debate requires conceding undeniable facts.\n"
    "4. News is presented in two sections — 'IN-HOUSE NEWS' (curated, classified by importance) and 'LIVE WEB SEARCH RESULTS' (real-time web data). Treat both as current and factual, but prioritize in-house news when available as it has been through classification.\n"
    "5. 'SMART MONEY' sections carry disclosed positioning and are factual filings, not opinion. Insider activity comes from SEC Form 4 and covers OPEN-MARKET buys and sells only — grants and option exercises are excluded because they are compensation, not conviction. A sale marked '10b5-1 pre-scheduled' was set up months in advance and is weak evidence of a view; an unscheduled purchase is strong evidence. A 13D means a holder intends to influence the company; a 13G is passive. Korean flows show what institutions (기관) and foreign investors (외국인) net bought or sold. Absence of insider buying is NOT the same as insider selling — do not treat 'no disclosures' as bearish.\n"
    "6. 'Technicals (ML Base)' is a statistical baseline. When it reports no measurable edge, its direction is the historical base rate, not a forecast: do not cite it as evidence for either side."
)

# Persona + rules composed into the system message so the ENTIRE invariant part
# of the prompt sits ahead of everything that changes between rounds.
#
# _COMMON_RULES used to be spliced into the user message *after* the debate
# history — which grows on every turn — so the shared prefix ended at the
# context block and the rules were re-billed at full price each round. Both
# providers price a cached prefix well below a fresh one, and the prefix only
# extends as far as the first byte that differs.
_BULL_SYSTEM = f"{_BULL_SYSTEM_MESSAGE}\n\n{_COMMON_RULES}"
_BEAR_SYSTEM = f"{_BEAR_SYSTEM_MESSAGE}\n\n{_COMMON_RULES}"


class AdvisoryState(TypedDict):
    ticker: str
    ml_prediction: dict
    past_lessons: dict  # Structured: {ticker_lessons: [...], sector_lessons: [...], market_lessons: [...]}
    news_context: str

    fundamentals_report: str
    technical_report: str
    smart_money_report: str
    thesis_context: str

    debate_history: list[str]
    debate_round_count: int

    final_advisory: str
    # Every key a node returns has to be declared here or LangGraph drops the
    # write silently. `executive_summary` was returned but undeclared for
    # weeks, so the watchlist card and the Telegram summary rendered blank
    # with nothing failing anywhere.
    executive_summary: str
    trader_direction: str
    trader_conviction: str

class AdvisoryGraph:
    def __init__(self, db, progress_callback=None, debate_chunk_callback: Optional[Callable[[str, int, str], Awaitable[None]]] = None):
        self.db = db
        self.progress_callback = progress_callback
        self.debate_chunk_callback = debate_chunk_callback

    @staticmethod
    def _format_lessons(lessons: dict) -> str:
        """Format the structured lessons dict into a relevance-ranked text block."""
        if not lessons or all(not v for v in lessons.values()):
            return "No relevant past lessons available."

        parts = []

        ticker = lessons.get("ticker_lessons", [])
        if ticker:
            lines = "\n".join(
                f"- {'[SUCCESS]' if l.get('was_successful') else '[FAILURE]'} "
                f"{l['lesson_learned']}"
                for l in ticker
            )
            parts.append(f"Ticker-Specific Lessons:\n{lines}")

        sector = lessons.get("sector_lessons", [])
        if sector:
            lines = "\n".join(
                f"- {'[SUCCESS]' if l.get('was_successful') else '[FAILURE]'} "
                f"[{l.get('sector', '?')}] {l['lesson_learned']}"
                for l in sector
            )
            parts.append(f"Sector Lessons (same sector as this ticker):\n{lines}")

        market = lessons.get("market_lessons", [])
        if market:
            lines = "\n".join(
                f"- {'[SUCCESS]' if l.get('was_successful') else '[FAILURE]'} "
                f"[Market] {l['lesson_learned']}"
                for l in market
            )
            parts.append(f"Market-Wide Lessons:\n{lines}")

        return "\n\n".join(parts) if parts else "No relevant past lessons available."

    async def _update_progress(self, msg: str):
        if self.progress_callback:
            try:
                await self.progress_callback(msg)
            except Exception as e:
                log.warning(f"Progress callback raised exception (e.g. print encoding error): {e}")

    async def aggregate_data_node(self, state: AdvisoryState) -> dict:
        """A fast, 0-token python script that formats data for the researchers."""
        await self._update_progress("✅ Aggregating Data (ML, News, Fundamentals)...")
        ticker = state["ticker"]

        # Fundamentals
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE", "N/A")
            fpe = info.get("forwardPE", "N/A")
            rev_growth = info.get("revenueGrowth", "N/A")
            margins = info.get("profitMargins", "N/A")
            fundamentals = f"Fundamentals for {ticker}: P/E={pe}, Fwd P/E={fpe}, Rev Growth={rev_growth}, Profit Margin={margins}."
        except Exception as e:
            fundamentals = f"Error fetching fundamentals: {e}"

        # Technicals / ML
        technicals = self._ml_baseline_report(state.get("ml_prediction") or {})

        # Smart money — disclosed positioning. US tickers get SEC Form 4 insider
        # trades and 13D/G stakes; Korean tickers get daily 기관/외국인 flows.
        # Both read from local tables, so this stays a zero-token, zero-network step.
        smart_money = self._build_smart_money_report(ticker)

        # Causal chain. If the Thesis Engine already placed this ticker inside a
        # bottleneck chain, the debate should argue that mechanism rather than
        # rediscover it from the news. Returns '' when the ticker is in no active
        # chain, so the prompt builder splices it in unconditionally.
        try:
            thesis_context = self.db.get_active_thesis_context(ticker)
        except Exception as e:
            log.warning("agents.thesis_context_failed", ticker=ticker, error=str(e))
            thesis_context = ""

        return {
            "fundamentals_report": fundamentals,
            "technical_report": technicals,
            "smart_money_report": smart_money,
            "thesis_context": thesis_context,
            "debate_round_count": 0,
            "debate_history": []
        }

    @staticmethod
    def _ml_baseline_report(ml: dict) -> str:
        """
        The ML baseline as the researchers read it, worded by what the model
        can actually claim.

        `prior` means walk-forward evaluation found no measurable edge at this
        horizon, so the direction and probability are the historical base rate.
        Printed like a forecast ("Predicts UP with 57% confidence") that base
        rate reads as a directional argument. `universal` is the pooled model's
        calibrated probability, stated beside the skill it measured out of
        sample. Rows from the retired per-ticker and sector tiers, which only
        old cached state still carries, keep their original wording.
        """
        direction = ml.get("predicted_direction") or "UNKNOWN"
        model_type = ml.get("model_type")
        meta = ml.get("model_meta") or {}

        if direction == "UNKNOWN" or model_type == "llm_only":
            return ("No trained ML model exists yet for this ticker. Running debate "
                    "using news, fundamentals, and general knowledge.")

        if model_type == "prior":
            text = "ML baseline: no measurable edge at this horizon"
            auc = _finite(meta.get("auc"))
            if auc is not None:
                low, high = _finite(meta.get("auc_ci_low")), _finite(meta.get("auc_ci_high"))
                ci = f", CI {low:.2f}-{high:.2f}" if low is not None and high is not None else ""
                text += f" (walk-forward AUC {auc:.2f}{ci})"
            base = _finite(meta.get("base_rate"))
            if base is None:
                base = _finite(ml.get("probability_up"))
            if base is None:
                return f"{text}. Treat direction as a coin flip."
            return (f"{text}; historical base rate {base:.0%} of windows closed higher. "
                    "Treat direction as a coin flip weighted by the base rate.")

        confidence = _finite(ml.get("confidence")) or 0.0
        if model_type == "universal":
            p_up = _finite(ml.get("probability_up"))
            if p_up is None:
                p_up = confidence if direction == "UP" else 1.0 - confidence
            features = _feature_snapshot(ml.get("feature_snapshot"))
            rsi = _finite(features.get("rsi_14"))
            vol_21 = _finite(features.get("vol_21"))
            rel_spy_21d = _finite(features.get("rel_spy_21d"))
            return (
                f"Quantitative ML (pooled walk-forward model, AUC {_or_na(_finite(meta.get('auc')), '.2f')}, "
                f"Brier skill {_or_na(_finite(meta.get('brier_skill')), '+.3f')}): "
                f"P(up) = {p_up:.0%} ({direction}). Key readings: RSI {_or_na(rsi, '.1f')}, "
                # vol_21 is the daily log-return standard deviation, and
                # rel_spy_21d the excess log return over SPY in units of it.
                f"21d vol {_or_na(vol_21, '.1%')}{'/day' if vol_21 is not None else ''}, "
                f"21d return vs SPY {_or_na(rel_spy_21d, '+.2f')}{'σ' if rel_spy_21d is not None else ''}."
            )

        return f"Quantitative ML Predicts {direction} with {int(confidence * 100)}% confidence."

    def _build_smart_money_report(self, ticker: str) -> str:
        """Insider / stake / flow / off-exchange context for the debate, by market.

        Sections are gathered independently so one empty or failing source
        degrades to its own "no data" line instead of blanking the others —
        a ticker with no Form 4 filings still has an off-exchange print record,
        and the market-wide regime applies to every instrument.
        """
        market = classify_market(ticker)
        sections: list[str] = []

        def add(label: str, build):
            try:
                text = build()
            except Exception as e:
                log.warning("agents.smart_money_section_failed", ticker=ticker,
                            section=label, error=str(e) or repr(e))
                return
            if text:
                sections.append(text)

        if market == US:
            add("insider", lambda: InsiderTracker(self.db).get_report(ticker, days=90))
            add("darkpool", lambda: DarkPoolTracker(self.db).get_report(ticker, days=20))
            # US-only: Yahoo's analyst coverage of Korean listings is too thin to
            # be worth showing the debate, and an unreliable consensus is worse
            # than none when the model will argue from it either way.
            add("analyst", lambda: AnalystRatingsTracker(self.db).get_report(ticker))
        elif market == KR:
            add("kr_flows", lambda: KrFlowTracker(self.db).get_report(ticker, days=20))

        # Market-wide, so they apply regardless of listing venue. The technical
        # rating is a pure function of price, so unlike everything above it works
        # for indices and crypto too.
        add("technical_rating",
            lambda: TechnicalRatingTracker(self.db).get_report(ticker))
        add("regime", lambda: MarketRegimeTracker(self.db).get_report(days=60))

        if not sections:
            return "No insider or institutional flow data applies to this instrument."
        return "\n\n".join(sections)

    def _build_debate_context(self, state: AdvisoryState) -> str:
        """The shared evidence block both researchers argue from.

        Bull and Bear read the same string by construction. While these were two
        copies, adding evidence to one side and not the other was a one-line
        mistake that would have quietly made the debate unfair rather than
        raising an error.
        """
        parts = [
            f"Ticker: {state['ticker']}",
            f"Fundamentals: {state.get('fundamentals_report')}",
            f"Technicals (ML Base): {state.get('technical_report')}",
            f"News Context (Sentiment/Urgency): {state.get('news_context')}",
            f"SMART MONEY (disclosed insider & institutional positioning):\n"
            f"{state.get('smart_money_report') or 'Not available.'}",
        ]

        thesis = state.get("thesis_context")
        if thesis:
            parts.append(
                "CAUSAL CHAIN (the Thesis Engine placed this ticker inside an "
                "active chain - how the theme reaches it, and what breaks the "
                f"link):\n{thesis}\n"
                "Read the crowding stage as timing, not conviction: EARLY means "
                "the move is not yet priced in, CROWDED means it largely is. "
                "Argue the mechanism and the stated falsifier - the theme being "
                "popular is not itself evidence."
            )

        parts.append(
            f"Past Lessons (relevance-ranked):\n"
            f"{self._format_lessons(state.get('past_lessons', {}))}"
        )
        return "\n".join(parts)

    # Appended to a turn the provider cut off mid-sentence. Reasoning tokens
    # share the completion budget, so this is reachable whenever the thinking
    # block crowds out the answer — see settings.debate_max_output_tokens.
    _TRUNCATION_MARKER = "\n\n_[turn truncated — output budget reached]_"

    async def _finalize_turn(
        self,
        content: Optional[str],
        finish_reason: Optional[str],
        completion_tokens: int,
        speaker: Optional[str] = None,
        round_num: Optional[int] = None,
    ) -> str:
        """Make a cut-off or empty debate turn visible instead of silent.

        A turn that hit the token ceiling used to be appended to
        `debate_history` and cached as though it had concluded, and an empty
        one appended a bare "Bull: " that the arena renders as a card which
        simply is not there. Both now say so, in the transcript and the log.
        """
        text = (content or "").strip()
        if not text:
            log.warning("agents.debate_turn_empty", speaker=speaker,
                        round_num=round_num, finish_reason=finish_reason,
                        completion_tokens=completion_tokens)
            addition = ("[No argument returned — the model spent its entire "
                        "completion budget on reasoning without emitting an "
                        "answer.]")
            text = addition
        elif finish_reason == "length":
            log.warning("agents.debate_turn_truncated", speaker=speaker,
                        round_num=round_num, completion_tokens=completion_tokens,
                        chars=len(text))
            addition = self._TRUNCATION_MARKER
            text = text + addition
        else:
            return text

        # Live viewers watched the stream stop mid-thought; say why on the same
        # channel, or the marker only surfaces once the verdict event lands.
        if self.debate_chunk_callback and speaker and round_num is not None:
            await self.debate_chunk_callback(speaker, round_num, addition)
        return text

    async def _call_researcher(
        self, prompt: str, speaker: Optional[str] = None,
        round_num: Optional[int] = None, system_message: Optional[str] = None,
    ) -> str:
        """
        One debate turn, streamed when somebody is watching the arena.

        Runs at xhigh reasoning effort, which is why `debate_max_output_tokens`
        is set so high: the thinking block is billed against the same
        completion budget as the answer, so a round-2 turn carrying the full
        history can exhaust it and come back empty. `_finalize_turn` is what
        turns that into a visible marker rather than a silent blank.
        """
        if not is_llm_configured() or not settings.model_debate:
            return "No model configured for the debate — set MODEL_DEBATE."

        model_name = settings.model_debate
        sys_msg = system_message or (
            "You are a top-tier financial researcher. "
            "Reason through the facts before answering."
        )
        common = dict(
            system=sys_msg,
            max_tokens=settings.debate_max_output_tokens,
            reasoning="xhigh",
        )

        try:
            if self.debate_chunk_callback and speaker and round_num is not None:
                collected: list[str] = []
                finish_reason = None
                usage = None
                with track_llm(self.db, model_name, "debate_research",
                               prompt_text=prompt, store_text=True) as u:
                    async for chunk in stream_complete(
                        model=model_name, prompt=prompt, **common
                    ):
                        if chunk.text:
                            collected.append(chunk.text)
                            await self.debate_chunk_callback(speaker, round_num, chunk.text)
                        if chunk.finish_reason:
                            finish_reason = chunk.finish_reason
                        if chunk.usage is not None:
                            usage = chunk.usage
                    # Usage rides the trailing chunk, so it is assigned onto the
                    # record rather than derived from a response object.
                    if usage is not None:
                        u.prompt_tokens = getattr(usage, "prompt_tokens", None)
                        u.candidate_tokens = getattr(usage, "completion_tokens", None)
                        u.cost = response_cost(usage)
                    u.response_text = "".join(collected)

                return await self._finalize_turn(
                    "".join(collected), finish_reason,
                    getattr(usage, "completion_tokens", 0) or 0,
                    speaker, round_num,
                )

            with track_llm(self.db, model_name, "debate_research") as u:
                u.response = response = await complete(
                    model=model_name, prompt=prompt, **common
                )
            return await self._finalize_turn(
                response.text, response.finish_reason,
                getattr(response.usage, "completion_tokens", 0) or 0,
                speaker, round_num,
            )
        except Exception as e:
            log.error(f"Debate call failed: {e}")
            return f"Error calling model: {e}"

    async def bull_researcher_node(self, state: AdvisoryState) -> dict:
        round_num = state["debate_round_count"] + 1
        await self._update_progress(f"🔄 Round {round_num}: Bullish Researcher speaking...")

        ticker = state['ticker']
        context = self._build_debate_context(state)

        history = state.get("debate_history", [])

        if not history:
            prompt = (
                f"{context}\n\n"
                f"Write your initial Bull Case for {ticker} over the next 1-3 months. "
                f"Structure your argument: (a) Key catalysts from the news with specific dates/figures, "
                f"(b) Fundamental or technical support, (c) Estimated upside magnitude with reasoning, "
                f"(d) One assumption your thesis depends on that could be wrong."
            )
        else:
            debate_log = "\n\n".join(history)
            prompt = (
                f"{context}\n\n"
                f"Debate History so far:\n{debate_log}\n\n"
                f"Write your rebuttal defending the Bull Case for {ticker}. "
                f"Address the Bear's specific criticisms directly. If the Bear raised a valid point "
                f"you cannot refute, concede it and explain why your thesis still holds despite it."
            )

        result = await self._call_researcher(prompt, speaker="bull", round_num=round_num, system_message=_BULL_SYSTEM)
        history.append(f"Bull: {result}")

        return {"debate_history": history}

    async def bear_researcher_node(self, state: AdvisoryState) -> dict:
        round_num = state["debate_round_count"] + 1
        await self._update_progress(f"🔄 Round {round_num}: Bearish Researcher attacking...")

        ticker = state['ticker']
        context = self._build_debate_context(state)

        history = state.get("debate_history", [])
        debate_log = "\n\n".join(history)

        prompt = (
            f"{context}\n\n"
            f"Debate History so far:\n{debate_log}\n\n"
            f"Write your Bear Case attacking the Bull's thesis for {ticker}. "
            f"Structure your argument: (a) Specific risks or catalysts the Bull ignored or downplayed, "
            f"(b) Why those risks could materialize (cite precedent or context), "
            f"(c) Estimated downside magnitude with reasoning, "
            f"(d) One part of the Bull case you concede is genuinely strong."
        )

        result = await self._call_researcher(prompt, speaker="bear", round_num=round_num, system_message=_BEAR_SYSTEM)
        history.append(f"Bear: {result}")

        return {
            "debate_history": history,
            "debate_round_count": state["debate_round_count"] + 1
        }

    def should_continue_debate(self, state: AdvisoryState) -> str:
        """Conditional routing: skip round 2 if Bull and Bear already agree on direction."""
        if state["debate_round_count"] >= 2:
            return "trader_risk_manager"

        # After round 1, check if there's genuine disagreement worth a second round
        if state["debate_round_count"] >= 1 and self._debate_has_consensus(state):
            log.info("debate.consensus_detected", ticker=state.get("ticker"), rounds=state["debate_round_count"])
            return "trader_risk_manager"

        return "bull_researcher"

    @staticmethod
    def _debate_has_consensus(state: AdvisoryState) -> bool:
        """
        Lightweight heuristic: check if Bull and Bear agree on directional sentiment.
        If both are bullish or both are bearish, there's no real debate — skip round 2.
        Saves the Bull rebuttal and the second Bear turn, both at xhigh reasoning.
        """
        history = state.get("debate_history", [])
        if len(history) < 2:
            return False

        # Get the last Bull and Bear statements
        bull_text = ""
        bear_text = ""
        for entry in history:
            if entry.startswith("Bull:"):
                bull_text = entry.lower()
            elif entry.startswith("Bear:"):
                bear_text = entry.lower()

        if not bull_text or not bear_text:
            return False

        bullish_keywords = [
            "bullish", "upside", "growth", "catalyst", "buy", "long", "outperform",
            "beat", "strong", "positive", "opportunity", "momentum", "rally"
        ]
        bearish_keywords = [
            "bearish", "downside", "risk", "headwind", "sell", "short", "underperform",
            "decline", "weak", "negative", "concern", "overvalued", "correction", "crash"
        ]

        def count_keywords(text, keywords):
            return sum(1 for kw in keywords if kw in text)

        def lean(text: str) -> int:
            """+1 leans bullish, -1 leans bearish, 0 is mixed or silent."""
            bullish = count_keywords(text, bullish_keywords)
            bearish = count_keywords(text, bearish_keywords)
            return (bullish > bearish) - (bullish < bearish)

        # Each side's lean on one shared scale. Two booleans cannot express
        # this: "Bull is bullish" and "Bear is bearish" answer different
        # questions, so both-bullish and both-bearish each read as one True and
        # one False, indistinguishable from a half-hearted Bear.
        bull_lean = lean(bull_text)
        bear_lean = lean(bear_text)

        # Consensus = both lean the same way (both bullish or both bearish).
        # Disagreement (the expected case, or the roles swapped) continues the
        # debate, and so does a mixed or weak signal on either side.
        return bull_lean != 0 and bull_lean == bear_lean

    async def trader_risk_manager_node(self, state: AdvisoryState) -> dict:
        await self._update_progress("✅ Trader/Risk Manager finalizing trade plan...")
        if not is_llm_configured() or not settings.model_trader:
            msg = "No model configured for the trader — set MODEL_TRADER."
            if self.debate_chunk_callback:
                await self.debate_chunk_callback("trader", 3, msg)
            return {
                "final_advisory": msg,
                "executive_summary": msg,
                "trader_direction": "",
                "trader_conviction": "",
            }

        ticker = state['ticker']
        debate_log = "\n\n".join(state.get("debate_history", []))
        lessons_text = self._format_lessons(state.get('past_lessons', {}))

        prompt = (
            f"Ticker: {ticker}\n\n"
            f"=== FULL DEBATE HISTORY ===\n{debate_log}\n\n"
            f"=== PAST LESSONS ===\n{lessons_text}\n\n"
            f"SYNTHESIS INSTRUCTIONS:\n"
            f"1. Weigh the Bull and Bear arguments. Which side has stronger evidence from the news context?\n"
            f"2. Factor in any specific catalysts, dates (IPOs, earnings, product launches), and figures discussed.\n"
            f"3. Consider the past lessons — do they suggest overconfidence, a blind spot, or a confirmed pattern?\n"
            f"4. Make a clear recommendation: BUY, SELL, or HOLD with conviction level (Low/Medium/High).\n"
            f"5. State your time horizon (days/weeks/months) and the #1 risk that would invalidate your call.\n"
            f"6. If relevant, cite historical precedent for similar setups.\n"
            f"7. Format the full advisory in clean Markdown with ### section headers. No emojis.\n\n"
            f"Return your answer in the four fields of the required response "
            f"schema: `direction` (BUY/SELL/HOLD) and `conviction` "
            f"(Low/Medium/High) carry the call itself and are rendered as the "
            f"headline, so they must agree with the prose in `executive_summary` "
            f"and `full_advisory`."
        )

        try:
            model_name = settings.model_trader

            with track_llm(self.db, model_name, "trader_advisory") as u:
                u.response = response = await complete(
                    model=model_name,
                    prompt=prompt,
                    system=_TRADER_SYSTEM_MESSAGE,
                    schema=TraderAdvisory,
                    reasoning="high",
                )

            advisory: Optional[TraderAdvisory] = None
            if isinstance(response.parsed, TraderAdvisory):
                advisory = response.parsed
            else:
                # Schema-constrained decoding did not populate `.parsed` — parse
                # the text ourselves, tolerating unescaped control characters.
                try:
                    advisory = parse_structured(response.text, TraderAdvisory)
                except Exception as parse_e:
                    log.error("agents.trader_parse_failed", error=str(parse_e))

            if advisory is not None:
                final_advisory = advisory.full_advisory
                executive_summary = advisory.executive_summary
                direction = advisory.direction
                conviction = advisory.conviction
            else:
                # Never surface a raw JSON blob to the UI: pull the prose out if
                # we can, and only fall back to the raw text if it is not JSON.
                salvaged = salvage_json_field(response.text, "full_advisory")
                final_advisory = salvaged or (
                    "Advisory generated but could not be formatted for display."
                    if response.text.lstrip().startswith("{")
                    else response.text
                )
                executive_summary = (
                    salvage_json_field(response.text, "executive_summary")
                    or "No executive summary available."
                )
                # Declared ahead of the prose in the schema, so these are the
                # fields most likely to have survived a truncated response.
                direction = salvage_json_field(response.text, "direction") or ""
                conviction = salvage_json_field(response.text, "conviction") or ""

            await self._stream_trader(final_advisory)
            return {
                "final_advisory": final_advisory,
                "executive_summary": executive_summary,
                "trader_direction": direction,
                "trader_conviction": conviction,
            }

        except Exception as e:
            log.error(f"Trader/Risk Manager Gemini call failed: {e}")
            final_advisory = f"Error generating final advisory: {e}"
            await self._stream_trader(final_advisory)
            return {
                "final_advisory": final_advisory,
                "executive_summary": "Error generating advisory.",
                "trader_direction": "",
                "trader_conviction": "",
            }

    async def _stream_trader(self, text: str) -> None:
        """Replay the finished advisory to the SSE client word by word."""
        if not self.debate_chunk_callback:
            return
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            await self.debate_chunk_callback("trader", 3, chunk)
            await asyncio.sleep(0.01)

    def build_graph(self):
        builder = StateGraph(AdvisoryState)

        builder.add_node("aggregate_data", self.aggregate_data_node)
        builder.add_node("bull_researcher", self.bull_researcher_node)
        builder.add_node("bear_researcher", self.bear_researcher_node)
        builder.add_node("trader_risk_manager", self.trader_risk_manager_node)

        # Sequence: START -> Aggregate -> Bull -> Bear
        builder.add_edge(START, "aggregate_data")
        builder.add_edge("aggregate_data", "bull_researcher")
        builder.add_edge("bull_researcher", "bear_researcher")

        # Conditional Edge after Bear:
        builder.add_conditional_edges(
            "bear_researcher",
            self.should_continue_debate,
            {
                "bull_researcher": "bull_researcher",
                "trader_risk_manager": "trader_risk_manager"
            }
        )

        builder.add_edge("trader_risk_manager", END)

        return builder.compile()

    async def run(self, ticker: str, ml_prediction: dict, past_lessons: dict, news_context: str) -> dict:
        graph = self.build_graph()
        initial_state = {
            "ticker": ticker,
            "ml_prediction": ml_prediction,
            "past_lessons": past_lessons,
            "news_context": news_context,
            "debate_history": [],
            "debate_round_count": 0
        }

        max_retries = 2
        for attempt in range(max_retries):
            try:
                final_state = await graph.ainvoke(initial_state)
                return final_state
            except Exception as e:
                log.warning(f"Graph execution attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    return {"final_advisory": f"Failed to generate multi-agent advice after {max_retries} attempts."}
                await asyncio.sleep(2)
