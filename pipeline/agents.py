import json
import asyncio
from typing import TypedDict, Optional, Dict, Callable, Awaitable
import yfinance as yf
from langgraph.graph import StateGraph, START, END

from config.llm import get_client, get_deepseek_client
from config.settings import settings
from config.logging_config import get_logger

log = get_logger(__name__)

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
    "a minor confirmatory signal only. For every recommendation, explicitly state "
    "your conviction level, time horizon, and the key risk that would invalidate "
    "your thesis. Format output as clean Markdown with ### headers."
)

# Shared debate rules — extracted to avoid duplication
_COMMON_RULES = (
    "CRITICAL RULES:\n"
    "1. Be extremely direct, concise, and fact-driven. No fluff, rhetoric, or sassy language.\n"
    "2. Ground your arguments primarily in the RECENT news context. You MUST explicitly call out and analyze any specific upcoming catalysts, exact dates (e.g., IPOs, earnings), and figures mentioned in the news.\n"
    "3. If the opposing side makes a valid point, intelligently acknowledge it. Meaningful debate requires conceding undeniable facts.\n"
    "4. News is presented in two sections — 'IN-HOUSE NEWS' (curated, classified by importance) and 'LIVE WEB SEARCH RESULTS' (real-time web data). Treat both as current and factual, but prioritize in-house news when available as it has been through classification."
)

class AdvisoryState(TypedDict):
    ticker: str
    ml_prediction: dict
    past_lessons: dict  # Structured: {ticker_lessons: [...], sector_lessons: [...], market_lessons: [...]}
    news_context: str

    fundamentals_report: str
    technical_report: str

    debate_history: list[str]
    debate_round_count: int

    final_advisory: str

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
        ml = state.get("ml_prediction", {})
        direction = ml.get("predicted_direction", "UNKNOWN")
        conf = ml.get("confidence", 0.0)

        if direction == "UNKNOWN":
            technicals = "No trained ML model exists yet for this ticker. Running debate using news, fundamentals, and general knowledge."
        else:
            try:
                feats = json.loads(ml.get("feature_snapshot", "{}"))
                rsi = feats.get("rsi_14", "N/A")
                vol = feats.get("volatility", "N/A")
                llm_acc = feats.get("llm_historical_accuracy", "N/A")
                technicals = f"Quantitative ML Predicts {direction} with {int(conf*100)}% confidence.\nFeatures: RSI={rsi}, Volatility={vol}, Hist LLM Accuracy={llm_acc}."
            except Exception:
                technicals = f"Quantitative ML Predicts {direction} with {int(conf*100)}% confidence."

        return {
            "fundamentals_report": fundamentals,
            "technical_report": technicals,
            "debate_round_count": 0,
            "debate_history": []
        }

    async def _call_deepseek(self, prompt: str, speaker: Optional[str] = None, round_num: Optional[int] = None, system_message: Optional[str] = None) -> str:
        client = get_deepseek_client()
        if not client:
            return "DeepSeek API key not configured."
        try:
            model_name = getattr(settings, "deepseek_model_reasoner", "deepseek-v4-pro")
            sys_msg = system_message or "You are a top-tier financial researcher. Reason through the facts before answering."
            if self.debate_chunk_callback and speaker and round_num is not None:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}},
                    stream=True
                )
                collected_chunks = []
                async for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            collected_chunks.append(delta)
                            await self.debate_chunk_callback(speaker, round_num, delta)
                full_content = "".join(collected_chunks)
                self.db.log_llm_usage(model_name=model_name, operation="debate_research", prompt_tokens=0, candidate_tokens=0)
                return full_content
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled"}}
                )
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                self.db.log_llm_usage(model_name=model_name, operation="debate_research", prompt_tokens=prompt_tokens, candidate_tokens=completion_tokens)
                return response.choices[0].message.content
        except Exception as e:
            log.error(f"DeepSeek call failed: {e}")
            return f"Error calling DeepSeek: {e}"

    async def bull_researcher_node(self, state: AdvisoryState) -> dict:
        round_num = state["debate_round_count"] + 1
        await self._update_progress(f"🔄 Round {round_num}: Bullish Researcher speaking...")

        ticker = state['ticker']
        context = (
            f"Ticker: {ticker}\n"
            f"Fundamentals: {state.get('fundamentals_report')}\n"
            f"Technicals (ML Base): {state.get('technical_report')}\n"
            f"News Context (Sentiment/Urgency): {state.get('news_context')}\n"
            f"Past Lessons (relevance-ranked):\n"
            f"{self._format_lessons(state.get('past_lessons', {}))}"
        )

        history = state.get("debate_history", [])

        if not history:
            prompt = (
                f"{context}\n\n"
                f"{_COMMON_RULES}\n\n"
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
                f"{_COMMON_RULES}\n\n"
                f"Write your rebuttal defending the Bull Case for {ticker}. "
                f"Address the Bear's specific criticisms directly. If the Bear raised a valid point "
                f"you cannot refute, concede it and explain why your thesis still holds despite it."
            )

        result = await self._call_deepseek(prompt, speaker="bull", round_num=round_num, system_message=_BULL_SYSTEM_MESSAGE)
        history.append(f"Bull: {result}")

        return {"debate_history": history}

    async def bear_researcher_node(self, state: AdvisoryState) -> dict:
        round_num = state["debate_round_count"] + 1
        await self._update_progress(f"🔄 Round {round_num}: Bearish Researcher attacking...")

        ticker = state['ticker']
        context = (
            f"Ticker: {ticker}\n"
            f"Fundamentals: {state.get('fundamentals_report')}\n"
            f"Technicals (ML Base): {state.get('technical_report')}\n"
            f"News Context (Sentiment/Urgency): {state.get('news_context')}\n"
            f"Past Lessons (relevance-ranked):\n"
            f"{self._format_lessons(state.get('past_lessons', {}))}"
        )

        history = state.get("debate_history", [])
        debate_log = "\n\n".join(history)

        prompt = (
            f"{context}\n\n"
            f"Debate History so far:\n{debate_log}\n\n"
            f"{_COMMON_RULES}\n\n"
            f"Write your Bear Case attacking the Bull's thesis for {ticker}. "
            f"Structure your argument: (a) Specific risks or catalysts the Bull ignored or downplayed, "
            f"(b) Why those risks could materialize (cite precedent or context), "
            f"(c) Estimated downside magnitude with reasoning, "
            f"(d) One part of the Bull case you concede is genuinely strong."
        )

        result = await self._call_deepseek(prompt, speaker="bear", round_num=round_num, system_message=_BEAR_SYSTEM_MESSAGE)
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
        Saves one full DeepSeek v4-pro reasoning call (~$0.03-0.06 per ticker).
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

        bull_bullish = count_keywords(bull_text, bullish_keywords)
        bull_bearish = count_keywords(bull_text, bearish_keywords)
        bear_bullish = count_keywords(bear_text, bullish_keywords)
        bear_bearish = count_keywords(bear_text, bearish_keywords)

        # Determine each agent's dominant sentiment
        bull_is_bullish = bull_bullish > bull_bearish
        bear_is_bearish = bear_bearish > bear_bullish

        # Consensus = both lean the same way (both bullish or both bearish)
        # Disagreement = Bull is bullish AND Bear is bearish (the expected case)
        if bull_is_bullish and bear_is_bearish:
            return False  # Genuine disagreement — continue debate
        if (not bull_is_bullish and not bear_is_bearish) or (bull_is_bullish == bear_is_bearish):
            return True   # Both lean same way — skip to trader

        # If signals are mixed/weak, default to continuing the debate
        return False

    async def trader_risk_manager_node(self, state: AdvisoryState) -> dict:
        await self._update_progress("✅ Trader/Risk Manager finalizing trade plan...")
        client = get_client()
        if not client:
            msg = "Gemini API key not configured."
            if self.debate_chunk_callback:
                await self.debate_chunk_callback("trader", 3, msg)
            return {"final_advisory": msg}

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
            f"OUTPUT FORMAT — valid JSON with exactly these two keys:\n"
            f'{{"executive_summary": "TLDR: [BUY/SELL/HOLD] — [1-2 sentence actionable reason, no markdown]", '
            f'"full_advisory": "Complete markdown analysis per instructions above"}}'
        )

        try:
            model_name = settings.gemini_model_chat or "gemini-3-flash-preview"
            from google.genai import types
            loop = asyncio.get_running_loop()

            def ask():
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'system_instruction': _TRADER_SYSTEM_MESSAGE,
                        'thinking_config': types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                        'response_mime_type': 'application/json'
                    }
                )

            response = await loop.run_in_executor(None, ask)
            prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            self.db.log_llm_usage(model_name=model_name, operation="trader_advisory", prompt_tokens=prompt_tokens, candidate_tokens=completion_tokens)

            try:
                text = response.text.strip()
                if text.startswith("```json"): text = text[7:]
                if text.startswith("```"): text = text[3:]
                if text.endswith("```"): text = text[:-3]
                data = json.loads(text.strip())

                final_advisory = data.get("full_advisory", response.text)
                executive_summary = data.get("executive_summary", "No executive summary provided.")

                if self.debate_chunk_callback:
                    words = final_advisory.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        await self.debate_chunk_callback("trader", 3, chunk)
                        await asyncio.sleep(0.01)

                return {
                    "final_advisory": final_advisory,
                    "executive_summary": executive_summary
                }
            except Exception as parse_e:
                log.error(f"Failed to parse JSON from Trader: {parse_e}")
                final_advisory = response.text
                if self.debate_chunk_callback:
                    words = final_advisory.split(" ")
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        await self.debate_chunk_callback("trader", 3, chunk)
                        await asyncio.sleep(0.01)
                return {"final_advisory": final_advisory, "executive_summary": "No executive summary available."}

        except Exception as e:
            log.error(f"Trader/Risk Manager Gemini call failed: {e}")
            final_advisory = f"Error generating final advisory: {e}"
            if self.debate_chunk_callback:
                words = final_advisory.split(" ")
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    await self.debate_chunk_callback("trader", 3, chunk)
                    await asyncio.sleep(0.01)
            return {"final_advisory": final_advisory, "executive_summary": "Error generating advisory."}

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
