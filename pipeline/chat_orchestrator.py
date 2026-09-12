import json
import asyncio
from typing import Any, AsyncIterator, TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from config.llm import (
    complete,
    is_llm_configured,
    response_cost,
    stream_complete,
    strip_code_fence,
)
from config.settings import settings
from config.logging_config import get_logger
from config.usage import track_llm
from data.database import Database
from pipeline.embedder import Embedder
from pipeline.grounded_answer import HONESTY_SENTENCE, GradeVerdict, grade_context
from pipeline.web_search import enrich_chat_context
import numpy as np

log = get_logger(__name__)

# Rule 8 of the analyst prompt, added only when the grader found nothing and the
# web found nothing either. Without it, "I have no dated catalyst" is expressed
# by the model as a confident paragraph about general market conditions, which
# is indistinguishable from an answer.
HONESTY_RULE = (
    f"HONESTY REQUIREMENT: the database context was graded insufficient for "
    f"this question and no live web results were found. Open your answer with "
    f"exactly this sentence: \"{HONESTY_SENTENCE}what follows is general "
    f"context, not an explanation.\" Then give whatever general context is "
    f"genuinely useful. Do NOT invent a cause, a catalyst, or a date."
)

# ── Shared prompt builder (used by REST SSE, WS, and graph nodes) ──────

def build_chat_prompt(query: str, context: str = "", *,
                      honesty_required: bool = False) -> str:
    """Build a consistent analyst prompt with optional RAG context.

    `honesty_required` is set when the grader judged the retrieved context
    insufficient and the web search that followed produced nothing. It is
    checked against the context itself as well: a prompt carrying a LIVE WEB
    SEARCH RESULTS block has grounding by definition, so the rule would be a
    lie even if a caller asked for it.
    """
    persona = (
        "You are a professional, precise, and highly analytical Wall Street analyst. "
        "Your methodology: ground claims in data, quantify impact when possible, "
        "distinguish context-sourced facts from general-knowledge inference."
    )

    formatting = (
        "CRITICAL MARKDOWN FORMATTING & LAYOUT RULES (CHATGPT STYLE):\n"
        "1. USE STANDARD MIXED-CASE / SENTENCE CASE ONLY. NEVER output ALL CAPS or block uppercase.\n"
        "2. MANDATORY HEADINGS: Begin every section header with '### ' and bold title on a NEW LINE preceded by a double blank line (\\n\\n### **Section Title**\\n\\n).\n"
        "3. NEVER attach headings to body text on the same line. ALWAYS put a blank line (\\n\\n) after every heading before the paragraph begins.\n"
        "4. MANDATORY BULLET SPACING: Place every sub-bullet point on a BRAND NEW LINE starting with '- **Sub-topic:** explanation'.\n"
        "5. SEPARATE PARAGRAPHS: Insert a blank line (\\n\\n) between every paragraph and list.\n\n"
        "EXACT TEMPLATE TO EMULATE:\n\n"
        "### **Executive Summary**\n\n"
        "A 2-3 sentence high-level overview answering the core user query directly.\n\n"
        "### 1. **Primary Volatility Driver**\n\n"
        "Detailed narrative paragraph explaining the primary factor with numbers, metrics, and quotes.\n\n"
        "- **Key Catalyst:** Specific event or quote from context.\n"
        "- **Market Impact:** Quantified percentage move or price effect.\n\n"
        "### 2. **Secondary Volatility Driver**\n\n"
        "Detailed narrative paragraph explaining the secondary factor.\n\n"
        "- **Key Sub-factor:** Specific detail.\n"
        "- **Trading Dynamics:** Market structure details.\n\n"
        "### **Analyst Outlook & Conclusion**\n\n"
        "Forward-looking summary statement with key risks to watch.\n"
    )

    be_honest = honesty_required and "LIVE WEB SEARCH RESULTS" not in (context or "")

    if context:
        return (
            f"{persona}\n\n"
            f"Answer the user's query using the recent news context below as your primary source.\n\n"
            f"=== DATABASE CONTEXT (ordered by relevance) ===\n{context}\n\n"
            f"ANALYTICAL RULES:\n"
            f"1. Lead with the most impactful information from the context.\n"
            f"2. Pull verbatim quotes where they strengthen your answer: '[DB Context: <date>] \"exact quote\"'\n"
            f"3. If two sources in the context contradict, present both sides and explain which has stronger evidence.\n"
            f"4. For claims not supported by the context, tag them: '[General Knowledge] your claim here'\n"
            f"5. If the most recent context item is over 4 hours old, warn: 'Note: latest data may be stale — prices/conditions may have changed.'\n"
            f"6. When asked about specific tickers, estimate magnitude where possible (e.g., 'could move 3-5%').\n"
            f"7. News context may include a 'LIVE WEB SEARCH RESULTS' section — treat this as real-time\n"
            f"   data potentially more current than in-house articles. Cite sources explicitly.\n"
            f"{f'8. {HONESTY_RULE}' if be_honest else ''}\n"
            f"{formatting}\n\n"
            f"User Query: {query}"
        )
    else:
        return (
            f"{persona}\n\n"
            f"No recent news is available in the database for this topic. "
            f"Answer using your general knowledge, but clearly state this limitation.\n\n"
            f"ANALYTICAL RULES:\n"
            f"1. State upfront: 'I don't have current data on this — this is based on general knowledge.'\n"
            f"2. When possible, cite well-known historical precedents or market patterns.\n"
            f"3. Quantify uncertainty explicitly.\n"
            f"{f'4. {HONESTY_RULE}' if be_honest else ''}\n"
            f"{formatting}\n\n"
            f"User Query: {query}"
        )


class ChatState(TypedDict, total=False):
    query: str
    context: str
    routing_decision: str
    final_answer: str
    # rag_node has always returned top_articles; nothing declared it, so
    # graph.ainvoke dropped the key and only the hand-rolled REST path ever saw
    # the citations. Declared now, which is what makes the graph and the
    # streaming path emit the same sources.
    top_articles: list[dict]
    grade: Optional[GradeVerdict]
    web_sources: list[dict]
    grounded_by: str

def _needs_honesty(state: ChatState) -> bool:
    """
    Whether the answer has to open by admitting it found nothing.

    True only when the grounding is actually absent: `grounded_by` is "db" while
    relevant articles exist, "web" once a search lands anything, and falls back
    to "none" when the grader rejected the context and the search came up empty.
    """
    return (state.get("grounded_by") or "none") == "none"


# Module-level embedder cache — shared across all ChatOrchestrator instances
# to avoid re-initializing the Gemini embedding client on every chat message.
_shared_embedder: Optional[Embedder] = None
_embedder_lock = asyncio.Lock()


class ChatOrchestrator:
    def __init__(self, db: Database, progress_callback=None,
                 embedder: Optional[Embedder] = None, research_callback=None):
        self.db = db
        self.progress_callback = progress_callback
        # web_search_node is a graph node, so it only receives the state — the
        # per-source streaming callback has to reach it off the instance. One
        # orchestrator serves one question, so there is nothing to race.
        self.research_callback = research_callback
        self._embedder = embedder  # Allow injection; falls back to shared singleton below

    async def _get_embedder(self) -> Embedder:
        """Return a ready-to-use embedder, reusing a module-level singleton."""
        global _shared_embedder
        if self._embedder is not None:
            return self._embedder
        if _shared_embedder is not None and _shared_embedder._initialized:
            return _shared_embedder
        async with _embedder_lock:
            if _shared_embedder is None:
                _shared_embedder = Embedder()
            if not _shared_embedder._initialized:
                await _shared_embedder.initialize()
            return _shared_embedder

    async def _update_progress(self, msg: str):
        if self.progress_callback:
            await self.progress_callback(msg)

    async def router_node(self, state: ChatState) -> dict:
        await self._update_progress("🧠 Classifying query complexity...")
        if not is_llm_configured():
            log.error("chat.router_unconfigured")
            return {"routing_decision": "shallow"}

        prompt = (
            "You are a highly accurate routing agent for a financial AI assistant.\n"
            "Your job is to classify the user's intent into one of two categories: 'shallow' or 'complex'.\n\n"
            "1. 'shallow': Use this for simple factual lookups, quick summaries, direct price inquiries, or general chatter. "
            "These queries do NOT require deep reasoning or connecting multiple data points.\n"
            "   Examples of 'shallow':\n"
            "   - 'What is the current price of AAPL?'\n"
            "   - 'Summarize the latest news on TSLA.'\n"
            "   - 'Is there any news about NVIDIA earnings?'\n"
            "   - 'Hello, how are you?'\n\n"
            "2. 'complex': Use this for deep financial analysis, multi-part questions, comparative queries, fundamental/technical synthesis, "
            "or anything requiring the AI to deeply evaluate risk, predict trends, or analyze market impact.\n"
            "   Examples of 'complex':\n"
            "   - 'Given the recent Fed rate hike, how will tech stocks like MSFT and GOOG perform over the next quarter?'\n"
            "   - 'Provide a deep fundamental analysis of PLTR considering its forward P/E and recent government contracts.'\n"
            "   - 'Why did the market drop yesterday, and what does it mean for my portfolio?'\n"
            "   - 'Compare the technical indicators of AMD and INTC.'\n\n"
            f"User Query: {state['query']}\n\n"
            "Return ONLY a valid JSON object with a single key 'decision' whose value is either 'shallow' or 'complex'."
        )
        
        try:
            model_name = settings.model_router

            with track_llm(self.db, model_name, "chat_router",
                           prompt_text=prompt, store_text=True) as u:
                u.response = resp = await complete(
                    model=model_name, prompt=prompt, json_mode=True,
                )
            result_text = resp.text.strip()
            try:
                data = json.loads(strip_code_fence(result_text))
                decision = data.get("decision", "shallow").lower()
                if decision not in ["shallow", "complex"]:
                    decision = "shallow"
            except json.JSONDecodeError:
                decision = "shallow"
                
            return {"routing_decision": decision}
        except Exception as e:
            log.error(f"Router node failed: {e}")
            return {"routing_decision": "shallow"}

    async def rag_node(self, state: ChatState) -> dict:
        await self._update_progress("🔍 Retrieving and ranking relevant news context...")

        embedder = await self._get_embedder()
        query = state['query']
        query_vec = await embedder.get_embedding(query)
        if query_vec is None:
            return {"context": ""}

        loop = asyncio.get_running_loop()
        
        def vector_search_and_rank():
            if getattr(self.db, 'has_sqlite_vec', False):
                query_bytes = query_vec.astype(np.float32).tobytes()
                with self.db.connection() as conn:
                    # Fetch top 20 by distance, then rerank by importance_score
                    rows = conn.execute(
                        """
                        SELECT headline, summary, importance_score, vec_distance_cosine(embedding, ?) as distance 
                        FROM articles 
                        WHERE embedding IS NOT NULL AND (event_type IS NULL OR event_type != 'noise') 
                        ORDER BY distance LIMIT 20
                        """,
                        (query_bytes,)
                    ).fetchall()
                    
                    results = [dict(row) for row in rows]
                    # Rerank by importance_score (None becomes 0.0)
                    results.sort(key=lambda x: x['importance_score'] or 0.0, reverse=True)
                    return results[:5]
            else:
                # Fallback to numpy if sqlite_vec is not available
                def calculate_similarity(v1, v2):
                    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                embeddings = self.db.get_all_embeddings(exclude_noise=True)
                scored = []
                for article_id, vec in embeddings:
                    sim = calculate_similarity(query_vec, vec)
                    scored.append((sim, article_id))
                scored.sort(key=lambda x: x[0], reverse=True)
                top_20_ids = [x[1] for x in scored[:20]]
                
                if not top_20_ids:
                    return []
                    
                with self.db.connection() as conn:
                    placeholders = ','.join('?' for _ in top_20_ids)
                    rows = conn.execute(
                        f"SELECT id, headline, summary, importance_score FROM articles WHERE id IN ({placeholders})",
                        top_20_ids
                    ).fetchall()
                    
                    results = [dict(row) for row in rows]
                    results.sort(key=lambda x: x['importance_score'] or 0.0, reverse=True)
                    return results[:5]

        top_articles = await loop.run_in_executor(None, vector_search_and_rank)
        
        context_texts = []
        for r in top_articles:
            text_content = r['summary'] if r['summary'] else "(No text available)"
            context_texts.append(f"Title: {r['headline']}\nContent: {text_content[:2000]}")
            
        context_str = "\n\n---\n\n".join(context_texts) if context_texts else ""
        return {
            "context": context_str,
            "top_articles": top_articles,
            "grounded_by": "db" if context_str else "none",
        }

    async def grade_node(self, state: ChatState) -> dict:
        """
        Does the retrieved context actually answer this question?

        A `complex` route skips the call entirely: that lane searches the web
        unconditionally, so grading it would be paying a model to produce a
        verdict nothing acts on.
        """
        await self._update_progress("🧪 Grading database context...")

        if state.get("routing_decision") == "complex":
            return {"grade": GradeVerdict(
                sufficient=False, specificity="none",
                reason="complex always searches",
            )}

        grade = await grade_context(
            self.db, state["query"], state.get("context", ""), purpose="chat"
        )
        # An insufficient grade demotes the grounding even though DB context
        # exists: having articles about the ticker is not the same as having the
        # one that answers the question, and the honesty rule keys off this.
        patch: dict = {"grade": grade}
        if not grade.sufficient:
            patch["grounded_by"] = "none"
        return patch

    async def web_search_node(self, state: ChatState) -> dict:
        """Top the context up with live web results."""
        await self._update_progress("🌐 Searching the web for latest information...")

        enriched, web_sources = await enrich_chat_context(
            query=state["query"],
            db_context=state.get("context", ""),
            max_results=settings.web_search_max_results,
            research_callback=self.research_callback,
        )
        patch: dict = {"context": enriched, "web_sources": web_sources}
        if web_sources:
            patch["grounded_by"] = "web"
        return patch

    def decide_after_grade(self, state: ChatState) -> str:
        """
        Pure: "web_search" or "answer".

        Kept separate from the graph edge below so the decision is testable
        without compiling a graph, and so the streaming path and the graph
        cannot disagree about when a search happens.
        """
        if state.get("routing_decision") == "complex":
            return "web_search"
        grade = state.get("grade")
        if grade is None or not getattr(grade, "sufficient", False):
            return "web_search"
        return "answer"

    def route_after_grade(self, state: ChatState) -> str:
        """Graph edge: the name of the next node to run."""
        if self.decide_after_grade(state) == "web_search":
            return "web_search"
        return state.get("routing_decision") or "shallow"

    def should_continue(self, state: ChatState) -> str:
        return state.get("routing_decision") or "shallow"

    async def shallow_agent_node(self, state: ChatState) -> dict:
        await self._update_progress("⚡ Answering via Shallow model...")
        return await self._generate_answer(
            state,
            model_name=settings.model_chat_shallow,
            reasoning=None,  # No thinking
            operation_name="chat_shallow"
        )

    async def complex_agent_node(self, state: ChatState) -> dict:
        # The inline enrich_chat_context call that used to live here is now
        # web_search_node, reached by an edge. It had to move: a shallow query
        # whose context was graded insufficient needs the same search, and
        # hiding it inside the complex agent made that impossible to express.
        await self._update_progress("🤔 Answering via Complex model (Medium Thinking)...")
        return await self._generate_answer(
            state,
            model_name=settings.model_chat_complex,
            reasoning="medium",
            operation_name="chat_complex"
        )

    async def _generate_answer(self, state: ChatState, model_name: str,
                               reasoning: Optional[str], operation_name: str) -> dict:
        """
        The shallow/complex split is now one argument: `reasoning` is None for
        the fast lane and "medium" for the thinking lane, where it used to be a
        provider-specific ThinkingConfig object.
        """
        if not is_llm_configured():
            return {"final_answer": "❌ LLM not configured."}

        prompt = build_chat_prompt(
            state['query'], state.get('context', ''),
            honesty_required=_needs_honesty(state),
        )

        try:
            with track_llm(self.db, model_name, operation_name,
                           prompt_text=prompt, store_text=True) as u:
                u.response = resp = await complete(
                    model=model_name, prompt=prompt, reasoning=reasoning,
                )
            return {"final_answer": resp.text.strip()}
        except Exception as e:
            log.error(f"{operation_name} failed: {e}")
            return {"final_answer": f"❌ Failed to generate answer: {str(e)}"}

    def build_graph(self):
        """
        START -> router -> rag -> grade -> (web_search) -> shallow | complex.

        Kept alongside `iter_events`, which is what actually serves requests.
        The graph is the structural statement of the routing — compiled in a
        test, and the thing to read when asking "when does this search?" — and
        `decide_after_grade` is shared by both, so they cannot disagree.
        """
        builder = StateGraph(ChatState)

        builder.add_node("router", self.router_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("grade", self.grade_node)
        builder.add_node("web_search", self.web_search_node)
        builder.add_node("shallow", self.shallow_agent_node)
        builder.add_node("complex", self.complex_agent_node)

        builder.add_edge(START, "router")
        builder.add_edge("router", "rag")
        builder.add_edge("rag", "grade")

        builder.add_conditional_edges(
            "grade",
            self.route_after_grade,
            {
                "web_search": "web_search",
                "shallow": "shallow",
                "complex": "complex",
            }
        )
        builder.add_conditional_edges(
            "web_search",
            self.should_continue,
            {
                "shallow": "shallow",
                "complex": "complex"
            }
        )

        builder.add_edge("shallow", END)
        builder.add_edge("complex", END)

        return builder.compile()

    async def iter_events(
        self, query: str, *, research_callback=None
    ) -> AsyncIterator[tuple[str, Any]]:
        """
        Run one chat turn, yielding (event, payload) as it goes.

        The single implementation of the chat turn. The REST SSE endpoint used to
        re-implement this sequence by hand — router, rag, web search, stream —
        which is why the graph's routing and the endpoint's routing drifted
        apart: only the endpoint ever searched the web, and only on `complex`.
        Telegram and the legacy WebSocket consume the same generator; `run()`
        joins the tokens.

        Events, all dicts except `error` which is a plain string:
          step/classification, step/retrieval, step/grading, step/web_search,
          research_start | research_source | research_complete (passed through
          from the search provider), sources, token, error, done.
        """
        self.research_callback = research_callback or self.research_callback

        state: ChatState = {
            "query": query, "context": "", "routing_decision": "shallow",
            "final_answer": "", "top_articles": [], "grade": None,
            "web_sources": [], "grounded_by": "none",
        }

        # ── Route ────────────────────────────────────────────────────────
        state.update(await self.router_node(state))
        decision = state.get("routing_decision") or "shallow"
        yield "step", {
            "step": "classification", "intent": decision,
            "reasoning": f"Routed to {decision} agent",
        }

        # ── Retrieve ─────────────────────────────────────────────────────
        state.update(await self.rag_node(state))
        yield "step", {
            "step": "retrieval",
            "context": state.get("context", ""),
            # The consumer offsets the web-search progress counter by this, so
            # the side panel counts DB citations alongside web ones.
            "count": len(state.get("top_articles") or []),
        }

        # ── Grade ────────────────────────────────────────────────────────
        state.update(await self.grade_node(state))
        grade = state.get("grade")
        yield "step", {
            "step": "grading",
            "sufficient": bool(grade and grade.sufficient),
            "specificity": grade.specificity if grade else "none",
            "reason": grade.reason if grade else "",
        }

        # ── Search, if the evidence was thin ─────────────────────────────
        if self.decide_after_grade(state) == "web_search":
            yield "step", {
                "step": "web_search", "intent": "searching",
                "reasoning": "Running live web search for latest information...",
            }
            before = state.get("context", "")
            state.update(await self.web_search_node(state))
            found = len(state.get("web_sources") or [])
            if state.get("context", "") != before:
                yield "step", {
                    "step": "web_search", "intent": "merged",
                    "reasoning": f"Merged {found} web search source(s) into context",
                }

        # ── Citations ────────────────────────────────────────────────────
        articles = list(state.get("top_articles") or []) + list(
            state.get("web_sources") or []
        )
        if articles:
            yield "sources", {"articles": articles}

        # ── Answer ───────────────────────────────────────────────────────
        model = (
            settings.model_chat_shallow if decision == "shallow"
            else settings.model_chat_complex
        )
        if not is_llm_configured() or not model:
            yield "error", "❌ LLM not configured."
            return

        prompt = build_chat_prompt(
            query, state.get("context", ""),
            honesty_required=_needs_honesty(state),
        )

        collected: list[str] = []
        try:
            # Streamed responses carry usage on a trailing chunk, so keep the
            # last one seen and log it after the stream drains.
            with track_llm(self.db, model, "chat_stream") as usage:
                async for chunk in stream_complete(
                    model=model,
                    prompt=prompt,
                    reasoning=None if decision == "shallow" else "medium",
                ):
                    if chunk.text:
                        collected.append(chunk.text)
                        yield "token", {"text": chunk.text}
                    if chunk.usage is not None:
                        usage.prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                        usage.candidate_tokens = getattr(chunk.usage, "completion_tokens", None)
                        usage.cost = response_cost(chunk.usage)
                usage.response_text = "".join(collected)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("chat.stream_failed", error=str(e))
            yield "error", f"❌ Failed to generate answer: {e}"
            return

        yield "done", {
            "answer": "".join(collected),
            "routing": decision,
            "grounded_by": state.get("grounded_by", "none"),
        }

    async def run(self, query: str) -> str:
        """Collect one full answer as a string, for Telegram and tests."""
        collected: list[str] = []
        error: Optional[str] = None
        try:
            async for event, data in self.iter_events(query):
                if event == "token":
                    collected.append(str(data.get("text", "")))
                elif event == "error":
                    error = data if isinstance(data, str) else str(data)
        except Exception as e:
            log.error(f"ChatOrchestrator run failed: {e}")
            return f"❌ Orchestrator failed: {str(e)}"

        answer = "".join(collected).strip()
        if answer:
            return answer
        return error or "Error: No answer generated."
