import json
import asyncio
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from google.genai import types
from config.llm import get_client, DEFAULT_SAFETY_SETTINGS
from config.settings import settings
from config.logging_config import get_logger
from data.database import Database
from pipeline.embedder import GeminiEmbedder
from pipeline.web_search import enrich_chat_context
import numpy as np

log = get_logger(__name__)

# ── Shared prompt builder (used by REST SSE, WS, and graph nodes) ──────

def build_chat_prompt(query: str, context: str = "") -> str:
    """Build a consistent analyst prompt with optional RAG context."""
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
            f"   data potentially more current than in-house articles. Cite sources explicitly.\n\n"
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
            f"3. Quantify uncertainty explicitly.\n\n"
            f"{formatting}\n\n"
            f"User Query: {query}"
        )


class ChatState(TypedDict):
    query: str
    context: str
    routing_decision: str
    final_answer: str

# Module-level embedder cache — shared across all ChatOrchestrator instances
# to avoid re-initializing the Gemini embedding client on every chat message.
_shared_embedder: Optional[GeminiEmbedder] = None
_embedder_lock = asyncio.Lock()


class ChatOrchestrator:
    def __init__(self, db: Database, progress_callback=None, embedder: Optional[GeminiEmbedder] = None):
        self.db = db
        self.progress_callback = progress_callback
        self._embedder = embedder  # Allow injection; falls back to shared singleton below

    async def _get_embedder(self) -> GeminiEmbedder:
        """Return a ready-to-use embedder, reusing a module-level singleton."""
        global _shared_embedder
        if self._embedder is not None:
            return self._embedder
        if _shared_embedder is not None and _shared_embedder._initialized:
            return _shared_embedder
        async with _embedder_lock:
            if _shared_embedder is None:
                _shared_embedder = GeminiEmbedder()
            if not _shared_embedder._initialized:
                await _shared_embedder.initialize()
            return _shared_embedder

    async def _update_progress(self, msg: str):
        if self.progress_callback:
            await self.progress_callback(msg)

    async def router_node(self, state: ChatState) -> dict:
        await self._update_progress("🧠 Classifying query complexity...")
        client = get_client()
        if not client:
            log.error("Gemini client not configured for router.")
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
            model_name = getattr(settings, "gemini_model_router", "gemini-2.5-flash-lite")
            
            loop = asyncio.get_running_loop()
            def ask():
                start_time = time.time()
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'safety_settings': DEFAULT_SAFETY_SETTINGS
                    }
                )
                latency = int((time.time() - start_time) * 1000)
                if resp.usage_metadata:
                    self.db.log_llm_usage(
                        model_name=model_name,
                        operation="chat_router",
                        prompt_tokens=resp.usage_metadata.prompt_token_count,
                        candidate_tokens=resp.usage_metadata.candidates_token_count,
                        latency_ms=latency,
                        is_error=False,
                        prompt_text=prompt,
                        response_text=resp.text
                    )
                return resp.text.strip()
                
            result_text = await loop.run_in_executor(None, ask)
            try:
                if result_text.startswith("```json"): result_text = result_text[7:]
                if result_text.startswith("```"): result_text = result_text[3:]
                if result_text.endswith("```"): result_text = result_text[:-3]
                data = json.loads(result_text.strip())
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
        return {"context": context_str, "top_articles": top_articles}

    def should_continue(self, state: ChatState) -> str:
        return state["routing_decision"]

    async def shallow_agent_node(self, state: ChatState) -> dict:
        await self._update_progress("⚡ Answering via Shallow model...")
        return await self._generate_answer(
            state, 
            model_name=getattr(settings, "gemini_model_chat_shallow", "gemini-3.1-flash-lite"),
            thinking_level=None,  # No thinking
            operation_name="chat_shallow"
        )

    async def complex_agent_node(self, state: ChatState) -> dict:
        await self._update_progress("🌐 Searching the web for latest information...")

        # Enrich DB context with live web search (complex queries only)
        db_context = state.get("context", "")
        enriched_context, _ = await enrich_chat_context(
            query=state["query"],
            db_context=db_context,
            max_results=settings.web_search_max_results,
        )

        await self._update_progress("🤔 Answering via Complex model (Medium Thinking)...")
        # Build enriched state — don't mutate the graph state to avoid side effects
        enriched_state = dict(state)
        enriched_state["context"] = enriched_context
        return await self._generate_answer(
            enriched_state,
            model_name=getattr(settings, "gemini_model_chat_complex", "gemini-3-flash-preview"),
            thinking_level=types.ThinkingLevel.MEDIUM,  # Medium thinking
            operation_name="chat_complex"
        )
        
    async def _generate_answer(self, state: ChatState, model_name: str, thinking_level: Optional[int], operation_name: str) -> dict:
        client = get_client()
        if not client:
            return {"final_answer": "❌ LLM not configured."}
            
        query = state['query']
        context_str = state['context']
        prompt = build_chat_prompt(query, context_str)
            
        config = {
            'safety_settings': DEFAULT_SAFETY_SETTINGS
        }
        
        if thinking_level is not None:
            config['thinking_config'] = types.ThinkingConfig(thinking_level=thinking_level)
            
        loop = asyncio.get_running_loop()
        def ask():
            start_time = time.time()
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            latency = int((time.time() - start_time) * 1000)
            if resp.usage_metadata:
                self.db.log_llm_usage(
                    model_name=model_name,
                    operation=operation_name,
                    prompt_tokens=resp.usage_metadata.prompt_token_count,
                    candidate_tokens=resp.usage_metadata.candidates_token_count,
                    latency_ms=latency,
                    is_error=False,
                    prompt_text=prompt,
                    response_text=resp.text
                )
            return resp.text.strip()
            
        try:
            result = await loop.run_in_executor(None, ask)
            return {"final_answer": result}
        except Exception as e:
            log.error(f"{operation_name} failed: {e}")
            return {"final_answer": f"❌ Failed to generate answer: {str(e)}"}

    def build_graph(self):
        builder = StateGraph(ChatState)
        
        builder.add_node("router", self.router_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("shallow", self.shallow_agent_node)
        builder.add_node("complex", self.complex_agent_node)
        
        builder.add_edge(START, "router")
        builder.add_edge("router", "rag")
        
        builder.add_conditional_edges(
            "rag",
            self.should_continue,
            {
                "shallow": "shallow",
                "complex": "complex"
            }
        )
        
        builder.add_edge("shallow", END)
        builder.add_edge("complex", END)
        
        return builder.compile()

    async def run(self, query: str) -> str:
        graph = self.build_graph()
        initial_state = {
            "query": query,
            "context": "",
            "routing_decision": "shallow",
            "final_answer": ""
        }
        
        try:
            final_state = await graph.ainvoke(initial_state)
            return final_state.get("final_answer", "Error: No answer generated.")
        except Exception as e:
            log.error(f"ChatOrchestrator run failed: {e}")
            return f"❌ Orchestrator failed: {str(e)}"
