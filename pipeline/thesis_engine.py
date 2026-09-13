"""
Deus — Thesis Engine

Takes an emerging theme and answers: what does this actually bottleneck on, and
who sits at that chokepoint?

    theme -> causal chain of bottlenecks -> companies at each one -> how
    priced-in each already is

The value is in hops two and three. When AI capex is the story, hop 1 is "GPU
demand rises" and the answer is NVDA — which is already priced, because that
inference is available to everyone reading the same headline. Hop 3 is "HBM
packaging capacity binds", and the answer is a name most people have not
connected to the story yet. The decomposition prompt is written to push
explicitly past the first hop, and the crowding score exists to catch the cases
where it failed to.

Graph shape (mirrors AdvisoryGraph in pipeline/agents.py):

    START -> decompose -> ground -> discover -> resolve -> score -> persist -> END
                  |                     |
                  +-> END               +-> persist  (nothing found: still
                                                      record the chain)

Every hop carries its own citations. They come from two places, and the split
is a cost decision:

  * ground — nearest-neighbour over the article embeddings ingest already paid
    for. Free, so it runs for EVERY node at every depth, hop 1 included;
  * discover — Tavily, billed per query, so it stays restricted to leaves at
    hop >= 2 as before.

Three guards live in Python rather than in the prompt, because prompts are
requests and these are requirements:

  * chain validation — cycles, dangling parents, over-deep hops and node count
    are enforced after parsing, not asked for politely;
  * the evidence gate — a candidate whose cited URLs appear in neither the
    search results nor the archive articles handed to that node is dropped.
    Models invent plausible sources, and this is a free, hard check;
  * company extraction skips hop 1 even when it is grounded: the names there
    are the consensus tickers the decomposition prompt already rules out.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from api.sse_manager import event_bus
from config.llm import (
    complete,
    config_error,
    is_llm_configured,
    parse_structured,
    response_cost,
    stream_complete,
    strip_code_fence,
)
from config.logging_config import get_logger
from config.settings import settings
from config.usage import track_llm
from data.database import Database
from pipeline.crowding import CrowdingScorer, edge_score, score_conviction
from pipeline.embedder import Embedder
from pipeline.price_feed import PriceFeed
from pipeline.personas import (
    BOTTLENECK_TAXONOMY,
    CONFIDENCE_CALIBRATION,
    EXPOSURE_GUIDANCE,
    JSON_ONLY,
    SECOND_ORDER_ANALYST,
)
from pipeline.theme_detector import ThemeDetector, ThemeSeed
from pipeline.ticker_resolver import TickerResolver
from pipeline.web_search import (
    _fallback_format,
    build_bottleneck_search_query,
    create_search_provider,
)

log = get_logger(__name__)

VALID_BOTTLENECKS = {
    "capacity", "raw_material", "energy", "regulatory",
    "talent", "logistics", "ip", "capital",
}


# ── Response schemas ─────────────────────────────────────────────────────
#
# Field descriptions are sent to the model as the schema description on every
# call, so each is kept to one line.


class ChainNode(BaseModel):
    """One causal hop in the chain."""

    node_key: str = Field(description="Short unique id for this node, e.g. 'n1'.")
    parent_key: Optional[str] = Field(
        default=None, description="node_key of the parent, or null for a root."
    )
    order_depth: int = Field(
        default=1, ge=1, le=5, description="1 for first-order, 2 or 3 for downstream."
    )
    claim: str = Field(description="The bottleneck, stated as one concrete sentence.")
    mechanism: str = Field(description="Why the parent causes this. Name what gets bought.")
    bottleneck_type: str = Field(default="", description="One of the bottleneck types listed.")
    falsifier: str = Field(description="The observation that would disprove this link.")
    lead_time: str = Field(default="", description="One of: now, 1-2q, 2-4q, 2y+.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ThesisChain(BaseModel):
    """A flat list of nodes; the tree is carried by parent_key."""

    nodes: list[ChainNode] = Field(default_factory=list)


class CandidateCompany(BaseModel):
    """A company positioned at one bottleneck."""

    node_key: str = Field(description="node_key of the bottleneck this company serves.")
    company_name: str = Field(description="Full company name.")
    ticker_guess: Optional[str] = Field(default=None, description="Best-guess symbol, or null.")
    is_listed: bool = Field(default=True, description="False if private or a subsidiary.")
    parent_company: Optional[str] = Field(default=None, description="Listed parent, if any.")
    role_in_chain: str = Field(description="What this company supplies at this bottleneck.")
    exposure_pct: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Estimated percent of total revenue exposed to this bottleneck.",
    )
    exposure_basis: Optional[str] = Field(default=None, description="Why that percentage.")
    substitutability: str = Field(
        default="oligopoly",
        description="One of: sole_source, duopoly, oligopoly, commoditized.",
    )
    evidence_urls: list[str] = Field(
        default_factory=list, description="URLs from the provided search results only."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateList(BaseModel):
    companies: list[CandidateCompany] = Field(default_factory=list)


# ── Prompts ──────────────────────────────────────────────────────────────

THESIS_DECOMPOSITION_PROMPT = """{persona}

THEME UNDER ANALYSIS
Title: {title}
Summary: {summary}

SUPPORTING HEADLINES
{headlines}

ALREADY PRICED IN — DO NOT PROPOSE THESE AS THE ANSWER
{consensus}

YOUR TASK
Decompose this theme into the chain of physical, economic and regulatory
bottlenecks it creates. Build a tree, at most {max_hops} hops deep, at most
{max_children} children per node, at most {max_nodes} nodes total.

Hop 1 is the obvious first-order consequence. It is already priced in and is
only there to anchor the chain — spend as little of the tree on it as possible.
Your value is entirely in hops 2 and 3: the chokepoints that bind once the
first-order demand is satisfied, and that the market has not yet connected to
this theme.

For every node you MUST provide:
- claim: the bottleneck as one concrete, checkable sentence. Not "demand rises"
  but "HBM stacking capacity limits accelerator output".
- mechanism: why the parent causes this, in terms of what physically gets
  bought. Name the purchase order.
- bottleneck_type: from the taxonomy below.
- falsifier: the specific observation that would prove this link wrong.
- lead_time: when it binds — now, 1-2q, 2-4q, or 2y+.
- confidence: calibrated per the guide below.

{taxonomy}

{calibration}

Quantify wherever the context supports it: what share of the parent's spend
does this step represent, and how long would new capacity take to add?

Return a FLAT list of nodes. Express the tree with parent_key, not by nesting.
Every parent_key must refer to a node_key that exists in your list, and the
graph must be acyclic.

{json_only}
Respond with an object of the form:
{{"nodes": [{{"node_key": "n1", "parent_key": null, "order_depth": 1,
"claim": "...", "mechanism": "...", "bottleneck_type": "capacity",
"falsifier": "...", "lead_time": "1-2q", "confidence": 0.7}}]}}
"""

THESIS_EXTRACTION_PROMPT = """{persona}

You are identifying the companies positioned at specific bottlenecks.

BOTTLENECKS AND THEIR SEARCH RESULTS
{blocks}

YOUR TASK
For each bottleneck, name up to {max_per_node} companies that supply or control
it. Prefer the company that would feel the effect most, not the largest company
adjacent to it.

{exposure_guidance}

Rules:
- Use ONLY companies that appear in the search results above. Do not add names
  from memory — sourcing changes faster than your training data.
- evidence_urls must contain URLs copied verbatim from the results shown for
  that bottleneck. A company you cannot cite will be discarded.
- Set is_listed false for private companies and subsidiaries, and name the
  listed parent in parent_company. Do not silently drop them: an unlisted
  chokepoint owner is still the correct answer to "who controls this".
- substitutability: sole_source, duopoly, oligopoly, or commoditized.

{json_only}
Respond with an object of the form:
{{"companies": [{{"node_key": "n3", "company_name": "...", "ticker_guess": "...",
"is_listed": true, "role_in_chain": "...", "exposure_pct": 45.0,
"exposure_basis": "...", "substitutability": "duopoly",
"evidence_urls": ["..."], "confidence": 0.6}}]}}
"""


# ── Graph state ──────────────────────────────────────────────────────────


class ThesisState(TypedDict):
    """Every key any node returns MUST be declared here.

    LangGraph filters node output down to the declared channels, silently
    dropping anything else — the same trap that loses `executive_summary` in
    agents.py and `top_articles` in chat_orchestrator.py today. tests/
    test_thesis_engine.py asserts this list stays complete.
    """

    seed: dict
    chain_nodes: list
    search_results: dict
    candidates: list
    resolved: list
    scored: list
    thesis_id: Optional[str]
    errors: list


ProgressCallback = Optional[Callable[[str], Awaitable[None]]]
ChunkCallback = Optional[Callable[[str], Awaitable[None]]]
ResearchCallback = Optional[Callable[[str, dict], Awaitable[None]]]


class ThesisGraph:
    """The reasoning pipeline for one thesis."""

    def __init__(
        self,
        db: Database,
        progress_callback: ProgressCallback = None,
        chunk_callback: ChunkCallback = None,
        research_callback: ResearchCallback = None,
        node_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        candidate_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        self.db = db
        self.progress_callback = progress_callback
        self.chunk_callback = chunk_callback
        self.research_callback = research_callback
        self.node_callback = node_callback
        self.candidate_callback = candidate_callback
        self.resolver = TickerResolver(db)
        self.scorer = CrowdingScorer(db)

    # ── Callback plumbing ────────────────────────────────────────────────

    async def _progress(self, message: str) -> None:
        if not self.progress_callback:
            return
        try:
            await self.progress_callback(message)
        except Exception as e:
            log.debug("thesis.progress_callback_failed", error=str(e))

    async def _emit(self, cb, payload) -> None:
        if not cb:
            return
        try:
            await cb(payload)
        except Exception as e:
            log.debug("thesis.callback_failed", error=str(e))

    async def _research(self, event: str, data: dict) -> None:
        if not self.research_callback:
            return
        try:
            await self.research_callback(event, data)
        except Exception:
            pass

    # ── Nodes ────────────────────────────────────────────────────────────

    async def decompose_node(self, state: ThesisState) -> dict:
        """Theme -> causal chain. The one expensive reasoning call."""
        seed = state["seed"]
        await self._progress(f"Decomposing: {seed.get('title', '')}")

        # Two distinct causes, reported distinctly. Collapsing them into one
        # "MODEL_THESIS_REASONER is not configured" sends the reader to edit a
        # setting that is already correct while the real gap — no key, or a
        # .env the process never found — goes unmentioned.
        if not is_llm_configured():
            return {"chain_nodes": [], "errors": [config_error("MODEL_THESIS_REASONER")]}
        if not settings.model_thesis_reasoner:
            return {"chain_nodes": [], "errors": ["MODEL_THESIS_REASONER is not set in .env"]}

        headlines = "\n".join(f"- {h}" for h in (seed.get("headlines") or [])[:12])
        consensus = ", ".join(seed.get("consensus_tickers") or []) or "(none identified)"
        prompt = THESIS_DECOMPOSITION_PROMPT.format(
            persona=SECOND_ORDER_ANALYST,
            title=seed.get("title", ""),
            summary=seed.get("summary", "") or "(no summary available)",
            headlines=headlines or "(no in-house headlines; reason from the title)",
            consensus=consensus,
            max_hops=settings.thesis_max_hops,
            max_children=settings.thesis_max_children_per_node,
            max_nodes=settings.thesis_max_nodes,
            taxonomy=BOTTLENECK_TAXONOMY,
            calibration=CONFIDENCE_CALIBRATION,
            json_only=JSON_ONLY,
        )

        try:
            raw, finish_reason = await self._call_reasoner(prompt)
        except Exception as e:
            log.error("thesis.decompose_failed", error=str(e))
            return {"chain_nodes": [], "errors": [f"decompose failed: {e}"]}

        # The model spent the whole budget thinking and never wrote the answer.
        # Reported as itself rather than as a parse failure: the fix is a larger
        # thesis_max_output_tokens, and "could not parse the chain" sends
        # whoever reads the log looking at the schema instead.
        if finish_reason == "length" and not raw.strip():
            log.error("thesis.decompose_budget_exhausted",
                      max_tokens=settings.thesis_max_output_tokens)
            await self._progress(
                "The model used its entire output budget on reasoning and never "
                "produced a chain. Raise THESIS_MAX_OUTPUT_TOKENS."
            )
            return {"chain_nodes": [],
                    "errors": ["reasoning used the whole output budget; "
                               "raise thesis_max_output_tokens"]}

        chain = None
        try:
            chain = parse_structured(strip_code_fence(raw), ThesisChain)
        except Exception as e:
            log.error("thesis.decompose_parse_failed", error=str(e),
                      finish_reason=finish_reason, chars=len(raw))

        if chain is None:
            detail = ("the chain was cut off mid-JSON (output budget reached)"
                      if finish_reason == "length" else
                      "could not parse the chain")
            await self._progress(f"Decomposition failed: {detail}")
            return {"chain_nodes": [], "errors": [detail]}

        nodes = validate_chain(
            [n.model_dump() for n in chain.nodes],
            max_nodes=settings.thesis_max_nodes,
            max_hops=settings.thesis_max_hops,
        )
        if not nodes:
            # Parsed, but validation kept nothing — every node was cyclic,
            # orphaned or too deep. has_chain() would route this to END with an
            # empty error list, which is the one failure mode that looks
            # identical to success from outside.
            log.error("thesis.chain_empty_after_validation",
                      proposed=len(chain.nodes))
            await self._progress("The proposed chain failed validation")
            return {"chain_nodes": [],
                    "errors": [f"all {len(chain.nodes)} proposed bottlenecks "
                               f"failed chain validation"]}

        for n in nodes:
            await self._emit(self.node_callback, n)
        await self._progress(f"Chain built: {len(nodes)} bottlenecks")
        log.info("thesis.decomposed", nodes=len(nodes),
                 leaves=sum(1 for n in nodes if n["is_leaf"]))
        return {"chain_nodes": nodes}

    async def _call_reasoner(self, prompt: str) -> tuple[str, Optional[str]]:
        """The decompose call, at xhigh effort, streamed when someone is listening.

        Returns the answer text and the provider's finish_reason. The caller
        needs the latter: reasoning tokens are drawn from the same completion
        budget as the answer, so at xhigh the thinking block can consume all of
        it and leave the content empty. That is indistinguishable from a
        malformed reply unless finish_reason is checked.
        """
        model = settings.model_thesis_reasoner
        kwargs = dict(
            model=model,
            prompt=prompt,
            system=SECOND_ORDER_ANALYST,
            json_mode=True,
            reasoning="xhigh",
            max_tokens=settings.thesis_max_output_tokens,
        )

        if not self.chunk_callback:
            with track_llm(self.db, model, "thesis_decompose") as u:
                u.response = response = await complete(**kwargs)
            return response.text, response.finish_reason

        collected: list[str] = []
        finish_reason: Optional[str] = None
        with track_llm(self.db, model, "thesis_decompose") as u:
            async for chunk in stream_complete(**kwargs):
                # This is the one call in the project that wants the chain of
                # thought back — surfacing it is the whole appeal of watching a
                # thesis run. It used to arrive on DeepSeek's own
                # `reasoning_content`; config.llm now normalises whatever the
                # route returns into `chunk.reasoning`.
                if chunk.reasoning:
                    await self._emit(self.chunk_callback, chunk.reasoning)
                if chunk.text:
                    collected.append(chunk.text)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage is not None:
                    # Usage arrives on a trailing chunk with no choices. Set the
                    # counts explicitly rather than handing over a response
                    # object: this is the only branch that logs the most
                    # expensive call in the feature, and a silently-zero token
                    # count is indistinguishable from a free one.
                    u.prompt_tokens = getattr(chunk.usage, "prompt_tokens", None)
                    u.candidate_tokens = getattr(chunk.usage, "completion_tokens", None)
                    u.cost = response_cost(chunk.usage)
        return "".join(collected), finish_reason

    async def ground_node(self, state: ThesisState) -> dict:
        """Attach supporting articles from our own corpus to every hop.

        Runs for all nodes at all depths, unlike the web search that follows.
        The reason searches stop at leaves is the per-query bill; nearest
        neighbour over embeddings ingest has already paid for costs nothing, so
        hop 1 gets citations too — and hop 1 is exactly where the reader most
        wants to check the premise the rest of the chain hangs off.

        Never fatal: a thesis with no internal coverage is still a thesis.
        """
        nodes = state.get("chain_nodes") or []
        if not nodes:
            return {}

        for n in nodes:
            n.setdefault("sources", [])

        # initialize() returns None on both paths, so the result cannot be
        # tested — get_embeddings() answers all-None when it failed, which is
        # handled below anyway.
        embedder = Embedder(db=self.db)
        await embedder.initialize()

        await self._progress(f"Grounding {len(nodes)} bottlenecks in the archive")

        # One batched call for the whole chain rather than one per node.
        # The mechanism is included because a bare claim is a fragment; the
        # two together read much more like the articles being matched against.
        texts = [
            f"{n.get('claim', '')} {n.get('mechanism', '')}".strip() for n in nodes
        ]
        vectors = await embedder.get_embeddings(texts)
        if not any(v is not None for v in vectors):
            log.info("thesis.grounding_unavailable")
            return {"chain_nodes": nodes}

        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=settings.thesis_evidence_lookback_days)
        ).isoformat()

        def lookup() -> list[list[dict]]:
            found = []
            for vec in vectors:
                if vec is None:
                    found.append([])
                    continue
                try:
                    found.append(self.db.find_articles_for_claim(
                        vec,
                        limit=settings.thesis_evidence_per_node,
                        min_similarity=settings.thesis_evidence_min_similarity,
                        since_iso=cutoff,
                    ))
                except Exception as e:
                    log.warning("thesis.grounding_query_failed", error=str(e))
                    found.append([])
            return found

        per_node = await asyncio.to_thread(lookup)

        grounded = 0
        for node, articles in zip(nodes, per_node):
            node["sources"] = [
                {
                    "kind": "internal",
                    "title": a.get("headline") or "",
                    "url": a.get("url") or "",
                    "source": a.get("source_name") or "",
                    "published_at": a.get("published_at"),
                    "article_id": a.get("id"),
                    "similarity": round(float(a.get("similarity") or 0.0), 3),
                }
                for a in articles
            ]
            if node["sources"]:
                grounded += 1
            for src in node["sources"]:
                await self._research("research_source", {
                    "title": src["title"], "url": src["url"],
                    "domain": src["source"], "node_key": node["node_key"],
                    "kind": "internal",
                })
            # Re-emitted so the live view can show citations under a card it
            # already drew. The page upserts on node_key, so this updates
            # rather than duplicates.
            await self._emit(self.node_callback, node)

        log.info("thesis.grounded", nodes=len(nodes), with_sources=grounded)
        await self._progress(
            f"Archive grounding: {grounded}/{len(nodes)} bottlenecks cited"
        )
        return {"chain_nodes": nodes}

    async def discover_node(self, state: ThesisState) -> dict:
        """Search each searchable bottleneck, then extract companies in one call.

        The search half can be skipped entirely — no targets, no provider — and
        extraction still runs against whatever the grounding step attached.
        Those two used to be one early return, which meant an unconfigured
        TAVILY_API_KEY produced a chain with no companies even when the archive
        had plenty to say about it.
        """
        nodes = state.get("chain_nodes") or []
        errors = list(state.get("errors", []))
        targets = searchable_nodes(nodes, settings.thesis_max_search_nodes)

        provider = create_search_provider() if targets else None
        if targets and provider is None:
            log.info("thesis.no_search_provider")
            errors.append("web search not configured")

        if provider is None:
            return await self._extract_and_gate(nodes, {}, errors)

        await self._progress(f"Researching {len(targets)} bottlenecks")
        await self._research("research_start", {"query": f"{len(targets)} bottlenecks"})

        semaphore = asyncio.Semaphore(3)

        async def search_one(node: dict) -> tuple[str, list]:
            async with semaphore:
                query = build_bottleneck_search_query(
                    node["claim"], node.get("bottleneck_type", "")
                )
                try:
                    results = await provider.search(
                        query, max_results=settings.web_search_max_results
                    )
                except Exception as e:
                    log.warning("thesis.search_failed",
                                node=node["node_key"], error=str(e))
                    return node["node_key"], []
                for r in results:
                    await self._research("research_source", {
                        "title": r.title, "url": r.url,
                        "domain": r.source, "node_key": node["node_key"],
                    })
                return node["node_key"], results

        pairs = await asyncio.gather(*(search_one(n) for n in targets))
        search_results = {k: v for k, v in pairs if v}
        found = sum(len(v) for v in search_results.values())
        await self._research("research_complete", {"sources_found": found})

        # Fold the web hits into the same per-node source list the archive
        # grounding filled, so one node carries one list of citations whatever
        # they came from, and persist_node has nothing extra to assemble.
        by_key = {n["node_key"]: n for n in nodes}
        for key, results in search_results.items():
            node = by_key.get(key)
            if not node:
                continue
            node["searched"] = True
            node.setdefault("sources", []).extend(
                {
                    "kind": "web",
                    "title": r.title,
                    "url": r.url,
                    "source": r.source,
                    "published_at": r.published_date,
                }
                for r in results
            )
            await self._emit(self.node_callback, node)

        return await self._extract_and_gate(nodes, search_results, errors)

    async def _extract_and_gate(
        self, nodes: list[dict], search_results: dict, errors: list
    ) -> dict:
        """Find companies across whatever sources the nodes ended up with.

        Extraction runs whenever ANY node has something to read. Before the
        grounding step that could only be a web result; a node carrying archive
        coverage and no web hits is worth extracting from too.
        """
        if not search_results and not any(n.get("sources") for n in nodes):
            return {"search_results": {}, "candidates": [], "errors": errors}

        await self._progress("Identifying companies at each chokepoint")
        candidates = await self._extract_companies(nodes, search_results)

        # Hallucination gate. A company whose cited URLs appear in none of the
        # sources shown to its node was invented, however plausible it reads.
        kept = filter_by_evidence(candidates, search_results, nodes)
        dropped = len(candidates) - len(kept)
        if dropped:
            log.info("thesis.evidence_gate_dropped", dropped=dropped)

        for c in kept:
            await self._emit(self.candidate_callback, c)
        return {
            "search_results": {
                k: [{"title": r.title, "url": r.url, "domain": r.source}
                    for r in v]
                for k, v in search_results.items()
            },
            "candidates": kept,
            "errors": errors,
        }

    async def _extract_companies(
        self, nodes: list[dict], search_results: dict
    ) -> list[dict]:
        """One batched call across every searched bottleneck."""
        if not is_llm_configured() or not settings.model_thesis_extract:
            return []

        by_key = {n["node_key"]: n for n in nodes}
        blocks = []
        # Hop 1 is the priced-in layer, so it is grounded for the reader but
        # deliberately not mined for names — the consensus tickers it would
        # return are the ones the decomposition prompt already rules out.
        keys = [
            n["node_key"] for n in nodes
            if n.get("order_depth", 1) >= 2
            and (search_results.get(n["node_key"]) or n.get("sources"))
        ]
        for key in keys:
            node = by_key.get(key)
            if not node:
                continue
            parts = [
                f"### Bottleneck {key}: {node['claim']}",
                f"Type: {node.get('bottleneck_type') or 'unspecified'}",
            ]
            results = search_results.get(key) or []
            if results:
                parts.append(f"Search results:\n{_fallback_format(results)}")
            archive = _format_internal_sources(node.get("sources") or [])
            if archive:
                parts.append(f"From our own news archive:\n{archive}")
            blocks.append("\n".join(parts))

        if not blocks:
            return []

        prompt = THESIS_EXTRACTION_PROMPT.format(
            persona=SECOND_ORDER_ANALYST,
            blocks="\n\n".join(blocks),
            max_per_node=settings.thesis_max_candidates_per_node,
            exposure_guidance=EXPOSURE_GUIDANCE,
            json_only=JSON_ONLY,
        )

        try:
            # Reasoning is set rather than left to the model: omitting it lets a
            # thinking-by-default slug think at its own budget. The causal work
            # happened in the decomposition; this reads cited names out of the
            # results. Low rather than none, since exposure_pct and
            # substitutability are still judgment calls.
            with track_llm(self.db, settings.model_thesis_extract, "thesis_extract") as u:
                u.response = response = await complete(
                    model=settings.model_thesis_extract,
                    prompt=prompt,
                    schema=CandidateList,
                    reasoning="low",
                )
            parsed = response.parsed
            if not isinstance(parsed, CandidateList):
                parsed = parse_structured(strip_code_fence(response.text), CandidateList)
            return [c.model_dump() for c in parsed.companies]
        except Exception as e:
            log.error("thesis.extract_failed", error=str(e))
            return []

    async def resolve_node(self, state: ThesisState) -> dict:
        """Confirm guessed tickers against live quotes."""
        candidates = state.get("candidates") or []
        if not candidates:
            return {"resolved": []}
        await self._progress(f"Resolving {len(candidates)} company names")
        resolved = await self.resolver.resolve_many(candidates)
        return {"resolved": resolved}

    async def score_node(self, state: ThesisState) -> dict:
        """Conviction, crowding and edge for every candidate. No LLM."""
        candidates = state.get("resolved") or []
        if not candidates:
            return {"scored": []}
        await self._progress("Scoring how priced-in each name already is")

        by_key = {n["node_key"]: n for n in (state.get("chain_nodes") or [])}

        def score_all() -> list[dict]:
            out = []
            for cand in candidates:
                node = by_key.get(cand.get("node_key"), {})
                conviction = score_conviction(
                    node_confidence=node.get("confidence", 0.5),
                    exposure_pct=cand.get("exposure_pct"),
                    substitutability=cand.get("substitutability", ""),
                    evidence_count=len(cand.get("evidence_urls") or []),
                )
                cand["conviction"] = conviction
                ticker = cand.get("ticker")
                if ticker:
                    result = self.scorer.score(ticker)
                    cand["crowding"] = result["crowding"]
                    cand["data_coverage"] = result["coverage"]
                    cand["rumour_stage"] = result["stage"]
                    cand["score_components"] = result["components"]
                    cand["score_raw"] = result["raw"]
                    cand["edge_score"] = edge_score(conviction, result["crowding"])
                else:
                    # Unlisted or unresolved: conviction still stands, but there
                    # is no price to be crowded into.
                    cand["crowding"] = None
                    cand["data_coverage"] = 0.0
                    cand["rumour_stage"] = "UNKNOWN"
                    cand["edge_score"] = None
                out.append(cand)
            return out

        scored = await asyncio.to_thread(score_all)
        scored.sort(key=lambda c: (c.get("edge_score") is None,
                                   -(c.get("edge_score") or 0.0)))
        return {"scored": scored}

    async def persist_node(self, state: ThesisState) -> dict:
        """Write the thesis, its chain and its candidates."""
        seed = state["seed"]
        nodes = state.get("chain_nodes") or []
        scored = state.get("scored") or []

        def write() -> Optional[str]:
            fingerprint = seed.get("fingerprint") or ""
            if fingerprint:
                self.db.deactivate_theses(fingerprint)
            row = dict(seed.get("thesis_row") or {})
            row["model_name"] = settings.model_thesis_reasoner
            thesis_id = self.db.insert_thesis(row)
            keymap = self.db.insert_thesis_nodes(thesis_id, nodes)
            self.db.upsert_thesis_candidates([
                {
                    "thesis_id": thesis_id,
                    "node_id": keymap.get(c.get("node_key")),
                    "company_name": c.get("company_name", ""),
                    "ticker": c.get("ticker"),
                    "ticker_guess": c.get("ticker_guess"),
                    "market": c.get("market"),
                    "resolution_status": c.get("resolution_status", "pending"),
                    "alt_tickers": c.get("alt_tickers", []),
                    "listing_status": c.get("listing_status", "unresolved"),
                    "parent_company": c.get("parent_company"),
                    "us_proxy": c.get("us_proxy"),
                    "role_in_chain": c.get("role_in_chain", ""),
                    "exposure": c.get("exposure_pct") or 0.0,
                    "exposure_rationale": c.get("exposure_basis") or "",
                    "substitutability": c.get("substitutability", "oligopoly"),
                    "evidence_urls": c.get("evidence_urls", []),
                    "conviction": c.get("conviction", 0.0),
                    "crowding": c.get("crowding"),
                    "rumour_stage": c.get("rumour_stage"),
                    "edge_score": c.get("edge_score"),
                    "data_coverage": c.get("data_coverage", 0.0),
                }
                for c in scored
            ])
            return thesis_id

        try:
            thesis_id = await asyncio.to_thread(write)
        except Exception as e:
            log.error("thesis.persist_failed", error=str(e))
            return {"thesis_id": None,
                    "errors": [*state.get("errors", []), f"persist failed: {e}"]}

        log.info("thesis.persisted", thesis_id=thesis_id,
                 nodes=len(nodes), candidates=len(scored))
        await self._progress("Thesis saved")
        return {"thesis_id": thesis_id}

    # ── Routing ──────────────────────────────────────────────────────────

    def has_chain(self, state: ThesisState) -> str:
        return "ground" if state.get("chain_nodes") else "end"

    def has_candidates(self, state: ThesisState) -> str:
        # A chain with no companies is still worth keeping: the bottlenecks are
        # the reusable part, and the next run can search them again.
        return "resolve" if state.get("candidates") else "persist"

    def build_graph(self):
        builder = StateGraph(ThesisState)

        builder.add_node("decompose", self.decompose_node)
        builder.add_node("ground", self.ground_node)
        builder.add_node("discover", self.discover_node)
        builder.add_node("resolve", self.resolve_node)
        builder.add_node("score", self.score_node)
        builder.add_node("persist", self.persist_node)

        builder.add_edge(START, "decompose")
        builder.add_conditional_edges(
            "decompose", self.has_chain, {"ground": "ground", "end": END}
        )
        builder.add_edge("ground", "discover")
        builder.add_conditional_edges(
            "discover", self.has_candidates, {"resolve": "resolve", "persist": "persist"}
        )
        builder.add_edge("resolve", "score")
        builder.add_edge("score", "persist")
        builder.add_edge("persist", END)
        return builder.compile()

    async def run(self, seed: ThemeSeed) -> dict:
        """Execute one thesis end to end."""
        initial: ThesisState = {
            "seed": {
                "title": seed.title,
                "summary": seed.summary,
                "headlines": seed.headlines,
                "consensus_tickers": seed.consensus_tickers,
                "fingerprint": seed.fingerprint,
                "thesis_row": seed.to_thesis_row(),
            },
            "chain_nodes": [],
            "search_results": {},
            "candidates": [],
            "resolved": [],
            "scored": [],
            "thesis_id": None,
            "errors": [],
        }
        graph = self.build_graph()
        return await graph.ainvoke(initial)


# ── Chain validation (Python-side, not prompt-side) ──────────────────────


def validate_chain(nodes: list[dict], max_nodes: int, max_hops: int) -> list[dict]:
    """Enforce the structural rules the prompt merely asks for.

    Drops nodes deeper than max_hops, nodes whose parent does not exist, and
    anything caught in a cycle; truncates to max_nodes; then marks leaves.
    Returns nodes ordered shallowest-first.
    """
    seen: dict[str, dict] = {}
    for n in nodes:
        key = (n.get("node_key") or "").strip()
        if not key or key in seen:
            continue
        if not (n.get("claim") or "").strip():
            continue
        depth = int(n.get("order_depth") or 1)
        if depth < 1 or depth > max_hops:
            continue
        btype = (n.get("bottleneck_type") or "").strip().lower()
        seen[key] = {
            "node_key": key,
            "parent_key": (n.get("parent_key") or None),
            "order_depth": depth,
            "claim": n.get("claim", "").strip(),
            "mechanism": (n.get("mechanism") or "").strip(),
            "bottleneck_type": btype if btype in VALID_BOTTLENECKS else "",
            "falsifier": (n.get("falsifier") or "").strip(),
            "lead_time": (n.get("lead_time") or "").strip(),
            "confidence": float(n.get("confidence") or 0.5),
            "is_leaf": False,
            "searched": False,
        }

    # Drop dangling parents, then anything that cannot reach a root without
    # revisiting itself.
    for node in seen.values():
        if node["parent_key"] not in seen:
            node["parent_key"] = None

    acyclic: dict[str, dict] = {}
    for key, node in seen.items():
        walker, hops, ok = node, 0, True
        while walker["parent_key"]:
            walker = seen[walker["parent_key"]]
            hops += 1
            if hops > len(seen):
                ok = False
                break
        if ok:
            acyclic[key] = node

    ordered = sorted(acyclic.values(), key=lambda n: (n["order_depth"], n["node_key"]))
    ordered = ordered[:max_nodes]

    kept = {n["node_key"] for n in ordered}
    for node in ordered:
        if node["parent_key"] not in kept:
            node["parent_key"] = None
    parents = {n["parent_key"] for n in ordered if n["parent_key"]}
    for node in ordered:
        node["is_leaf"] = node["node_key"] not in parents
    return ordered


def searchable_nodes(nodes: list[dict], limit: int) -> list[dict]:
    """Leaves at hop >= 2, deepest and most confident first.

    Hop 1 is the priced-in layer by definition, so searching it spends the
    per-thesis budget on the answer everyone already has.
    """
    targets = [n for n in nodes if n.get("is_leaf") and n.get("order_depth", 1) >= 2]
    if not targets:
        # A one-hop chain is a weak thesis, but returning nothing at all would
        # produce a thesis with no companies whatsoever.
        targets = [n for n in nodes if n.get("is_leaf")]
    targets.sort(key=lambda n: (-n.get("order_depth", 1), -n.get("confidence", 0.0)))
    selected = targets[:limit]
    for node in selected:
        node["searched"] = True
    return selected


def _format_internal_sources(sources: list[dict]) -> str:
    """Render archive articles for the extraction prompt.

    Same shape as _fallback_format's web block so the model reads one
    consistent citation format, with the URL spelled out because the evidence
    gate only keeps a company whose cited URL appears verbatim here.
    """
    internal = [s for s in sources if s.get("kind") == "internal"]
    if not internal:
        return ""
    lines = []
    for i, s in enumerate(internal, 1):
        date = (s.get("published_at") or "")[:10] or "recent"
        lines.append(
            f"{i}. [{s.get('source') or 'archive'}] {s.get('title') or ''} "
            f"({date})\n   {s.get('url') or ''}"
        )
    return "\n".join(lines)


def filter_by_evidence(
    candidates: list[dict],
    search_results: dict,
    nodes: Optional[list[dict]] = None,
) -> list[dict]:
    """Drop companies not backed by a URL actually shown to their node.

    Models produce confident, plausible, entirely fabricated sources. This is
    free to check and catches the failure the extraction prompt cannot prevent.

    `nodes` carries the archive articles the grounding step attached. They are
    shown to the model alongside the web results, so a company cited from one
    is legitimately evidenced — omitting them here would silently discard every
    name that came from our own corpus.
    """
    urls_by_node = {
        key: {r.url for r in results if getattr(r, "url", None)}
        for key, results in search_results.items()
    }
    for node in nodes or []:
        allowed = urls_by_node.setdefault(node.get("node_key"), set())
        allowed.update(
            s["url"] for s in (node.get("sources") or []) if s.get("url")
        )
    kept = []
    for cand in candidates:
        allowed = urls_by_node.get(cand.get("node_key"), set())
        cited = [u for u in (cand.get("evidence_urls") or []) if u in allowed]
        if not cited:
            log.debug("thesis.candidate_unevidenced",
                      company=cand.get("company_name"))
            continue
        cand["evidence_urls"] = cited
        kept.append(cand)
    return kept


# ── Facade ───────────────────────────────────────────────────────────────


class ThesisEngine:
    """Entry point used by the scheduler and the API."""

    def __init__(self, db: Database):
        self.db = db
        self.detector = ThemeDetector(db)

    async def generate(self, limit: Optional[int] = None) -> list[str]:
        """Detect themes and build a thesis for the most accelerated ones."""
        if not settings.thesis_enabled:
            return []
        limit = limit or settings.thesis_per_run
        seeds = await self.detector.collect(limit=limit)
        if not seeds:
            log.info("thesis.no_seeds")
            return []

        ids = []
        for seed in seeds[:limit]:
            graph = ThesisGraph(self.db)
            result = await graph.run(seed)
            if result.get("thesis_id"):
                ids.append(result["thesis_id"])
            else:
                # Nobody is watching a scheduled run, so a graph that ends
                # without persisting has to say so here or the morning simply
                # produces nothing and leaves no trace of why.
                log.warning("thesis.generation_produced_nothing",
                            title=seed.title,
                            errors=result.get("errors") or ["unknown"])

        if ids:
            # The scheduled path had no counterpart to rescore_all's publish,
            # so a thesis built at 06:30 stayed invisible until someone
            # reloaded the page by hand.
            try:
                await event_bus.publish("thesis_update", {
                    "kind": "generated",
                    "count": len(ids),
                    "thesis_ids": ids,
                })
            except Exception as e:
                log.warning("thesis.sse_publish_failed", error=str(e))
        return ids

    async def generate_from_text(self, text: str, **callbacks) -> dict:
        """Build a thesis from a user-supplied topic.

        A first-class entry point rather than a fallback: this corpus ingests
        around 20 articles a day, which is often too thin for clustering to
        surface a specific theme on demand.
        """
        seed = self.detector.seed_from_text(text)
        graph = ThesisGraph(self.db, **callbacks)
        return await graph.run(seed)

    async def rescore_all(self, limit: Optional[int] = None) -> dict:
        """Re-score every live candidate. No LLM calls — prices and SQL only.

        Deliberately separate from generation. Generation is the expensive
        half and runs once a day; this is the half that produces the trade
        timing, because a candidate only becomes actionable when it *moves*
        between stages. Running it daily is what makes EARLY -> BUILDING ->
        CROWDED observable at all.
        """
        if not settings.thesis_enabled:
            return {"scored": 0}

        limit = limit or settings.thesis_rescore_max_tickers
        candidates = await asyncio.to_thread(self.db.get_scoreable_candidates, limit)
        if not candidates:
            return {"scored": 0}

        tickers = sorted({c["ticker"] for c in candidates if c.get("ticker")})
        # price_history_sync only covers the watchlist, and thesis candidates
        # are off-watchlist by construction, so this job fetches its own bars.
        try:
            await PriceFeed(self.db).refresh_history_for(tickers, history_range="1y")
        except Exception as e:
            log.warning("thesis.rescore_price_refresh_failed", error=str(e))

        today = datetime.now(timezone.utc).date().isoformat()
        scorer = CrowdingScorer(self.db)

        def score_and_write() -> dict:
            snapshots, transitions, promoted = [], [], 0
            for cand in candidates:
                ticker = cand["ticker"]
                prior = self.db.get_latest_snapshot(cand["id"], before_date=today)
                result = scorer.score(ticker)
                conviction = float(cand.get("conviction") or 0.0)
                edge = edge_score(conviction, result["crowding"])

                if prior and prior.get("rumour_stage") != result["stage"]:
                    transitions.append({
                        "ticker": ticker,
                        "company_name": cand.get("company_name"),
                        "thesis_title": cand.get("thesis_title"),
                        "old_stage": prior.get("rumour_stage"),
                        "new_stage": result["stage"],
                        "edge_score": edge,
                    })

                self.db.update_candidate_score(
                    cand["id"], result["stage"], result["crowding"],
                    edge, result["coverage"],
                )
                raw = result["raw"]
                snapshots.append({
                    "candidate_id": cand["id"], "ticker": ticker,
                    "as_of_date": today, "price": raw.get("price"),
                    "crowding": result["crowding"], "rumour_stage": result["stage"],
                    "edge_score": edge, "data_coverage": result["coverage"],
                    "components": result["components"],
                    "ret_1m": raw.get("ret_1m"), "ret_3m": raw.get("ret_3m"),
                    "dist_from_52w_high": raw.get("dist_from_52w_high"),
                    "volume_ratio": raw.get("volume_ratio"),
                    "mentions_30d": (raw.get("mentions_recent") or 0)
                                    + (raw.get("mentions_base") or 0),
                    "mention_accel": raw.get("mention_accel"),
                })

                # Auto-track quiet, high-conviction names so ingest starts
                # accumulating the mention and price history the *next*
                # re-score needs. Without this the coverage never improves and
                # every discovered name stays permanently unmeasurable.
                if (
                    result["crowding"] is not None
                    and result["crowding"] <= settings.thesis_promote_max_crowding
                    and conviction >= settings.thesis_promote_min_conviction
                    and cand.get("listing_status") in ("us_listed", "adr")
                ):
                    self.db.promote_thesis_ticker(
                        ticker, cand["thesis_id"],
                        f"[Thesis] {cand.get('thesis_title') or ''}: "
                        f"{cand.get('role_in_chain') or ''}".strip(),
                    )
                    promoted += 1

            self.db.upsert_thesis_snapshots(snapshots)
            return {"scored": len(snapshots), "transitions": transitions,
                    "promoted": promoted}

        stats = await asyncio.to_thread(score_and_write)
        log.info("thesis.rescored", scored=stats["scored"],
                 transitions=len(stats["transitions"]), promoted=stats["promoted"])

        try:
            await event_bus.publish("thesis_update", {
                "kind": "rescore",
                "scored": stats["scored"],
                "promoted": stats["promoted"],
                "transitions": stats["transitions"][:10],
            })
        except Exception as e:
            log.warning("thesis.sse_publish_failed", error=str(e))
        return stats
