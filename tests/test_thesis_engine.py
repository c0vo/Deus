"""Tests for the thesis engine.

No live LLM calls: every client is mocked. The focus is on the three guards
that live in Python rather than in a prompt — chain validation, search
targeting, and the evidence gate — plus the mechanical check that no node
returns a state key LangGraph would silently discard, and the once-a-day rule
the scheduler applies to the morning run.
"""

import ast
import inspect
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from orchestrator import scheduler
from orchestrator.scheduler import PipelineOrchestrator
from pipeline import thesis_engine as te
from pipeline.thesis_engine import (
    THESIS_DECOMPOSITION_PROMPT,
    THESIS_EXTRACTION_PROMPT,
    CandidateList,
    ThesisChain,
    ThesisGraph,
    ThesisState,
    filter_by_evidence,
    searchable_nodes,
    validate_chain,
)


def node(key, parent=None, depth=1, claim="a claim", conf=0.5, btype="capacity"):
    return {
        "node_key": key, "parent_key": parent, "order_depth": depth,
        "claim": claim, "mechanism": "m", "bottleneck_type": btype,
        "falsifier": "f", "lead_time": "1-2q", "confidence": conf,
    }


class Result:
    """Stands in for WebSearchResult (only .url is read by the gate)."""

    def __init__(self, url, title="t", source="d"):
        self.url = url
        self.title = title
        self.source = source


# ── The LangGraph silent-drop guard ──────────────────────────────────────


def test_every_returned_state_key_is_declared():
    """LangGraph filters node output to the declared channels.

    Anything a node returns that is not in ThesisState is dropped without a
    warning — the bug that currently loses `executive_summary` in agents.py and
    `top_articles` in chat_orchestrator.py. This walks the AST of every *_node
    method and checks each returned dict literal's keys.
    """
    declared = set(ThesisState.__annotations__)
    source = Path(inspect.getfile(te)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    graph_cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "ThesisGraph"
    )
    checked = 0
    node_methods = 0
    for method in graph_cls.body:
        if not isinstance(method, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        # Every method, not just *_node: a graph node may tail-return the dict
        # built by a helper (discover_node -> _extract_and_gate), and checking
        # only the node methods would leave that helper's channels unverified.
        if method.name.endswith("_node"):
            node_methods += 1
        checked += 1
        for stmt in ast.walk(method):
            if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Dict):
                continue
            for k in stmt.value.keys:
                assert isinstance(k, ast.Constant), (
                    f"{method.name} returns a non-literal key; it cannot be verified"
                )
                assert k.value in declared, (
                    f"{method.name} returns '{k.value}', which is not declared in "
                    f"ThesisState and will be silently dropped by LangGraph"
                )
    assert node_methods >= 6, "expected to inspect every graph node method"


def test_graph_compiles():
    graph = ThesisGraph(MagicMock()).build_graph()
    assert graph is not None


# ── Chain validation ─────────────────────────────────────────────────────


def test_valid_chain_survives_and_marks_leaves():
    nodes = validate_chain(
        [node("n1"), node("n2", "n1", 2), node("n3", "n2", 3)],
        max_nodes=12, max_hops=3,
    )
    assert [n["node_key"] for n in nodes] == ["n1", "n2", "n3"]
    assert [n["is_leaf"] for n in nodes] == [False, False, True]


def test_dangling_parent_is_reparented_to_root():
    nodes = validate_chain([node("n2", "ghost", 2)], max_nodes=12, max_hops=3)
    assert nodes[0]["parent_key"] is None


def test_cycles_are_removed():
    """A -> B -> A must not survive; it would loop the renderer."""
    nodes = validate_chain(
        [node("a", "b", 1), node("b", "a", 2)], max_nodes=12, max_hops=3
    )
    assert all(n["parent_key"] is None or n["parent_key"] in
               {x["node_key"] for x in nodes} for n in nodes)
    for n in nodes:
        assert n["node_key"] != n["parent_key"]


def test_over_deep_nodes_are_dropped():
    nodes = validate_chain(
        [node("n1"), node("n9", "n1", 9)], max_nodes=12, max_hops=3
    )
    assert [n["node_key"] for n in nodes] == ["n1"]


def test_node_count_is_capped():
    nodes = validate_chain([node(f"n{i}") for i in range(40)], max_nodes=5, max_hops=3)
    assert len(nodes) == 5


def test_nodes_without_a_claim_are_dropped():
    nodes = validate_chain([node("n1", claim="  ")], max_nodes=12, max_hops=3)
    assert nodes == []


def test_duplicate_keys_are_collapsed():
    nodes = validate_chain([node("n1"), node("n1")], max_nodes=12, max_hops=3)
    assert len(nodes) == 1


def test_unknown_bottleneck_type_is_blanked_not_kept():
    nodes = validate_chain([node("n1", btype="vibes")], max_nodes=12, max_hops=3)
    assert nodes[0]["bottleneck_type"] == ""


# ── Search targeting ─────────────────────────────────────────────────────


def test_search_skips_hop_one_which_is_already_priced_in():
    nodes = validate_chain(
        [node("n1"), node("n2", "n1", 2), node("n3", "n2", 3)],
        max_nodes=12, max_hops=3,
    )
    targets = searchable_nodes(nodes, limit=6)
    assert [n["node_key"] for n in targets] == ["n3"]


def test_search_prefers_deeper_nodes_first():
    nodes = validate_chain(
        [node("n1"), node("a", "n1", 2), node("b", "n1", 2), node("c", "a", 3)],
        max_nodes=12, max_hops=3,
    )
    targets = searchable_nodes(nodes, limit=6)
    assert targets[0]["order_depth"] == 3


def test_search_is_capped_and_marks_searched():
    nodes = validate_chain(
        [node("n1")] + [node(f"x{i}", "n1", 2) for i in range(10)],
        max_nodes=20, max_hops=3,
    )
    targets = searchable_nodes(nodes, limit=3)
    assert len(targets) == 3
    assert all(n["searched"] for n in targets)


def test_single_hop_chain_still_yields_a_search_target():
    nodes = validate_chain([node("n1")], max_nodes=12, max_hops=3)
    assert len(searchable_nodes(nodes, limit=6)) == 1


# ── The hallucination gate ───────────────────────────────────────────────


def test_unevidenced_company_is_dropped():
    """Models invent plausible sources; a citation not in the results is one."""
    results = {"n3": [Result("https://real.test/a")]}
    cands = [
        {"node_key": "n3", "company_name": "Real Co",
         "evidence_urls": ["https://real.test/a"]},
        {"node_key": "n3", "company_name": "Invented Co",
         "evidence_urls": ["https://fabricated.test/x"]},
    ]
    kept = filter_by_evidence(cands, results)
    assert [c["company_name"] for c in kept] == ["Real Co"]


def test_company_with_no_citations_is_dropped():
    results = {"n3": [Result("https://real.test/a")]}
    kept = filter_by_evidence([{"node_key": "n3", "evidence_urls": []}], results)
    assert kept == []


def test_evidence_from_another_node_does_not_count():
    """Citations must come from the results shown for that bottleneck."""
    results = {"n2": [Result("https://real.test/a")], "n3": [Result("https://real.test/b")]}
    kept = filter_by_evidence(
        [{"node_key": "n3", "evidence_urls": ["https://real.test/a"]}], results
    )
    assert kept == []


def test_surviving_citations_are_pruned_to_the_verified_set():
    results = {"n3": [Result("https://real.test/a")]}
    kept = filter_by_evidence(
        [{"node_key": "n3", "evidence_urls": ["https://real.test/a", "https://fake/x"]}],
        results,
    )
    assert kept[0]["evidence_urls"] == ["https://real.test/a"]


# ── Archive grounding ────────────────────────────────────────────────────


def test_archive_citations_count_as_evidence():
    """A company found in our own corpus must survive the hallucination gate.

    The gate was built when web results were the only sources a node had. Left
    unchanged it would have silently discarded every name that came from the
    archive, which is the whole point of grounding.
    """
    nodes = [{
        "node_key": "n3",
        "sources": [{"kind": "internal", "url": "https://archive.test/a"}],
    }]
    kept = filter_by_evidence(
        [{"node_key": "n3", "company_name": "Archive Co",
          "evidence_urls": ["https://archive.test/a"]}],
        {},                     # no web results at all
        nodes,
    )
    assert [c["company_name"] for c in kept] == ["Archive Co"]


def test_archive_citations_are_still_scoped_to_their_node():
    """Widening the gate must not make it node-blind."""
    nodes = [
        {"node_key": "n2",
         "sources": [{"kind": "internal", "url": "https://archive.test/a"}]},
        {"node_key": "n3", "sources": []},
    ]
    kept = filter_by_evidence(
        [{"node_key": "n3", "evidence_urls": ["https://archive.test/a"]}], {}, nodes
    )
    assert kept == []


def test_gate_still_works_without_node_context():
    """nodes is optional; the pre-grounding call shape must keep behaving."""
    results = {"n3": [Result("https://real.test/a")]}
    kept = filter_by_evidence(
        [{"node_key": "n3", "evidence_urls": ["https://real.test/a"]}], results
    )
    assert len(kept) == 1


def test_internal_sources_render_with_url_for_the_gate():
    """The prompt block must spell out the URL the gate then matches on."""
    block = te._format_internal_sources([
        {"kind": "internal", "title": "HBM supply tightens", "source": "Reuters",
         "published_at": "2026-07-01T00:00:00Z", "url": "https://archive.test/a"},
        {"kind": "web", "title": "not archive", "url": "https://web.test/b"},
    ])
    assert "https://archive.test/a" in block
    assert "Reuters" in block
    assert "2026-07-01" in block
    assert "not archive" not in block    # web hits are formatted separately


def test_grounding_runs_for_every_hop_not_just_leaves():
    """searchable_nodes gates the paid search; grounding must not inherit it.

    Hop 1 is the premise the rest of the chain hangs off, so it is exactly
    where a reader wants a citation — and it costs nothing to provide.
    """
    src = inspect.getsource(ThesisGraph.ground_node)
    assert "searchable_nodes" not in src
    assert "order_depth" not in src


@pytest.mark.asyncio
async def test_ground_node_attaches_sources_and_reemits(monkeypatch):
    db = MagicMock()
    db.find_articles_for_claim.return_value = [{
        "id": "a1", "headline": "HBM capacity tightens", "url": "https://arch/1",
        "source_name": "Reuters", "published_at": "2026-07-01T00:00:00Z",
        "similarity": 0.81,
    }]

    class _Embedder:
        def __init__(self, *a, **k):
            pass

        async def initialize(self):
            return None          # mirrors the real one, which returns None

        async def get_embeddings(self, texts):
            return [object() for _ in texts]

    monkeypatch.setattr(te, "Embedder", _Embedder)

    emitted = []

    async def on_node(n):
        emitted.append(n["node_key"])

    graph = ThesisGraph(db, node_callback=on_node)
    out = await graph.ground_node({"chain_nodes": [node("n1"), node("n2", "n1", 2)]})

    nodes = out["chain_nodes"]
    assert [n["node_key"] for n in nodes] == ["n1", "n2"]
    for n in nodes:
        assert n["sources"][0]["kind"] == "internal"
        assert n["sources"][0]["url"] == "https://arch/1"
        assert n["sources"][0]["similarity"] == 0.81
    # Hop 1 included — grounding is not restricted the way searching is.
    assert emitted == ["n1", "n2"]


@pytest.mark.asyncio
async def test_ground_node_survives_an_embedder_outage(monkeypatch):
    """No embeddings must mean no citations, not a dead thesis."""
    class _Dead:
        def __init__(self, *a, **k):
            pass

        async def initialize(self):
            return None

        async def get_embeddings(self, texts):
            return [None for _ in texts]

    monkeypatch.setattr(te, "Embedder", _Dead)
    db = MagicMock()

    out = await ThesisGraph(db).ground_node({"chain_nodes": [node("n1")]})

    assert out["chain_nodes"][0]["sources"] == []
    db.find_articles_for_claim.assert_not_called()


# ── The streamed reasoning call ──────────────────────────────────────────


class _Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Chunk:
    def __init__(self, content=None, reasoning=None, usage=None, finish_reason=None):
        # finish_reason is set explicitly rather than left to MagicMock's
        # auto-attribute, which is truthy and would make every chunk look like
        # a terminal one.
        self.choices = (
            [MagicMock(delta=_Delta(content, reasoning),
                       finish_reason=finish_reason)]
            if usage is None else []
        )
        self.usage = usage


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


async def test_streamed_reasoner_records_tokens_and_emits_thinking(monkeypatch):
    """The trailing usage chunk must reach llm_usage_log.

    This is the most expensive call in the feature, and a silently-zero token
    count would be indistinguishable from a free one.
    """
    from config.llm import StreamChunk

    usage = MagicMock(prompt_tokens=1234, completion_tokens=567, cost=0.0042)
    chunks = [
        StreamChunk(reasoning="thinking out loud "),
        StreamChunk(text='{"nodes": '),
        StreamChunk(text="[]}"),
        StreamChunk(usage=usage),
    ]

    seen_kwargs = {}

    def fake_stream(**kwargs):
        seen_kwargs.update(kwargs)

        async def gen():
            for c in chunks:
                yield c

        return gen()

    monkeypatch.setattr(te, "stream_complete", fake_stream)
    monkeypatch.setattr(te.settings, "model_thesis_reasoner", "test/reasoner")

    db = MagicMock()
    seen_thinking = []

    async def on_chunk(text):
        seen_thinking.append(text)

    graph = ThesisGraph(db, chunk_callback=on_chunk)
    out, finish_reason = await graph._call_reasoner("prompt")

    assert seen_kwargs["reasoning"] == "xhigh"
    assert out == '{"nodes": []}'
    assert finish_reason is None
    assert "".join(seen_thinking) == "thinking out loud "
    db.log_llm_usage.assert_called_once()
    logged = db.log_llm_usage.call_args.kwargs
    assert logged["prompt_tokens"] == 1234
    assert logged["candidate_tokens"] == 567
    assert logged["operation"] == "thesis_decompose"
    assert logged["cost_usd"] == pytest.approx(0.0042)


@pytest.mark.asyncio
async def test_streamed_reasoner_reports_finish_reason(monkeypatch):
    """finish_reason has to survive the stream, or truncation is invisible."""
    from config.llm import StreamChunk

    chunks = [
        StreamChunk(reasoning="thinking and thinking"),
        StreamChunk(text="", finish_reason="length"),
    ]

    def fake_stream(**kwargs):
        async def gen():
            for c in chunks:
                yield c

        return gen()

    monkeypatch.setattr(te, "stream_complete", fake_stream)
    monkeypatch.setattr(te.settings, "model_thesis_reasoner", "test/reasoner")

    async def on_chunk(_):
        pass

    out, finish_reason = await ThesisGraph(
        MagicMock(), chunk_callback=on_chunk
    )._call_reasoner("prompt")

    assert out == ""
    assert finish_reason == "length"



# ── Dead ends must announce themselves ───────────────────────────────────
#
# Each of these used to leave decompose_node returning an empty chain, which
# has_chain() routes straight to END. From outside — the dashboard, the log,
# the scheduled morning run — that is indistinguishable from a thesis that
# simply had nothing to say.


async def _decompose_with(monkeypatch, raw, finish_reason):
    """Drive decompose_node over a canned reasoner reply."""
    graph = ThesisGraph(MagicMock())
    monkeypatch.setattr(te, "is_llm_configured", lambda: True)
    monkeypatch.setattr(te.settings, "model_thesis_reasoner", "test/reasoner")

    async def fake_reasoner(_prompt):
        return raw, finish_reason

    monkeypatch.setattr(graph, "_call_reasoner", fake_reasoner)
    state: ThesisState = {"seed": {"title": "t", "summary": "s", "headlines": [],
                                   "consensus_tickers": []}}
    return await graph.decompose_node(state)


@pytest.mark.asyncio
async def test_budget_exhausted_is_reported_as_itself(monkeypatch):
    """Reasoning tokens share the completion budget with the answer.

    At xhigh the thinking block can consume all of it, leaving empty content.
    That is the failure the 4000-token default produced, and calling it a parse
    error sends the next reader to the schema instead of the token ceiling.
    """
    out = await _decompose_with(monkeypatch, "", "length")

    assert out["chain_nodes"] == []
    assert "output budget" in out["errors"][0]


@pytest.mark.asyncio
async def test_truncated_json_says_it_was_cut_off(monkeypatch):
    out = await _decompose_with(monkeypatch, '{"nodes": [{"node_key": "n1"', "length")

    assert out["chain_nodes"] == []
    assert "cut off" in out["errors"][0]


@pytest.mark.asyncio
async def test_unparseable_reply_still_reports_a_plain_parse_failure(monkeypatch):
    out = await _decompose_with(monkeypatch, "I cannot help with that.", "stop")

    assert out["chain_nodes"] == []
    assert out["errors"] == ["could not parse the chain"]


@pytest.mark.asyncio
async def test_chain_emptied_by_validation_is_an_error_not_a_silence(monkeypatch):
    """Nodes that all fail validation must not read as a successful empty run.

    A cycle, because that is the one defect validate_chain deletes outright —
    a dangling parent is re-rooted and a too-deep node is only dropped if the
    whole chain is too deep.
    """
    cyclic = ('{"nodes": ['
              '{"node_key": "n1", "parent_key": "n2", "order_depth": 1, '
              '"claim": "c1", "mechanism": "m", "falsifier": "f"}, '
              '{"node_key": "n2", "parent_key": "n1", "order_depth": 2, '
              '"claim": "c2", "mechanism": "m", "falsifier": "f"}]}')
    out = await _decompose_with(monkeypatch, cyclic, "stop")

    assert out["chain_nodes"] == []
    assert "failed chain validation" in out["errors"][0]


# ── Prompts and schemas ──────────────────────────────────────────────────


def test_decomposition_prompt_formats_without_brace_errors():
    """Literal JSON braces in a .format() template must be doubled."""
    out = THESIS_DECOMPOSITION_PROMPT.format(
        persona="p", title="t", summary="s", headlines="h", consensus="c",
        max_hops=3, max_children=3, max_nodes=12, taxonomy="tax",
        calibration="cal", json_only="j",
    )
    assert '"node_key"' in out


def test_extraction_prompt_formats_without_brace_errors():
    out = THESIS_EXTRACTION_PROMPT.format(
        persona="p", blocks="b", max_per_node=4, exposure_guidance="e", json_only="j"
    )
    assert '"company_name"' in out


def test_decomposition_prompt_pushes_past_the_first_hop():
    """The instruction the entire feature depends on."""
    lowered = THESIS_DECOMPOSITION_PROMPT.lower()
    assert "already priced in" in lowered
    assert "hops 2 and 3" in lowered
    for required in ("falsifier", "mechanism", "parent_key"):
        assert required in lowered


def test_extraction_prompt_forbids_uncited_companies():
    lowered = THESIS_EXTRACTION_PROMPT.lower()
    assert "only companies that appear in the search results" in lowered
    assert "discarded" in lowered


def test_chain_schema_parses_a_model_response():
    payload = """
    {"nodes": [{"node_key": "n1", "parent_key": null, "order_depth": 1,
     "claim": "c", "mechanism": "m", "bottleneck_type": "capacity",
     "falsifier": "f", "lead_time": "now", "confidence": 0.7}]}
    """
    chain = ThesisChain.model_validate_json(payload)
    assert chain.nodes[0].node_key == "n1"


def test_candidate_schema_tolerates_missing_optionals():
    payload = """
    {"companies": [{"node_key": "n3", "company_name": "Amkor",
     "role_in_chain": "OSAT packaging"}]}
    """
    parsed = CandidateList.model_validate_json(payload)
    assert parsed.companies[0].is_listed is True
    assert parsed.companies[0].exposure_pct is None


def test_exposure_pct_is_bounded():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CandidateList.model_validate(
            {"companies": [{"node_key": "n", "company_name": "x",
                            "role_in_chain": "r", "exposure_pct": 500}]}
        )


# ── The morning run and its start-up catch-up ────────────────────────────
#
# The catch-up used to decide from saved theses alone, so a morning whose
# paid decomposition failed to persist was re-bought on every worker restart —
# and deploy.sh restarts the worker on every push.


class _ConfigDB:
    """user_config as a dict, plus the one theses query the catch-up makes."""

    def __init__(self, theses_today=0, config=None):
        self.config = dict(config or {})
        self.theses_today = theses_today

    def get_config(self, key, default="{}"):
        return self.config.get(key, default)

    def set_config(self, key, value):
        self.config[key] = value

    def count_theses_since(self, cutoff_utc):
        return self.theses_today


def _orchestrator(db, generate):
    """Only what the thesis schedule reads, without building every tracker."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.db = db
    orch.tz = ZoneInfo(scheduler.settings.timezone)
    orch.thesis_engine = MagicMock()
    orch.thesis_engine.generate = generate
    return orch


def _today_local():
    return datetime.now(ZoneInfo(scheduler.settings.timezone)).strftime("%Y-%m-%d")


@pytest.fixture
def thesis_due_now(monkeypatch):
    """Enabled and due from midnight, so the catch-up acts whenever this runs."""
    monkeypatch.setattr(scheduler.settings, "thesis_enabled", True)
    monkeypatch.setattr(scheduler, "THESIS_GENERATION_HOUR", 0)
    monkeypatch.setattr(scheduler, "THESIS_GENERATION_MINUTE", 0)


@pytest.mark.asyncio
async def test_scheduled_run_records_the_attempt_before_generating():
    """Stamped before the paid call, so a restart mid-run cannot repeat it."""
    db = _ConfigDB()
    stamp_seen_by_generate = []

    async def generate():
        stamp_seen_by_generate.append(
            db.config.get(PipelineOrchestrator.THESIS_ATTEMPT_KEY)
        )
        return []

    await _orchestrator(db, generate).run_thesis_generation()

    assert stamp_seen_by_generate == [_today_local()]


@pytest.mark.asyncio
async def test_catchup_skips_a_day_already_attempted_with_nothing_saved(thesis_due_now):
    db = _ConfigDB(
        theses_today=0,
        config={PipelineOrchestrator.THESIS_ATTEMPT_KEY: _today_local()},
    )
    generate = AsyncMock(return_value=[])

    with patch.object(scheduler, "log") as mock_log:
        await _orchestrator(db, generate)._catchup_thesis()

    generate.assert_not_awaited()
    events = [c.args[0] for c in mock_log.info.call_args_list]
    assert "orchestrator.startup_catchup.thesis_already_attempted" in events


@pytest.mark.asyncio
async def test_catchup_runs_when_the_last_attempt_was_another_day(thesis_due_now):
    db = _ConfigDB(config={PipelineOrchestrator.THESIS_ATTEMPT_KEY: "2000-01-01"})
    generate = AsyncMock(return_value=["t1"])

    await _orchestrator(db, generate)._catchup_thesis()

    generate.assert_awaited_once()
    assert db.config[PipelineOrchestrator.THESIS_ATTEMPT_KEY] == _today_local()


@pytest.mark.asyncio
async def test_a_catchup_that_saved_nothing_is_not_repeated_on_restart(thesis_due_now):
    db = _ConfigDB(theses_today=0)
    generate = AsyncMock(return_value=[])   # the decomposition persisted nothing
    orch = _orchestrator(db, generate)

    await orch._catchup_thesis()
    await orch._catchup_thesis()            # the next deploy restarts the worker

    assert generate.await_count == 1
