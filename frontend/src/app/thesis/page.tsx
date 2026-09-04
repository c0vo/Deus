"use client";

import { useState, useCallback, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { getApiUrl, fetchJson } from "../utils/api";
import { Waypoints, Search, Loader2, ExternalLink, Plus } from "lucide-react";

// ── Types ────────────────────────────────────────────────────────────────

type Stage = "EARLY" | "BUILDING" | "CROWDED" | "POST_NEWS" | "UNKNOWN";

interface NodeSource {
  kind: "internal" | "web";
  title: string;
  url: string;
  source: string;
  published_at?: string | null;
  article_id?: string;
  similarity?: number;
}

interface ChainNode {
  node_key: string;
  parent_key: string | null;
  order_depth: number;
  claim: string;
  mechanism: string;
  bottleneck_type: string;
  falsifier: string;
  lead_time: string;
  confidence: number;
  sources?: NodeSource[];
}

interface Candidate {
  id?: string;
  node_key?: string;
  node_id?: string;
  company_name: string;
  ticker: string | null;
  listing_status: string;
  us_proxy?: string | null;
  role_in_chain: string;
  exposure?: number;
  exposure_pct?: number;
  substitutability: string;
  conviction: number;
  crowding: number | null;
  rumour_stage: Stage | null;
  edge_score: number | null;
  data_coverage: number;
  evidence_urls?: string[];
}

interface Thesis {
  id: string;
  title: string;
  summary: string;
  acceleration: number | null;
  acceleration_basis: string;
  article_count_recent: number;
  article_count_base: number;
  node_count?: number;
  candidate_count?: number;
  nodes?: ChainNode[];
  candidates?: Candidate[];
  top_candidates?: Candidate[];
}

// ── Presentation helpers ─────────────────────────────────────────────────

// Stage maps onto the existing pill primitives in globals.css — no new CSS.
const STAGE_PILL: Record<Stage, string> = {
  EARLY: "pill pill-ok",
  BUILDING: "pill pill-signal",
  CROWDED: "pill pill-warn",
  POST_NEWS: "pill pill-bad",
  UNKNOWN: "pill",
};

const STAGE_HINT: Record<Stage, string> = {
  EARLY: "Quiet — the rumour stage",
  BUILDING: "Attention is arriving",
  CROWDED: "Largely priced in",
  POST_NEWS: "Attention rolling over — the exit",
  UNKNOWN: "Not enough data to judge",
};

const HOP_LABEL: Record<number, string> = {
  1: "1st order — already priced in",
  2: "2nd order",
  3: "3rd order",
};

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`;
}

/** Coverage is rendered explicitly everywhere a score is, because a strong
 *  edge on thin data is a guess and must not look like a measurement. */
function CoverageDots({ coverage }: { coverage: number }) {
  const filled = Math.max(1, Math.round((coverage || 0) * 3));
  return (
    <span
      className="inline-flex gap-0.5 align-middle"
      title={`Data coverage ${pct(coverage)} — how much of the crowding score is backed by real data`}
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className={`w-1 h-1 rounded-full ${
            i < filled ? "bg-terminal-muted" : "bg-border-dim"
          }`}
        />
      ))}
    </span>
  );
}

/** Citations for one hop, archive first.
 *
 *  Same "source • time" line the markets news feed uses, so a source reads the
 *  same wherever it appears. Archive rows are tagged because provenance
 *  matters here: one is an article this system ingested and scored, the other
 *  is a search hit it saw once. */
function NodeSources({ sources }: { sources?: NodeSource[] }) {
  if (!sources || sources.length === 0) return null;
  const ordered = [
    ...sources.filter((s) => s.kind === "internal"),
    ...sources.filter((s) => s.kind !== "internal"),
  ];
  return (
    <div className="pt-1.5 border-t border-border-dim space-y-1">
      <span className="label">sources</span>
      {ordered.map((s, i) => (
        <a
          key={`${s.url}-${i}`}
          href={s.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-start gap-1.5 text-[11px] group tap"
        >
          <ExternalLink
            size={10}
            strokeWidth={1.7}
            className="mt-0.5 shrink-0 text-terminal-muted-alt"
          />
          <span className="min-w-0">
            <span className="text-terminal-muted group-hover:text-terminal-text">
              {s.title || s.url}
            </span>
            <span className="block text-[10px] text-terminal-muted-alt">
              {s.source || "unknown source"}
              {s.published_at ? ` • ${formatSourceDate(s.published_at)}` : ""}
              {s.kind === "internal" ? " • archive" : ""}
            </span>
          </span>
        </a>
      ))}
    </div>
  );
}

function formatSourceDate(raw: string): string {
  const d = new Date(raw);
  return isNaN(d.getTime()) ? raw.slice(0, 10) : d.toLocaleDateString();
}

function CandidateCard({
  candidate,
  onTrack,
}: {
  candidate: Candidate;
  onTrack: (c: Candidate) => void;
}) {
  const stage = (candidate.rumour_stage || "UNKNOWN") as Stage;
  const exposure = candidate.exposure ?? candidate.exposure_pct ?? 0;
  const thin = (candidate.data_coverage || 0) < 0.35;
  const tradeable =
    candidate.listing_status === "us_listed" || candidate.listing_status === "adr";

  return (
    <div className="border border-border-dim rounded p-2.5 bg-bg-surface space-y-1.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-medium text-sm truncate">
              {candidate.company_name}
            </span>
            {candidate.ticker ? (
              <span className="num text-xs text-terminal-signal">
                {candidate.ticker}
              </span>
            ) : (
              <span
                className="tag"
                title={
                  candidate.listing_status === "foreign_unlisted"
                    ? "Not US-listed — kept because it owns the chokepoint"
                    : "No tradeable listing found"
                }
              >
                {candidate.listing_status === "private" ? "private" : "unlisted"}
              </span>
            )}
          </div>
          <p className="text-xs text-terminal-muted mt-0.5">
            {candidate.role_in_chain}
          </p>
        </div>
        <span className={STAGE_PILL[stage]} title={STAGE_HINT[stage]}>
          {stage.replace("_", " ")}
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs text-terminal-muted-alt flex-wrap">
        <span title="Estimated share of revenue exposed to this bottleneck">
          exposure <span className="num text-terminal-text">{Math.round(exposure)}%</span>
        </span>
        <span title="How replaceable this supplier is">
          {candidate.substitutability?.replace("_", " ")}
        </span>
        <span title="Conviction x (1 - crowding). Higher = strong position, still quiet.">
          edge{" "}
          <span className="num text-terminal-text">
            {candidate.edge_score === null || candidate.edge_score === undefined
              ? "—"
              : candidate.edge_score.toFixed(2)}
          </span>
        </span>
        <CoverageDots coverage={candidate.data_coverage || 0} />
        {thin && (
          <span className="text-terminal-muted-alt" title="Most crowding signals were unavailable">
            thin data
          </span>
        )}
      </div>

      {candidate.us_proxy && (
        <p className="text-xs text-terminal-muted-alt">
          nearest tradeable proxy:{" "}
          <span className="num text-terminal-signal">{candidate.us_proxy}</span>
        </p>
      )}

      <div className="flex items-center gap-2 flex-wrap">
        {(candidate.evidence_urls || []).slice(0, 2).map((url) => (
          <a
            key={url}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-terminal-muted hover:text-terminal-signal inline-flex items-center gap-1 tap"
          >
            <ExternalLink size={10} strokeWidth={1.7} />
            source
          </a>
        ))}
        {tradeable && candidate.id && (
          <button
            onClick={() => onTrack(candidate)}
            className="btn-ghost text-xs inline-flex items-center gap-1 ml-auto tap"
            title="Add to watchlist"
          >
            <Plus size={11} strokeWidth={1.7} />
            track
          </button>
        )}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────

function ThesisContent() {
  const searchParams = useSearchParams();
  const [theses, setTheses] = useState<Thesis[]>([]);
  const [selected, setSelected] = useState<Thesis | null>(null);
  const [seedInput, setSeedInput] = useState("");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [reasoning, setReasoning] = useState("");
  const [liveNodes, setLiveNodes] = useState<ChainNode[]>([]);
  const [liveCandidates, setLiveCandidates] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const loadTheses = useCallback(async () => {
    try {
      const res = await fetchJson<{ data: Thesis[] }>("/api/thesis?limit=20");
      setTheses(res.data || []);
      return res.data || [];
    } catch {
      return [];
    }
  }, []);

  const loadDetail = useCallback(async (id: string) => {
    try {
      const res = await fetchJson<{ data: Thesis }>(`/api/thesis/${id}`);
      setSelected(res.data);
      setLiveNodes([]);
      setLiveCandidates([]);
    } catch {
      setError("Could not load that thesis.");
    }
  }, []);

  useEffect(() => {
    // Guarded so a slow initial load cannot write state after unmount.
    let cancelled = false;
    (async () => {
      const list = await loadTheses();
      if (cancelled) return;
      const wanted = searchParams.get("id");
      if (wanted) await loadDetail(wanted);
      else if (list.length) await loadDetail(list[0].id);
    })();
    return () => {
      cancelled = true;
    };
  }, [loadTheses, loadDetail, searchParams]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, reasoning]);

  const runThesis = useCallback(async () => {
    const topic = seedInput.trim();
    if (!topic || running) return;

    setRunning(true);
    setError(null);
    setLogs([]);
    setReasoning("");
    setLiveNodes([]);
    setLiveCandidates([]);
    setSelected(null);

    try {
      // Hand-rolled over fetch rather than EventSource: the Next dev rewrite
      // proxy buffers streams, so getApiUrl retargets port 8000 directly.
      const response = await fetch(
        getApiUrl(`/api/thesis/stream?seed=${encodeURIComponent(topic)}`)
      );
      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response stream");
      const decoder = new TextDecoder("utf-8");
      let partial = "";
      let currentEvent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const lines = (partial + decoder.decode(value, { stream: true })).split("\n");
        partial = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.substring(7).trim();
            continue;
          }
          if (!line.startsWith("data: ")) continue;
          const raw = line.substring(6);

          if (currentEvent === "agent_update") {
            setLogs((p) => [...p, raw]);
          } else if (currentEvent === "reasoning_chunk") {
            try {
              setReasoning((p) => p + JSON.parse(raw).text);
            } catch {}
          } else if (currentEvent === "chain_node") {
            try {
              const node = JSON.parse(raw) as ChainNode;
              // Upsert, not append: decompose emits each node as soon as the
              // chain parses, then grounding and the web search re-emit the
              // same node once its citations are attached.
              setLiveNodes((p) => {
                const i = p.findIndex((n) => n.node_key === node.node_key);
                if (i === -1) return [...p, node];
                const next = [...p];
                next[i] = node;
                return next;
              });
            } catch {}
          } else if (currentEvent === "candidate") {
            try {
              setLiveCandidates((p) => [...p, JSON.parse(raw) as Candidate]);
            } catch {}
          } else if (currentEvent === "research_source") {
            try {
              const s = JSON.parse(raw);
              setLogs((p) => [...p, `  ↳ ${s.domain || s.url}`]);
            } catch {}
          } else if (currentEvent === "verdict") {
            try {
              const v = JSON.parse(raw);
              if (v.thesis_id) {
                await loadTheses();
                await loadDetail(v.thesis_id);
              }
            } catch {}
          } else if (currentEvent === "error") {
            setError(raw);
          }
          currentEvent = "";
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Thesis generation failed");
    } finally {
      setRunning(false);
    }
  }, [seedInput, running, loadTheses, loadDetail]);

  const trackCandidate = useCallback(async (c: Candidate) => {
    if (!c.id) return;
    try {
      await fetch(getApiUrl(`/api/thesis/candidates/${c.id}/track`), {
        method: "POST",
      });
      setLogs((p) => [...p, `Added ${c.ticker} to the watchlist.`]);
    } catch {
      setError(`Could not track ${c.ticker}.`);
    }
  }, []);

  const nodes = liveNodes.length ? liveNodes : selected?.nodes || [];
  const candidates = liveCandidates.length
    ? liveCandidates
    : selected?.candidates || [];

  // Chain rendering is grouped by hop depth rather than drawn as a graph: hop
  // depth is the thing that matters here (hop 1 is priced in, hops 2-3 are the
  // point), and columns stay legible on a phone where a canvas graph would not.
  const byHop = [1, 2, 3].map((depth) => ({
    depth,
    nodes: nodes.filter((n) => n.order_depth === depth),
  }));

  const candidatesForNode = (node: ChainNode) =>
    candidates.filter(
      (c) => c.node_key === node.node_key || c.node_id === node.node_key
    );

  return (
    <div className="p-4 md:p-6 space-y-4 md:space-y-6">
      <div className="border-b border-border-dim pb-4 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <Waypoints size={20} strokeWidth={1.7} />
            Thesis Engine
          </h1>
          <p className="text-xs text-terminal-muted mt-1">
            Break a developing story into the bottlenecks it creates, then find who
            sits at each one — and how much is already priced in.
          </p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-2">
        <div className="relative flex-1">
          <Search
            size={14}
            strokeWidth={1.7}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-terminal-muted"
          />
          <input
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runThesis()}
            placeholder="A developing situation — e.g. China restricts gallium exports"
            className="w-full bg-bg-surface border border-border-dim rounded pl-8 pr-3 py-2 text-sm focus:outline-none focus:border-terminal-signal"
            disabled={running}
          />
        </div>
        <button
          onClick={runThesis}
          disabled={running || !seedInput.trim()}
          className="btn-signal text-sm px-4 py-2 disabled:opacity-40 inline-flex items-center gap-2 tap"
        >
          {running && <Loader2 size={14} className="animate-spin" strokeWidth={1.7} />}
          {running ? "Reasoning…" : "Build thesis"}
        </button>
      </div>

      {error && (
        <div className="card border-terminal-red">
          <div className="card-body text-xs text-terminal-red">{error}</div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr] gap-4">
        {/* Active theses */}
        <div className="card">
          <div className="card-head">
            <span className="card-title">Active theses</span>
          </div>
          <div className="card-body space-y-1 max-h-none lg:max-h-[600px] lg:overflow-y-auto scrollbar-thin">
            {theses.length === 0 && (
              <p className="text-xs text-terminal-muted">
                None yet. One is generated each morning, or type a topic above.
              </p>
            )}
            {theses.map((t) => (
              <button
                key={t.id}
                onClick={() => loadDetail(t.id)}
                className={`w-full text-left p-2 rounded border text-xs tap ${
                  selected?.id === t.id
                    ? "border-terminal-signal bg-bg-surface"
                    : "border-border-dim hover:border-border-soft"
                }`}
              >
                <div className="font-medium truncate">{t.title}</div>
                <div className="text-terminal-muted-alt mt-0.5 flex items-center gap-2">
                  {t.acceleration !== null ? (
                    <span
                      className="num"
                      title="Share-of-voice growth vs the baseline window"
                    >
                      {t.acceleration.toFixed(1)}× accel
                    </span>
                  ) : (
                    <span title="Corpus too thin to measure acceleration">
                      accel n/a
                    </span>
                  )}
                  <span className="num">{t.candidate_count ?? 0} names</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* The chain */}
        <div className="space-y-4">
          {(running || logs.length > 0) && (
            <div className="card">
              <div className="card-head">
                <span className="card-title">Reasoning</span>
              </div>
              <div className="card-body">
                <div className="max-h-40 overflow-y-auto scrollbar-thin space-y-0.5">
                  {logs.map((l, i) => (
                    <p key={i} className="text-xs text-terminal-muted font-mono">
                      {l}
                    </p>
                  ))}
                  {reasoning && (
                    <p className="text-xs text-terminal-muted-alt whitespace-pre-wrap mt-2">
                      {reasoning.slice(-1200)}
                      {running && (
                        <span className="inline-block w-1.5 h-3 bg-terminal-signal animate-pulse ml-0.5 align-middle" />
                      )}
                    </p>
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          )}

          {selected && !running && (
            <div className="card">
              <div className="card-body">
                <h2 className="font-medium">{selected.title}</h2>
                {selected.summary && (
                  <p className="text-xs text-terminal-muted mt-1">{selected.summary}</p>
                )}
                <div className="flex gap-3 text-xs text-terminal-muted-alt mt-2 flex-wrap">
                  <span className="num">
                    {selected.article_count_recent} recent /{" "}
                    {selected.article_count_base} baseline articles
                  </span>
                  {selected.acceleration_basis === "thin" && (
                    <span title="Not enough articles in the baseline window to compute acceleration">
                      acceleration unmeasurable
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {nodes.length === 0 && !running && (
            <div className="card">
              <div className="card-body text-xs text-terminal-muted">
                Select a thesis, or enter a developing situation above to build one.
              </div>
            </div>
          )}

          {byHop.map(
            ({ depth, nodes: hopNodes }) =>
              hopNodes.length > 0 && (
                <div key={depth} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="label">{HOP_LABEL[depth] || `hop ${depth}`}</span>
                    <div className="flex-1 h-px bg-border-dim" />
                  </div>

                  {hopNodes.map((node) => (
                    <div
                      key={node.node_key}
                      className={`card ${depth === 1 ? "opacity-70" : ""}`}
                    >
                      <div className="card-body space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <p className="text-sm font-medium">{node.claim}</p>
                          <div className="flex gap-1 shrink-0">
                            {node.bottleneck_type && (
                              <span className="tag">
                                {node.bottleneck_type.replace("_", " ")}
                              </span>
                            )}
                            {node.lead_time && <span className="tag">{node.lead_time}</span>}
                          </div>
                        </div>

                        {node.mechanism && (
                          <p className="text-xs text-terminal-muted">
                            <span className="text-terminal-muted-alt">why: </span>
                            {node.mechanism}
                          </p>
                        )}
                        {node.falsifier && (
                          <p className="text-xs text-terminal-muted-alt">
                            <span>breaks if: </span>
                            {node.falsifier}
                          </p>
                        )}

                        <NodeSources sources={node.sources} />

                        {candidatesForNode(node).length > 0 && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                            {candidatesForNode(node).map((c, i) => (
                              <CandidateCard
                                key={c.id || `${c.company_name}-${i}`}
                                candidate={c}
                                onTrack={trackCandidate}
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )
          )}
        </div>
      </div>
    </div>
  );
}

export default function ThesisPage() {
  return (
    <Suspense
      fallback={
        <div className="p-6 text-xs text-terminal-muted">Loading thesis engine…</div>
      }
    >
      <ThesisContent />
    </Suspense>
  );
}
