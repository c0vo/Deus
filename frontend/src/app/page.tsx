"use client";

import { useState, useCallback, useEffect, Suspense } from "react";
import { getApiUrl } from "./utils/api";
import { useSystemStatus } from "./hooks/useSystemStatus";
import TickerTape from "./components/TickerTape";
import StatStrip, { StatTile } from "./components/StatStrip";
import SectorHeatmap from "./components/SectorHeatmap";
import LiveNewsFeed from "./components/LiveNewsFeed";
import SmartMoneyPanel from "./components/SmartMoneyPanel";
import EventsCalendar from "./components/EventsCalendar";
import MacroThemesCard from "./components/MacroThemesCard";
import TrendForecaster from "./components/TrendForecaster";
import IPOWatchlist from "./components/IPOWatchlist";
import PipelineTelemetry from "./components/PipelineTelemetry";
import { useBrainSSE, BrainSnapshot } from "./brain/hooks/useBrainSSE";
import { AlertCircle } from "lucide-react";

function DashboardContent() {
  const [snapshot, setSnapshot] = useState<BrainSnapshot | null>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [ipos, setIpos] = useState<any[]>([]);
  const [themes, setThemes] = useState<any[]>([]);
  const [themesLoading, setThemesLoading] = useState(false);
  const [pipelineMetrics, setPipelineMetrics] = useState<any[]>([]);
  const [liveArticles, setLiveArticles] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [sseConnected, setSseConnected] = useState(false);

  const { status, getDbSizeStr } = useSystemStatus();

  // SSE connection
  const onSnapshot = useCallback((data: BrainSnapshot) => {
    setSnapshot(data);
    setPipelineMetrics(data.pipeline_metrics || []);
    setLiveArticles(data.articles || []);
    setSseConnected(true);
    setError(null);
  }, []);

  const onNewArticles = useCallback((data: any) => {
    const newArts = data.articles || [data];
    setLiveArticles((prev) => [...newArts, ...prev].slice(0, 100));
  }, []);

  const onPipelineStatus = useCallback((data: any) => {
    setPipelineMetrics((prev) => [data, ...prev].slice(0, 20));
    setIsRunning(true);
    setTimeout(() => setIsRunning(false), 10000);
  }, []);

  const onSectorHeatmap = useCallback((data: any) => {
    setSnapshot((prev) => (prev ? { ...prev, sector_heatmap: data } : prev));
  }, []);

  const { connect } = useBrainSSE({
    onSnapshot,
    onNewArticles,
    onPipelineStatus,
    onSectorHeatmap,
    // The event tracker publishes on its own 6-hourly schedule, so the card
    // refreshes on the push rather than waiting out the 5-minute poll.
    onEventsUpdated: () => fetchEvents(),
    onError: (msg) => {
      setError(msg);
      setSseConnected(false);
    },
  });

  useEffect(() => {
    const timeout = setTimeout(() => connect(), 500);
    return () => clearTimeout(timeout);
  }, [connect]);

  // Events
  const fetchEvents = useCallback(async () => {
    try {
      const res = await fetch(getApiUrl("/api/brain/events"));
      if (res.ok) {
        const json = await res.json();
        setEvents(json.data || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchEvents]);

  // IPOs
  const fetchIpos = useCallback(async () => {
    try {
      const res = await fetch(getApiUrl("/api/brain/ipos"));
      if (res.ok) {
        const json = await res.json();
        setIpos(json.data || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchIpos();
    const interval = setInterval(fetchIpos, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchIpos]);

  const handleRemoveEvent = useCallback(async (id: number) => {
    if (!confirm("Dismiss this event?")) return;
    try {
      const res = await fetch(getApiUrl(`/api/brain/events/${id}`), {
        method: "DELETE",
      });
      if (res.ok) setEvents((prev) => prev.filter((e) => e.id !== id));
    } catch {}
  }, []);

  const handleRemoveIpo = useCallback(async (id: number) => {
    if (!confirm("Remove this IPO from the watchlist?")) return;
    try {
      const res = await fetch(getApiUrl(`/api/brain/ipos/${id}`), {
        method: "DELETE",
      });
      if (res.ok) setIpos((prev) => prev.filter((ipo) => ipo.id !== id));
    } catch {}
  }, []);

  // Macro themes
  const fetchThemes = useCallback(async (forceRefresh = false) => {
    setThemesLoading(true);
    try {
      const url = forceRefresh
        ? getApiUrl("/api/brain/macro-themes?refresh=true")
        : getApiUrl("/api/brain/macro-themes");
      const res = await fetch(url);
      if (res.ok) {
        const json = await res.json();
        setThemes(json.data || []);
      }
    } catch {}
    setThemesLoading(false);
  }, []);

  useEffect(() => {
    fetchThemes();
  }, [fetchThemes]);

  const handleFetchForecast = useCallback(async (sector: string) => {
    const res = await fetch(
      getApiUrl(`/api/brain/trend-forecast/${encodeURIComponent(sector)}`)
    );
    if (!res.ok) throw new Error("Could not load the forecast. Try again.");
    const json = await res.json();
    return json.data;
  }, []);

  const sectors = snapshot?.sector_heatmap
    ? snapshot.sector_heatmap.map((s: any) => s.sector)
    : [];

  // ── KPI strip ─────────────────────────────────────────────────────────────
  // Every tile reads from data we actually have; nothing here is estimated.
  const lastMetric = pipelineMetrics.length ? pipelineMetrics[0] : null;
  const dist = snapshot?.sentiment_distribution;
  const skew =
    dist && dist.total > 0 ? (dist.bullish - dist.bearish) / dist.total : null;

  // pipelineMetrics arrives newest-first; a sparkline reads oldest to newest.
  const cycleSeries = pipelineMetrics
    .map((m) => m?.cycle_duration_seconds)
    .filter((n): n is number => typeof n === "number")
    .slice(0, 12)
    .reverse();

  const tiles: StatTile[] = [
    {
      label: "Articles indexed",
      value: status.total_articles ? status.total_articles.toLocaleString() : "—",
      note: status.online ? undefined : "offline",
      tone: "neutral",
    },
    {
      label: "Sentiment skew",
      value: skew === null ? "—" : `${skew >= 0 ? "+" : "−"}${Math.abs(skew).toFixed(2)}`,
      note: skew === null ? undefined : skew >= 0 ? "Bullish" : "Bearish",
      tone: skew === null ? "neutral" : skew >= 0 ? "up" : "down",
    },
    {
      label: "Cycle time",
      value: lastMetric?.cycle_duration_seconds
        ? `${lastMetric.cycle_duration_seconds.toFixed(1)}s`
        : "—",
      note: lastMetric ? "last cycle" : undefined,
      tone: "neutral",
      series: cycleSeries.length > 1 ? cycleSeries : undefined,
    },
    {
      // Coverage sat at ~100% because duplicates were kept, so it never said
      // anything. What's actually useful is how much of the intake is unique.
      label: "Duplicates caught",
      value:
        snapshot?.embedding_status?.duplicate_articles !== undefined
          ? snapshot.embedding_status.duplicate_articles.toLocaleString()
          : "—",
      note:
        snapshot?.embedding_status?.duplicate_articles && status.total_articles
          ? `${((snapshot.embedding_status.duplicate_articles / status.total_articles) * 100).toFixed(1)}% of intake`
          : undefined,
      tone: "neutral",
    },
  ];

  return (
    <div className="flex flex-col">
      <TickerTape />

      <div className="p-5 flex flex-col gap-4">
        {/* Title */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
            <p className="mt-0.5 text-[13px] text-terminal-muted">
              Live surveillance, sector intelligence, and real-time analytics
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`pill ${sseConnected ? "pill-ok" : "pill-bad"}`}>
              <span className="pill-dot" />
              {sseConnected ? "Streaming" : "Disconnected"}
            </span>
            <span className={`pill ${isRunning ? "pill-signal" : ""}`}>
              {isRunning ? "Pipeline running" : "Pipeline idle"}
            </span>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 px-3.5 py-2.5 rounded border border-terminal-red/40 bg-terminal-red-soft text-xs text-terminal-red">
            <AlertCircle size={14} strokeWidth={1.7} />
            <span>{error} — reconnecting automatically.</span>
          </div>
        )}

        <StatStrip tiles={tiles} />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-7">
            <SectorHeatmap sectors={snapshot?.sector_heatmap || []} />
          </div>
          <div className="lg:col-span-5">
            <LiveNewsFeed initialArticles={liveArticles} />
          </div>
        </div>

        {/* Sits directly under the classified-news feed so disclosed
            institutional conviction reads against news sentiment. */}
        <SmartMoneyPanel />

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          <EventsCalendar events={events} onRemove={handleRemoveEvent} />
          <IPOWatchlist ipos={ipos} onRemove={handleRemoveIpo} />
          <MacroThemesCard
            themes={themes}
            loading={themesLoading}
            onRefresh={() => fetchThemes(true)}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-7">
            <TrendForecaster
              sectors={sectors}
              onFetchForecast={handleFetchForecast}
            />
          </div>
          <div className="lg:col-span-5">
            <PipelineTelemetry
              pipelineMetrics={pipelineMetrics}
              embeddingStatus={snapshot?.embedding_status}
              sentimentDistribution={snapshot?.sentiment_distribution}
              dbSizeLabel={getDbSizeStr(status.db_size_bytes)}
              totalArticles={status.total_articles}
              isRunning={isRunning}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  return (
    <Suspense
      fallback={
        <div className="p-5 text-sm text-terminal-muted">Loading dashboard…</div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
