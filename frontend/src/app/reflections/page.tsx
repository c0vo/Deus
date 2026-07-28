"use client";

import { useEffect, useState, useCallback } from "react";
import {
  BookOpen, CheckCircle2, XCircle, Trash2, Plus, Filter,
  Globe, Building2, Tag
} from "lucide-react";

interface Reflection {
  id: number;
  ticker: string;
  prediction_id?: number | null;
  date: string;
  lesson_learned: string;
  was_successful: boolean | number;
  scope: string;
  sector?: string | null;
  tags?: string | null;
}

type ScopeFilter = "ALL" | "ticker" | "sector" | "market";
type SuccessFilter = "ALL" | "SUCCESS" | "FAILURE";

const SCOPE_LABELS: Record<string, string> = {
  ticker: "TICKER",
  sector: "SECTOR",
  market: "MARKET",
};

const SCOPE_COLORS: Record<string, string> = {
  ticker: "border-l-terminal-signal",
  sector: "border-l-terminal-yellow",
  market: "border-l-terminal-violet",
};

export default function Reflections() {
  // Filters
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>("ALL");
  const [sectorFilter, setSectorFilter] = useState("");
  const [tickerFilter, setTickerFilter] = useState("");
  const [successFilter, setSuccessFilter] = useState<SuccessFilter>("ALL");

  // Data
  const [allReflections, setAllReflections] = useState<Reflection[]>([]);
  const [sectors, setSectors] = useState<string[]>([]);
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // Form state
  const [lessonText, setLessonText] = useState("");
  const [isSuccess, setIsSuccess] = useState(true);
  const [newScope, setNewScope] = useState<"ticker" | "sector" | "market">("ticker");
  const [newTicker, setNewTicker] = useState("AAPL");
  const [newSector, setNewSector] = useState("");
  const [customSector, setCustomSector] = useState("");
  const [useCustomSector, setUseCustomSector] = useState(false);
  const [newTags, setNewTags] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Fetch data sources
  const fetchReflections = useCallback(async (filters?: {
    scope?: string; sector?: string; ticker?: string; was_successful?: boolean;
  }) => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters?.scope) params.set("scope", filters.scope);
      if (filters?.sector) params.set("sector", filters.sector);
      if (filters?.ticker) params.set("ticker", filters.ticker);
      if (filters?.was_successful !== undefined) params.set("was_successful", String(filters.was_successful));

      const res = await fetch(`/api/reflections?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setAllReflections(data.data || []);
      }
    } catch (err) {
      console.error("Failed to fetch reflections", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSectors = useCallback(async () => {
    try {
      const res = await fetch("/api/reflections/sectors");
      if (res.ok) {
        const data = await res.json();
        setSectors(data.data || []);
      }
    } catch {}
  }, []);

  const fetchWatchlist = useCallback(async () => {
    try {
      const res = await fetch("/api/watchlist");
      if (res.ok) {
        const data = await res.json();
        const list = data.data || [];
        setWatchlist(list);
        if (list.length > 0 && !newTicker) setNewTicker(list[0]);
      }
    } catch {}
  }, [newTicker]);

  // Initial loads
  useEffect(() => {
    fetchReflections();
    fetchSectors();
    fetchWatchlist();
  }, [fetchReflections, fetchSectors, fetchWatchlist]);

  // Apply filters
  const applyFilters = () => {
    const f: any = {};
    if (scopeFilter !== "ALL") f.scope = scopeFilter;
    if (sectorFilter) f.sector = sectorFilter;
    if (tickerFilter) f.ticker = tickerFilter;
    if (successFilter !== "ALL") f.was_successful = successFilter === "SUCCESS";
    fetchReflections(f);
  };

  // Delete handler
  const handleDelete = async (id: number) => {
    if (!confirm("Delete this reflection?")) return;
    try {
      const res = await fetch(`/api/reflections/${id}`, { method: "DELETE" });
      if (res.ok) {
        setAllReflections((prev) => prev.filter((r) => r.id !== id));
      }
    } catch (err) {
      console.error("Failed to delete reflection", err);
    }
  };

  // Submit handler
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!lessonText.trim()) return;

    setSubmitting(true);
    try {
      const effectiveSector = useCustomSector ? customSector : newSector;
      const payload: any = {
        lesson_learned: lessonText.trim(),
        was_successful: isSuccess,
        scope: newScope,
        tags: newTags.trim() || null,
      };
      if (newScope === "ticker") payload.scope = "ticker";
      if (newScope === "sector") payload.sector = effectiveSector || null;
      if (newScope === "market") payload.scope = "market";

      const tickerParam = newScope === "ticker" ? newTicker : "MARKET";

      const res = await fetch(`/api/reflections/${tickerParam}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setLessonText("");
        setNewTags("");
        applyFilters();
      } else {
        alert("Failed to submit reflection.");
      }
    } catch (err) {
      console.error(err);
      alert("Error submitting reflection.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="p-6 space-y-6">

      {/* Title */}
      <div className="border-b border-border-dim pb-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Reflections
        </h1>
        <p className="text-xs text-terminal-muted">
          Log and review prediction lessons — per ticker, sector, or market-wide
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

        {/* ─── Left Column: Log Form ─── */}
        <div className="lg:col-span-1 border border-border-dim bg-bg-card p-4 space-y-4">
          <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider border-b border-border-dim pb-2 flex items-center gap-1.5">
            <Plus size={14} className="text-terminal-violet" />
            Log reflection
          </h3>
          <form onSubmit={handleSubmit} className="space-y-4 text-xs">

            {/* Scope selector */}
            <div className="space-y-1">
              <label className="block text-terminal-muted font-bold uppercase">SCOPE</label>
              <div className="flex gap-2">
                {(["ticker", "sector", "market"] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setNewScope(s)}
                    className={`flex-1 py-1.5 border text-xs font-bold uppercase transition-colors ${
                      newScope === s
                        ? "border-terminal-text text-terminal-text bg-bg-surface"
                        : "border-border-dim text-terminal-muted hover:text-terminal-text"
                    }`}
                  >
                    {SCOPE_LABELS[s]}
                  </button>
                ))}
              </div>
            </div>

            {/* Ticker selector (only when scope === ticker) */}
            {newScope === "ticker" && (
              <div className="space-y-1">
                <label className="block text-terminal-muted font-bold uppercase">TICKER</label>
                <select
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value)}
                  className="w-full bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none font-mono"
                >
                  {watchlist.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                  {watchlist.length === 0 && <option value="AAPL">AAPL</option>}
                </select>
              </div>
            )}

            {/* Sector selector (only when scope === sector) */}
            {newScope === "sector" && (
              <div className="space-y-1">
                <label className="block text-terminal-muted font-bold uppercase">SECTOR</label>
                {!useCustomSector ? (
                  <div className="flex gap-1">
                    <select
                      value={newSector}
                      onChange={(e) => setNewSector(e.target.value)}
                      className="flex-1 bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none font-mono"
                    >
                      <option value="">Select sector...</option>
                      {sectors.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => setUseCustomSector(true)}
                      className="px-2 border border-border-dim text-terminal-muted hover:text-terminal-text text-[10px]"
                      title="Custom sector"
                    >
                      Other
                    </button>
                  </div>
                ) : (
                  <div className="flex gap-1">
                    <input
                      type="text"
                      value={customSector}
                      onChange={(e) => setCustomSector(e.target.value)}
                      placeholder="e.g. Real Estate"
                      className="flex-1 bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none font-mono"
                    />
                    <button
                      type="button"
                      onClick={() => { setUseCustomSector(false); setCustomSector(""); }}
                      className="px-2 border border-border-dim text-terminal-muted hover:text-terminal-text text-[10px]"
                    >
                      Pick
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Outcome */}
            <div className="space-y-1">
              <label className="block text-terminal-muted font-bold uppercase">OUTCOME</label>
              <div className="flex items-center gap-4 bg-bg-surface p-2 border border-border-dim/50">
                <label className="flex items-center gap-1.5 cursor-pointer text-terminal-signal">
                  <input
                    type="radio"
                    checked={isSuccess === true}
                    onChange={() => setIsSuccess(true)}
                    className="accent-terminal-signal"
                  />
                  <span>SUCCESSFUL</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer text-terminal-red">
                  <input
                    type="radio"
                    checked={isSuccess === false}
                    onChange={() => setIsSuccess(false)}
                    className="accent-terminal-red"
                  />
                  <span>FAILED</span>
                </label>
              </div>
            </div>

            {/* Tags */}
            <div className="space-y-1">
              <label className="block text-terminal-muted font-bold uppercase uppercase flex items-center gap-1">
                <Tag size={10} /> TAGS (OPTIONAL)
              </label>
              <input
                type="text"
                value={newTags}
                onChange={(e) => setNewTags(e.target.value)}
                placeholder="e.g. war, macro, earnings"
                className="w-full bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none font-mono"
              />
            </div>

            {/* Lesson text */}
            <div className="space-y-1">
              <label className="block text-terminal-muted font-bold uppercase">LESSON LEARNED</label>
              <textarea
                value={lessonText}
                onChange={(e) => setLessonText(e.target.value)}
                disabled={submitting}
                placeholder={
                  newScope === "ticker" ? "e.g. RSI divergence on AAPL was a false signal during low volume..."
                  : newScope === "sector" ? "e.g. War in Ukraine affects defense sector bullishly..."
                  : "e.g. Rising rate environment broadly bearish for growth stocks..."
                }
                rows={5}
                className="w-full bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none font-sans"
                required
              />
            </div>

            <button
              type="submit"
              disabled={submitting || !lessonText.trim() || (newScope === "sector" && !newSector && !customSector)}
              className="w-full py-2 border border-terminal-violet text-terminal-violet hover:bg-terminal-violet/10 text-xs font-bold flex items-center justify-center gap-1.5 transition-all disabled:opacity-50"
            >
              <Plus size={14} />
              {submitting ? "STORING..." : "STORE REFLECTION"}
            </button>
          </form>
        </div>

        {/* ─── Right Column: Reflections Feed ─── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Filter Bar */}
          <div className="border border-border-dim bg-bg-card p-4 space-y-3">
            {/* Scope + Success filters */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <Globe size={12} className="text-terminal-muted" />
                <span className="text-[10px] text-terminal-muted uppercase font-bold">Scope:</span>
                {(["ALL", "ticker", "sector", "market"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setScopeFilter(s)}
                    className={`px-2 py-0.5 border text-[10px] uppercase ${
                      scopeFilter === s
                        ? "border-terminal-text text-terminal-text"
                        : "border-transparent text-terminal-muted hover:text-terminal-text"
                    }`}
                  >
                    {s === "ALL" ? "ALL" : SCOPE_LABELS[s]}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1.5">
                <Filter size={12} className="text-terminal-muted" />
                <span className="text-[10px] text-terminal-muted uppercase font-bold">Outcome:</span>
                {(["ALL", "SUCCESS", "FAILURE"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSuccessFilter(s)}
                    className={`px-2 py-0.5 border text-[10px] uppercase ${
                      successFilter === s
                        ? s === "SUCCESS" ? "border-terminal-green text-terminal-green"
                          : s === "FAILURE" ? "border-terminal-red text-terminal-red"
                          : "border-terminal-text text-terminal-text"
                        : "border-transparent text-terminal-muted hover:text-terminal-text"
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Sector + Ticker filters */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5">
                <Building2 size={12} className="text-terminal-muted" />
                <select
                  value={sectorFilter}
                  onChange={(e) => setSectorFilter(e.target.value)}
                  className="bg-bg-surface border border-border-dim text-[10px] text-terminal-text px-2 py-0.5 font-mono focus:outline-none"
                >
                  <option value="">All sectors</option>
                  {sectors.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-terminal-muted uppercase font-bold">Ticker:</span>
                <select
                  value={tickerFilter}
                  onChange={(e) => setTickerFilter(e.target.value)}
                  className="bg-bg-surface border border-border-dim text-[10px] text-terminal-text px-2 py-0.5 font-mono focus:outline-none"
                >
                  <option value="">All tickers</option>
                  {watchlist.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={applyFilters}
                className="px-3 py-0.5 border border-terminal-text text-terminal-text text-[10px] uppercase font-bold hover:bg-bg-surface transition-colors"
              >
                Apply
              </button>
            </div>
          </div>

          {/* Reflections List */}
          {loading ? (
            <div className="border border-border-dim bg-bg-card p-8 text-center text-xs text-terminal-muted animate-pulse">
              Querying reflection_log...
            </div>
          ) : allReflections.length === 0 ? (
            <div className="border border-border-dim bg-bg-card p-8 text-center text-xs text-terminal-muted">
              No reflections match your filters.
            </div>
          ) : (
            <div className="space-y-4">
              {allReflections.map((ref) => {
                const isSucc = Boolean(ref.was_successful);
                const scopeKey = ref.scope || "ticker";
                const colorBorder = SCOPE_COLORS[scopeKey] || "border-l-terminal-signal";

                return (
                  <div key={ref.id} className={`border border-border-dim bg-bg-card p-4 relative border-l-4 ${colorBorder}`}>
                    {/* Header row */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2 text-xs flex-wrap">
                        {/* Scope badge */}
                        <span className={`text-[9px] uppercase font-bold px-1 border ${
                          scopeKey === "ticker" ? "border-terminal-signal text-terminal-signal"
                          : scopeKey === "sector" ? "border-terminal-yellow text-terminal-yellow"
                          : "border-terminal-violet text-terminal-violet"
                        }`}>
                          {SCOPE_LABELS[scopeKey] || "TICKER"}
                        </span>

                        {/* Sector badge */}
                        {ref.sector && scopeKey !== "ticker" && (
                          <span className="text-[9px] text-terminal-yellow font-mono">
                            [{ref.sector}]
                          </span>
                        )}

                        {/* Ticker */}
                        <span className="font-bold text-terminal-text uppercase">
                          {ref.ticker}
                        </span>

                        {/* Prediction ID */}
                        {ref.prediction_id && (
                          <span className="text-[10px] text-terminal-muted">
                            (ID: {ref.prediction_id})
                          </span>
                        )}

                        {/* Tags */}
                        {ref.tags && (
                          <span className="text-[9px] text-terminal-muted font-mono">
                            #{ref.tags.replace(/,/g, " #")}
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {/* Success/Failure icon */}
                        {isSucc ? (
                          <CheckCircle2 size={12} className="text-terminal-green" />
                        ) : (
                          <XCircle size={12} className="text-terminal-red" />
                        )}
                        {/* Formatted date */}
                        <span className="text-[10px] text-terminal-muted font-mono whitespace-nowrap">
                          {ref.date ? new Date(ref.date).toLocaleDateString("en-US", {
                            year: "numeric", month: "short", day: "numeric"
                          }) : "—"}
                        </span>
                        {/* Delete button */}
                        <button
                          onClick={() => handleDelete(ref.id)}
                          className="text-terminal-muted hover:text-terminal-red transition-colors"
                          title="Delete reflection"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </div>

                    {/* Lesson text */}
                    <p className="text-xs text-terminal-text leading-relaxed font-sans whitespace-pre-wrap mt-2">
                      {ref.lesson_learned}
                    </p>
                  </div>
                );
              })}

              {/* Count summary */}
              <div className="text-[10px] text-terminal-muted text-center font-mono">
                Showing {allReflections.length} reflection{allReflections.length !== 1 ? "s" : ""}
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
