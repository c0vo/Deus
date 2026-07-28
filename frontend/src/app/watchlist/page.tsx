"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Plus, Trash2, ChevronDown, ChevronUp, AlertCircle, RefreshCw, ExternalLink, Brain } from "lucide-react";
import FormattedText from "../components/FormattedText";
import GeneralNewsFeed from "../components/GeneralNewsFeed";
import PriceChart from "../components/PriceChart";

interface PredictionInfo {
  direction: "UP" | "DOWN" | "TRAINING";
  confidence: number;
}

interface AdvisoryInfo {
  final_advisory?: string;
  executive_summary?: string;
  _cache_date?: string;
}

interface MarketTickerData {
  ticker: string;
  sector: string;
  current_price: number;
  daily_change_pct: number;
  predictions: Record<string, PredictionInfo>;
  cached_advisory?: AdvisoryInfo;
}

interface ChartPoint {
  time: string;
  close: number;
  sma20?: number | null;
}

interface NewsItem {
  id: string;
  headline: string;
  summary?: string;
  classification_summary?: string;
  source_name: string;
  url: string;
  published_at: string;
  importance_score?: number | null;
  sentiment_score?: number | null;
  suggested_direction?: string | null;
}

interface TickerDetail {
  loading: boolean;
  chart: ChartPoint[];
  news: NewsItem[];
  error?: string | null;
}

const RANGE_TO_DAYS: Record<string, number> = {
  "5d": 5,
  "1m": 30,
  "3m": 90,
  "1y": 252,
};

const money = (value?: number | null) => `$${Number(value ?? 0).toFixed(2)}`;
const pct = (value?: number | null) => `${Number(value ?? 0) >= 0 ? "+" : ""}${Number(value ?? 0).toFixed(2)}%`;

// Client-side memory cache to persist markets data across route changes (Stale-While-Revalidate pattern)
let cachedWatchlist: string[] | null = null;
let cachedMarketsData: MarketTickerData[] | null = null;

export default function WatchlistPage() {
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [marketsData, setMarketsData] = useState<MarketTickerData[]>([]);
  const [newTicker, setNewTicker] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTickers, setExpandedTickers] = useState<Record<string, boolean>>({});
  const [chartRanges, setChartRanges] = useState<Record<string, string>>({});
  const [tickerDetails, setTickerDetails] = useState<Record<string, TickerDetail>>({});

  useEffect(() => {
    if (cachedWatchlist && cachedMarketsData) {
      setWatchlist(cachedWatchlist);
      setMarketsData(cachedMarketsData);
      setLoading(false);
      // Run background silent revalidation to get fresh indicators/prices
      silentRevalidate();
    } else {
      fetchWatchlistAndMarkets();
    }
  }, []);

  useEffect(() => {
    const hasAnyTraining = marketsData.some((item) =>
      Object.values(item.predictions || {}).some((pred) => pred.direction === "TRAINING")
    );

    if (hasAnyTraining) {
      const interval = setInterval(() => {
        silentRevalidate();
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [marketsData]);

  const silentRevalidate = async () => {
    try {
      const wlRes = await fetch("/api/watchlist");
      if (wlRes.ok) {
        const wlData = await wlRes.json();
        const list = wlData.data || [];
        setWatchlist(list);
        cachedWatchlist = list;
      }
      const mRes = await fetch("/api/markets");
      if (mRes.ok) {
        const mData = await mRes.json();
        const data = mData.data || [];
        setMarketsData(data);
        cachedMarketsData = data;
      }
    } catch (err) {
      console.error("Silent background markets revalidation failed:", err);
    }
  };

  const fetchWatchlistAndMarkets = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch Watchlist
      const wlRes = await fetch("/api/watchlist");
      if (!wlRes.ok) throw new Error("Failed to fetch watchlist");
      const wlData = await wlRes.json();
      const list = wlData.data || [];
      setWatchlist(list);
      cachedWatchlist = list;

      // Fetch Market Grid
      const mRes = await fetch("/api/markets");
      if (!mRes.ok) throw new Error("Failed to fetch market data");
      const mData = await mRes.json();
      const data = mData.data || [];
      setMarketsData(data);
      cachedMarketsData = data;
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load dashboard data");
    } finally {
      setLoading(false);
    }
  };

  const handleAddTicker = async (e: React.FormEvent) => {
    e.preventDefault();
    const ticker = newTicker.toUpperCase().trim();
    if (!ticker) return;

    try {
      const res = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      const data = await res.json();
      if (!res.ok || data.success === false) {
        throw new Error(data.detail || "Failed to add ticker");
      }
      setNewTicker("");
      fetchWatchlistAndMarkets();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleRemoveTicker = async (ticker: string) => {
    if (!confirm(`Are you sure you want to remove ${ticker} from watchlist?`)) return;

    try {
      const res = await fetch(`/api/watchlist/${ticker}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok || data.success === false) {
        throw new Error(data.detail || "Failed to remove ticker");
      }
      fetchWatchlistAndMarkets();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const fetchTickerDetail = async (ticker: string, range = chartRanges[ticker] || "1m") => {
    setTickerDetails((prev) => ({
      ...prev,
      [ticker]: { ...(prev[ticker] || { chart: [], news: [] }), loading: true, error: null },
    }));

    try {
      const days = RANGE_TO_DAYS[range] || 30;
      const [chartRes, newsRes] = await Promise.all([
        fetch(`/api/charts/${ticker}?days=${days}`),
        fetch(`/api/news/${ticker}?hours=168&limit=8`),
      ]);
      if (!chartRes.ok || !newsRes.ok) {
        throw new Error("Ticker detail endpoints failed to load.");
      }
      const chartJson = await chartRes.json();
      const newsJson = await newsRes.json();
      setTickerDetails((prev) => ({
        ...prev,
        [ticker]: {
          loading: false,
          chart: chartJson.data || [],
          news: newsJson.data || [],
          error: null,
        },
      }));
    } catch (err: any) {
      setTickerDetails((prev) => ({
        ...prev,
        [ticker]: {
          ...(prev[ticker] || { chart: [], news: [] }),
          loading: false,
          error: err.message || "Failed to load ticker detail.",
        },
      }));
    }
  };

  const handleRangeChange = (ticker: string, range: string) => {
    setChartRanges((prev) => ({ ...prev, [ticker]: range }));
    fetchTickerDetail(ticker, range);
  };

  const toggleExpand = (ticker: string) => {
    const shouldExpand = !expandedTickers[ticker];
    setExpandedTickers((prev) => ({
      ...prev,
      [ticker]: shouldExpand,
    }));
    if (shouldExpand && !tickerDetails[ticker]) {
      fetchTickerDetail(ticker);
    }
  };

  // Group tickers by sector
  const sectors: Record<string, MarketTickerData[]> = {};
  marketsData.forEach((item) => {
    const s = item.sector || "Unassigned";
    if (!sectors[s]) sectors[s] = [];
    sectors[s].push(item);
  });

  return (
    <div className="p-6 space-y-6">

      {/* Top Title Banner */}
      <div className="flex items-center justify-between border-b border-border-dim pb-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Markets
          </h1>
          <p className="text-xs text-terminal-muted">
            Tracked symbols, predictions, and detailed ticker analysis
          </p>
        </div>
        <button
          onClick={fetchWatchlistAndMarkets}
          className="flex items-center gap-2 px-3 py-1 text-xs border border-border-dim hover:border-terminal-text hover:bg-bg-card transition-all"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          REFRESH
        </button>
      </div>

      {error && (
        <div className="border border-terminal-error bg-bg-card p-3 flex items-center gap-3 text-sm text-terminal-error">
          <AlertCircle size={18} />
          <span>Error: {error}</span>
        </div>
      )}

      {/* Watchlist Quickbar & Add Symbol Form */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Watchlist Panel */}
        <div className="md:col-span-2 border border-border-dim bg-bg-card p-4">
          <h3 className="text-xs text-terminal-muted uppercase tracking-wider mb-3">
            [Tracked Symbols]
          </h3>
          <div className="flex flex-wrap gap-2">
            {watchlist.length === 0 ? (
              <span className="text-xs text-terminal-muted">No symbols tracked. Add one on the right.</span>
            ) : (
              watchlist.map((ticker) => {
                const marketInfo = marketsData.find((m) => m.ticker === ticker);
                const priceStr = marketInfo ? money(marketInfo.current_price) : "";
                const changeStr = marketInfo ? pct(marketInfo.daily_change_pct) : "";
                const changeColor = marketInfo
                  ? (marketInfo.daily_change_pct > 0 ? "text-terminal-green" : marketInfo.daily_change_pct < 0 ? "text-terminal-red" : "text-terminal-muted")
                  : "text-terminal-muted";

                return (
                  <div
                    key={ticker}
                    className="flex items-center gap-2.5 px-2.5 py-1 text-sm border border-border-dim bg-bg-surface"
                  >
                    <span className="font-bold text-terminal-text">{ticker}</span>
                    {marketInfo && (
                      <span className="font-mono text-xs text-terminal-muted">
                        {priceStr}{" "}
                        <span className={changeColor}>
                          ({changeStr})
                        </span>
                      </span>
                    )}
                    <button
                      onClick={() => handleRemoveTicker(ticker)}
                      className="text-terminal-muted hover:text-terminal-red transition-colors"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Add Ticker Form */}
        <div className="border border-border-dim bg-bg-card p-4">
          <h3 className="text-xs text-terminal-muted uppercase tracking-wider mb-3">
            [Add Ticker Symbol]
          </h3>
          <form onSubmit={handleAddTicker} className="flex gap-2">
            <input
              type="text"
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value)}
              placeholder="e.g. TSLA, NVDA"
              className="flex-1 bg-bg-surface border border-border-dim text-sm text-terminal-text px-3 py-1.5 focus:border-terminal-text focus:outline-none uppercase"
            />
            <button
              type="submit"
              className="px-3 py-1.5 rounded border border-terminal-signal text-terminal-signal hover:bg-terminal-signal/10 text-sm font-medium flex items-center gap-1.5 transition-colors"
            >
              <Plus size={16} />
              ADD
            </button>
          </form>
        </div>

      </div>
      {/* General News Section — macro/geopolitical/global market context */}
      <GeneralNewsFeed />

      {loading && marketsData.length === 0 ? (
        <div className="text-center py-10 text-sm text-terminal-muted animate-pulse">
          Loading market data and predictions…
        </div>
      ) : (
        <div className="space-y-8">
          {Object.keys(sectors).map((sectorName) => (
            <div key={sectorName} className="space-y-3">
              {/* Sector Name */}
              <h2 className="label border-b border-border-soft pb-1.5">
                {sectorName}
              </h2>

              {/* Tickers in Sector */}
              <div className="space-y-3">
                {sectors[sectorName].map((item) => {
                  const isExpanded = !!expandedTickers[item.ticker];
                  const currentPrice = Number(item.current_price ?? 0);
                  const dailyChange = Number(item.daily_change_pct ?? 0);
                  const changeColor = dailyChange >= 0 ? "text-terminal-green" : "text-terminal-red";
                  const detail = tickerDetails[item.ticker];
                  const activeRange = chartRanges[item.ticker] || "1m";

                  return (
                    <div
                      key={item.ticker}
                      className="border border-border-dim bg-bg-card hover:border-terminal-text transition-all duration-200"
                    >
                      {/* Grid Header Info Row */}
                      <div className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 text-sm">

                        <div className="flex items-center gap-4 min-w-[180px]">
                          <span className="text-lg font-bold text-terminal-text">{item.ticker}</span>
                          <span className="text-xs text-terminal-muted px-2 py-0.5 border border-border-dim">
                            {item.sector}
                          </span>
                        </div>

                        <div className="flex items-center gap-6">
                          <div>
                            <span className="text-xs text-terminal-muted block">PRICE</span>
                            <span className="font-bold text-terminal-text">{money(currentPrice)}</span>
                          </div>
                          <div>
                            <span className="text-xs text-terminal-muted block">24H CHANGE</span>
                            <span className={`font-bold ${changeColor}`}>
                              {pct(dailyChange)}
                            </span>
                          </div>
                        </div>

                        {/* 4 Horizons Predictions Grid */}
                        <div className="flex flex-wrap items-center gap-3">
                          {/* ML model source indicator */}
                          <div className="w-full mb-0.5">
                            <span className="text-[9px] text-terminal-muted/50 uppercase tracking-wider flex items-center gap-1">
                              <Brain size={10} />
                              ML Forecast &rarr;
                            </span>
                          </div>
                          {["5d", "1m", "3m", "1y"].map((horizon) => {
                            const pred = item.predictions?.[horizon];
                            if (!pred) {
                              return (
                                <div key={horizon} className="text-center px-2.5 py-1 border border-border-dim bg-bg-surface min-w-[70px]">
                                  <span className="text-[10px] text-terminal-muted block">{horizon.toUpperCase()}</span>
                                  <span className="text-xs font-bold text-terminal-muted">N/A</span>
                                </div>
                              );
                            }

                            if (pred.direction === "TRAINING") {
                              return (
                                <div key={horizon} className="text-center px-2.5 py-1 border border-terminal-violet bg-bg-surface min-w-[70px] animate-pulse flex flex-col items-center justify-center">
                                  <span className="text-[10px] text-terminal-muted block">{horizon.toUpperCase()}</span>
                                  <span className="text-[9px] font-bold text-terminal-violet tracking-wider">TRAINING...</span>
                                </div>
                              );
                            }

                            const isUp = pred.direction === "UP";
                            const badgeBg = isUp ? "border-terminal-green text-terminal-green" : "border-terminal-red text-terminal-red";

                            return (
                              <div key={horizon} className={`text-center px-2.5 py-1 border bg-bg-surface min-w-[70px] ${badgeBg}`}>
                                <span className="text-[10px] text-terminal-muted block">{horizon.toUpperCase()}</span>
                                <span className="text-xs font-bold">{pred.direction} ({(pred.confidence * 100).toFixed(0)}%)</span>
                              </div>
                            );
                          })}
                        </div>

                        {/* Collapsible Trigger */}
                        <button
                          onClick={() => toggleExpand(item.ticker)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-border-dim hover:bg-bg-surface text-terminal-text transition-colors self-start md:self-auto"
                        >
                          {isExpanded ? (
                            <>
                              <span>HIDE REPORT</span>
                              <ChevronUp size={14} />
                            </>
                          ) : (
                            <>
                              <span>SHOW REPORT</span>
                              <ChevronDown size={14} />
                            </>
                          )}
                        </button>

                      </div>

                      {/* Expandable AI Narrative Section */}
                      {isExpanded && (
                        <div className="border-t border-border-dim bg-bg-surface/50 p-4 space-y-4">

                          <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">

                            {/* Analysis Meta Info */}
                            <div className="xl:col-span-1 xl:border-r border-border-dim xl:pr-4 text-xs space-y-2 text-terminal-muted">
                              <h4 className="font-bold text-terminal-text uppercase tracking-wider mb-2">
                                [Report Info]
                              </h4>
                              <div>
                                <span className="block">REPORT SOURCE:</span>
                                <span className="text-terminal-violet font-bold">GRADIENT BOOSTING ML + GEMINI CO-AGENT DEBATE</span>
                              </div>
                              <div>
                                <span className="block">CACHE DATE:</span>
                                <span className="text-terminal-text">{item.cached_advisory?._cache_date || "LIVE GENERATED"}</span>
                              </div>
                            </div>

                            {/* Executive Summary Narrative */}
                            <div className="xl:col-span-2 space-y-2">
                              <h4 className="text-xs font-bold text-terminal-text uppercase tracking-wider">
                                [AI Executive Summary & Verdict]
                              </h4>
                              {item.cached_advisory?.final_advisory ? (
                                <div className="space-y-3">
                                  <FormattedText
                                    text={item.cached_advisory.final_advisory}
                                    className="text-sm text-terminal-text leading-relaxed font-sans"
                                  />
                                  {item.cached_advisory.executive_summary && (
                                    <FormattedText
                                      text={`Summary: ${item.cached_advisory.executive_summary}`}
                                      className="text-xs text-terminal-muted italic font-sans"
                                    />
                                  )}
                                </div>
                              ) : (
                                <div className="flex flex-col items-start gap-3">
                                  <p className="text-xs text-terminal-muted">
                                    No cached analysis report exists for this ticker. Run a live agent debate to generate one.
                                  </p>
                                  <Link
                                    href={`/predict?ticker=${item.ticker}`}
                                    className="px-3 py-1 border border-terminal-violet text-terminal-violet hover:bg-terminal-violet/10 text-xs font-bold transition-all"
                                  >
                                    RUN ANALYSIS DEBATE
                                  </Link>
                                </div>
                              )}
                            </div>

                            {/* Price Chart */}
                            <div className="xl:col-span-2 card p-3 space-y-3">
                              <div className="flex items-center justify-between gap-3">
                                <h4 className="card-title">Price</h4>
                                <div className="seg" role="tablist">
                                  {Object.keys(RANGE_TO_DAYS).map((range) => (
                                    <button
                                      key={range}
                                      role="tab"
                                      aria-selected={activeRange === range}
                                      onClick={() => handleRangeChange(item.ticker, range)}
                                    >
                                      {range}
                                    </button>
                                  ))}
                                </div>
                              </div>

                              {detail?.loading ? (
                                <div className="h-52 flex items-center justify-center text-xs text-terminal-muted animate-pulse">
                                  Loading chart and ticker news...
                                </div>
                              ) : detail?.chart?.length ? (
                                <PriceChart data={detail.chart} />
                              ) : (
                                <div className="h-52 flex items-center justify-center text-xs text-terminal-muted">
                                  {detail?.error || "No chart history available for this range."}
                                </div>
                              )}
                            </div>

                          </div>

                          <div className="border border-border-dim bg-bg-card">
                            <div className="p-3 border-b border-border-dim flex items-center justify-between gap-3">
                              <h4 className="text-xs font-bold text-terminal-text uppercase tracking-wider">
                                [Relevant News, Sentiment & Sources]
                              </h4>
                              <Link
                                href={`/chat?query=${encodeURIComponent(`Summarize recent ${item.ticker} news and sentiment`)}`}
                                className="text-[10px] border border-terminal-accent text-terminal-accent px-2 py-1 hover:bg-terminal-accent/10"
                              >
                                ASK ANALYST
                              </Link>
                            </div>
                            {detail?.news?.length ? (
                              <div className="divide-y divide-border-dim">
                                {detail.news.map((news) => {
                                  const sentiment = Number(news.sentiment_score ?? 0);
                                  const sentimentClass =
                                    sentiment > 0.2 ? "text-terminal-green" : sentiment < -0.2 ? "text-terminal-red" : "text-terminal-muted";
                                  return (
                                    <article key={news.id} className="p-3 grid grid-cols-1 lg:grid-cols-[1fr_160px] gap-3">
                                      <div className="space-y-1.5">
                                        <a
                                          href={news.url}
                                          target="_blank"
                                          rel="noreferrer"
                                          className="text-sm font-bold text-terminal-text hover:text-terminal-signal inline-flex items-start gap-1.5"
                                        >
                                          <span>{news.headline}</span>
                                          <ExternalLink size={12} className="mt-1 shrink-0" />
                                        </a>
                                        <FormattedText
                                          text={news.classification_summary || news.summary || "No summary available."}
                                          className="text-xs text-terminal-muted leading-relaxed font-sans"
                                        />
                                      </div>
                                      <div className="text-[10px] text-terminal-muted space-y-1 lg:text-right">
                                        <div>
                                          SOURCE: <span className="text-terminal-text font-bold">{news.source_name}</span>
                                        </div>
                                        <div>
                                          SENTIMENT: <span className={`font-bold ${sentimentClass}`}>{sentiment.toFixed(2)}</span>
                                        </div>
                                        <div>
                                          IMPORTANCE: <span className="text-terminal-text">{news.importance_score ?? "N/A"}</span>
                                        </div>
                                        {news.suggested_direction && (
                                          <div>
                                            DIRECTION: <span className="text-terminal-violet">{news.suggested_direction}</span>
                                          </div>
                                        )}
                                      </div>
                                    </article>
                                  );
                                })}
                              </div>
                            ) : (
                              <div className="p-4 text-xs text-terminal-muted">
                                {detail?.loading ? "Loading relevant news..." : "No recent sourced news found for this ticker in the local database."}
                              </div>
                            )}
                          </div>

                        </div>
                      )}

                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

    </div>
  );
}
