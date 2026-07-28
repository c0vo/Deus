"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw, TrendingUp, AlertCircle, MessageSquare, Play, ChevronDown, ChevronUp, ExternalLink, Newspaper } from "lucide-react";
import FormattedText from "../components/FormattedText";
import { getApiUrl } from "../utils/api";

interface TrendingArticle {
  headline: string;
  source_name: string;
  published_at: string;
  sentiment_score: number | null;
  classification_summary: string | null;
  summary: string | null;
  url: string;
  importance_score: number | null;
  suggested_direction: string | null;
}

interface TrendingTicker {
  ticker: string;
  mention_count: number;
  avg_sentiment: number;
  summary: string;
  articles?: TrendingArticle[];
}

export default function Trending() {
  const [hours, setHours] = useState<number>(24);
  const [data, setData] = useState<TrendingTicker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTickers, setExpandedTickers] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchTrending(hours);
  }, [hours]);

  const toggleExpanded = (ticker: string) => {
    setExpandedTickers((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) {
        next.delete(ticker);
      } else {
        next.add(ticker);
      }
      return next;
    });
  };

  const fetchTrending = async (h: number, forceRefresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(getApiUrl(`/api/trending?hours=${h}${forceRefresh ? "&refresh=true" : ""}`));
      if (!res.ok) {
        throw new Error("Failed to fetch trending tickers telemetry.");
      }
      const json = await res.json();
      setData(json.data || []);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load trending tickers.");
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (ts: string) => {
    try {
      const date = new Date(ts);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffH = Math.floor(diffMs / (1000 * 60 * 60));
      if (diffH < 1) {
        const diffM = Math.floor(diffMs / (1000 * 60));
        return `${diffM}m ago`;
      }
      if (diffH < 24) return `${diffH}h ago`;
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    } catch {
      return ts;
    }
  };

  const sentimentBadge = (score: number | null) => {
    if (score === null || score === undefined) return null;
    const color =
      score > 0.2
        ? "text-terminal-green border-terminal-green"
        : score < -0.2
        ? "text-terminal-red border-terminal-red"
        : "text-terminal-muted border-border-dim";
    return (
      <span className={`px-1.5 py-0.5 border text-[10px] font-bold rounded-sm ${color}`}>
        {score > 0 ? "+" : ""}{score.toFixed(2)}
      </span>
    );
  };

  return (
    <div className="p-6 space-y-6">
      {/* Title */}
      <div className="border-b border-border-dim pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Trending
          </h1>
          <p className="text-xs text-terminal-muted">
            Top mentioned tickers in the database with AI summary explanations
          </p>
        </div>

        {/* Hour filter controls */}
        <div className="flex items-center gap-3 text-xs self-start sm:self-auto">
          <span className="label">Time window</span>
          <div className="flex border border-border-dim">
            {[12, 24, 48, 72].map((h) => (
              <button
                key={h}
                onClick={() => setHours(h)}
                disabled={loading}
                className={`px-3 py-1 border-r border-border-dim last:border-0 font-bold uppercase transition-colors ${
                  hours === h
                    ? "bg-terminal-signal text-terminal-on-signal"
                    : "hover:bg-bg-surface text-terminal-text"
                }`}
              >
                {h}H
              </button>
            ))}
          </div>

          <button
            onClick={() => fetchTrending(hours, true)}
            className="flex items-center gap-2 px-3 py-1.5 border border-border-dim hover:border-terminal-text hover:bg-bg-card transition-all"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            REFRESH
          </button>
        </div>
      </div>

      {error && (
        <div className="border border-terminal-error bg-bg-card p-3 flex items-center gap-3 text-xs text-terminal-error">
          <AlertCircle size={16} />
          <span>Error: {error}</span>
        </div>
      )}

      {loading ? (
        <div className="text-center py-12 text-xs text-terminal-muted animate-pulse">
          Calculating mentions and generating explanations…
        </div>
      ) : data.length === 0 ? (
        <div className="border border-border-dim bg-bg-card p-10 text-center text-xs text-terminal-muted">
          No trending tickers logged in the last {hours} hours.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {data.map((item, idx) => {
            const sentiment = item.avg_sentiment;
            const sentimentColor =
              sentiment > 0.2
                ? "text-terminal-green border-terminal-green"
                : sentiment < -0.2
                ? "text-terminal-red border-terminal-red"
                : "text-terminal-muted border-border-dim";

            const sentimentLabel = sentiment > 0.2 ? "Bullish" : sentiment < -0.2 ? "Bearish" : "Neutral";
            const isExpanded = expandedTickers.has(item.ticker);
            const articleCount = item.articles?.length || 0;

            return (
              <div
                key={item.ticker}
                className="border border-border-dim bg-bg-card hover:border-terminal-text transition-all duration-200"
              >
                <div className="p-4">
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                    {/* Ticker Rank, Name, Mentions */}
                    <div className="flex flex-wrap items-center gap-4">
                      <span className="num text-terminal-muted-alt font-medium text-lg">
                        #{idx + 1}
                      </span>
                      <span className="num text-xl font-medium tracking-tight">
                        ${item.ticker}
                      </span>
                      <span className="tag">
                        <span className="num">{item.mention_count}</span> mentions
                      </span>
                      <span className={`px-2 py-0.5 border text-xs font-bold rounded-sm ${sentimentColor}`}>
                        {sentimentLabel} <span className="num">{sentiment.toFixed(2)}</span>
                      </span>
                    </div>

                    {/* Quick Action buttons */}
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/predict?ticker=${item.ticker}`}
                        className="px-3 py-1 rounded border border-terminal-signal text-terminal-signal hover:bg-terminal-signal/10 text-xs font-medium flex items-center gap-1.5 transition-colors"
                      >
                        <Play size={12} />
                        Run debate
                      </Link>
                      <Link
                        href={`/chat?query=${encodeURIComponent(`Why is ${item.ticker} trending in the news?`)}`}
                        className="px-3 py-1 rounded border border-border-dim text-terminal-muted hover:bg-bg-surface hover:text-terminal-text text-xs font-medium flex items-center gap-1.5 transition-colors"
                      >
                        <MessageSquare size={12} />
                        Ask analyst
                      </Link>
                    </div>
                  </div>

                  {/* AI Explanation / Context Summary */}
                  <div className="mt-4 pt-3 border-t border-border-dim/40 space-y-1">
                    <span className="label block">
                      AI trend rationale
                    </span>
                    <FormattedText
                      text={item.summary}
                      className="text-sm text-terminal-text font-sans leading-relaxed"
                    />
                  </div>

                  {/* Expandable Articles Section */}
                  {articleCount > 0 && (
                    <div className="mt-3 pt-3 border-t border-border-dim/30">
                      <button
                        onClick={() => toggleExpanded(item.ticker)}
                        className="w-full flex items-center justify-between text-xs font-bold text-terminal-muted uppercase tracking-wider hover:text-terminal-text transition-colors group"
                      >
                        <span className="flex items-center gap-1.5">
                          <Newspaper size={12} className="text-terminal-muted-alt" />
                          Top {articleCount} Related Articles
                        </span>
                        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </button>

                      {isExpanded && (
                        <div className="mt-3 divide-y divide-border-dim border border-border-dim bg-bg-card">
                          {item.articles!.map((article, aIdx) => {
                            const sentimentVal = Number(article.sentiment_score ?? 0);
                            const sentimentClass =
                              sentimentVal > 0.2 ? "text-terminal-green" : sentimentVal < -0.2 ? "text-terminal-red" : "text-terminal-muted";
                            const directionClass =
                              article.suggested_direction === "bullish"
                                ? "text-terminal-green"
                                : article.suggested_direction === "bearish"
                                ? "text-terminal-red"
                                : "text-terminal-muted";

                            return (
                              <article
                                key={aIdx}
                                className="p-3 grid grid-cols-1 lg:grid-cols-[1fr_160px] gap-3 hover:bg-bg-surface/30 transition-all"
                              >
                                <div className="space-y-1.5">
                                  <a
                                    href={article.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-sm font-bold text-terminal-text hover:text-terminal-signal inline-flex items-start gap-1.5"
                                  >
                                    <span>{article.headline}</span>
                                    <ExternalLink size={12} className="mt-1 shrink-0" />
                                  </a>
                                  <FormattedText
                                    text={article.classification_summary || article.summary || "No summary available."}
                                    className="text-xs text-terminal-muted leading-relaxed font-sans"
                                  />
                                </div>
                                <div className="text-[10px] text-terminal-muted space-y-1 lg:text-right">
                                  <div>
                                    SOURCE: <span className="text-terminal-text font-bold">{article.source_name}</span>
                                  </div>
                                  <div>
                                    SENTIMENT: <span className={`font-bold ${sentimentClass}`}>{sentimentVal.toFixed(2)}</span>
                                  </div>
                                  <div>
                                    IMPORTANCE: <span className="text-terminal-text">{article.importance_score ?? "N/A"}</span>
                                  </div>
                                  {article.suggested_direction && (
                                    <div>
                                      DIRECTION: <span className={directionClass}>{article.suggested_direction}</span>
                                    </div>
                                  )}
                                  <div>
                                    PUBLISHED: <span className="text-terminal-text">{formatTimestamp(article.published_at)}</span>
                                  </div>
                                </div>
                              </article>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
