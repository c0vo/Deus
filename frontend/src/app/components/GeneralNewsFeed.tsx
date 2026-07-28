"use client";

import { useState, useEffect } from "react";

interface GeneralArticle {
  id: string;
  headline: string;
  summary?: string;
  classification_summary?: string;
  source_name: string;
  url: string;
  published_at: string;
  importance_score: number | null;
  sentiment_score: number | null;
  urgency?: string | null;
  suggested_direction?: string | null;
  event_type?: string | null;
  affected_sectors?: string | null;
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function parseSectors(raw: string | null | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function getSentimentBadge(score: number | null) {
  if (score === null || score === undefined) return null;
  if (score > 0.15) return { label: "BULL", class: "text-terminal-green border-terminal-green" };
  if (score < -0.15) return { label: "BEAR", class: "text-terminal-red border-terminal-red" };
  return { label: "NEUT", class: "text-terminal-muted border-terminal-muted" };
}

function getImportanceBadge(score: number | null) {
  if (score === null || score === undefined) return null;
  if (score >= 8.0) return { label: "HIGH", class: "text-terminal-yellow border-terminal-yellow" };
  if (score >= 5.0) return { label: "MED", class: "text-terminal-muted border-terminal-muted" };
  return { label: "LOW", class: "text-terminal-muted border-border-dim/50" };
}

function getEventTypeLabel(eventType: string | null | undefined) {
  if (!eventType) return null;
  const colors: Record<string, string> = {
    macro: "text-terminal-violet border-terminal-violet",
    geopolitical: "text-terminal-yellow border-terminal-yellow",
    general: "text-terminal-muted border-terminal-muted",
  };
  const cls = colors[eventType.toLowerCase()] || "text-terminal-muted border-terminal-muted";
  return (
    <span className={`border px-1 text-[9px] uppercase ${cls}`}>
      {eventType}
    </span>
  );
}

export default function GeneralNewsFeed() {
  const [articles, setArticles] = useState<GeneralArticle[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [initialLoad, setInitialLoad] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/news/general?hours=168&limit=10");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) {
          setArticles(json.data || []);
          setError(null);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch");
        }
      } finally {
        if (!cancelled) {
          setInitialLoad(false);
        }
      }
    }

    load();

    const interval = setInterval(load, 300_000); // 5 min auto-refresh
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="border border-border-dim bg-bg-card">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border-dim">
        <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-terminal-violet animate-pulse" />
          General news
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-terminal-muted">{articles.length} articles</span>
        </div>
      </div>

      {/* Error banner — shown above content, does not replace it */}
      {error && (
        <div className="px-3 py-2 text-[10px] text-terminal-red bg-terminal-red/5 border-b border-terminal-red/20">
          Error refreshing: {error}
        </div>
      )}

      {/* Content */}
      {initialLoad ? (
        <div className="p-6 text-center text-xs text-terminal-muted animate-pulse">
          Loading macro, geopolitical, and general market news...
        </div>
      ) : articles.length === 0 ? (
        <div className="p-6 text-center text-xs text-terminal-muted">
          No macro or general news articles available yet. Articles tagged as macro, geopolitical, or
          general by the pipeline will appear here.
        </div>
      ) : (
        <div className="divide-y divide-border-soft text-[11px]">
          {articles.map((art) => {
            const sentimentBadge = getSentimentBadge(art.sentiment_score);
            const importanceBadge = getImportanceBadge(art.importance_score);
            const sectors = parseSectors(art.affected_sectors);

            return (
              <div key={art.id} className="transition-colors">
                <div className="p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <span className="text-terminal-text font-bold leading-tight block">
                        {art.headline}
                      </span>
                      <div className="flex items-center gap-1.5 mt-1.5 text-[10px] flex-wrap">
                        <span className="text-terminal-muted">{art.source_name}</span>
                        <span className="text-terminal-muted">•</span>
                        <span className="text-terminal-muted">
                          {formatTime(art.published_at)}
                        </span>
                        {sentimentBadge && (
                          <span className={`border px-1 ${sentimentBadge.class}`}>
                            {sentimentBadge.label}
                          </span>
                        )}
                        {importanceBadge && (
                          <span className={`border px-1 ${importanceBadge.class}`}>
                            {importanceBadge.label}
                          </span>
                        )}
                        {getEventTypeLabel(art.event_type)}
                      </div>

                      {/* Classification summary shown inline */}
                      {art.classification_summary && (
                        <p className="mt-2 text-xs text-terminal-muted leading-relaxed border-l-2 border-terminal-violet/40 pl-3">
                          {art.classification_summary}
                        </p>
                      )}
                    </div>
                  </div>

                  {sectors.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {sectors.map((s) => (
                        <span
                          key={s}
                          className="text-[9px] border border-border-dim px-1.5 py-0.5 text-terminal-muted"
                        >
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                  {art.url && (
                    <a
                      href={art.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-block mt-2 text-[10px] text-terminal-green hover:text-terminal-green/80 underline underline-offset-2"
                    >
                      Read full article →
                    </a>
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
