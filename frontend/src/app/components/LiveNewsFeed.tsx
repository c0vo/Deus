"use client";

import { useState, useEffect, useRef } from "react";

interface NewsArticle {
  id: string;
  headline: string;
  source_name: string;
  published_at: string;
  importance_score: number | null;
  sentiment_score: number | null;
  event_type?: string | null;
  suggested_direction?: string | null;
  classification_summary?: string | null;
  affected_sectors?: string | null;
  url?: string | null;
}

interface LiveNewsFeedProps {
  initialArticles?: NewsArticle[];
}

function parseSectors(raw: string | null | undefined | string[]): string[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** Importance becomes a left severity stripe — readable before any text is. */
function severityClass(score: number | null | undefined): string {
  if (score === null || score === undefined) return "bg-border-dim";
  if (score >= 8.0) return "bg-terminal-yellow";
  if (score >= 5.0) return "bg-terminal-muted-alt";
  return "bg-border-dim";
}

function sentimentTag(score: number | null | undefined) {
  if (score === null || score === undefined) return null;
  if (score > 0.15) return <span className="tag tag-up">Bullish</span>;
  if (score < -0.15) return <span className="tag tag-down">Bearish</span>;
  return <span className="tag">Neutral</span>;
}

export default function LiveNewsFeed({ initialArticles = [] }: LiveNewsFeedProps) {
  const [articles, setArticles] = useState<NewsArticle[]>(initialArticles);
  const [autoScroll, setAutoScroll] = useState(true);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setArticles(initialArticles);
  }, [initialArticles]);

  useEffect(() => {
    if (autoScroll && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [articles, autoScroll]);

  return (
    <div className="card h-full">
      <div className="card-head">
        <h3 className="card-title">
          <span className="live-dot text-terminal-green" />
          Live news
        </h3>
        <div className="flex items-center gap-2">
          <span className="num text-[11px] text-terminal-muted-alt">
            {articles.length}
          </span>
          <button
            type="button"
            onClick={() => setAutoScroll(!autoScroll)}
            aria-pressed={autoScroll}
            className={`btn-ghost ${
              autoScroll ? "text-terminal-signal border-terminal-signal" : ""
            }`}
          >
            Auto
          </button>
        </div>
      </div>

      <div ref={feedRef} className="flex-1 overflow-y-auto max-h-96">
        {articles.length === 0 ? (
          <div className="p-4 text-center text-xs text-terminal-muted">
            Waiting for news ingestion…
          </div>
        ) : (
          articles.map((art, idx) => {
            const sectors = parseSectors(art.affected_sectors);
            const body = (
              <>
                <span
                  className={`absolute left-0 top-0 bottom-0 w-0.5 ${severityClass(
                    art.importance_score
                  )}`}
                />
                <p className="text-[13px] font-medium leading-snug text-pretty">
                  {art.headline}
                </p>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-terminal-muted-alt">
                  <span>{art.source_name}</span>
                  <span className="opacity-50">/</span>
                  <span className="num">
                    {new Date(art.published_at).toLocaleTimeString("en-GB", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                  <span className="opacity-50">/</span>
                  {sentimentTag(art.sentiment_score)}
                  {art.event_type && <span className="tag">{art.event_type}</span>}
                </div>

                {art.classification_summary && (
                  <p className="mt-1.5 text-[11px] leading-relaxed text-terminal-muted">
                    {art.classification_summary}
                  </p>
                )}

                {sectors.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {sectors.map((s) => (
                      <span key={s} className="tag">
                        {s}
                      </span>
                    ))}
                  </div>
                )}
              </>
            );

            const className =
              "relative block border-b border-border-soft last:border-b-0 py-2.5 pl-4 pr-3.5 hover:bg-bg-surface transition-colors";

            return art.url ? (
              <a
                key={art.id || idx}
                href={art.url}
                target="_blank"
                rel="noopener noreferrer"
                className={className}
              >
                {body}
              </a>
            ) : (
              <div key={art.id || idx} className={className}>
                {body}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
