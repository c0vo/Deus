"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { getApiUrl } from "../utils/api";
import { AlertCircle, ExternalLink } from "lucide-react";

interface BriefingArticle {
  headline: string;
  summary: string | null;
  classification_summary: string | null;
  importance_score: number | null;
  url: string | null;
  affected_sectors: string | null;
  affected_tickers: string | null;
  source_name: string;
  published_at: string;
  sentiment_score: number | null;
  suggested_direction: string | null;
}

const WINDOWS = [
  { label: "24h", hours: 24 },
  { label: "48h", hours: 48 },
  { label: "7d", hours: 168 },
] as const;

function parseList(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function sentimentTag(score: number | null) {
  if (score === null || score === undefined) return null;
  if (score > 0.15) return <span className="tag tag-up">Bullish</span>;
  if (score < -0.15) return <span className="tag tag-down">Bearish</span>;
  return <span className="tag">Neutral</span>;
}

/** Importance drives a left severity stripe, as in the live feed. */
function severityClass(score: number | null): string {
  if (score === null || score === undefined) return "bg-border-dim";
  if (score >= 8.0) return "bg-terminal-yellow";
  if (score >= 5.0) return "bg-terminal-muted-alt";
  return "bg-border-dim";
}

function BriefingContent() {
  const [sections, setSections] = useState<Record<string, BriefingArticle[]>>({});
  const [hours, setHours] = useState<number>(24);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBriefing = useCallback(async (windowHours: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        getApiUrl(`/api/briefing?hours=${windowHours}&limit=40`)
      );
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const json = await res.json();
      setSections(json.data || {});
    } catch (err: any) {
      setError(err.message || "Could not load the briefing.");
      setSections({});
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBriefing(hours);
  }, [fetchBriefing, hours]);

  // Busiest sector first — that ordering is itself a signal.
  const sectors = Object.keys(sections).sort(
    (a, b) => (sections[b]?.length || 0) - (sections[a]?.length || 0)
  );
  const totalArticles = sectors.reduce(
    (sum, s) => sum + (sections[s]?.length || 0),
    0
  );

  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Briefing</h1>
          <p className="mt-0.5 text-[13px] text-terminal-muted">
            The same digest the bot sends each morning, grouped by sector and
            ranked by importance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="seg" role="tablist">
            {WINDOWS.map((w) => (
              <button
                key={w.label}
                role="tab"
                aria-selected={hours === w.hours}
                onClick={() => setHours(w.hours)}
              >
                {w.label}
              </button>
            ))}
          </div>
          <span className="pill">
            <span className="num">{totalArticles}</span> stories
          </span>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded border border-terminal-red/40 bg-terminal-red-soft text-xs text-terminal-red">
          <AlertCircle size={14} strokeWidth={1.7} />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="card card-body text-center text-xs text-terminal-muted">
          Assembling the briefing…
        </div>
      ) : sectors.length === 0 ? (
        <div className="card card-body text-center text-xs text-terminal-muted">
          No ranked stories in the last {hours >= 168 ? "7 days" : `${hours} hours`}.
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 items-start">
          {sectors.map((sector) => (
            <section key={sector} className="card">
              <div className="card-head">
                <h2 className="card-title">{sector}</h2>
                <span className="num text-[11px] text-terminal-muted-alt">
                  {sections[sector].length}
                </span>
              </div>

              <div className="flex flex-col">
                {sections[sector].map((article, idx) => {
                  const tickers = parseList(article.affected_tickers);
                  return (
                    <article
                      key={`${article.url || article.headline}-${idx}`}
                      className="relative border-b border-border-soft last:border-b-0 py-2.5 pl-4 pr-3.5 hover:bg-bg-surface transition-colors"
                    >
                      <span
                        className={`absolute left-0 top-0 bottom-0 w-0.5 ${severityClass(
                          article.importance_score
                        )}`}
                      />

                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-[13px] font-medium leading-snug text-pretty">
                          {article.url ? (
                            <a
                              href={article.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="hover:text-terminal-signal inline-flex items-start gap-1.5"
                            >
                              <span>{article.headline}</span>
                              <ExternalLink
                                size={11}
                                strokeWidth={1.7}
                                className="mt-1 shrink-0 opacity-60"
                              />
                            </a>
                          ) : (
                            article.headline
                          )}
                        </h3>
                        {article.importance_score !== null && (
                          <span className="num shrink-0 text-[11px] text-terminal-muted-alt">
                            {article.importance_score.toFixed(1)}
                          </span>
                        )}
                      </div>

                      {article.classification_summary && (
                        <p className="mt-1.5 text-[11px] leading-relaxed text-terminal-muted">
                          {article.classification_summary}
                        </p>
                      )}

                      <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-terminal-muted-alt">
                        <span>{article.source_name}</span>
                        <span className="opacity-50">/</span>
                        <span className="num">
                          {new Date(article.published_at).toLocaleString("en-GB", {
                            day: "2-digit",
                            month: "short",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                        {sentimentTag(article.sentiment_score) && (
                          <>
                            <span className="opacity-50">/</span>
                            {sentimentTag(article.sentiment_score)}
                          </>
                        )}
                        {tickers.slice(0, 4).map((t) => (
                          <span key={t} className="tag num">
                            {t}
                          </span>
                        ))}
                      </div>
                    </article>
                  );
                })}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

export default function BriefingPage() {
  return (
    <Suspense
      fallback={
        <div className="p-5 text-sm text-terminal-muted">Loading briefing…</div>
      }
    >
      <BriefingContent />
    </Suspense>
  );
}
