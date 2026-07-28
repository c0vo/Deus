"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { getApiUrl } from "../utils/api";
import NewsGlobe, { GeoCountry } from "../components/NewsGlobe";
import { AlertCircle, ExternalLink } from "lucide-react";

interface GeoArticle {
  id: string;
  headline: string;
  source_name: string;
  published_at: string;
  countries: string[];
  sentiment_score: number | null;
  importance_score: number | null;
  event_type: string | null;
  url: string | null;
}

interface GeoPayload {
  countries: GeoCountry[];
  recent: GeoArticle[];
  window_hours: number;
  total_tagged: number;
  untagged_backlog: number;
}

const WINDOWS = [
  { label: "6h", hours: 6 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 168 },
] as const;

function sentimentColour(score: number | null | undefined): string {
  if (score === null || score === undefined) return "text-terminal-muted-alt";
  if (score > 0.15) return "text-terminal-green";
  if (score < -0.15) return "text-terminal-red";
  return "text-terminal-muted-alt";
}

function GlobeContent() {
  const [data, setData] = useState<GeoPayload | null>(null);
  const [hours, setHours] = useState<number>(24);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGeo = useCallback(async (windowHours: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        getApiUrl(`/api/brain/news-geo?hours=${windowHours}`)
      );
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const json = await res.json();
      setData(json.data);
    } catch (err: any) {
      setError(err.message || "Could not load geographic news data.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGeo(hours);
    const interval = setInterval(() => fetchGeo(hours), 60_000);
    return () => clearInterval(interval);
  }, [fetchGeo, hours]);

  const countries = data?.countries || [];
  const maxCount = Math.max(1, ...countries.map((c) => c.count));

  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Globe</h1>
          <p className="mt-0.5 text-[13px] text-terminal-muted">
            Where the news is landing — country fill by volume, markers by
            sentiment
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
            <span className="num">{data?.total_tagged ?? 0}</span> tagged
          </span>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded border border-terminal-red/40 bg-terminal-red-soft text-xs text-terminal-red">
          <AlertCircle size={14} strokeWidth={1.7} />
          <span>{error}</span>
        </div>
      )}

      {!!data?.untagged_backlog && (
        <div className="px-3.5 py-2.5 rounded border border-border-dim bg-bg-card text-[11px] text-terminal-muted">
          <span className="num">{data.untagged_backlog.toLocaleString()}</span>{" "}
          older articles are still being tagged in the background. Coverage will
          fill in as the backfill runs.
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        <div className="lg:col-span-7 card">
          <div className="card-head">
            <h2 className="card-title">News origin</h2>
            <span className="text-[11px] text-terminal-muted-alt">
              drag to rotate
            </span>
          </div>
          <div className="card-body">
            {loading && !data ? (
              <div className="h-[380px] grid place-items-center text-xs text-terminal-muted">
                Loading globe…
              </div>
            ) : countries.length === 0 ? (
              <div className="h-[380px] grid place-items-center text-xs text-terminal-muted">
                No geo-tagged stories in this window yet.
              </div>
            ) : (
              <NewsGlobe countries={countries} />
            )}

            <div className="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-terminal-muted-alt">
              <span className="inline-flex items-center gap-1.5">
                <i className="w-3 h-1.5 rounded-sm bg-terminal-signal" />
                Story volume
              </span>
              <span className="inline-flex items-center gap-1.5">
                <i className="w-1.5 h-1.5 rounded-full bg-terminal-green" />
                Net bullish
              </span>
              <span className="inline-flex items-center gap-1.5">
                <i className="w-1.5 h-1.5 rounded-full bg-terminal-red" />
                Net bearish
              </span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 flex flex-col gap-4">
          <div className="card">
            <div className="card-head">
              <h2 className="card-title">By country</h2>
              <span className="num text-[11px] text-terminal-muted-alt">
                {countries.length}
              </span>
            </div>
            <div className="flex flex-col max-h-72 overflow-y-auto">
              {countries.slice(0, 20).map((c) => (
                <div
                  key={c.country}
                  className="grid grid-cols-[minmax(88px,1fr)_1fr_44px] items-center gap-3 px-3.5 py-2 border-b border-border-soft last:border-b-0"
                >
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium truncate">
                      {c.name || c.country}
                    </div>
                    <div className="num text-[11px] text-terminal-muted-alt">
                      {c.country}
                    </div>
                  </div>
                  <div className="h-1.5 rounded-sm bg-bg-surface overflow-hidden">
                    <span
                      className="block h-full rounded-sm bg-terminal-signal"
                      style={{ width: `${(c.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <div
                    className={`num text-right text-xs font-medium ${sentimentColour(
                      c.avg_sentiment
                    )}`}
                  >
                    {c.count}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <h2 className="card-title">Latest tagged</h2>
            </div>
            <div className="flex flex-col max-h-80 overflow-y-auto">
              {(data?.recent || []).slice(0, 15).map((a) => (
                <article
                  key={a.id}
                  className="px-3.5 py-2.5 border-b border-border-soft last:border-b-0 hover:bg-bg-surface transition-colors"
                >
                  <h3 className="text-[13px] font-medium leading-snug text-pretty">
                    {a.url ? (
                      <a
                        href={a.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:text-terminal-signal inline-flex items-start gap-1.5"
                      >
                        <span>{a.headline}</span>
                        <ExternalLink
                          size={11}
                          strokeWidth={1.7}
                          className="mt-1 shrink-0 opacity-60"
                        />
                      </a>
                    ) : (
                      a.headline
                    )}
                  </h3>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-terminal-muted-alt">
                    <span>{a.source_name}</span>
                    <span className="opacity-50">/</span>
                    <span className="num">
                      {new Date(a.published_at).toLocaleTimeString("en-GB", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                    {a.countries.map((code) => (
                      <span key={code} className="tag num">
                        {code}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
              {!data?.recent?.length && !loading && (
                <div className="p-4 text-center text-xs text-terminal-muted">
                  Nothing tagged in this window.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function GlobePage() {
  return (
    <Suspense
      fallback={
        <div className="p-5 text-sm text-terminal-muted">Loading globe…</div>
      }
    >
      <GlobeContent />
    </Suspense>
  );
}
