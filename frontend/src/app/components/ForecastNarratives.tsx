"use client";

import { useEffect, useState } from "react";
import FormattedText from "./FormattedText";
import { fetchJson } from "../utils/api";

/** One horizon's entry from /api/predictions/{ticker}/narratives. */
interface HorizonNarrative {
  direction: string;
  confidence: number;
  horizon_days: number;
  created_at: string | null;
  model_type: string;
  narrative: string;
}

interface NarrativesData {
  ticker: string;
  horizons: Record<string, HorizonNarrative>;
}

/** Display order, matching the badge row this section explains. */
const HORIZONS = ["5d", "1m", "3m", "1y"];

/**
 * Who made the call each narrative interprets. The keys mirror
 * NARRATIVE_MODEL_TYPES in api/server.py, the only types the endpoint serves.
 */
const CALL_SOURCE: Record<string, string> = {
  per_ticker: "per-ticker model",
  sector: "sector model",
  universal: "universal model",
  llm_only: "LLM call, no trained model",
};

const directionTone = (direction: string) =>
  direction === "UP"
    ? "text-terminal-green"
    : direction === "DOWN"
      ? "text-terminal-red"
      : "text-terminal-muted";

function madeAt(raw: string | null): string {
  if (!raw) return "";
  // SQLite writes CURRENT_TIMESTAMP as "YYYY-MM-DD HH:MM:SS" in UTC with no
  // zone, which the browser would read as local time. Same fix as clockLabel
  // in AlertsCard.
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(raw);
  const when = new Date(hasZone ? raw : `${raw.replace(" ", "T")}Z`);
  if (Number.isNaN(when.getTime())) return "";
  return when.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * The written reading behind each ML forecast badge, for one ticker's report.
 *
 * Fetches itself on open, the way AnalystPanel does, because the text is not
 * on /api/markets: it would ride on every grid poll and is only read here.
 * `refreshKey` should change whenever the grid's badges do, so a report left
 * open picks up a prediction that lands while it is open.
 *
 * Renders nothing until there is at least one narrative to show. A horizon with
 * no narrative is omitted rather than shown as a blank.
 */
export default function ForecastNarratives({
  ticker,
  refreshKey = "",
}: {
  ticker: string;
  refreshKey?: string;
}) {
  const [data, setData] = useState<NarrativesData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;

    const load = async () => {
      try {
        const json = await fetchJson<{ data?: NarrativesData | null }>(
          `/api/predictions/${encodeURIComponent(ticker)}/narratives`
        );
        if (cancelled) return;
        setData(json.data || null);
        setError(false);
      } catch {
        if (!cancelled) setError(true);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [ticker, refreshKey]);

  const entries: [string, HorizonNarrative][] = [];
  for (const horizon of HORIZONS) {
    const entry = data?.horizons?.[horizon];
    if (entry && entry.narrative && entry.narrative.trim()) {
      entries.push([horizon, entry]);
    }
  }

  // A failed refetch keeps showing the last good answer: each entry carries its
  // own timestamp, so older text cannot pass for newer.
  if (entries.length === 0 && !error) return null;

  return (
    <div className="border-b border-border-dim pb-3 space-y-2">
      <div className="space-y-0.5">
        <h4 className="text-xs font-bold text-terminal-text uppercase tracking-wider">
          [ML Forecast Narrative]
        </h4>
        <p className="text-[10px] text-terminal-muted uppercase tracking-wider">
          Each horizon&apos;s call, interpreted by an LLM with recent news as context
        </p>
      </div>

      {entries.length === 0 ? (
        <p className="text-xs text-terminal-red">Failed to load forecast narratives.</p>
      ) : (
        entries.map(([horizon, entry]) => {
          const made = madeAt(entry.created_at);
          return (
            <div key={horizon} className="space-y-1">
              <p className="text-xs">
                <span className="font-bold text-terminal-muted">{horizon.toUpperCase()}</span>{" "}
                <span className={`font-bold ${directionTone(entry.direction)}`}>
                  {entry.direction} ({(Number(entry.confidence ?? 0) * 100).toFixed(0)}%)
                </span>{" "}
                <span className="text-terminal-muted">
                  &middot; {CALL_SOURCE[entry.model_type] ?? entry.model_type}
                  {made ? ` · made ${made}` : ""}
                </span>
              </p>
              <FormattedText
                text={entry.narrative}
                className="text-sm text-terminal-text leading-relaxed font-sans"
              />
            </div>
          );
        })
      )}
    </div>
  );
}
