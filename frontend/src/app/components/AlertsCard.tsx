"use client";

import { useEffect, useState } from "react";
import { ChevronDown, ExternalLink } from "lucide-react";
import { fetchJson } from "../utils/api";

export interface AlertSource {
  title?: string;
  url?: string;
  source?: string;
  published_at?: string;
  kind?: string;
}

export interface AlertRow {
  id: number;
  ticker: string | null;
  kind: string;
  pct: number | null;
  price: number | null;
  severity: string | null;
  title: string;
  summary: string | null;
  grounded_by: string;
  sources: AlertSource[];
  created_at: string | null;
}

/**
 * Narrow one SSE payload into an AlertRow.
 *
 * The `alert` topic carries the stored row, so this is a shape check rather
 * than a translation — but it is the boundary where untyped JSON becomes typed,
 * and a malformed event has to be dropped rather than rendered as blanks.
 */
export function normalizeAlert(data: unknown): AlertRow | null {
  if (!data || typeof data !== "object") return null;
  const row = data as Record<string, unknown>;
  if (typeof row.id !== "number" || typeof row.kind !== "string") return null;

  const sources = Array.isArray(row.sources)
    ? (row.sources as AlertSource[])
    : [];

  return {
    id: row.id,
    ticker: typeof row.ticker === "string" ? row.ticker : null,
    kind: row.kind,
    pct: typeof row.pct === "number" ? row.pct : null,
    price: typeof row.price === "number" ? row.price : null,
    severity: typeof row.severity === "string" ? row.severity : null,
    title: typeof row.title === "string" ? row.title : "",
    summary: typeof row.summary === "string" ? row.summary : null,
    grounded_by:
      typeof row.grounded_by === "string" ? row.grounded_by : "none",
    sources,
    created_at: typeof row.created_at === "string" ? row.created_at : null,
  };
}

const GROUNDING_LABELS: Record<string, string> = {
  db: "in-house",
  web: "web",
  none: "no catalyst",
};

const KIND_LABELS: Record<string, string> = {
  price_drop: "Drop",
  price_move: "Move",
  volume: "Volume",
  earnings_whisper: "Earnings",
  breaking: "Breaking",
};

function clockLabel(iso: string | null): string {
  if (!iso) return "—";
  // SQLite writes CURRENT_TIMESTAMP as "YYYY-MM-DD HH:MM:SS" with no zone, and
  // it is UTC. Parsed as-is the browser reads it as local time, which on the
  // deployment timezone would put every alert nine hours in the future.
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(iso);
  const when = new Date(hasZone ? iso : `${iso.replace(" ", "T")}Z`);
  if (Number.isNaN(when.getTime())) return "—";
  return when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

interface AlertsCardProps {
  alerts: AlertRow[];
  onLoaded: (rows: AlertRow[]) => void;
}

export default function AlertsCard({ alerts, onLoaded }: AlertsCardProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [error, setError] = useState(false);

  // The loader lives inside the effect, with a cancelled flag, the way every
  // other fetch-on-mount card here does it. Calling an async loader from the
  // effect body instead trips react-hooks/set-state-in-effect and leaves a
  // real unmount race behind the lint error.
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const json = await fetchJson<{ data?: unknown[] }>(
          "/api/alerts?limit=10",
        );
        if (cancelled) return;
        const rows = (json.data || [])
          .map(normalizeAlert)
          .filter((r): r is AlertRow => r !== null);
        onLoaded(rows);
        setError(false);
      } catch {
        if (!cancelled) setError(true);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [onLoaded]);

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Pushed alerts</h3>
        <span className="text-[11px] text-terminal-muted-alt">
          {alerts.length > 0 ? `${alerts.length} recent` : ""}
        </span>
      </div>

      {alerts.length === 0 ? (
        <div className="card-body text-center text-xs text-terminal-muted">
          {error
            ? "Could not load alerts."
            : "No alerts pushed yet — nothing has cleared the thresholds."}
        </div>
      ) : (
        <div className="flex flex-col max-h-80 overflow-y-auto overscroll-contain">
          {alerts.map((alert) => {
            const isExpanded = expandedId === alert.id;
            const down = (alert.pct ?? 0) < 0;
            return (
              <div
                key={alert.id}
                className="border-b border-border-soft last:border-b-0"
              >
                <button
                  type="button"
                  onClick={() => setExpandedId(isExpanded ? null : alert.id)}
                  aria-expanded={isExpanded}
                  className="w-full text-left px-3.5 py-2.5 hover:bg-bg-surface transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="num text-[11px] text-terminal-muted-alt shrink-0">
                      {clockLabel(alert.created_at)}
                    </span>
                    {alert.ticker && (
                      <span className="tag num shrink-0">{alert.ticker}</span>
                    )}
                    {alert.pct !== null && (
                      <span
                        className={`num text-[11px] font-medium shrink-0 ${
                          down ? "text-terminal-red" : "text-terminal-green"
                        }`}
                      >
                        {alert.pct > 0 ? "+" : ""}
                        {alert.pct.toFixed(2)}%
                      </span>
                    )}
                    <span className="ml-auto shrink-0 flex items-center gap-2">
                      <span className="text-[10px] uppercase tracking-wide text-terminal-muted-alt">
                        {KIND_LABELS[alert.kind] || alert.kind}
                      </span>
                      <ChevronDown
                        size={13}
                        strokeWidth={1.7}
                        className={`text-terminal-muted-alt transition-transform ${
                          isExpanded ? "rotate-180" : ""
                        }`}
                      />
                    </span>
                  </div>

                  <div className="mt-1 flex items-start gap-2">
                    <span className="text-[13px] leading-snug text-pretty">
                      {alert.title}
                    </span>
                  </div>

                  <span
                    className={`mt-1.5 inline-block text-[10px] px-1.5 py-[1px] rounded-sm border ${
                      alert.grounded_by === "none"
                        ? "border-terminal-muted-alt/40 text-terminal-muted-alt"
                        : "border-terminal-signal/50 text-terminal-signal"
                    }`}
                  >
                    {GROUNDING_LABELS[alert.grounded_by] || alert.grounded_by}
                  </span>
                </button>

                {isExpanded && (
                  <div className="px-3.5 pb-3 flex flex-col gap-2">
                    {alert.summary && (
                      <p className="text-xs leading-relaxed text-terminal-muted">
                        {alert.summary}
                      </p>
                    )}
                    {alert.sources.length > 0 ? (
                      <div className="flex flex-col gap-1">
                        {alert.sources.slice(0, 5).map((source, idx) => (
                          <a
                            key={`${alert.id}-${idx}`}
                            href={source.url || "#"}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[11px] text-terminal-muted hover:text-terminal-signal transition-colors flex items-start gap-1.5"
                          >
                            <ExternalLink
                              size={11}
                              strokeWidth={1.7}
                              className="mt-[2px] shrink-0"
                            />
                            <span className="min-w-0">
                              {source.title || source.url}
                              {source.source ? ` — ${source.source}` : ""}
                            </span>
                          </a>
                        ))}
                      </div>
                    ) : (
                      <p className="text-[11px] text-terminal-muted-alt">
                        No dated source supports a cause for this alert.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
