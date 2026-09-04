"use client";

import { useEffect, useState } from "react";
import { fetchJson, nonOverlapping } from "../utils/api";

interface ConsensusRow {
  session_date: string;
  strong_buy: number | null;
  buy: number | null;
  hold: number | null;
  sell: number | null;
  strong_sell: number | null;
  analyst_count: number | null;
  recommendation_key: string | null;
  recommendation_mean: number | null;
  target_mean: number | null;
  target_high: number | null;
  target_low: number | null;
  target_median: number | null;
  spot_price: number | null;
  published_at: string;
}

interface Consensus {
  ticker: string;
  covered: boolean;
  latest: ConsensusRow | null;
  total_ratings?: number;
}

interface TechnicalRating {
  timeframe: string;
  summary_score: number | null;
  summary_label: string | null;
  ma_score: number | null;
  ma_label: string | null;
  osc_score: number | null;
  osc_label: string | null;
  buy_votes: number | null;
  neutral_votes: number | null;
  sell_votes: number | null;
  bars_available: number | null;
}

interface AnalystData {
  ticker: string;
  spot_price: number | null;
  consensus: Consensus;
  upside_pct: number | null;
  technical: Record<string, TechnicalRating>;
  min_bars: number;
}

/** Display order and labels. Matches TIMEFRAMES in pipeline/technical_rating.py. */
const TIMEFRAMES: [string, string][] = [
  ["short", "Short term"],
  ["medium", "Medium term"],
  ["long", "Long term"],
];

const BUCKETS: [keyof ConsensusRow, string, string][] = [
  ["strong_buy", "Strong Buy", "bg-terminal-green"],
  ["buy", "Buy", "bg-terminal-green-soft"],
  ["hold", "Hold", "bg-terminal-muted"],
  ["sell", "Sell", "bg-terminal-red-soft"],
  ["strong_sell", "Strong Sell", "bg-terminal-red"],
];

/** Rating label -> tone. Two greens and two reds so strength is visible. */
function toneFor(label: string | null | undefined): string {
  switch (label) {
    case "STRONG_BUY":
      return "text-terminal-green";
    case "BUY":
      return "text-terminal-green-soft";
    case "SELL":
      return "text-terminal-red-soft";
    case "STRONG_SELL":
      return "text-terminal-red";
    default:
      return "text-terminal-muted";
  }
}

function labelText(label: string | null | undefined): string {
  if (!label) return "—";
  return label.replace(/_/g, " ");
}

function money(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `$${value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function signedPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

/**
 * Where spot sits inside the analyst target range, as a 0-100 percentage.
 *
 * Clamped because spot genuinely trades outside the range — a stock can fall
 * below every target on the street — and an unclamped value would position the
 * marker outside the bar.
 */
function rangePosition(
  spot: number | null,
  low: number | null | undefined,
  high: number | null | undefined
): number | null {
  if (spot === null || low === null || low === undefined) return null;
  if (high === null || high === undefined || high <= low) return null;
  return Math.min(100, Math.max(0, ((spot - low) / (high - low)) * 100));
}

export default function AnalystPanel({ ticker }: { ticker: string }) {
  const [data, setData] = useState<AnalystData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Same shape as DarkPoolPanel: nonOverlapping guards the poll, and the
  // cancelled flag does the different job of stopping a slow response for an
  // old ticker from overwriting a newer one. Both are needed.
  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;

    const load = nonOverlapping(async () => {
      try {
        const json = await fetchJson<{ data?: AnalystData | null }>(
          `/api/analysts/${encodeURIComponent(ticker)}`
        );
        if (cancelled) return;
        setData(json.data || null);
        setError(null);
      } catch {
        if (cancelled) return;
        setError("Failed to load analyst data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    });

    void load();
    // Both sides refresh once a day, so this only needs to catch a manual
    // backfill landing while the row is open.
    const timer = setInterval(load, 10 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [ticker]);

  if (loading) {
    return (
      <div className="text-xs text-terminal-muted font-mono py-2">
        Loading analyst data…
      </div>
    );
  }
  if (error) {
    return <div className="text-xs text-terminal-red font-mono py-2">{error}</div>;
  }

  const row = data?.consensus?.latest ?? null;
  const covered = Boolean(data?.consensus?.covered && row);
  const total = data?.consensus?.total_ratings ?? 0;
  const spot = data?.spot_price ?? null;
  const marker = rangePosition(spot, row?.target_low, row?.target_high);
  const technical = data?.technical ?? {};
  const hasTechnical = Object.keys(technical).length > 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 py-2">
      {/* ── Consensus ─────────────────────────────────────────────── */}
      <div>
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-[10px] uppercase tracking-wider text-terminal-muted font-mono">
            Analyst Consensus
          </span>
          {covered && (
            <span className={`text-xs font-mono ${toneFor(
              row?.recommendation_key?.toUpperCase() ?? null
            )}`}>
              {labelText(row?.recommendation_key?.toUpperCase() ?? null)}
            </span>
          )}
        </div>

        {!covered ? (
          <div className="text-xs text-terminal-muted font-mono">
            No analyst coverage on record.
          </div>
        ) : (
          <>
            <div className="space-y-1">
              {BUCKETS.map(([key, label, bar]) => {
                const count = (row?.[key] as number | null) ?? 0;
                const width = total > 0 ? (count / total) * 100 : 0;
                return (
                  <div key={key} className="flex items-center gap-2 text-[11px] font-mono">
                    <span className="w-20 text-terminal-muted shrink-0">{label}</span>
                    <div className="flex-1 h-2 bg-bg-input rounded-sm overflow-hidden">
                      <div className={`h-full ${bar}`} style={{ width: `${width}%` }} />
                    </div>
                    <span className="w-6 text-right text-terminal-text">{count}</span>
                  </div>
                );
              })}
            </div>

            <div className="mt-2 text-[10px] text-terminal-muted font-mono">
              {total} rating{total === 1 ? "" : "s"}
              {row?.recommendation_mean !== null &&
                row?.recommendation_mean !== undefined &&
                ` · mean ${row.recommendation_mean.toFixed(2)}/5 (1 = most bullish)`}
            </div>

            {/* Target range. The bar is low -> high with a marker at spot, so
                the question "is the street above or below here" is answered
                without reading any number. */}
            {row?.target_mean !== null && row?.target_mean !== undefined && (
              <div className="mt-3">
                <div className="flex items-baseline justify-between text-[11px] font-mono">
                  <span className="text-terminal-muted">Price target</span>
                  <span className="text-terminal-text">
                    {money(row.target_mean)}
                    {data?.upside_pct !== null && data?.upside_pct !== undefined && (
                      <span
                        className={
                          data.upside_pct >= 0
                            ? "text-terminal-green ml-1"
                            : "text-terminal-red ml-1"
                        }
                      >
                        {signedPct(data.upside_pct)}
                      </span>
                    )}
                  </span>
                </div>

                <div className="relative h-1.5 my-2 bg-bg-input rounded-sm">
                  {marker !== null && (
                    <div
                      className="absolute -top-1 w-0.5 h-3.5 bg-terminal-signal"
                      style={{ left: `${marker}%` }}
                      title={`Spot ${money(spot)}`}
                    />
                  )}
                </div>

                <div className="flex justify-between text-[10px] text-terminal-muted font-mono">
                  <span>low {money(row.target_low)}</span>
                  <span>spot {money(spot)}</span>
                  <span>high {money(row.target_high)}</span>
                </div>
                {row.analyst_count !== null && (
                  <div className="mt-1 text-[10px] text-terminal-muted font-mono">
                    {row.analyst_count} analysts · as of{" "}
                    {String(row.published_at).slice(0, 10)}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Technical rating ──────────────────────────────────────── */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-terminal-muted font-mono mb-2">
          Technical Rating
        </div>

        {!hasTechnical ? (
          <div className="text-xs text-terminal-muted font-mono">
            Not enough price history — needs {data?.min_bars ?? "—"} bars per
            timeframe. Run scripts/manual/backfill_price_history.py.
          </div>
        ) : (
          <div className="space-y-2">
            {TIMEFRAMES.map(([key, label]) => {
              const rating = technical[key];
              if (!rating) {
                return (
                  <div
                    key={key}
                    className="flex items-center justify-between text-[11px] font-mono"
                  >
                    <span className="text-terminal-muted">{label}</span>
                    <span className="text-terminal-muted">not enough history</span>
                  </div>
                );
              }
              const score = rating.summary_score ?? 0;
              // -1..+1 mapped onto a centre-origin bar, so direction reads as
              // left/right rather than needing the sign.
              const halfWidth = Math.abs(score) * 50;
              return (
                <div key={key}>
                  <div className="flex items-baseline justify-between text-[11px] font-mono">
                    <span className="text-terminal-muted">{label}</span>
                    <span className={toneFor(rating.summary_label)}>
                      {labelText(rating.summary_label)}
                      <span className="text-terminal-muted ml-1">
                        {score >= 0 ? "+" : ""}
                        {score.toFixed(2)}
                      </span>
                    </span>
                  </div>

                  <div className="relative h-1.5 my-1 bg-bg-input rounded-sm">
                    <div className="absolute left-1/2 top-0 w-px h-full bg-border-dim" />
                    <div
                      className={`absolute top-0 h-full ${
                        score >= 0 ? "bg-terminal-green" : "bg-terminal-red"
                      }`}
                      style={
                        score >= 0
                          ? { left: "50%", width: `${halfWidth}%` }
                          : { right: "50%", width: `${halfWidth}%` }
                      }
                    />
                  </div>

                  <div className="text-[10px] text-terminal-muted font-mono">
                    {rating.buy_votes}▲ {rating.neutral_votes}● {rating.sell_votes}▼
                    {" · MAs "}
                    <span className={toneFor(rating.ma_label)}>
                      {labelText(rating.ma_label)}
                    </span>
                    {" · osc "}
                    <span className={toneFor(rating.osc_label)}>
                      {labelText(rating.osc_label)}
                    </span>
                  </div>
                </div>
              );
            })}
            <div className="text-[10px] text-terminal-muted font-mono pt-1">
              26 indicators, TradingView methodology. Momentum and trend only —
              not a valuation view.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
