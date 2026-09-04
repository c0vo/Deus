"use client";

import { useCallback, useEffect, useState } from "react";
import { fetchJson, nonOverlapping } from "../utils/api";

interface Session {
  session_date: string;
  off_exchange_volume: number;
  consolidated_volume: number | null;
  off_exchange_share: number | null;
  short_ratio: number | null;
}

interface Summary {
  ticker: string;
  sessions: number;
  latest_session: string | null;
  short_ratio: number | null;
  short_ratio_avg: number | null;
  offexch_share: number | null;
  offexch_share_avg: number | null;
}

interface RegimeMetric {
  latest: number | null;
  average: number | null;
  sessions: number;
  latest_session: string | null;
}

interface DarkPool {
  ticker: string;
  summary: Summary;
  series: Session[];
  regime_summary: Record<string, RegimeMetric> & { window_days: number };
}

const WINDOWS = [30, 60, 90];

function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** Signed percentage-point delta against a trailing average. */
function delta(latest: number | null, average: number | null): string {
  if (latest === null || average === null) return "";
  const diff = (latest - average) * 100;
  return `${diff >= 0 ? "+" : ""}${diff.toFixed(1)}pp`;
}

function toneFor(latest: number | null, average: number | null): string {
  if (latest === null || average === null) return "text-terminal-muted";
  return latest >= average ? "text-terminal-green" : "text-terminal-red";
}

export default function DarkPoolPanel() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [ticker, setTicker] = useState("");
  const [days, setDays] = useState(60);
  const [data, setData] = useState<DarkPool | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Self-contained like SmartMoneyPanel: the dashboard holds no selected-ticker
  // state, and threading one through just for this panel would couple every
  // other card on the page to it.
  useEffect(() => {
    let cancelled = false;

    const loadTickers = async () => {
      try {
        const json = await fetchJson<{ data?: string[] }>("/api/watchlist");
        if (cancelled) return;
        const list: string[] = json.data || [];
        setTickers(list);
        setTicker((current) => current || list[0] || "");
        if (list.length === 0) setLoading(false);
      } catch {
        if (!cancelled) setLoading(false);
      }
    };

    void loadTickers();
    return () => {
      cancelled = true;
    };
  }, []);

  // Mirrors SmartMoneyPanel: no synchronous setState in the effect body, and a
  // cancelled flag so a slow response for an old window or ticker cannot
  // overwrite a newer one.
  useEffect(() => {
    if (!ticker) return;
    let cancelled = false;

    // nonOverlapping guards the 10-minute poll; the cancelled flag is still
    // needed alongside it, for the different job of stopping a slow response
    // for an old ticker or window from overwriting a newer one.
    const load = nonOverlapping(async () => {
      try {
        const json = await fetchJson<{ data?: DarkPool | null }>(
          `/api/darkpool/${encodeURIComponent(ticker)}?days=${days}`
        );
        if (cancelled) return;
        setData(json.data || null);
        setError(null);
      } catch {
        if (cancelled) return;
        // Deliberately not err.message: fetchJson reports the raw path and
        // status, and a timeout reports "signal timed out". Neither belongs on
        // the card.
        setError("Failed to load off-exchange data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    });

    void load();
    const timer = setInterval(load, 10 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [ticker, days]);

  const selectWindow = useCallback((next: number) => {
    setDays(next);
    setLoading(true);
  }, []);

  const summary = data?.summary;
  // Sessions with no matching price row report a null share. They stay in the
  // series so the date axis is honest, but they cannot be plotted.
  const plotted = (data?.series ?? []).filter((s) => s.off_exchange_share !== null);
  const peak = Math.max(0.6, ...plotted.map((s) => s.off_exchange_share ?? 0));
  const dix = data?.regime_summary?.dix;
  const gex = data?.regime_summary?.gex;

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          Dark Pool
          <span className="label ml-2">off-exchange prints · FINRA</span>
        </h3>
        <div className="flex items-center gap-2">
          <select
            aria-label="Ticker"
            value={ticker}
            onChange={(e) => {
              setTicker(e.target.value);
              setLoading(true);
            }}
            className="bg-transparent border border-border-soft rounded-sm text-[11px] px-1 py-0.5"
          >
            {tickers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <div className="seg">
            {WINDOWS.map((w) => (
              <button
                key={w}
                onClick={() => selectWindow(w)}
                className={days === w ? "is-active" : ""}
              >
                {w}d
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div className="card-body text-[11px] text-terminal-red">{error}</div>
      )}

      {!error && loading && !data && (
        <div className="card-body text-[11px] text-terminal-muted">Loading…</div>
      )}

      {!error && data && summary && summary.sessions === 0 && (
        <div className="card-body text-[11px] text-terminal-muted">
          No off-exchange history for {data.ticker} yet.
        </div>
      )}

      {!error && summary && summary.sessions > 0 && (
        <>
          <div className="card-body border-b border-border-soft">
            <div className="flex justify-between text-[11px] mb-1.5">
              <span className="text-terminal-muted">
                Off-exchange{" "}
                <span className="num text-terminal-fg">
                  {pct(summary.offexch_share)}
                </span>{" "}
                <span className={toneFor(summary.offexch_share, summary.offexch_share_avg)}>
                  {delta(summary.offexch_share, summary.offexch_share_avg)}
                </span>
              </span>
              <span className="text-terminal-muted">
                Short of that{" "}
                <span className="num text-terminal-fg">
                  {pct(summary.short_ratio)}
                </span>{" "}
                <span className={toneFor(summary.short_ratio, summary.short_ratio_avg)}>
                  {delta(summary.short_ratio, summary.short_ratio_avg)}
                </span>
              </span>
            </div>

            {/* Deviation from this ticker's own trailing average is the whole
                signal — the levels are near-constant per name, so a bar chart
                of raw values would look identical for every ticker. */}
            {plotted.length > 0 ? (
              <div className="flex items-end gap-px h-8">
                {plotted.map((s) => {
                  const share = s.off_exchange_share ?? 0;
                  const hot = share >= (summary.offexch_share_avg ?? share);
                  return (
                    <span
                      key={s.session_date}
                      title={`${s.session_date} · ${pct(share)} off-exchange · ${pct(
                        s.short_ratio
                      )} short`}
                      className={`flex-1 ${
                        hot ? "bg-terminal-green" : "bg-terminal-muted"
                      }`}
                      style={{ height: `${Math.max(4, (share / peak) * 100)}%` }}
                    />
                  );
                })}
              </div>
            ) : (
              <div className="text-[11px] text-terminal-muted">
                No priced sessions in this window.
              </div>
            )}
          </div>

          <div className="card-body text-[11px] text-terminal-muted space-y-1">
            {dix?.latest !== null && dix?.latest !== undefined && (
              <div className="flex justify-between">
                <span>Dark Index (DIX), market-wide</span>
                <span className={toneFor(dix.latest, dix.average)}>
                  <span className="num">{dix.latest.toFixed(3)}</span>
                  {dix.average !== null && (
                    <span className="ml-1 opacity-70">
                      vs {dix.average.toFixed(3)} avg
                    </span>
                  )}
                </span>
              </div>
            )}
            {gex?.latest !== null && gex?.latest !== undefined && (
              <div className="flex justify-between">
                <span>Gamma exposure (GEX)</span>
                <span
                  className={
                    gex.latest < 0 ? "text-terminal-red" : "text-terminal-green"
                  }
                >
                  <span className="num">
                    {(gex.latest / 1e9).toFixed(2)}B
                  </span>
                  <span className="ml-1 opacity-70">
                    {gex.latest < 0 ? "dealers short gamma" : "dealers long gamma"}
                  </span>
                </span>
              </div>
            )}
            <p className="pt-1 opacity-70">
              Off-exchange short volume is not short interest — most of it is
              market-maker and wholesaler hedging, so only the move against this
              ticker&apos;s own average is meaningful.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
