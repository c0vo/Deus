"use client";

import { useCallback, useEffect, useState } from "react";
import { getApiUrl } from "../utils/api";

interface InsiderTxn {
  ticker: string;
  insider_name: string | null;
  insider_title: string | null;
  is_officer: number;
  is_director: number;
  is_ten_pct_owner: number;
  transaction_date: string;
  filed_at: string;
  transaction_code: string;
  shares: number | null;
  price_per_share: number | null;
  value_usd: number | null;
  is_10b5_1: number | null;
}

interface TickerRoll {
  ticker: string;
  buy_value: number;
  sell_value: number;
  buy_count: number;
  sell_count: number;
  distinct_buyers: number;
  net_value: number;
}

interface Stake {
  ticker: string;
  filer_name: string | null;
  form_type: string;
  is_activist: number;
  is_amendment: number;
  pct_of_class: number | null;
  filed_at: string;
}

interface SmartMoney {
  window_days: number;
  totals: {
    buy_value: number;
    sell_value: number;
    net_value: number;
    buy_ratio: number | null;
    transaction_count: number;
  };
  by_ticker: TickerRoll[];
  transactions: InsiderTxn[];
  stakes: Stake[];
}

const WINDOWS = [7, 30, 90];

/** Compact USD — insider values span $20k director buys to $500m block sales. */
function usd(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
  return `$${value.toFixed(0)}`;
}

function role(t: InsiderTxn): string {
  if (t.insider_title) return t.insider_title;
  if (t.is_officer) return "Officer";
  if (t.is_director) return "Director";
  if (t.is_ten_pct_owner) return "10% owner";
  return "Insider";
}

export default function SmartMoneyPanel() {
  const [days, setDays] = useState(30);
  const [data, setData] = useState<SmartMoney | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The effect never calls setState synchronously — loading is seeded by
  // useState and flipped in the window-change handler, both outside the effect
  // body. Every state write here happens after an await, and the cancelled flag
  // stops a slow in-flight response from overwriting a newer window's data.
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await fetch(getApiUrl(`/api/brain/smart-money?days=${days}`));
        if (!res.ok) throw new Error("Failed to load smart money data.");
        const json = await res.json();
        if (cancelled) return;
        setData(json.data || null);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load smart money data."
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void load();
    const timer = setInterval(load, 10 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [days]);

  const selectWindow = useCallback((next: number) => {
    setDays(next);
    setLoading(true);
  }, []);

  const totals = data?.totals;
  const gross = (totals?.buy_value ?? 0) + (totals?.sell_value ?? 0);
  const net = totals?.net_value ?? 0;
  const ranked = (data?.by_ticker ?? []).filter((r) => r.net_value !== 0);
  const peak = Math.max(1, ...ranked.map((r) => Math.abs(r.net_value)));

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          Smart Money
          <span className="label ml-2">insider &amp; institutional</span>
        </h3>
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

      {error && (
        <div className="card-body text-[11px] text-terminal-red">{error}</div>
      )}

      {!error && loading && !data && (
        <div className="card-body text-[11px] text-terminal-muted">Loading…</div>
      )}

      {!error && data && totals && (
        <>
          {/* Open-market buy vs sell split. Grants and option exercises are
              excluded server-side — they are compensation, not conviction. */}
          <div className="card-body border-b border-border-soft">
            <div className="flex justify-between text-[11px] mb-1.5">
              <span className="text-terminal-green">
                Bought <span className="num">{usd(totals.buy_value)}</span>
              </span>
              <span
                className={
                  net >= 0 ? "text-terminal-green" : "text-terminal-red"
                }
              >
                Net <span className="num">{usd(net)}</span>
              </span>
              <span className="text-terminal-red">
                Sold <span className="num">{usd(totals.sell_value)}</span>
              </span>
            </div>
            {gross > 0 ? (
              <div className="flex h-1.5 rounded-sm overflow-hidden">
                <span
                  className="bg-terminal-green"
                  style={{ width: `${(totals.buy_value / gross) * 100}%` }}
                />
                <span
                  className="bg-terminal-red"
                  style={{ width: `${(totals.sell_value / gross) * 100}%` }}
                />
              </div>
            ) : (
              <div className="h-1.5 rounded-sm bg-border-dim" />
            )}
            <div className="mt-1.5 text-[10px] text-terminal-muted-alt">
              <span className="num">{totals.transaction_count}</span> open-market
              transactions disclosed in {data.window_days}d
              {totals.buy_ratio !== null && (
                <>
                  {" · "}buy ratio{" "}
                  <span className="num">
                    {(totals.buy_ratio * 100).toFixed(0)}%
                  </span>
                </>
              )}
            </div>
          </div>

          {/* Net conviction by ticker, diverging off a centre axis. */}
          {ranked.length > 0 && (
            <div className="card-body border-b border-border-soft">
              <div className="label mb-2">Net insider flow by ticker</div>
              <div className="flex flex-col gap-1">
                {ranked.slice(0, 8).map((r) => {
                  const up = r.net_value >= 0;
                  const width = (Math.abs(r.net_value) / peak) * 50;
                  return (
                    <div key={r.ticker} className="flex items-center gap-2">
                      <span className="w-12 text-[11px] num">{r.ticker}</span>
                      <div className="flex-1 flex h-3 items-center">
                        <div className="w-1/2 flex justify-end">
                          {!up && (
                            <span
                              className="h-2 bg-terminal-red rounded-l-sm"
                              style={{ width: `${width * 2}%` }}
                            />
                          )}
                        </div>
                        <span className="w-px h-3 bg-border-soft" />
                        <div className="w-1/2">
                          {up && (
                            <span
                              className="block h-2 bg-terminal-green rounded-r-sm"
                              style={{ width: `${width * 2}%` }}
                            />
                          )}
                        </div>
                      </div>
                      <span
                        className={`w-16 text-right text-[11px] num ${
                          up ? "text-terminal-green" : "text-terminal-red"
                        }`}
                      >
                        {usd(r.net_value)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* >5% stake filings. 13D means intent to influence; 13G is passive. */}
          {data.stakes.length > 0 && (
            <div className="card-body border-b border-border-soft">
              <div className="label mb-2">Ownership stakes (13D / 13G)</div>
              <div className="flex flex-col gap-1">
                {data.stakes.slice(0, 4).map((s, i) => (
                  <div
                    key={`${s.ticker}-${s.filed_at}-${i}`}
                    className="flex items-center gap-2 text-[11px]"
                  >
                    <span className="num w-12">{s.ticker}</span>
                    <span className={`tag ${s.is_activist ? "tag-warn" : ""}`}>
                      {s.is_activist ? "13D activist" : "13G passive"}
                    </span>
                    <span className="text-terminal-muted truncate flex-1">
                      {s.filer_name || "Unknown filer"}
                    </span>
                    <span className="num text-terminal-muted-alt">
                      {String(s.filed_at).slice(0, 10)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card-body max-h-56 overflow-y-auto">
            <div className="label mb-2">Recent disclosures</div>
            {data.transactions.length === 0 ? (
              <div className="text-[11px] text-terminal-muted">
                No open-market insider transactions disclosed in this window.
              </div>
            ) : (
              <div className="flex flex-col gap-1.5">
                {data.transactions.slice(0, 12).map((t, i) => {
                  const buy = t.transaction_code === "P";
                  return (
                    <div
                      key={`${t.ticker}-${t.filed_at}-${i}`}
                      className="flex items-start gap-2 text-[11px]"
                    >
                      <span className="num w-12 shrink-0">{t.ticker}</span>
                      <span className={`tag ${buy ? "tag-up" : "tag-down"} shrink-0`}>
                        {buy ? "BUY" : "SELL"}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="truncate">
                          {t.insider_name || "Unknown"}
                          <span className="text-terminal-muted-alt">
                            {" "}
                            · {role(t)}
                          </span>
                        </div>
                        <div className="text-[10px] text-terminal-muted-alt">
                          <span className="num">{t.transaction_date}</span>
                          {t.shares !== null && (
                            <>
                              {" · "}
                              <span className="num">
                                {t.shares.toLocaleString()}
                              </span>{" "}
                              sh
                            </>
                          )}
                          {/* A pre-scheduled sale was set up months earlier and
                              says little about a current view. */}
                          {t.is_10b5_1 === 1 && " · 10b5-1 pre-scheduled"}
                        </div>
                      </div>
                      <span
                        className={`num shrink-0 ${
                          buy ? "text-terminal-green" : "text-terminal-red"
                        }`}
                      >
                        {t.value_usd !== null ? usd(Math.abs(t.value_usd)) : "—"}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
