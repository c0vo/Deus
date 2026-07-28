"use client";

import { useState, useEffect } from "react";
import { getApiUrl } from "../utils/api";

interface SectorData {
  sector: string;
  avg_sentiment: number;
  article_count: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  sentiment_momentum: number;
  top_tickers: { ticker: string; count: number }[];
}

interface SectorHeatmapProps {
  sectors: SectorData[];
  onSectorClick?: (sector: string) => void;
}

const PERIODS = ["1d", "7d", "1m", "1y"] as const;

export default function SectorHeatmap({ sectors, onSectorClick }: SectorHeatmapProps) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("1d");
  const [fetchedSectors, setFetchedSectors] = useState<SectorData[] | null>(null);
  const [periodLoading, setPeriodLoading] = useState(false);

  useEffect(() => {
    if (selectedPeriod === "1d") {
      setFetchedSectors(null);
      return;
    }
    let cancelled = false;
    setPeriodLoading(true);
    fetch(getApiUrl(`/api/brain/sector-heatmap?period=${selectedPeriod}`))
      .then((r) => r.json())
      .then((j) => {
        if (!cancelled) {
          setFetchedSectors(j.data || []);
          setPeriodLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) setPeriodLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPeriod]);

  const displaySectors = selectedPeriod === "1d" ? sectors : fetchedSectors || [];

  const periodTabs = (
    <div className="seg" role="tablist">
      {PERIODS.map((p) => (
        <button
          key={p}
          role="tab"
          aria-selected={selectedPeriod === p}
          onClick={() => setSelectedPeriod(p)}
        >
          {p}
        </button>
      ))}
    </div>
  );

  if (!displaySectors || displaySectors.length === 0) {
    return (
      <div className="card">
        <div className="card-head">
          <h3 className="card-title">Sector heatmap</h3>
          {periodTabs}
        </div>
        <div className="card-body text-xs text-terminal-muted">
          {periodLoading
            ? `Loading ${selectedPeriod.toUpperCase()} sector data…`
            : "No sector data available yet."}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Sector heatmap</h3>
        {periodTabs}
      </div>

      <div className="flex flex-col">
        {displaySectors.slice(0, 8).map((s) => {
          const isExpanded = expanded === s.sector;
          const isPositive = s.avg_sentiment >= 0;
          // Bars read off a true zero axis, so direction is legible before the number is.
          const width = Math.min(Math.abs(s.avg_sentiment) * 100, 50);

          return (
            <div
              key={s.sector}
              className="border-b border-border-soft last:border-b-0"
            >
              <button
                type="button"
                onClick={() => {
                  setExpanded(isExpanded ? null : s.sector);
                  onSectorClick?.(s.sector);
                }}
                className="w-full grid grid-cols-[minmax(88px,108px)_1fr_52px] items-center gap-3 px-3.5 py-2.5 text-left hover:bg-bg-surface transition-colors"
              >
                <div className="min-w-0">
                  <div className="text-[13px] font-medium truncate">{s.sector}</div>
                  <div className="num text-[11px] text-terminal-muted-alt">
                    {s.article_count.toLocaleString()} articles
                  </div>
                </div>

                <div className="relative h-[18px]">
                  <span className="absolute left-1/2 top-0 bottom-0 w-px bg-border-dim" />
                  <span
                    className={`absolute top-1 bottom-1 rounded-[1px] transition-all duration-500 ${
                      isPositive ? "bg-terminal-green" : "bg-terminal-red"
                    }`}
                    style={{
                      left: isPositive ? "50%" : `${50 - width}%`,
                      width: `${width}%`,
                    }}
                  />
                </div>

                <div
                  className={`num text-right text-xs font-medium ${
                    isPositive ? "text-terminal-green" : "text-terminal-red"
                  }`}
                >
                  {isPositive ? "+" : "−"}
                  {Math.abs(s.avg_sentiment).toFixed(2)}
                </div>
              </button>

              {isExpanded && (
                <div className="px-3.5 pb-3 -mt-0.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-terminal-muted">
                  <span>
                    Momentum{" "}
                    <span
                      className={`num ${
                        s.sentiment_momentum >= 0
                          ? "text-terminal-green"
                          : "text-terminal-red"
                      }`}
                    >
                      {s.sentiment_momentum >= 0 ? "+" : "−"}
                      {Math.abs(s.sentiment_momentum).toFixed(3)}
                    </span>
                  </span>
                  <span className="flex items-center gap-2.5">
                    <span className="num text-terminal-green">{s.bullish_count} bull</span>
                    <span className="num text-terminal-muted-alt">{s.neutral_count} neut</span>
                    <span className="num text-terminal-red">{s.bearish_count} bear</span>
                  </span>
                  {s.top_tickers && s.top_tickers.length > 0 && (
                    <span className="flex items-center gap-1">
                      {s.top_tickers.slice(0, 3).map((t) => (
                        <span key={t.ticker} className="tag num">
                          {t.ticker}
                        </span>
                      ))}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
