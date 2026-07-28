"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

interface MacroTheme {
  title: string;
  explanation: string;
  impacted_sectors: string[];
  impacted_tickers: string[];
  confidence: number;
}

interface MacroThemesCardProps {
  themes: MacroTheme[];
  loading?: boolean;
  onRefresh?: () => void;
}

export default function MacroThemesCard({
  themes,
  loading,
  onRefresh,
}: MacroThemesCardProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Macro themes</h3>
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="btn-ghost"
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        )}
      </div>

      {loading && themes.length === 0 ? (
        <div className="card-body text-center text-xs text-terminal-muted">
          Generating macro themes from news context…
        </div>
      ) : themes.length === 0 ? (
        <div className="card-body text-center text-xs text-terminal-muted">
          No macro themes available yet.
        </div>
      ) : (
        <div className="flex flex-col max-h-80 overflow-y-auto">
          {themes.map((theme, idx) => {
            const isExpanded = expandedIdx === idx;
            return (
              <div
                key={idx}
                className="border-b border-border-soft last:border-b-0"
              >
                <button
                  type="button"
                  onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                  aria-expanded={isExpanded}
                  className="w-full text-left px-3.5 py-2.5 hover:bg-bg-surface transition-colors"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[13px] font-medium leading-snug text-pretty">
                      {theme.title}
                    </span>
                    <span className="shrink-0 flex items-center gap-2">
                      <span className="num text-[11px] text-terminal-muted-alt">
                        {(theme.confidence * 100).toFixed(0)}%
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

                  {/* Confidence as a meter — the number alone doesn't rank at a glance. */}
                  <div className="mt-1.5 h-[3px] rounded-sm bg-bg-surface overflow-hidden">
                    <span
                      className="block h-full rounded-sm bg-terminal-signal"
                      style={{ width: `${Math.min(theme.confidence * 100, 100)}%` }}
                    />
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-3.5 pb-3 flex flex-col gap-2">
                    <p className="text-xs leading-relaxed text-terminal-muted">
                      {theme.explanation}
                    </p>

                    {theme.impacted_sectors.length > 0 && (
                      <div className="text-[11px] text-terminal-muted-alt">
                        <span className="label">Sectors</span>{" "}
                        <span className="text-terminal-muted">
                          {theme.impacted_sectors.join(", ")}
                        </span>
                      </div>
                    )}

                    {theme.impacted_tickers.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {theme.impacted_tickers.map((t) => (
                          <span key={t} className="tag num">
                            {t}
                          </span>
                        ))}
                      </div>
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
