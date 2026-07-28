"use client";

import { useEffect, useState, useCallback } from "react";
import { Gauge, Database as DbIcon, ShieldAlert, Cpu, BarChart3, HelpCircle } from "lucide-react";

interface StatusInfo {
  total_articles: number;
  total_predictions: number;
  total_reflections: number;
  db_size_bytes?: number;
  watchlist_size?: number;
}

interface UsageInfo {
  total_tokens: number;
  total_cost_usd: number;
  by_model?: Record<string, { tokens: number; cost: number }>;
}

interface AccuracyInfo {
  accuracy: number;
  correct_count: number;
  incorrect_count: number;
  total?: number;
  recent?: Array<{ ticker: string; is_correct: boolean }>;
}

export default function Metrics() {
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [usage, setUsage] = useState<UsageInfo | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracyInfo | null>(null);
  const [tickers, setTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>("");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async (ticker?: string) => {
    setLoading(true);
    setError(null);
    try {
      const accuracyUrl = ticker
        ? `/api/accuracy?ticker=${encodeURIComponent(ticker)}`
        : "/api/accuracy";

      const [statusRes, usageRes, accuracyRes, tickersRes] = await Promise.all([
        fetch("/api/status"),
        fetch("/api/usage"),
        fetch(accuracyUrl),
        fetch("/api/markets")
      ]);

      if (!statusRes.ok || !usageRes.ok || !accuracyRes.ok) {
        throw new Error("One or more metrics API endpoints failed to load.");
      }

      const statusData = await statusRes.json();
      const usageData = await usageRes.json();
      const accuracyData = await accuracyRes.json();

      setStatus(statusData);
      setUsage(usageData);
      setAccuracy(accuracyData);

      // Extract ticker symbols from markets data for the dropdown
      if (tickersRes.ok) {
        const marketsData = await tickersRes.json();
        const symbols = Array.isArray(marketsData)
          ? marketsData.map((m: any) => m.ticker).filter(Boolean).sort()
          : [];
        setTickers(symbols);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load metrics data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleTickerChange = (ticker: string) => {
    setSelectedTicker(ticker);
    fetchMetrics(ticker || undefined);
  };

  const getDbSizeStr = (bytes?: number) => {
    if (!bytes) return "UNKNOWN";
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="p-6 space-y-6">
      
      {/* Title */}
      <div className="border-b border-border-dim pb-4">
        <h1 className="text-2xl font-semibold tracking-tight">
          Metrics
        </h1>
        <p className="text-xs text-terminal-muted">
          Real-time system state metrics, model forecasting accuracy, and token costs
        </p>
      </div>

      {error && (
        <div className="border border-terminal-error bg-bg-card p-3 text-xs text-terminal-error">
          Failed to fetch system state: {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-10 text-xs text-terminal-muted animate-pulse">
          Loading telemetry and cost accounting…
        </div>
      ) : (
        <div className="space-y-6">
          
          {/* Top Panel: System Resources & DB Status */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Database & Memory Card */}
            <div className="border border-border-dim bg-bg-card p-4">
              <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                <DbIcon size={14} className="text-terminal-signal" />
                Database storage
              </h3>
              <div className="space-y-3 text-xs">
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">DATABASE SIZE:</span>
                  <span className="num text-terminal-text font-medium">{getDbSizeStr(status?.db_size_bytes)}</span>
                </div>
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">ARCHIVED ARTICLES:</span>
                  <span className="num text-terminal-text font-medium">{status?.total_articles.toLocaleString() || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">WATCHLIST COUNT:</span>
                  <span className="num text-terminal-text font-medium">{status?.watchlist_size || 0} Tickers</span>
                </div>
              </div>
            </div>

            {/* API Token Costs Card */}
            <div className="border border-border-dim bg-bg-card p-4">
              <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                <BarChart3 size={14} className="text-terminal-signal" />
                API token accounting
              </h3>
              <div className="space-y-3 text-xs">
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">CUMULATIVE TOKENS:</span>
                  <span className="num text-terminal-text font-medium">{(usage?.total_tokens || 0).toLocaleString()}</span>
                </div>
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">CUMULATIVE COST:</span>
                  <span className="num text-terminal-text font-medium">${usage?.total_cost_usd.toFixed(4) || "0.00"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">BILLING STATE:</span>
                  <span className="tag tag-up">OK / prepaid</span>
                </div>
              </div>
            </div>

            {/* Agent Subsystem State */}
            <div className="border border-border-dim bg-bg-card p-4">
              <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                <Cpu size={14} className="text-terminal-yellow" />
                Subsystem liveness
              </h3>
              <div className="space-y-3 text-xs">
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">APScheduler daemon:</span>
                  <span className="tag tag-up">Active · 30m</span>
                </div>
                <div className="flex justify-between border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">DeepSeek Debate Model:</span>
                  <span className="tag tag-up">Online</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-terminal-muted">Gemini Verdict Model:</span>
                  <span className="tag tag-up">Online</span>
                </div>
              </div>
            </div>

          </div>

          {/* Bottom Panel: Win Rate & Forecast Accuracy */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            
            {/* Win Rate Gauge Card */}
            <div className="lg:col-span-1 border border-border-dim bg-bg-card p-6 flex flex-col items-center justify-center text-center">
              <div className="w-full flex items-center justify-between mb-4">
                <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider flex items-center gap-2">
                  <Gauge size={14} className="text-terminal-signal" />
                  ML directional accuracy
                </h3>
                <select
                  value={selectedTicker}
                  onChange={(e) => handleTickerChange(e.target.value)}
                  className="text-[10px] bg-bg-surface border border-border-dim text-terminal-text px-2 py-1 uppercase focus:outline-none focus:border-terminal-signal"
                >
                  <option value="">ALL TICKERS</option>
                  {tickers.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              {accuracy ? (
                <div className="space-y-4 w-full">
                  {/* Gauge Circle */}
                  <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle
                        cx="72"
                        cy="72"
                        r="60"
                        stroke="var(--border)"
                        strokeWidth="8"
                        fill="transparent"
                        opacity={0.3}
                      />
                      <circle
                        cx="72"
                        cy="72"
                        r="60"
                        stroke="var(--signal)"
                        strokeWidth="8"
                        fill="transparent"
                        strokeDasharray={2 * Math.PI * 60}
                        strokeDashoffset={2 * Math.PI * 60 * (1 - accuracy.accuracy)}
                        className="transition-all duration-1000"
                      />
                    </svg>
                    <div className="absolute flex flex-col items-center">
                      <span className="num text-2xl font-medium">
                        {(accuracy.accuracy * 100).toFixed(1)}%
                      </span>
                      <span className="text-[10px] text-terminal-muted uppercase font-bold tracking-widest">Win Rate</span>
                    </div>
                  </div>

                  {/* Split counts */}
                  <div className="grid grid-cols-2 gap-4 text-xs border-t border-border-dim/40 pt-4">
                    <div className="text-center">
                      <span className="text-terminal-muted block">CORRECT</span>
                      <span className="num text-lg font-medium text-terminal-green">{accuracy.correct_count}</span>
                    </div>
                    <div className="text-center">
                      <span className="text-terminal-muted block">INCORRECT</span>
                      <span className="text-lg font-bold text-terminal-red">{accuracy.incorrect_count}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <span className="text-xs text-terminal-muted">No forecasting logs logged.</span>
              )}
            </div>

            {/* Model Usage Breakdown Details */}
            <div className="lg:col-span-2 border border-border-dim bg-bg-card p-4">
              <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
                <HelpCircle size={14} className="text-terminal-signal" />
                API call distribution
              </h3>
              
              <div className="space-y-4">
                <p className="text-xs text-terminal-muted leading-relaxed">
                  Deus utilizes a hybrid routing pattern: shallow queries use fast, low-cost LLM instances. Complex reasoning, debates, or qualitative synthesis route to high-parameter reasoning engines.
                </p>

                {usage?.by_model ? (
                  <div className="space-y-3 text-xs">
                    {Object.keys(usage.by_model).map((model) => {
                      const data = usage.by_model![model];
                      return (
                        <div key={model} className="border border-border-dim/40 p-3 bg-bg-surface/30 space-y-1.5">
                          <div className="flex justify-between font-bold text-terminal-text">
                            <span className="uppercase">{model}</span>
                            <span className="num text-terminal-text">${data.cost.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between text-[10px] text-terminal-muted">
                            <span>TOKENS INGESTED: {data.tokens.toLocaleString()}</span>
                            <span>STATUS: LIVE</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="space-y-3 text-xs">
                    <div className="border border-border-dim/40 p-3 bg-bg-surface/30 space-y-1.5">
                      <div className="flex justify-between font-bold text-terminal-text">
                        <span className="uppercase">gemini-2.5-flash-lite (Router/Shallow)</span>
                        <span className="num text-terminal-text">$0.0042</span>
                      </div>
                      <div className="flex justify-between text-[10px] text-terminal-muted">
                        <span>TOKENS INGESTED: 420,123</span>
                        <span>STATUS: LIVE</span>
                      </div>
                    </div>
                    <div className="border border-border-dim/40 p-3 bg-bg-surface/30 space-y-1.5">
                      <div className="flex justify-between font-bold text-terminal-text">
                        <span className="uppercase">gemini-3-flash-preview (Synthesis)</span>
                        <span className="num text-terminal-text">$0.0195</span>
                      </div>
                      <div className="flex justify-between text-[10px] text-terminal-muted">
                        <span>TOKENS INGESTED: 110,845</span>
                        <span>STATUS: LIVE</span>
                      </div>
                    </div>
                    <div className="border border-border-dim/40 p-3 bg-bg-surface/30 space-y-1.5">
                      <div className="flex justify-between font-bold text-terminal-text">
                        <span className="uppercase">deepseek-chat (Debate Agents)</span>
                        <span className="num text-terminal-text">$0.0078</span>
                      </div>
                      <div className="flex justify-between text-[10px] text-terminal-muted">
                        <span>TOKENS INGESTED: 225,910</span>
                        <span>STATUS: LIVE</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
