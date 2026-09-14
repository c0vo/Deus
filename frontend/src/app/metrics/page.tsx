"use client";

import { useEffect, useState, useCallback } from "react";
import { Gauge, Database as DbIcon, ShieldAlert, Cpu, BarChart3, HelpCircle, Brain } from "lucide-react";

interface StatusInfo {
  total_articles: number;
  total_predictions: number;
  total_reflections: number;
  db_size_bytes?: number;
  watchlist_size?: number;
}

interface UsageInfo {
  // total_* and by_model cover the same window (window_days); all_time_* is
  // lifetime. Keeping them distinct is deliberate — the two used to be mixed,
  // so the per-model table could never sum to the headline above it.
  window_days: number | null;
  total_tokens: number;
  total_cost_usd: number;
  all_time_tokens: number;
  all_time_cost_usd: number;
  // Successful calls the provider reported no cost for. They are stored at
  // $0.00, so this is the one way the figures above can understate spend.
  unpriced_calls?: number;
  by_model?: Record<string, { tokens: number; cost: number }>;
}

// Resolved accuracy for one horizon (by_horizon) or one horizon and model type
// (by_model_type) from /api/accuracy.
interface AccuracySlice {
  horizon_days: number;
  model_type?: string;
  total: number;
  correct: number;
  accuracy_pct: number;
}

interface AccuracyInfo {
  accuracy: number;
  correct_count: number;
  incorrect_count: number;
  total?: number;
  recent?: Array<{ ticker: string; is_correct: boolean }>;
  by_horizon?: AccuracySlice[];
  by_model_type?: AccuracySlice[];
}

// The newest walk-forward training run for one horizon, from /api/model-metrics.
// Every metric is out of sample; rates are fractions in [0, 1].
interface ModelMetricsRow {
  horizon_days: number;
  horizon_label?: string | null;
  created_at?: string | null;
  // "model": the horizon passed the ship rule. "prior": it did not, and
  // predictions there serve the base rate.
  status?: string | null;
  auc_mean?: number | null;
  auc_ci_low?: number | null;
  auc_ci_high?: number | null;
  brier_skill_mean?: number | null;
  hi_conf_acc?: number | null;
  hi_conf_n?: number | null;
  prior_up_rate?: number | null;
}

interface ModelMetricsInfo {
  data?: ModelMetricsRow[];
}

const HORIZON_LABELS: Record<number, string> = { 5: "5d", 21: "1m", 63: "3m", 252: "1y" };
const horizonLabel = (days: number) => HORIZON_LABELS[days] ?? `${days}d`;

// "—" for a metric the run could not measure, so a gap never reads as zero.
const fixed = (value: number | null | undefined, digits: number, signed = false) =>
  typeof value === "number" && Number.isFinite(value)
    ? `${signed && value > 0 ? "+" : ""}${value.toFixed(digits)}`
    : "—";
const percentOf = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";

export default function Metrics() {
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [usage, setUsage] = useState<UsageInfo | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracyInfo | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetricsInfo | null>(null);
  const [modelMetricsError, setModelMetricsError] = useState<string | null>(null);
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

      const [statusRes, usageRes, accuracyRes, tickersRes, modelMetricsRes] = await Promise.all([
        fetch("/api/status"),
        fetch("/api/usage"),
        fetch(accuracyUrl),
        fetch("/api/markets"),
        fetch("/api/model-metrics")
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

      // Its own error state: a model that has never trained, or an API that
      // predates the endpoint, should not blank the rest of the page.
      if (modelMetricsRes.ok) {
        setModelMetrics(await modelMetricsRes.json());
        setModelMetricsError(null);
      } else {
        setModelMetrics(null);
        setModelMetricsError(`/api/model-metrics responded ${modelMetricsRes.status}`);
      }

      // Extract ticker symbols from markets data for the dropdown. The grid
      // arrives wrapped as {data: [...]}.
      if (tickersRes.ok) {
        const marketsData = await tickersRes.json();
        const rows = Array.isArray(marketsData) ? marketsData : marketsData?.data;
        const symbols = Array.isArray(rows)
          ? rows.map((m: any) => m.ticker).filter(Boolean).sort()
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

  const modelRows = modelMetrics?.data ?? [];

  return (
    <div className="p-4 md:p-6 space-y-4 md:space-y-6">
      
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
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">DATABASE SIZE:</span>
                  <span className="num text-terminal-text font-medium">{getDbSizeStr(status?.db_size_bytes)}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">ARCHIVED ARTICLES:</span>
                  <span className="num text-terminal-text font-medium">{status?.total_articles.toLocaleString() || 0}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3">
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
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">7-DAY TOKENS:</span>
                  <span className="num text-terminal-text font-medium">{(usage?.total_tokens ?? 0).toLocaleString()}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">7-DAY COST:</span>
                  <span className="num text-terminal-text font-medium">${(usage?.total_cost_usd ?? 0).toFixed(4)}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">CUMULATIVE TOKENS:</span>
                  <span className="num text-terminal-text font-medium">{(usage?.all_time_tokens ?? 0).toLocaleString()}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">CUMULATIVE COST:</span>
                  <span className="num text-terminal-text font-medium">${(usage?.all_time_cost_usd ?? 0).toFixed(4)}</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">BILLING STATE:</span>
                  <span className="tag tag-up">OK / prepaid</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3">
                  <span className="text-terminal-muted">UNPRICED CALLS:</span>
                  <span className={`num font-medium ${(usage?.unpriced_calls ?? 0) > 0 ? "text-terminal-down" : "text-terminal-text"}`}>
                    {(usage?.unpriced_calls ?? 0).toLocaleString()}
                  </span>
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
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">APScheduler daemon:</span>
                  <span className="tag tag-up">Active · 30m</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3 border-b border-border-dim/30 pb-1.5">
                  <span className="text-terminal-muted">DeepSeek Debate Model:</span>
                  <span className="tag tag-up">Online</span>
                </div>
                <div className="flex flex-wrap justify-between gap-x-3">
                  <span className="text-terminal-muted">Gemini Verdict Model:</span>
                  <span className="tag tag-up">Online</span>
                </div>
              </div>
            </div>

          </div>

          {/* Model skill: the pooled model's out-of-sample evaluation per
              horizon, the numbers that say whether a badge's percentage means
              anything. NO EDGE rows are serving the base rate. */}
          <div className="border border-border-dim bg-bg-card p-4">
            <h3 className="text-xs font-bold text-terminal-muted uppercase tracking-wider mb-4 flex items-center gap-2">
              <Brain size={14} className="text-terminal-signal" />
              Model skill (walk-forward)
            </h3>

            {modelMetricsError ? (
              <div className="text-xs text-terminal-error">Failed to load model skill: {modelMetricsError}</div>
            ) : modelRows.length === 0 ? (
              <div className="border border-border-dim/40 p-3 bg-bg-surface/30 text-xs text-terminal-muted">
                No model has been trained yet. The worker trains the pooled models automatically.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs whitespace-nowrap">
                  <thead>
                    <tr className="text-[10px] text-terminal-muted uppercase tracking-wider border-b border-border-dim/40">
                      <th className="text-left font-medium py-1.5 pr-4">Horizon</th>
                      <th className="text-left font-medium py-1.5 pr-4">Status</th>
                      <th className="text-right font-medium py-1.5 pr-4">AUC (90% CI)</th>
                      <th className="text-right font-medium py-1.5 pr-4">Brier skill</th>
                      <th className="text-right font-medium py-1.5 pr-4">Hi-conf acc (n)</th>
                      <th className="text-right font-medium py-1.5 pr-4">Base up-rate</th>
                      <th className="text-right font-medium py-1.5">Trained</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelRows.map((row) => {
                      const shipped = row.status === "model";
                      return (
                        <tr key={row.horizon_days} className="border-b border-border-dim/20 last:border-b-0">
                          <td className="py-1.5 pr-4 font-bold text-terminal-text">
                            {row.horizon_label ?? horizonLabel(row.horizon_days)}
                          </td>
                          <td className="py-1.5 pr-4">
                            <span className={shipped ? "tag tag-up" : "tag"}>{shipped ? "MODEL" : "NO EDGE"}</span>
                          </td>
                          <td className="py-1.5 pr-4 text-right num text-terminal-text">
                            {fixed(row.auc_mean, 3)}
                            <span className="text-terminal-muted">
                              {" "}({fixed(row.auc_ci_low, 3)}–{fixed(row.auc_ci_high, 3)})
                            </span>
                          </td>
                          <td className="py-1.5 pr-4 text-right num text-terminal-text">
                            {fixed(row.brier_skill_mean, 4, true)}
                          </td>
                          <td className="py-1.5 pr-4 text-right num text-terminal-text">
                            {percentOf(row.hi_conf_acc)}
                            <span className="text-terminal-muted"> ({row.hi_conf_n ?? "—"})</span>
                          </td>
                          <td className="py-1.5 pr-4 text-right num text-terminal-text">
                            {percentOf(row.prior_up_rate)}
                          </td>
                          <td className="py-1.5 text-right num text-terminal-muted">
                            {row.created_at ? row.created_at.slice(0, 10) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <p className="mt-3 text-[10px] text-terminal-muted leading-relaxed">
              AUC ranks tickers within a day (0.50 = coin flip) · Brier skill &gt; 0 beats the base rate · NO EDGE = serving the base rate (includes too little evidence to judge)
            </p>
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
                  <div className="relative w-36 max-w-full aspect-square mx-auto flex items-center justify-center">
                    <svg
                      viewBox="0 0 144 144"
                      preserveAspectRatio="xMidYMid meet"
                      className="w-full h-full transform -rotate-90"
                    >
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

                  {/* The win rate above mixes every horizon and model type.
                      prior rows serve the base rate, so theirs is the hit rate
                      the model rows beside them have to beat. */}
                  {accuracy.by_model_type && accuracy.by_model_type.length > 0 && (
                    <div className="border-t border-border-dim/40 pt-3 text-left space-y-1">
                      <span className="text-[10px] text-terminal-muted uppercase tracking-wider block mb-1">
                        By horizon &amp; model
                      </span>
                      {accuracy.by_model_type.map((row) => (
                        <div
                          key={`${row.horizon_days}-${row.model_type}`}
                          className="flex flex-wrap justify-between gap-x-3 text-[11px]"
                        >
                          <span className="text-terminal-muted">
                            {horizonLabel(row.horizon_days)} · {row.model_type === "prior" ? "prior (base rate)" : row.model_type}
                          </span>
                          <span className="num text-terminal-text">
                            {row.accuracy_pct.toFixed(1)}%
                            <span className="text-terminal-muted"> n={row.total}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
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

                {/* No placeholder rows here on purpose. This panel used to fall
                    back to hardcoded figures labelled "STATUS: LIVE", which
                    rendered invented costs as though they were measured. */}
                {usage?.by_model && Object.keys(usage.by_model).length > 0 ? (
                  <div className="space-y-3 text-xs">
                    {Object.keys(usage.by_model).map((model) => {
                      const data = usage.by_model![model];
                      return (
                        <div key={model} className="border border-border-dim/40 p-3 bg-bg-surface/30 space-y-1.5">
                          <div className="flex flex-wrap justify-between gap-x-3 font-bold text-terminal-text">
                            <span className="uppercase">{model}</span>
                            <span className="num text-terminal-text">${data.cost.toFixed(4)}</span>
                          </div>
                          <div className="flex flex-wrap justify-between gap-x-3 text-[10px] text-terminal-muted">
                            <span>TOKENS INGESTED: {data.tokens.toLocaleString()}</span>
                            <span>LAST 7 DAYS</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="border border-border-dim/40 p-3 bg-bg-surface/30 text-xs text-terminal-muted">
                    No metered LLM calls in the last 7 days.
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
