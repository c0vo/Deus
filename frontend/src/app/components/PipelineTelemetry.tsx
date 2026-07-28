"use client";

interface PipelineTelemetryProps {
  pipelineMetrics?: any[];
  embeddingStatus?: {
    total_articles: number;
    embedded_articles: number;
    pending_articles: number;
    success_rate_pct: number;
    duplicate_articles?: number;
    dedup_pending?: number;
  };
  sentimentDistribution?: {
    bullish: number;
    bearish: number;
    neutral: number;
    total: number;
  };
  /** Moved here from the global header — these are diagnostics, not navigation. */
  dbSizeLabel?: string;
  totalArticles?: number;
  isRunning?: boolean;
}

function Cell({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="bg-bg-card px-3.5 py-2.5">
      <div className="label">{label}</div>
      <div className="num mt-1 text-lg font-medium">{value}</div>
      {note && <div className="num text-[11px] text-terminal-muted-alt">{note}</div>}
    </div>
  );
}

export default function PipelineTelemetry({
  pipelineMetrics,
  embeddingStatus,
  sentimentDistribution,
  dbSizeLabel,
  totalArticles,
  isRunning = false,
}: PipelineTelemetryProps) {
  const last = pipelineMetrics?.length ? pipelineMetrics[0] : null;
  const total = sentimentDistribution?.total || 0;

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          {isRunning && <span className="live-dot text-terminal-green" />}
          Pipeline telemetry
        </h3>
        <span className={`pill ${isRunning ? "pill-ok" : ""}`}>
          {isRunning ? "Running" : "Idle"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-px bg-border-soft">
        <Cell
          label="Embedded"
          value={embeddingStatus?.embedded_articles?.toLocaleString() ?? "—"}
          note={
            embeddingStatus?.pending_articles
              ? `${embeddingStatus.pending_articles.toLocaleString()} pending`
              : undefined
          }
        />
        <Cell
          label="LLM calls"
          value={last?.llm_calls_count?.toLocaleString() ?? "—"}
          note={last ? "last cycle" : undefined}
        />
        <Cell
          label="Classified"
          value={last?.articles_classified?.toLocaleString() ?? "—"}
          note={
            last?.articles_fetched
              ? `of ${last.articles_fetched.toLocaleString()} fetched`
              : undefined
          }
        />
        <Cell
          label="Database"
          value={dbSizeLabel || "—"}
          note={
            totalArticles ? `${totalArticles.toLocaleString()} articles` : undefined
          }
        />
      </div>

      {total > 0 && sentimentDistribution && (
        <div className="card-body border-t border-border-soft">
          <div className="flex justify-between text-[11px] mb-1.5">
            <span className="text-terminal-green">
              Bullish <span className="num">{sentimentDistribution.bullish}</span>
            </span>
            <span className="text-terminal-muted-alt">
              Neutral <span className="num">{sentimentDistribution.neutral}</span>
            </span>
            <span className="text-terminal-red">
              Bearish <span className="num">{sentimentDistribution.bearish}</span>
            </span>
          </div>
          <div className="flex h-1.5 rounded-sm overflow-hidden">
            <span
              className="bg-terminal-green"
              style={{ width: `${(sentimentDistribution.bullish / total) * 100}%` }}
            />
            <span
              className="bg-border-dim"
              style={{ width: `${(sentimentDistribution.neutral / total) * 100}%` }}
            />
            <span
              className="bg-terminal-red"
              style={{ width: `${(sentimentDistribution.bearish / total) * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
