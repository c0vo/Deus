"use client";

import type { ClassificationStatus } from "../brain/hooks/useBrainSSE";

interface PipelineTelemetryProps {
  pipelineMetrics?: any[];
  embeddingStatus?: {
    total_articles: number;
    embedded_articles: number;
    pending_articles: number;
    success_rate_pct: number;
    embedding_skipped_noise?: number;
    duplicate_articles?: number;
    dedup_pending?: number;
  };
  /**
   * Classification queue. Absent until the first snapshot or backlog publish
   * arrives, which is why every read below is optional.
   */
  classificationStatus?: ClassificationStatus;
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
  wide = false,
}: {
  label: string;
  value: string;
  note?: string;
  /** Fill the row. There are five cells in a two-column grid, so one must. */
  wide?: boolean;
}) {
  return (
    <div
      className={`bg-bg-card px-3.5 py-2.5 min-w-0${wide ? " col-span-2" : ""}`}
    >
      <div className="label">{label}</div>
      <div className="num mt-1 text-lg font-medium">{value}</div>
      {note && (
        <div className="num text-[11px] text-terminal-muted-alt break-words">{note}</div>
      )}
    </div>
  );
}

export default function PipelineTelemetry({
  pipelineMetrics,
  embeddingStatus,
  classificationStatus,
  sentimentDistribution,
  dbSizeLabel,
  totalArticles,
  isRunning = false,
}: PipelineTelemetryProps) {
  const last = pipelineMetrics?.length ? pipelineMetrics[0] : null;
  const total = sentimentDistribution?.total || 0;

  // An unset MODEL_CLASSIFIER is the one failure where every other number here
  // still looks healthy — articles arrive, embeddings are written — so it gets a
  // pill of its own rather than a subtitle. Strict `=== false`: undefined means
  // the status has not arrived yet, which is not the same as "not configured".
  const classifierMissing = classificationStatus?.classifier_configured === false;

  const unclassifiedNote = [
    classificationStatus?.last_run_classified
      ? `+${classificationStatus.last_run_classified.toLocaleString()} last run`
      : null,
    classificationStatus?.exhausted
      ? `${classificationStatus.exhausted.toLocaleString()} exhausted`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">
          {isRunning && <span className="live-dot text-terminal-green" />}
          Pipeline telemetry
        </h3>
        <div className="flex items-center gap-2">
          {classifierMissing && (
            <span className="pill pill-bad">
              <span className="pill-dot" />
              Classifier model not set
            </span>
          )}
          <span className={`pill ${isRunning ? "pill-ok" : ""}`}>
            {isRunning ? "Running" : "Idle"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-px bg-border-soft">
        <Cell
          label="Embedded"
          value={embeddingStatus?.embedded_articles?.toLocaleString() ?? "—"}
          // "pending" used to be total − embedded, which included every noise
          // row the embed pass skips on purpose and so never went down.
          note={
            embeddingStatus?.pending_articles
              ? `${embeddingStatus.pending_articles.toLocaleString()} to embed`
              : undefined
          }
        />
        <Cell
          label="LLM calls"
          // Real now: every tracked call increments config.usage.tally and the
          // cycle records the difference. This read 0 on every row for the life
          // of the table because nothing ever incremented the counter.
          value={last?.llm_calls_count?.toLocaleString() ?? "—"}
          note={last ? "last cycle" : undefined}
        />
        <Cell
          label="Unclassified"
          value={classificationStatus?.pending?.toLocaleString() ?? "—"}
          note={unclassifiedNote || undefined}
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
          wide
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
