"use client";

import { useEffect, useRef, useCallback } from "react";
import { getApiUrl } from "../../utils/api";

/**
 * State of the classification queue, published by the worker's backlog job and
 * also included in the SSE snapshot so a freshly-opened page is not blank.
 *
 * `pending` is the real unclassified count. Nothing on the dashboard reported it
 * before: the "pending" number was `total_articles - embedded_articles`, which
 * counts pre-filtered noise forever and says nothing about classification.
 */
export interface ClassificationStatus {
  /** Unclassified, non-duplicate rows — the backlog. */
  pending: number;
  /** Of those, the ones inside the age window with retries left. */
  pending_in_window: number;
  /** Unclassified rows that have used up their retry budget. */
  exhausted: number;
  stale: number;
  error: number;
  noise: number;
  /** Corpus total, excluding noise/error/stale. */
  classified: number;
  embedding_pending: number;
  embedding_exhausted: number;
  embedding_skipped_noise: number;
  /** False when no MODEL_CLASSIFIER slug is set — ingest is embedding-only. */
  classifier_configured?: boolean;
  /** Fields below come from the last backlog run, when there has been one. */
  last_run_at?: string;
  last_run_classified?: number;
  trigger?: string;
  candidates?: number;
  failed?: number;
  stale_marked?: number;
  error_requeued?: number;
  duration_s?: number;
  configured?: boolean;
}

export interface BrainSnapshot {
  articles: any[];
  embedding_status: {
    total_articles: number;
    embedded_articles: number;
    pending_articles: number;
    success_rate_pct: number;
    /** Rows skipped by the embed pass because they are already noise. */
    embedding_skipped_noise?: number;
    /** Articles flagged as semantic duplicates of an earlier story. */
    duplicate_articles?: number;
    unique_articles?: number;
    dedup_pending?: number;
  };
  classification_status?: ClassificationStatus;
  sentiment_distribution: {
    bullish: number;
    bearish: number;
    neutral: number;
    total: number;
  };
  sector_heatmap: any[];
  pipeline_metrics: any[];
}

export interface SSEOptions {
  onSnapshot?: (data: BrainSnapshot) => void;
  onPipelineStatus?: (data: any) => void;
  onNewArticles?: (data: any) => void;
  onSectorHeatmap?: (data: any) => void;
  onRotationSignal?: (data: any) => void;
  onIpoAlert?: (data: any) => void;
  onEventsUpdated?: (data: any) => void;
  onTrendForecast?: (data: any) => void;
  onHotTickers?: (data: any) => void;
  onMarketTicker?: (data: any) => void;
  onSentimentDistribution?: (data: any) => void;
  onEmbeddingStatus?: (data: any) => void;
  onClassificationStatus?: (data: ClassificationStatus) => void;
  onThesisUpdate?: (data: any) => void;
  onMacroThemes?: (data: unknown) => void;
  /** One pushed alert row, same shape as a /api/alerts entry. */
  onAlert?: (data: unknown) => void;
  /** The worker published a new weekly-tip digest; refetch /api/digests/latest. */
  onWeeklyTip?: (data: unknown) => void;
  onError?: (error: string) => void;
  onReconnect?: () => void;
}

export function useBrainSSE(options: SSEOptions) {
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (readerRef.current) {
      readerRef.current.cancel().catch(() => {});
      readerRef.current = null;
    }
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const connect = useCallback(async () => {
    disconnect();

    abortRef.current = new AbortController();

    try {
      const response = await fetch(getApiUrl("/api/brain/stream"), {
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`SSE connection failed: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No readable stream");
      readerRef.current = reader;

      const decoder = new TextDecoder("utf-8");
      let partialLine = "";
      let currentEvent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunk).split("\n");
        partialLine = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.substring(7).trim();
            continue;
          }
          if (line.startsWith("data: ")) {
            const dataStr = line.substring(6).trim();
            if (!dataStr) continue;

            let data: any;
            try {
              data = JSON.parse(dataStr);
            } catch {
              continue;
            }

            const { onSnapshot, onPipelineStatus, onNewArticles, onSectorHeatmap, onRotationSignal, onIpoAlert, onEventsUpdated, onTrendForecast, onHotTickers, onMarketTicker, onSentimentDistribution, onEmbeddingStatus, onClassificationStatus, onThesisUpdate, onMacroThemes, onAlert, onWeeklyTip, onReconnect } = optionsRef.current;

            switch (currentEvent) {
              case "snapshot":
                onSnapshot?.(data);
                onReconnect?.();
                break;
              case "pipeline_status":
                onPipelineStatus?.(data);
                break;
              case "new_articles":
                onNewArticles?.(data);
                break;
              case "sector_heatmap":
                onSectorHeatmap?.(data);
                break;
              case "rotation_signal":
                onRotationSignal?.(data);
                break;
              case "ipo_alert":
                onIpoAlert?.(data);
                break;
              case "events_updated":
                onEventsUpdated?.(data);
                break;
              case "trend_forecast":
                onTrendForecast?.(data);
                break;
              case "hot_tickers":
                onHotTickers?.(data);
                break;
              case "market_ticker":
                onMarketTicker?.(data);
                break;
              case "sentiment_distribution":
                onSentimentDistribution?.(data);
                break;
              case "embedding_status":
                onEmbeddingStatus?.(data);
                break;
              case "classification_status":
                onClassificationStatus?.(data as ClassificationStatus);
                break;
              case "thesis_update":
                onThesisUpdate?.(data);
                break;
              case "macro_themes":
                onMacroThemes?.(data);
                break;
              case "alert":
                onAlert?.(data);
                break;
              case "weekly_tip":
                onWeeklyTip?.(data);
                break;
              case "heartbeat":
                // keep-alive, no action needed
                break;
            }
            currentEvent = "";
          }
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        optionsRef.current.onError?.(err.message || "SSE disconnected");
        // Auto-reconnect after 10 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 10000);
      }
    }
  }, [disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  return { connect, disconnect };
}
