"use client";

import { useEffect, useRef, useCallback } from "react";
import { getApiUrl } from "../../utils/api";

export interface BrainSnapshot {
  articles: any[];
  embedding_status: {
    total_articles: number;
    embedded_articles: number;
    pending_articles: number;
    success_rate_pct: number;
    /** Articles flagged as semantic duplicates of an earlier story. */
    duplicate_articles?: number;
    unique_articles?: number;
    dedup_pending?: number;
  };
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

            const { onSnapshot, onPipelineStatus, onNewArticles, onSectorHeatmap, onRotationSignal, onIpoAlert, onEventsUpdated, onTrendForecast, onHotTickers, onMarketTicker, onSentimentDistribution, onEmbeddingStatus, onReconnect } = optionsRef.current;

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
