"use client";

import { useState, useEffect, useRef } from "react";

export interface SystemStatus {
  total_articles: number;
  total_predictions: number;
  total_reflections: number;
  db_size_bytes: number;
  watchlist_size: number;
  online: boolean;
}

const INITIAL: SystemStatus = {
  total_articles: 0,
  total_predictions: 0,
  total_reflections: 0,
  db_size_bytes: 0,
  watchlist_size: 0,
  online: false,
};

export function useSystemStatus() {
  const [status, setStatus] = useState<SystemStatus>(INITIAL);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/status");
        if (!res.ok) throw new Error("Status endpoint unavailable");
        const data = await res.json();
        if (mounted.current) {
          setStatus({
            total_articles: data.total_articles ?? 0,
            total_predictions: data.total_predictions ?? 0,
            total_reflections: data.total_reflections ?? 0,
            db_size_bytes: data.db_size_bytes ?? 0,
            watchlist_size: data.watchlist_size ?? 0,
            online: true,
          });
        }
      } catch {
        if (mounted.current) {
          setStatus((prev) => ({ ...prev, online: false }));
        }
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30_000);
    return () => {
      mounted.current = false;
      clearInterval(interval);
    };
  }, []);

  const getDbSizeStr = (bytes?: number) => {
    if (!bytes) return "—";
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return { status, getDbSizeStr };
}
