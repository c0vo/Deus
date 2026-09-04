"use client";

import { useEffect, useRef, useState } from "react";
import { fetchJson, nonOverlapping } from "../utils/api";
import { useReducedMotion } from "../hooks/useReducedMotion";

/** Scroll rate in px/sec. Real brokerage tapes run roughly 60-100. */
const TAPE_SPEED = 70;

interface TickerSnapshot {
  ticker: string;
  price: number;
  daily_change_pct: number;
}

export default function TickerTape() {
  const [items, setItems] = useState<TickerSnapshot[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const reducedMotion = useReducedMotion();
  const trackRef = useRef<HTMLDivElement>(null);
  const offsetRef = useRef(0);

  useEffect(() => {
    const fetchPrices = nonOverlapping(async () => {
      // A backgrounded tab has nothing to show; polling from it is pure cost.
      if (document.hidden) return;
      try {
        const json = await fetchJson<{ data?: any[] }>("/api/markets", {
          timeoutMs: 6000,
        });
        const data = (json.data || []).map((d: any) => ({
          ticker: d.ticker,
          price: d.current_price || d.price || 0,
          daily_change_pct: d.daily_change_pct || 0,
        }));
        setItems(data);
      } catch {
        // silently fail — the tape is ambient, not critical
      }
    });

    fetchPrices();
    // 30s rather than 10s: quotes are refreshed server-side on a timer now, so
    // polling faster than that only multiplies requests without adding data.
    const interval = setInterval(fetchPrices, 30000);
    // Catch up immediately when the tab comes back to the foreground.
    document.addEventListener("visibilitychange", fetchPrices);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", fetchPrices);
    };
  }, []);

  // The scroll is driven from JS rather than a CSS keyframe animation, so the
  // blanket reduced-motion rule in globals.css cannot silently freeze it.
  // Motion now follows the app-level toggle in the header rather than the OS
  // media query directly — see utils/motion.ts for why.
  useEffect(() => {
    if (reducedMotion || isPaused || items.length === 0) return;
    const el = trackRef.current;
    if (!el) return;

    let frame = 0;
    let last = performance.now();

    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;

      // One third of the strip is a full copy of the ticker list, so wrapping
      // at that width makes the loop seamless.
      const copyWidth = el.scrollWidth / 3;
      if (copyWidth > 0) {
        offsetRef.current = (offsetRef.current + TAPE_SPEED * dt) % copyWidth;
        el.style.transform = `translate3d(${-offsetRef.current}px, 0, 0)`;
      }

      frame = requestAnimationFrame(step);
    };

    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [reducedMotion, isPaused, items.length]);

  if (items.length === 0) return null;

  const renderItem = (item: TickerSnapshot, idx: number) => {
    const isUp = item.daily_change_pct >= 0;
    return (
      <span
        key={`${item.ticker}-${idx}`}
        className="inline-flex items-baseline gap-1.5 px-4 text-xs whitespace-nowrap border-r border-border-soft"
      >
        <span className="font-medium text-terminal-text">{item.ticker}</span>
        <span className="num text-terminal-muted">{item.price.toFixed(2)}</span>
        <span className={`num ${isUp ? "text-terminal-green" : "text-terminal-red"}`}>
          {isUp ? "+" : "−"}
          {Math.abs(item.daily_change_pct).toFixed(2)}%
        </span>
      </span>
    );
  };

  // Under reduced motion a frozen marquee would hide every ticker past the
  // fold, so the strip becomes manually scrollable instead of animated.
  if (reducedMotion) {
    return (
      <div
        className="h-[34px] overflow-x-auto border-b border-border-dim bg-bg-card"
        role="region"
        aria-label="Tracked tickers"
      >
        <div className="flex items-center h-full w-max">
          {items.map(renderItem)}
        </div>
      </div>
    );
  }

  // Tripled so the wrap point is never visible.
  const tickerItems = [...items, ...items, ...items];

  return (
    <div
      className="h-[34px] overflow-hidden border-b border-border-dim bg-bg-card"
      onMouseEnter={() => setIsPaused(true)}
      onTouchStart={() => setIsPaused(true)}
      onTouchEnd={() => setIsPaused(false)}
      onTouchCancel={() => setIsPaused(false)}
      onMouseLeave={() => setIsPaused(false)}
      role="region"
      aria-label="Tracked tickers"
    >
      <div ref={trackRef} className="flex items-center h-full w-max will-change-transform">
        {tickerItems.map(renderItem)}
      </div>
    </div>
  );
}
