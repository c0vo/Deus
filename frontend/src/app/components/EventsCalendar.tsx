"use client";

import { useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { EVENT_LABELS, daysUntil } from "../utils/calendar";
import { confidenceTag, DateChip } from "./calendar/EventChrome";
import { fetchJson, nonOverlapping } from "../utils/api";

interface TickerEvent {
  id: number;
  ticker: string;
  event_type: string;
  event_date: string;
  event_title: string;
  confidence: string;
  sector?: string | null;
}

/** One macro_events row as /api/brain/macro-events returns it. */
interface MacroEvent {
  id: number;
  date: string;
  time_et: string | null;
  name: string;
  kind: string;
  importance: number;
  source: string;
}

interface EventsCalendarProps {
  events: TickerEvent[];
  onRemove?: (id: number) => void;
}

/** How many macro rows sit above the ticker events. The card is a glance, not a page. */
const MACRO_ROWS = 5;

/** Days of macro lookahead. Matches the default on /api/brain/macro-events. */
const MACRO_DAYS_AHEAD = 14;

/**
 * Importance as filled dots out of three — the same 1-3 scale the seed uses, so
 * a CPI print (3) reads differently from a month end (1) without needing a
 * second colour. The calendar page's KindMark deliberately does not encode this;
 * here there is room.
 */
function ImportanceDots({ importance }: { importance: number }) {
  const level = Math.max(1, Math.min(importance || 1, 3));
  return (
    <span
      className="shrink-0 inline-flex items-center gap-0.5"
      title={`Importance ${level} of 3`}
      aria-label={`Importance ${level} of 3`}
    >
      {[1, 2, 3].map((i) => (
        <i
          key={i}
          className={`w-1 h-1 rounded-full ${
            i <= level ? "bg-terminal-signal" : "bg-border-dim"
          }`}
        />
      ))}
    </span>
  );
}

export default function EventsCalendar({ events, onRemove }: EventsCalendarProps) {
  const [macro, setMacro] = useState<MacroEvent[]>([]);

  // The macro table is written by the worker (a monthly web top-up, plus the
  // seed on every boot) and read here, so a slow poll is enough — this only
  // needs to catch a deploy that extended the schedule while a tab stayed open.
  useEffect(() => {
    let cancelled = false;

    const load = nonOverlapping(async () => {
      try {
        const json = await fetchJson<{ data?: MacroEvent[] }>(
          `/api/brain/macro-events?days_ahead=${MACRO_DAYS_AHEAD}`
        );
        if (cancelled) return;
        setMacro(json.data || []);
      } catch {
        // A failed macro fetch must not blank the ticker events below it, which
        // arrive as a prop and have nothing to do with this request.
        if (!cancelled) setMacro([]);
      }
    });

    void load();
    const timer = setInterval(load, 10 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const sorted = [...(events || [])].sort((a, b) =>
    a.event_date.localeCompare(b.event_date)
  );
  const macroRows = macro.slice(0, MACRO_ROWS);
  const total = sorted.length + macro.length;

  if (macroRows.length === 0 && sorted.length === 0) {
    return (
      <div className="card">
        <div className="card-head">
          <h3 className="card-title">Economic calendar</h3>
        </div>
        <div className="card-body text-xs text-terminal-muted">
          No upcoming events found.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Economic calendar</h3>
        <span className="num text-[11px] text-terminal-muted-alt">{total}</span>
      </div>

      <div className="flex flex-col max-h-80 overflow-y-auto overscroll-contain">
        {macroRows.map((ev) => {
          const days = daysUntil(ev.date);

          return (
            <div
              key={`macro:${ev.id}`}
              className="flex items-center gap-3 px-3.5 py-2.5 border-b border-border-soft hover:bg-bg-surface transition-colors"
            >
              <DateChip date={ev.date} />

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <ImportanceDots importance={ev.importance} />
                  <span className="text-[13px] font-medium truncate">
                    {EVENT_LABELS[ev.kind] || ev.kind}
                  </span>
                  {ev.source === "web" && (
                    <span className="tag tag-warn">Estimated</span>
                  )}
                </div>
                <div className="text-[11px] text-terminal-muted truncate">
                  {ev.name}
                  {ev.time_et && (
                    <span className="num text-terminal-muted-alt">
                      {" · "}
                      {ev.time_et} ET
                    </span>
                  )}
                </div>
              </div>

              {days >= 0 && (
                <span
                  className={`num shrink-0 text-[11px] ${
                    days <= 3 ? "text-terminal-yellow" : "text-terminal-muted-alt"
                  }`}
                >
                  {days === 0 ? "today" : `${days}d`}
                </span>
              )}
            </div>
          );
        })}

        {sorted.slice(0, 12).map((ev) => {
          const days = daysUntil(ev.event_date);

          return (
            <div
              key={ev.id}
              className="group flex items-center gap-3 px-3.5 py-2.5 border-b border-border-soft last:border-b-0 hover:bg-bg-surface transition-colors"
            >
              <DateChip date={ev.event_date} />

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="num text-[13px] font-medium">{ev.ticker}</span>
                  {confidenceTag(ev.confidence)}
                </div>
                <div className="text-[11px] text-terminal-muted truncate">
                  {ev.event_title || EVENT_LABELS[ev.event_type] || ev.event_type}
                </div>
              </div>

              <div className="shrink-0 flex items-center gap-2">
                {days >= 0 && (
                  <span
                    className={`num text-[11px] ${
                      days <= 3 ? "text-terminal-yellow" : "text-terminal-muted-alt"
                    }`}
                  >
                    {days === 0 ? "today" : `${days}d`}
                  </span>
                )}
                {onRemove && (
                  <button
                    type="button"
                    onClick={() => onRemove(ev.id)}
                    title="Dismiss event"
                    aria-label={`Dismiss ${ev.ticker} event`}
                    className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 text-terminal-muted-alt hover:text-terminal-red transition-all"
                  >
                    <Trash2 size={12} strokeWidth={1.7} />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
