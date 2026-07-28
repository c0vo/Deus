"use client";

import { Trash2 } from "lucide-react";

interface TickerEvent {
  id: number;
  ticker: string;
  event_type: string;
  event_date: string;
  event_title: string;
  confidence: string;
  sector?: string | null;
}

interface EventsCalendarProps {
  events: TickerEvent[];
  onRemove?: (id: number) => void;
}

const EVENT_LABELS: Record<string, string> = {
  earnings: "Earnings",
  product_launch: "Product launch",
  investor_day: "Investor day",
  fda_decision: "FDA decision",
  conference: "Conference",
  dividend: "Dividend",
  split: "Stock split",
  acquisition: "Acquisition",
};

/** Confidence is the only thing here that warrants colour. */
function confidenceTag(confidence: string) {
  if (confidence === "confirmed") return <span className="tag tag-up">Confirmed</span>;
  if (confidence === "estimated") return <span className="tag tag-warn">Estimated</span>;
  return <span className="tag">{confidence || "Unconfirmed"}</span>;
}

export default function EventsCalendar({ events, onRemove }: EventsCalendarProps) {
  if (!events || events.length === 0) {
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

  const daysUntil = (dateStr: string): number => {
    const eventDate = new Date(dateStr);
    if (isNaN(eventDate.getTime())) return -1;
    return Math.ceil((eventDate.getTime() - Date.now()) / 86_400_000);
  };

  const sorted = [...events].sort((a, b) =>
    a.event_date.localeCompare(b.event_date)
  );

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Economic calendar</h3>
        <span className="num text-[11px] text-terminal-muted-alt">
          {sorted.length}
        </span>
      </div>

      <div className="flex flex-col max-h-80 overflow-y-auto">
        {sorted.slice(0, 12).map((ev) => {
          const days = daysUntil(ev.event_date);
          const date = new Date(ev.event_date);
          const valid = !isNaN(date.getTime());

          return (
            <div
              key={ev.id}
              className="group flex items-center gap-3 px-3.5 py-2.5 border-b border-border-soft last:border-b-0 hover:bg-bg-surface transition-colors"
            >
              <div className="shrink-0 w-10 py-1 rounded bg-bg-surface text-center leading-tight">
                <span className="block text-[9px] font-semibold tracking-widest text-terminal-muted-alt">
                  {valid
                    ? date.toLocaleDateString("en-US", { month: "short" }).toUpperCase()
                    : "—"}
                </span>
                <span className="num block text-[13px] font-medium">
                  {valid ? date.getDate() : "—"}
                </span>
              </div>

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
