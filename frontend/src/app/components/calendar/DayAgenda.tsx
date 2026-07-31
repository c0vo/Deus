"use client";

import { EVENT_LABELS, daysUntil, formatDayTitle } from "../../utils/calendar";
import { confidenceTag, DateChip } from "./EventChrome";
import type { CalendarItem } from "./MonthGrid";

export function EventRow({ item }: { item: CalendarItem }) {
  const days = daysUntil(item.date);

  return (
    <div className="flex items-start gap-3 px-3.5 py-2.5 border-b border-border-soft last:border-b-0 hover:bg-bg-surface transition-colors">
      <DateChip date={item.date} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          {item.ticker && (
            <span className="num text-[13px] font-medium">{item.ticker}</span>
          )}
          {confidenceTag(item.confidence)}
          <span className="tag">
            {EVENT_LABELS[item.event_type] || item.event_type}
          </span>
        </div>
        <div className="mt-0.5 text-[11px] text-terminal-muted">{item.title}</div>
        {item.detail && (
          <div className="text-[11px] text-terminal-muted-alt">{item.detail}</div>
        )}
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
}

interface DayAgendaProps {
  date: string;
  items: CalendarItem[];
}

export default function DayAgenda({ date, items }: DayAgendaProps) {
  return (
    <div className="card">
      <div className="card-head">
        <h2 className="card-title">{formatDayTitle(date)}</h2>
        <span className="num text-[11px] text-terminal-muted-alt">
          {items.length}
        </span>
      </div>

      {items.length === 0 ? (
        <div className="card-body text-xs text-terminal-muted">
          Nothing scheduled on this day.
        </div>
      ) : (
        <div className="flex flex-col max-h-[520px] overflow-y-auto">
          {items.map((item) => (
            <EventRow key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}
