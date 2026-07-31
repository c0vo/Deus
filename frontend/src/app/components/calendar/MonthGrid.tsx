"use client";

import { isoDate, WEEKDAY_LABELS } from "../../utils/calendar";
import { KindMark } from "./EventChrome";

export interface CalendarItem {
  id: string;
  kind: "event" | "ipo";
  date: string;
  ticker: string | null;
  title: string;
  event_type: string;
  confidence: string;
  source: string | null;
  sector: string | null;
  detail: string;
  raw_id: number;
}

interface MonthGridProps {
  weeks: Date[][];
  /** 0-11. Days outside it are dimmed as leading/trailing filler. */
  anchorMonth: number;
  itemsByDate: Map<string, CalendarItem[]>;
  selected: string;
  onSelect: (iso: string) => void;
  /** Week mode gets taller cells and room for more rows. */
  dense?: boolean;
}

const MAX_ROWS = 3;
const MAX_ROWS_DENSE = 8;

export default function MonthGrid({
  weeks,
  anchorMonth,
  itemsByDate,
  selected,
  onSelect,
  dense = false,
}: MonthGridProps) {
  const today = isoDate(new Date());
  const maxRows = dense ? MAX_ROWS_DENSE : MAX_ROWS;

  return (
    <div className="card overflow-hidden">
      <div className="grid grid-cols-7 border-b border-border-soft">
        {WEEKDAY_LABELS.map((label, i) => (
          // Keyed by index — "T" and "S" each appear twice.
          <div key={i} className="label text-center py-2">
            {label}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-7 gap-px bg-border-soft">
        {weeks.flat().map((day) => {
          const iso = isoDate(day);
          const items = itemsByDate.get(iso) || [];
          const inMonth = day.getMonth() === anchorMonth;
          const isToday = iso === today;
          const isSelected = iso === selected;

          return (
            <button
              key={iso}
              type="button"
              aria-pressed={isSelected}
              aria-label={`${day.toLocaleDateString("en-GB", {
                weekday: "long",
                day: "numeric",
                month: "long",
                year: "numeric",
              })}, ${items.length} ${items.length === 1 ? "item" : "items"}`}
              onClick={() => onSelect(iso)}
              className={`flex flex-col gap-1 p-1.5 text-left transition-colors ${
                dense ? "min-h-[220px]" : "min-h-[84px]"
              } ${
                isSelected
                  ? "bg-bg-surface-hover"
                  : inMonth
                  ? "bg-bg-card hover:bg-bg-surface"
                  : "bg-bg-main hover:bg-bg-surface"
              }`}
            >
              <span
                className={`num text-[11px] ${
                  isToday
                    ? "text-terminal-signal font-semibold"
                    : inMonth
                    ? "text-terminal-muted"
                    : "text-terminal-muted-alt/50"
                }`}
              >
                {day.getDate()}
              </span>

              {items.slice(0, maxRows).map((item) => (
                <span
                  key={item.id}
                  className="flex items-center gap-1 min-w-0 text-[10px] leading-tight text-terminal-text"
                >
                  <KindMark kind={item.kind} confidence={item.confidence} />
                  <span className="num truncate">
                    {item.ticker || item.title}
                  </span>
                </span>
              ))}

              {items.length > maxRows && (
                <span className="num text-[10px] text-terminal-muted-alt">
                  +{items.length - maxRows}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
