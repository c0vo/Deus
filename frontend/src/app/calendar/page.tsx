"use client";

import { useState, useEffect, useCallback, useMemo, Suspense } from "react";
import { getApiUrl } from "../utils/api";
import MonthGrid, { CalendarItem } from "../components/calendar/MonthGrid";
import DayAgenda, { EventRow } from "../components/calendar/DayAgenda";
import {
  addMonths,
  addDays,
  buildWeeks,
  formatMonthTitle,
  isoDate,
  startOfMonth,
} from "../utils/calendar";
import { AlertCircle, ChevronLeft, ChevronRight } from "lucide-react";

interface CalendarPayload {
  from: string;
  to: string;
  items: CalendarItem[];
  counts: { event: number; ipo: number };
  sources: { finnhub: boolean };
}

type ViewMode = "month" | "week" | "list";

const MODES: { label: string; value: ViewMode }[] = [
  { label: "Month", value: "month" },
  { label: "Week", value: "week" },
  { label: "List", value: "list" },
];

function CalendarContent() {
  const [anchor, setAnchor] = useState<Date>(() => new Date());
  const [mode, setMode] = useState<ViewMode>("month");
  const [selected, setSelected] = useState<string>(() => isoDate(new Date()));
  const [data, setData] = useState<CalendarPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);

  const weeks = useMemo(
    () => buildWeeks(anchor, mode === "week" ? "week" : "month"),
    [anchor, mode]
  );

  // Fetch the whole visible grid, not the calendar month — otherwise the
  // leading and trailing days of adjacent months render blank and look broken.
  const rangeStart = isoDate(weeks[0][0]);
  const rangeEnd = isoDate(weeks[weeks.length - 1][6]);

  const fetchCalendar = useCallback(async (from: string, to: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(getApiUrl(`/api/calendar?from=${from}&to=${to}`));
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const json = await res.json();
      setData(json.data);
    } catch (err: any) {
      setError(err.message || "Could not load the calendar.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCalendar(rangeStart, rangeEnd);
  }, [fetchCalendar, rangeStart, rangeEnd]);

  const items = useMemo(() => data?.items || [], [data]);

  const itemsByDate = useMemo(() => {
    const map = new Map<string, CalendarItem[]>();
    for (const item of items) {
      const bucket = map.get(item.date);
      if (bucket) bucket.push(item);
      else map.set(item.date, [item]);
    }
    return map;
  }, [items]);

  const step = (direction: number) => {
    const next =
      mode === "week"
        ? addDays(anchor, direction * 7)
        : addMonths(startOfMonth(anchor), direction);
    setAnchor(next);
  };

  const goToday = () => {
    const now = new Date();
    setAnchor(now);
    setSelected(isoDate(now));
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshNote(null);
    try {
      const res = await fetch(getApiUrl("/api/calendar/refresh"), {
        method: "POST",
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || `Server returned ${res.status}`);
      setRefreshNote(
        `Added ${json.data.earnings_added} earnings, ${json.data.ipos_added} IPOs.`
      );
      await fetchCalendar(rangeStart, rangeEnd);
    } catch (err: any) {
      setRefreshNote(err.message || "Refresh failed.");
    } finally {
      setRefreshing(false);
    }
  };

  const total = items.length;
  const selectedItems = itemsByDate.get(selected) || [];
  const finnhubMissing = data ? !data.sources.finnhub : false;

  return (
    <div className="p-5 flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Calendar</h1>
          <p className="mt-0.5 text-[13px] text-terminal-muted">
            Earnings, IPOs and scheduled events across the watchlist
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="btn-ghost"
              onClick={() => step(-1)}
              aria-label="Previous"
            >
              <ChevronLeft size={13} strokeWidth={1.7} />
            </button>
            <button type="button" className="btn-ghost" onClick={goToday}>
              Today
            </button>
            <button
              type="button"
              className="btn-ghost"
              onClick={() => step(1)}
              aria-label="Next"
            >
              <ChevronRight size={13} strokeWidth={1.7} />
            </button>
          </div>

          <div className="seg" role="tablist">
            {MODES.map((m) => (
              <button
                key={m.value}
                role="tab"
                aria-selected={mode === m.value}
                onClick={() => setMode(m.value)}
              >
                {m.label}
              </button>
            ))}
          </div>

          <span className="pill">
            <span className="num">{total}</span> scheduled
          </span>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 px-3.5 py-2.5 rounded border border-terminal-red/40 bg-terminal-red-soft text-xs text-terminal-red">
          <AlertCircle size={14} strokeWidth={1.7} />
          <span>{error}</span>
        </div>
      )}

      {finnhubMissing && (
        <div className="px-3.5 py-2.5 rounded border border-border-dim bg-bg-card text-[11px] text-terminal-muted">
          No calendar sources configured. Set <span className="num">FINNHUB_API_KEY</span>{" "}
          in <span className="num">.env</span> and restart to pull earnings and IPO
          dates.
        </div>
      )}

      {!loading && !error && total === 0 && !finnhubMissing && (
        <div className="card">
          <div className="card-body flex flex-wrap items-center justify-between gap-3">
            <div className="text-xs text-terminal-muted">
              Nothing scheduled between{" "}
              <span className="num">{data?.from}</span> and{" "}
              <span className="num">{data?.to}</span>.
              {refreshNote && (
                <span className="ml-2 text-[11px] text-terminal-muted-alt">
                  {refreshNote}
                </span>
              )}
            </div>
            <button
              type="button"
              className="btn-ghost"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              {refreshing ? "Fetching…" : "Fetch from Finnhub"}
            </button>
          </div>
        </div>
      )}

      <div className="text-[11px] text-terminal-muted-alt">
        {mode === "week" ? "Week of " : ""}
        {formatMonthTitle(anchor)}
        {loading && <span className="ml-2">· loading…</span>}
      </div>

      {mode === "list" ? (
        <div className="card">
          <div className="card-head">
            <h2 className="card-title">Agenda</h2>
            <span className="num text-[11px] text-terminal-muted-alt">{total}</span>
          </div>
          {total === 0 ? (
            <div className="card-body text-xs text-terminal-muted">
              Nothing scheduled in this range.
            </div>
          ) : (
            <div className="flex flex-col">
              {items.map((item) => (
                <EventRow key={item.id} item={item} />
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          <div className="lg:col-span-8">
            <MonthGrid
              weeks={weeks}
              anchorMonth={anchor.getMonth()}
              itemsByDate={itemsByDate}
              selected={selected}
              onSelect={setSelected}
              dense={mode === "week"}
            />
            <div className="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-terminal-muted-alt">
              <span className="inline-flex items-center gap-1.5">
                <i className="w-1.5 h-1.5 rounded-[1px] bg-terminal-signal" />
                Event
              </span>
              <span className="inline-flex items-center gap-1.5">
                <i className="w-1.5 h-1.5 rotate-45 bg-terminal-signal" />
                IPO
              </span>
              <span className="inline-flex items-center gap-1.5">
                <i className="w-1.5 h-1.5 rounded-[1px] bg-terminal-muted-alt" />
                Unconfirmed
              </span>
            </div>
          </div>

          <div className="lg:col-span-4">
            <DayAgenda date={selected} items={selectedItems} />
          </div>
        </div>
      )}
    </div>
  );
}

export default function CalendarPage() {
  return (
    <Suspense
      fallback={
        <div className="p-5 text-sm text-terminal-muted">Loading calendar…</div>
      }
    >
      <CalendarContent />
    </Suspense>
  );
}
