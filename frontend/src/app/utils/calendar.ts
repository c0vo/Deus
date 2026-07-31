/**
 * Date maths for the calendar page.
 *
 * Two traps, both handled here rather than at each call site:
 *
 * 1. `new Date("2026-08-26")` parses as UTC midnight, so `.getDate()` returns
 *    the 25th anywhere west of Greenwich. Every date string is parsed at local
 *    noon instead — also immune to DST shifts.
 * 2. `toISOString().slice(0,10)` converts back through UTC and can shift the
 *    day again. Grid cells are keyed by date string, so a one-day shift files
 *    every event into the wrong cell. Format from the local getters instead.
 */

export const EVENT_LABELS: Record<string, string> = {
  earnings: "Earnings",
  product_launch: "Product launch",
  investor_day: "Investor day",
  fda_decision: "FDA decision",
  conference: "Conference",
  dividend: "Dividend",
  split: "Stock split",
  acquisition: "Acquisition",
  ipo: "IPO",
};

export const WEEKDAY_LABELS = ["M", "T", "W", "T", "F", "S", "S"];

/** Local-noon parse. Returns null for anything unusable. */
export function parseISODate(value: string | null | undefined): Date | null {
  if (!value) return null;
  const parts = value.slice(0, 10).split("-");
  if (parts.length !== 3) return null;
  const [y, m, d] = parts.map(Number);
  if (!y || !m || !d) return null;
  const date = new Date(y, m - 1, d, 12);
  return isNaN(date.getTime()) ? null : date;
}

/** Local YYYY-MM-DD. Never round-trips through UTC. */
export function isoDate(date: Date): string {
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${m}-${d}`;
}

export function startOfMonth(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), 1, 12);
}

export function addMonths(date: Date, n: number): Date {
  return new Date(date.getFullYear(), date.getMonth() + n, 1, 12);
}

export function addDays(date: Date, n: number): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + n, 12);
}

/** Monday of the week containing `date`. Weeks run Mon-Sun. */
export function mondayOf(date: Date): Date {
  const offset = (date.getDay() + 6) % 7;
  return addDays(date, -offset);
}

/**
 * The visible grid: whole Mon-Sun weeks covering `anchor`'s month (5 or 6 rows),
 * or the single week containing `anchor`.
 */
export function buildWeeks(anchor: Date, mode: "month" | "week"): Date[][] {
  if (mode === "week") {
    const start = mondayOf(anchor);
    return [Array.from({ length: 7 }, (_, i) => addDays(start, i))];
  }

  const first = startOfMonth(anchor);
  const start = mondayOf(first);
  const monthEnd = new Date(anchor.getFullYear(), anchor.getMonth() + 1, 0, 12);

  const weeks: Date[][] = [];
  let cursor = start;
  while (cursor <= monthEnd || weeks.length < 5) {
    weeks.push(Array.from({ length: 7 }, (_, i) => addDays(cursor, i)));
    cursor = addDays(cursor, 7);
    if (weeks.length >= 6) break;
  }
  return weeks;
}

/** Whole days from today to `iso`. Negative for past dates. */
export function daysUntil(iso: string): number {
  const target = parseISODate(iso);
  if (!target) return -1;
  const today = new Date();
  today.setHours(12, 0, 0, 0);
  return Math.round((target.getTime() - today.getTime()) / 86_400_000);
}

export function formatMonthTitle(date: Date): string {
  return date.toLocaleDateString("en-US", { month: "long", year: "numeric" });
}

export function formatDayTitle(iso: string): string {
  const date = parseISODate(iso);
  if (!date) return iso;
  return date.toLocaleDateString("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
  });
}
