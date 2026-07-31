"use client";

import { parseISODate } from "../../utils/calendar";

/** Confidence is the only thing here that warrants colour. */
export function confidenceTag(confidence: string) {
  if (confidence === "confirmed") return <span className="tag tag-up">Confirmed</span>;
  if (confidence === "estimated") return <span className="tag tag-warn">Estimated</span>;
  return <span className="tag">{confidence || "Unconfirmed"}</span>;
}

/** The month/day block used by every event row. */
export function DateChip({ date }: { date: string }) {
  const parsed = parseISODate(date);

  return (
    <div className="shrink-0 w-10 py-1 rounded bg-bg-surface text-center leading-tight">
      <span className="block text-[9px] font-semibold tracking-widest text-terminal-muted-alt">
        {parsed
          ? parsed.toLocaleDateString("en-US", { month: "short" }).toUpperCase()
          : "—"}
      </span>
      <span className="num block text-[13px] font-medium">
        {parsed ? parsed.getDate() : "—"}
      </span>
    </div>
  );
}

/**
 * Shape carries kind, colour carries confidence — so the grid stays inside the
 * one-accent rule instead of inventing a hue per event type.
 */
export function KindMark({
  kind,
  confidence,
}: {
  kind: string;
  confidence: string;
}) {
  const shape = kind === "ipo" ? "rotate-45" : "rounded-[1px]";
  const tone =
    confidence === "confirmed" ? "bg-terminal-signal" : "bg-terminal-muted-alt";
  return <i className={`w-1.5 h-1.5 shrink-0 ${shape} ${tone}`} />;
}
