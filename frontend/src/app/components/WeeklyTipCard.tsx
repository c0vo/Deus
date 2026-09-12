"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

/** One tip as stored in the digest's facts_json by pipeline/weekly_tip.py. */
export interface WeeklyTip {
  title: string;
  precedent: string;
  evidence: string;
  action: string;
  severity: "info" | "watch" | "warning";
}

/** A seasonality effect — the stat_line is already a finished sentence. */
export interface WeeklyTipEffect {
  name: string;
  window: string;
  stat_line: string;
}

export interface WeeklyTipDigest {
  id: number | null;
  period_start: string | null;
  period_end: string | null;
  model: string | null;
  created_at: string | null;
  tips: WeeklyTip[];
  tip_status: string | null;
  facts: { seasonality?: WeeklyTipEffect[] } | null;
}

interface WeeklyTipCardProps {
  digest: WeeklyTipDigest | null;
  loading?: boolean;
}

// Why the digest carries no tips. The backend distinguishes these so the card
// can say which one happened rather than rendering an empty list that looks
// like a quiet week.
const STATUS_NOTES: Record<string, string> = {
  not_configured: "MODEL_WEEKLY_TIP not configured — facts only",
  failed: "The tip model call failed — facts only",
  empty: "No tip survived fact-checking — facts only",
};

const SEVERITY_CLASS: Record<WeeklyTip["severity"], string> = {
  warning: "text-terminal-red",
  watch: "text-terminal-signal",
  info: "text-terminal-muted-alt",
};

export default function WeeklyTipCard({ digest, loading }: WeeklyTipCardProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const tips = digest?.tips ?? [];
  const effects = digest?.facts?.seasonality ?? [];
  const note = digest?.tip_status ? STATUS_NOTES[digest.tip_status] : undefined;

  const period =
    digest?.period_start && digest?.period_end
      ? `${digest.period_start} → ${digest.period_end}`
      : digest?.created_at?.slice(0, 10);

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Weekly tip</h3>
        {period && (
          <span className="num text-[11px] text-terminal-muted-alt">{period}</span>
        )}
      </div>

      {loading && !digest ? (
        <div className="card-body text-center text-xs text-terminal-muted">
          Loading the latest weekly tip…
        </div>
      ) : !digest ? (
        <div className="card-body text-center text-xs text-terminal-muted">
          No weekly tip generated yet — it runs every Sunday evening.
        </div>
      ) : (
        <div className="flex flex-col max-h-80 overflow-y-auto overscroll-contain">
          {note && (
            <div className="px-3.5 py-2 text-[11px] text-terminal-red border-b border-border-soft">
              {note}
            </div>
          )}

          {tips.map((tip, idx) => {
            const isExpanded = expandedIdx === idx;
            return (
              <div
                key={`${tip.title}-${idx}`}
                className="border-b border-border-soft last:border-b-0"
              >
                <button
                  type="button"
                  onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                  aria-expanded={isExpanded}
                  className="w-full text-left px-3.5 py-2.5 hover:bg-bg-surface transition-colors"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[13px] font-medium leading-snug text-pretty">
                      {tip.title}
                    </span>
                    <span className="shrink-0 flex items-center gap-2">
                      <span
                        className={`tag uppercase text-[10px] ${SEVERITY_CLASS[tip.severity]}`}
                      >
                        {tip.severity}
                      </span>
                      <ChevronDown
                        size={13}
                        strokeWidth={1.7}
                        className={`text-terminal-muted-alt transition-transform ${
                          isExpanded ? "rotate-180" : ""
                        }`}
                      />
                    </span>
                  </div>

                  <p className="mt-1 text-[11px] leading-relaxed text-terminal-muted">
                    {tip.precedent}
                  </p>
                </button>

                {isExpanded && (
                  <div className="px-3.5 pb-3 flex flex-col gap-2">
                    <div className="text-[11px] text-terminal-muted-alt">
                      <span className="label">Evidence</span>{" "}
                      <span className="text-terminal-muted">{tip.evidence}</span>
                    </div>
                    <div className="text-[11px] text-terminal-muted-alt">
                      <span className="label">Action</span>{" "}
                      <span className="text-terminal-muted">{tip.action}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* The measured precedents, shown whether or not a model turned them
              into tips — they are the part that is always true. */}
          {effects.length > 0 && (
            <div className="px-3.5 py-2.5 flex flex-col gap-1.5 border-t border-border-soft">
              <span className="label">Seasonality</span>
              {effects.map((effect, idx) => (
                <div key={`${effect.name}-${idx}`} className="text-[11px] leading-relaxed">
                  <span className="text-terminal-muted-alt">{effect.name}</span>{" "}
                  <span className="num text-terminal-muted">{effect.stat_line}</span>
                </div>
              ))}
            </div>
          )}

          {tips.length === 0 && effects.length === 0 && (
            <div className="card-body text-center text-xs text-terminal-muted">
              This week&apos;s digest carried no computable precedents.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
