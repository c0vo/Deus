"use client";

export interface StatTile {
  label: string;
  value: string;
  /** Optional secondary reading — a rate, a pending count, a direction. */
  note?: string;
  tone?: "up" | "down" | "neutral";
  /** Only supplied when a real series exists; we never draw an invented trend. */
  series?: number[];
}

interface StatStripProps {
  tiles: StatTile[];
}

const TONE_CLASS: Record<string, string> = {
  up: "text-terminal-green",
  down: "text-terminal-red",
  neutral: "text-terminal-muted-alt",
};

function Sparkline({ points, tone }: { points: number[]; tone: string }) {
  if (points.length < 2) return null;

  const w = 100;
  const h = 22;
  const pad = 2;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = max - min || 1;

  const x = (i: number) => (i / (points.length - 1)) * w;
  const y = (v: number) => pad + (1 - (v - min) / span) * (h - pad * 2);

  const line = points
    .map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)} ${y(v).toFixed(2)}`)
    .join(" ");
  const area = `${line} L${w} ${h} L0 ${h} Z`;
  const stroke = tone === "down" ? "var(--red)" : "var(--green)";

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="mt-2 h-[22px] w-full"
      aria-hidden="true"
    >
      <path d={area} fill={stroke} opacity="0.1" />
      <path
        d={line}
        fill="none"
        stroke={stroke}
        strokeWidth="1.25"
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={w} cy={y(points[points.length - 1])} r="1.6" fill={stroke} />
    </svg>
  );
}

export default function StatStrip({ tiles }: StatStripProps) {
  return (
    <section
      className="grid gap-4 grid-cols-1 sm:grid-cols-2 xl:grid-cols-4"
      aria-label="Key metrics"
    >
      {tiles.map((t) => (
        <div key={t.label} className="card px-3.5 py-3">
          <div className="label">{t.label}</div>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="num text-2xl font-medium tracking-tight">{t.value}</span>
            {t.note && (
              <span className={`text-xs font-medium ${TONE_CLASS[t.tone || "neutral"]}`}>
                {t.note}
              </span>
            )}
          </div>
          {t.series && t.series.length > 1 && (
            <Sparkline points={t.series} tone={t.tone || "neutral"} />
          )}
        </div>
      ))}
    </section>
  );
}
