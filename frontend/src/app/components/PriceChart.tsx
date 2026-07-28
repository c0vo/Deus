"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface ChartPoint {
  time: string;
  close: number;
  sma20?: number | null;
}

interface PriceChartProps {
  data: ChartPoint[];
}

/**
 * Price history styled the way brokerages do it: the series is green when the
 * period closed above where it opened and red when below, measured against a
 * dashed baseline at the opening price.
 */
export default function PriceChart({ data }: PriceChartProps) {
  const open = data[0]?.close ?? 0;
  const last = data[data.length - 1]?.close ?? 0;
  const isUp = last >= open;
  const stroke = isUp ? "var(--green)" : "var(--red)";

  return (
    <div className="h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 8, left: -20, bottom: 0 }}>
          <XAxis
            dataKey="time"
            stroke="var(--muted-alt)"
            fontSize={10}
            tickLine={false}
            minTickGap={22}
          />
          <YAxis
            stroke="var(--muted-alt)"
            fontSize={10}
            tickLine={false}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--panel)",
              borderColor: "var(--border)",
              borderRadius: "4px",
              color: "var(--foreground)",
              fontSize: "11px",
            }}
            labelStyle={{ color: "var(--muted)" }}
          />
          {open > 0 && (
            <ReferenceLine
              y={open}
              stroke="var(--muted-alt)"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
          )}
          <Line
            type="monotone"
            dataKey="close"
            stroke={stroke}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="sma20"
            stroke="var(--muted-alt)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
