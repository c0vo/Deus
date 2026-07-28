"use client";

import { useState } from "react";
import FormattedText from "./FormattedText";

interface TrendForecastProps {
  sectors: string[];
  onFetchForecast: (sector: string) => Promise<any>;
}

export default function TrendForecaster({
  sectors,
  onFetchForecast,
}: TrendForecastProps) {
  const [selectedSector, setSelectedSector] = useState("");
  const [forecast, setForecast] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelect = async (sector: string) => {
    setSelectedSector(sector);
    if (!sector) {
      setForecast(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    setForecast(null);
    try {
      const data = await onFetchForecast(sector);
      setForecast(data);
    } catch (err: any) {
      setError(err.message || "Could not load the forecast. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">Trend forecast</h3>
        <select
          value={selectedSector}
          onChange={(e) => handleSelect(e.target.value)}
          aria-label="Sector"
          className="px-2 py-1 rounded border border-border-dim bg-bg-card text-xs text-terminal-text cursor-pointer focus:outline-none focus:border-terminal-signal"
        >
          <option value="">Select sector…</option>
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="card-body flex flex-col gap-3">
        {loading && (
          <div className="py-6 text-center text-xs text-terminal-muted">
            Generating sector outlook…
          </div>
        )}

        {error && (
          <div className="px-3 py-2 rounded border border-terminal-red/40 bg-terminal-red-soft text-xs text-terminal-red">
            {error}
          </div>
        )}

        {forecast && !loading && (
          <>
            {forecast.narrative && (
              <FormattedText text={forecast.narrative} className="formatted-text" />
            )}

            {forecast.key_drivers?.length > 0 && (
              <div>
                <div className="label mb-1.5">Key drivers</div>
                <div className="flex flex-wrap gap-1">
                  {forecast.key_drivers.map((d: string, idx: number) => (
                    <span key={idx} className="tag">
                      {d}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {(forecast.confidence || forecast.time_horizon) && (
              <div className="flex items-center gap-4 pt-2.5 border-t border-border-soft text-[11px] text-terminal-muted">
                {forecast.confidence && (
                  <span>
                    Confidence{" "}
                    <span className="num text-terminal-text">
                      {(forecast.confidence * 100).toFixed(0)}%
                    </span>
                  </span>
                )}
                {forecast.time_horizon && (
                  <span>
                    Horizon{" "}
                    <span className="text-terminal-text">{forecast.time_horizon}</span>
                  </span>
                )}
              </div>
            )}
          </>
        )}

        {!selectedSector && !loading && !forecast && (
          <p className="py-6 text-center text-xs text-terminal-muted">
            Select a sector to generate an outlook.
          </p>
        )}
      </div>
    </div>
  );
}
