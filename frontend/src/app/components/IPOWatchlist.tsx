"use client";

import { Trash2 } from "lucide-react";

interface IPOItem {
  id: number;
  company_name: string;
  ticker: string | null;
  ipo_date: string | null;
  offering_price: string | null;
  status: string;
  sector: string | null;
}

interface IPOWatchlistProps {
  ipos: IPOItem[];
  onRemove?: (id: number) => void;
}

function statusTag(status: string) {
  switch (status) {
    case "priced":
    case "listed":
      return <span className="tag tag-up">{status}</span>;
    case "withdrawn":
      return <span className="tag tag-down">{status}</span>;
    case "upcoming":
      return <span className="tag tag-warn">{status}</span>;
    default:
      return <span className="tag">{status || "unknown"}</span>;
  }
}

export default function IPOWatchlist({ ipos, onRemove }: IPOWatchlistProps) {
  if (!ipos || ipos.length === 0) {
    return (
      <div className="card">
        <div className="card-head">
          <h3 className="card-title">IPO watchlist</h3>
        </div>
        <div className="card-body text-xs text-terminal-muted">
          No IPOs detected yet.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-head">
        <h3 className="card-title">IPO watchlist</h3>
        <span className="num text-[11px] text-terminal-muted-alt">{ipos.length}</span>
      </div>

      <div className="flex flex-col max-h-80 overflow-y-auto">
        {ipos.map((ipo) => (
          <div
            key={ipo.id}
            className="group flex items-start gap-3 px-3.5 py-2.5 border-b border-border-soft last:border-b-0 hover:bg-bg-surface transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium truncate">
                  {ipo.company_name}
                </span>
                {statusTag(ipo.status)}
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] text-terminal-muted-alt">
                {ipo.ticker && <span className="num">{ipo.ticker}</span>}
                {ipo.ticker && ipo.sector && <span className="opacity-50">/</span>}
                {ipo.sector && <span>{ipo.sector}</span>}
              </div>
            </div>

            <div className="shrink-0 text-right">
              <div className="num text-[11px] text-terminal-muted">
                {ipo.ipo_date
                  ? new Date(ipo.ipo_date).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })
                  : "TBA"}
              </div>
              {ipo.offering_price && (
                <div className="num text-[11px] text-terminal-muted-alt">
                  {ipo.offering_price}
                </div>
              )}
            </div>

            {onRemove && (
              <button
                type="button"
                onClick={() => onRemove(ipo.id)}
                title="Remove from watchlist"
                aria-label={`Remove ${ipo.company_name} from watchlist`}
                className="shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 text-terminal-muted-alt hover:text-terminal-red transition-all"
              >
                <Trash2 size={12} strokeWidth={1.7} />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
