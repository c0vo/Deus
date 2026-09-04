"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Waypoints, ArrowRight } from "lucide-react";
import { fetchJson, nonOverlapping } from "../utils/api";

interface Candidate {
  id: string;
  thesis_id: string;
  thesis_title?: string;
  company_name: string;
  ticker: string | null;
  role_in_chain: string;
  exposure: number;
  rumour_stage: string | null;
  edge_score: number | null;
  data_coverage: number;
}

interface Transition {
  ticker: string;
  company_name?: string;
  old_stage: string;
  new_stage: string;
}

/**
 * Dashboard widget: the least-crowded high-conviction names across every
 * active thesis, plus any stage changes from the last re-score.
 *
 * Sorted by edge_score — conviction discounted by how much of the move has
 * already happened — so what surfaces here is what is still early.
 */
export default function ThesisPanel({ transitions = [] }: { transitions?: Transition[] }) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [recentTransitions, setRecentTransitions] = useState<Transition[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // `cancelled` guards against setting state after unmount — the panel polls
    // every five minutes, so an in-flight request outliving the page is the
    // normal case rather than an edge one.
    let cancelled = false;

    const load = async () => {
      // Settled independently: a failure fetching one of these should not blank
      // the other, since they answer different questions.
      const [cands, trans] = await Promise.allSettled([
        fetchJson<{ data: Candidate[] }>(
          "/api/thesis/candidates?stage=EARLY&limit=6"
        ),
        fetchJson<{ data: Transition[] }>("/api/thesis/transitions?days=3"),
      ]);
      if (cancelled) return;
      setCandidates(cands.status === "fulfilled" ? cands.value.data || [] : []);
      setRecentTransitions(
        trans.status === "fulfilled" ? trans.value.data || [] : []
      );
      setLoading(false);
    };

    const guarded = nonOverlapping(load);
    guarded();
    const id = setInterval(guarded, 5 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // A live re-score is fresher than anything on disk, so SSE wins when it has
  // fired this session; otherwise show what the last few days recorded, so a
  // cold load is not blank until the next 09:10 job.
  const shownTransitions =
    transitions.length > 0 ? transitions : recentTransitions;

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title flex items-center gap-1.5">
          <Waypoints size={13} strokeWidth={1.7} />
          Emerging theses
        </span>
        <Link
          href="/thesis"
          className="text-xs text-terminal-muted hover:text-terminal-signal inline-flex items-center gap-1 tap"
        >
          all <ArrowRight size={11} strokeWidth={1.7} />
        </Link>
      </div>
      <div className="card-body space-y-2">
        {shownTransitions.length > 0 && (
          <div className="space-y-1 pb-2 border-b border-border-dim">
            {shownTransitions.slice(0, 3).map((t, i) => (
              <p key={i} className="text-xs">
                <span className="num text-terminal-signal">{t.ticker}</span>{" "}
                <span className="text-terminal-muted">
                  {t.old_stage} → {t.new_stage}
                </span>
              </p>
            ))}
          </div>
        )}

        {loading && <p className="text-xs text-terminal-muted">Loading…</p>}

        {!loading && candidates.length === 0 && (
          <p className="text-xs text-terminal-muted">
            No early candidates yet. A thesis is built each morning.
          </p>
        )}

        {candidates.map((c) => (
          <Link
            key={c.id}
            href={`/thesis?id=${c.thesis_id}`}
            className="block p-2 rounded border border-border-dim hover:border-border-soft tap"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <span className="num text-xs text-terminal-signal">
                  {c.ticker || "—"}
                </span>{" "}
                <span className="text-xs truncate">{c.company_name}</span>
              </div>
              <span
                className="num text-xs text-terminal-muted shrink-0"
                title="Conviction x (1 - crowding)"
              >
                {c.edge_score === null ? "—" : c.edge_score.toFixed(2)}
              </span>
            </div>
            <p className="text-xs text-terminal-muted-alt truncate mt-0.5">
              {c.role_in_chain}
            </p>
            {(c.data_coverage ?? 0) < 0.35 && (
              <p className="text-xs text-terminal-muted-alt mt-0.5">thin data</p>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}
