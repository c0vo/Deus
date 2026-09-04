"use client";

import { useSystemStatus } from "../hooks/useSystemStatus";
import MobileNav from "./MobileNav";
import LiveClock from "./LiveClock";
import MotionToggle from "./MotionToggle";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  const { status } = useSystemStatus();

  return (
    <header className="h-12 shrink-0 border-b border-border-dim bg-bg-card px-3 md:px-4 flex items-center justify-between gap-3 md:gap-4">
      <div className="flex items-center gap-2 min-w-0">
        <MobileNav />
        <span className="text-[15px] font-semibold tracking-[0.14em]">DEUS</span>
      </div>

      {/* Connection state is the only thing that belongs in global chrome —
          counts and DB size live on the dashboard telemetry card. */}
      <div className="flex items-center gap-2.5 md:gap-3.5 text-xs text-terminal-muted">
        {status.online ? (
          <span className="inline-flex items-center gap-2 text-terminal-green">
            <span className="live-dot" />
            <span className="text-terminal-muted">Live</span>
          </span>
        ) : (
          <span className="inline-flex items-center gap-2 text-terminal-red">
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            <span>Offline</span>
          </span>
        )}
        <span className="num hidden sm:inline text-terminal-muted-alt">
          <LiveClock />
        </span>
        <MotionToggle />
        <ThemeToggle />
      </div>
    </header>
  );
}
