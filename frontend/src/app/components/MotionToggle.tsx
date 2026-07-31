"use client";

import { Zap, ZapOff } from "lucide-react";
import { setMotionEnabled } from "../utils/motion";
import { useReducedMotion } from "../hooks/useReducedMotion";

export default function MotionToggle() {
  const reduced = useReducedMotion();

  return (
    <button
      onClick={() => setMotionEnabled(reduced)}
      title={reduced ? "Enable animation" : "Reduce animation"}
      aria-label={reduced ? "Enable animation" : "Reduce animation"}
      aria-pressed={reduced}
      className="grid place-items-center w-7 h-7 rounded border border-border-dim text-terminal-muted hover:bg-bg-surface hover:text-terminal-text transition-colors"
    >
      {reduced ? <ZapOff size={14} strokeWidth={1.7} /> : <Zap size={14} strokeWidth={1.7} />}
    </button>
  );
}
