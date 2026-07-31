"use client";

import { useSyncExternalStore } from "react";
import { isMotionReduced, subscribeMotion } from "../utils/motion";

/**
 * Returns true when motion should be SUPPRESSED.
 *
 * Pages are prerendered at build time (`output: "export"`), so the server
 * snapshot reports motion-on; React reconciles after hydration, and the
 * pre-paint script in layout.tsx has already stamped the real value by then.
 */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeMotion,
    isMotionReduced,
    () => false
  );
}
