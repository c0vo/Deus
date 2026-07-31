/**
 * Motion preference, as an app-level setting rather than an OS one.
 *
 * The OS `prefers-reduced-motion` query is deliberately NOT consulted as a
 * default: an accessibility setting left on for the desktop shell was silently
 * freezing the ticker tape and the globe, with no in-app way to notice or undo
 * it. The app defaults to motion ON and the user opts out here instead.
 *
 * State lives on `<html data-motion>` so CSS can key off it (see the two
 * reduced-motion blocks in globals.css) and so the pre-paint script in
 * layout.tsx can stamp it before the first frame.
 */

export const MOTION_KEY = "deus-motion";

const MOTION_EVENT = "deus:motionchange";

/** True when motion should be SUPPRESSED. */
export function isMotionReduced(): boolean {
  if (typeof document === "undefined") return false;
  return document.documentElement.dataset.motion === "off";
}

export function setMotionEnabled(on: boolean): void {
  document.documentElement.dataset.motion = on ? "on" : "off";
  try {
    localStorage.setItem(MOTION_KEY, on ? "on" : "off");
  } catch {
    /* localStorage unavailable */
  }
  window.dispatchEvent(new Event(MOTION_EVENT));
}

/**
 * `storage` only fires in OTHER tabs, so the custom event covers this one and
 * the storage listener keeps other tabs in sync for free.
 */
export function subscribeMotion(callback: () => void): () => void {
  window.addEventListener(MOTION_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(MOTION_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}
