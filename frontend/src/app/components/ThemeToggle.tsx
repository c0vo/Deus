"use client";

import { Sun, Moon } from "lucide-react";
import { useState, useEffect } from "react";

export default function ThemeToggle() {
  const [isLight, setIsLight] = useState(false);

  useEffect(() => {
    setIsLight(document.documentElement.classList.contains("light"));
  }, []);

  const toggle = () => {
    const next = !isLight;
    setIsLight(next);
    document.documentElement.classList.toggle("light", next);
    // Matches the pre-paint bootstrap in layout.tsx — the browser chrome
    // follows the app theme, not prefers-color-scheme.
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", next ? "#f4f6f9" : "#0b0d10");
    try {
      localStorage.setItem("deus-theme", next ? "light" : "dark");
    } catch {
      /* localStorage unavailable */
    }
  };

  return (
    <button
      onClick={toggle}
      title={isLight ? "Switch to dark theme" : "Switch to light theme"}
      aria-label={isLight ? "Switch to dark theme" : "Switch to light theme"}
      className="tap-halo grid place-items-center w-7 h-7 rounded border border-border-dim text-terminal-muted hover:bg-bg-surface hover:text-terminal-text transition-colors"
    >
      {isLight ? <Moon size={14} strokeWidth={1.7} /> : <Sun size={14} strokeWidth={1.7} />}
    </button>
  );
}
