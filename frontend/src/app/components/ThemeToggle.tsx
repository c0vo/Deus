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
      className="grid place-items-center w-7 h-7 rounded border border-border-dim text-terminal-muted hover:bg-bg-surface hover:text-terminal-text transition-colors"
    >
      {isLight ? <Moon size={14} strokeWidth={1.7} /> : <Sun size={14} strokeWidth={1.7} />}
    </button>
  );
}
