"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { Menu, X } from "lucide-react";
import SideNav from "./SideNav";

/**
 * Below `md` the permanent sidebar in layout.tsx is `display: none`, and this
 * takes over: a hamburger in the header plus an off-canvas drawer holding the
 * same SideNav.
 *
 * Both halves live in one component on purpose. The button belongs inside
 * <header> and the panel outside it, so a CSS-only drawer (peer/:target) can't
 * connect them — and a <Link> tap can't uncheck a checkbox, so the drawer would
 * stay open across navigation. Header is already a client component, so owning
 * the state here costs no new client boundary and no context.
 *
 * The panel is `position: fixed`, which escapes the `overflow: hidden` on
 * <body> and is viewport-relative only because nothing above it creates a
 * containing block. If a transform/filter/will-change is ever added to <body>
 * or <header>, this drawer breaks silently — it would become relative to that
 * ancestor instead.
 */
export default function MobileNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // Covers back/forward and any navigation that isn't a drawer link tap.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!open) return;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        buttonRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open navigation"
        aria-expanded={open}
        aria-controls="mobile-nav-panel"
        className="tap-halo md:hidden grid place-items-center w-7 h-7 -ml-1 rounded text-terminal-muted hover:bg-bg-surface hover:text-terminal-text transition-colors"
      >
        <Menu size={18} strokeWidth={1.7} />
      </button>

      {/* Scrim. touch-none stops a drag over it from scrolling the page behind. */}
      <div
        onClick={() => setOpen(false)}
        aria-hidden="true"
        className={`md:hidden fixed inset-0 z-40 bg-black/55 touch-none transition-opacity duration-200 ${
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      />

      {/* `inert` keeps the closed panel out of the tab order and the a11y tree
          without unmounting it, so the slide transition still runs. */}
      <div
        id="mobile-nav-panel"
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation"
        tabIndex={-1}
        inert={!open}
        className={`md:hidden fixed inset-y-0 left-0 z-50 w-64 max-w-[82vw] bg-bg-card border-r border-border-dim flex flex-col overscroll-contain overflow-y-auto safe-t safe-b transition-transform duration-200 ease-out ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="h-12 shrink-0 px-3 flex items-center justify-between border-b border-border-dim">
          <span className="text-[15px] font-semibold tracking-[0.14em]">DEUS</span>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
            className="tap-halo grid place-items-center w-7 h-7 rounded text-terminal-muted hover:bg-bg-surface hover:text-terminal-text transition-colors"
          >
            <X size={18} strokeWidth={1.7} />
          </button>
        </div>

        <SideNav onNavigate={() => setOpen(false)} touch />

        <div className="p-3 border-t border-border-soft text-[11px] leading-relaxed text-terminal-muted-alt">
          Deus v2.0.0
          <br />
          Gemini · DeepSeek
        </div>
      </div>
    </>
  );
}
