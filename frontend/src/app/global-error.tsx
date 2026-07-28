"use client";

import { useEffect } from "react";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

// This boundary replaces the root layout, so it has to bring its own
// stylesheet and fonts — otherwise it renders completely unstyled.
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Global error boundary caught:", error);
  }, [error]);

  return (
    <html
      lang="en"
      className={`h-full bg-bg-main text-terminal-text antialiased ${plexSans.variable} ${plexMono.variable}`}
    >
      <body className="h-full flex items-center justify-center p-6">
        <div className="card max-w-lg w-full p-7 text-center">
          <h1 className="text-lg font-semibold text-terminal-red">
            Deus could not start
          </h1>
          <p className="mt-2 text-sm text-terminal-muted break-words">
            {error.message || "An unexpected error occurred."}
          </p>
          {error.digest && (
            <p className="num mt-1 text-[11px] text-terminal-muted-alt">
              {error.digest}
            </p>
          )}
          <button onClick={reset} className="btn-signal mt-5">
            Reload
          </button>
        </div>
      </body>
    </html>
  );
}
