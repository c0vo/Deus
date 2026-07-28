"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Root error boundary caught:", error);
  }, [error]);

  return (
    <div className="p-6 flex items-center justify-center min-h-[400px]">
      <div className="card max-w-lg w-full p-7 text-center">
        <h2 className="text-lg font-semibold text-terminal-red">
          Something went wrong
        </h2>
        <p className="mt-2 text-sm text-terminal-muted break-words">
          {error.message || "The page could not be rendered."}
        </p>
        {error.digest && (
          <p className="num mt-1 text-[11px] text-terminal-muted-alt">
            {error.digest}
          </p>
        )}
        <button onClick={reset} className="btn-signal mt-5">
          Try again
        </button>
      </div>
    </div>
  );
}
