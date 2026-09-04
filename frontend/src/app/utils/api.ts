export function getApiUrl(path: string): string {
  if (typeof window !== "undefined") {
    // In development mode, route SSE requests directly to port 8000 (FastAPI)
    // to bypass the Next.js dev server proxy buffering on Server-Sent Events.
    // Uses window.location.hostname so it works over LAN (e.g., 192.168.x.x).
    if (window.location.port === "3000") {
      return `http://${window.location.hostname}:8000${path}`;
    }
  }
  return path;
}

export type FetchOptions = RequestInit & { timeoutMs?: number };

const DEFAULT_TIMEOUT_MS = 8000;

/**
 * fetch() with a deadline, routed through getApiUrl().
 *
 * Every request in this app used to hang indefinitely if the backend was slow
 * to answer. Combined with the polling intervals below that was actively
 * harmful: a stalled endpoint left panels blank forever while setInterval
 * stacked another request on top every few seconds.
 *
 * A caller-supplied signal is honoured alongside the timeout, so components
 * that abort on unmount keep working.
 */
export async function fetchWithTimeout(
  path: string,
  options: FetchOptions = {},
): Promise<Response> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...rest } = options;
  const timeoutSignal = AbortSignal.timeout(timeoutMs);

  return fetch(getApiUrl(path), {
    ...rest,
    signal: signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal,
  });
}

/** fetchWithTimeout that parses JSON and throws on a non-2xx response. */
export async function fetchJson<T = unknown>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const res = await fetchWithTimeout(path, options);
  if (!res.ok) {
    throw new Error(`${path} responded ${res.status}`);
  }
  return (await res.json()) as T;
}

/**
 * Wrap a polling function so a slow response cannot pile up behind itself.
 *
 * setInterval fires on a fixed schedule regardless of whether the previous
 * request finished. Once the backend is slower than the interval, that turns a
 * steady poll into an ever-growing queue of in-flight requests — exactly when
 * the backend can least afford it.
 */
export function nonOverlapping(fn: () => Promise<void>): () => Promise<void> {
  let inFlight = false;
  return async () => {
    if (inFlight) return;
    inFlight = true;
    try {
      await fn();
    } finally {
      inFlight = false;
    }
  };
}
