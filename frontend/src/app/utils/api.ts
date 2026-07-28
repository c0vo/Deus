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
