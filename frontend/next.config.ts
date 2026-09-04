import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable static HTML export for production / Termux deployment.
  // The built files in out/ are served directly by FastAPI on port 8000.
  output: "export",

  // `next dev` blocks cross-origin requests to its /_next/* and /__nextjs*
  // dev endpoints. The dev server runs with --hostname 0.0.0.0, and 0.0.0.0 is
  // what Next adds to its own allowlist — that never matches a LAN IP, so
  // loading the dev server on a phone silently kills Fast Refresh and the
  // error overlay (the page itself still loads, which makes it easy to miss).
  // Hostnames only: no scheme, no port. Wildcards match per dot-segment.
  // 127.0.0.1 is NOT allowlisted by default (only localhost/*.localhost).
  allowedDevOrigins: ["127.0.0.1", "192.168.123.108", "192.168.123.*"],

  // Rewrites only apply in dev mode (next dev). They proxy /api/* requests
  // to the FastAPI backend. In static export mode, the frontend is served
  // from FastAPI itself, so /api/* routes are on the same origin.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
