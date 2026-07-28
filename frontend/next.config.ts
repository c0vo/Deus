import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable static HTML export for production / Termux deployment.
  // The built files in out/ are served directly by FastAPI on port 8000.
  output: "export",

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
