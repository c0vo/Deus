import type { MetadataRoute } from "next";

/**
 * Emitted as out/manifest.webmanifest under `output: "export"` — metadata
 * routes are exempt from the static-generation bailout that would otherwise
 * skip a route handler. Next also injects <link rel="manifest"> automatically,
 * so layout.tsx must NOT also set `metadata.manifest` or there would be two
 * competing sources of truth.
 *
 * Note: Chrome only offers "install" from a secure context. Over plain LAN
 * http:// this yields a home-screen bookmark that ignores `display`. Reaching
 * the server via `adb reverse` (so the phone sees http://localhost:8000) or
 * any HTTPS front makes the standalone window work.
 */
// Required under `output: "export"`: without it Next treats this route handler
// as dynamic and fails the export rather than emitting the file.
export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    id: "/",
    name: "Deus — Financial Terminal",
    short_name: "Deus",
    description:
      "Autonomous financial news aggregator, market monitor, and stock prediction engine.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "portrait",
    // Baked into the install-time splash screen, so these cannot follow the
    // runtime theme toggle. Dark is the default (globals.css :root).
    background_color: "#0b0d10",
    theme_color: "#0b0d10",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
      {
        src: "/icon-maskable-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      { name: "Debate", url: "/predict" },
      { name: "Analyst", url: "/chat" },
      { name: "Markets", url: "/watchlist" },
    ],
  };
}
