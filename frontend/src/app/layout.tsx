import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import Header from "./components/Header";
import SideNav from "./components/SideNav";
import "./globals.css";

// Plex Sans ships a variable build — omit `weight` to get the full axis.
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-plex-sans",
  display: "swap",
});

// Plex Mono is static only, so the weights we use must be declared.
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Deus — Financial Terminal",
  description:
    "Autonomous financial news aggregator, market monitor, and stock prediction engine.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Lets the app paint under the Android gesture bar / notch. Anything pinned
  // to an edge must pair this with the safe-* utilities in globals.css.
  viewportFit: "cover",
  // Chrome's default (resizes-visual) leaves the layout viewport at full
  // height when the soft keyboard opens, which slides the chat input under it.
  // resizes-content shrinks the layout viewport so the h-dvh shell reflows.
  interactiveWidget: "resizes-content",
  // Single value, not the prefers-color-scheme array form: this app's theme is
  // a class driven by localStorage, not the OS setting, so the media form
  // would be wrong for anyone whose OS and app themes disagree. The bootstrap
  // script below and ThemeToggle keep this tag in sync at runtime instead.
  themeColor: "#0b0d10",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`h-full bg-bg-main text-terminal-text antialiased ${plexSans.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){var d=document.documentElement;try{var t=localStorage.getItem('deus-theme');if(t==='light')d.classList.add('light');var c=document.querySelector('meta[name="theme-color"]');if(c)c.setAttribute('content',t==='light'?'#f4f6f9':'#0b0d10');var m=localStorage.getItem('deus-motion');d.setAttribute('data-motion',m==='off'?'off':'on')}catch(e){d.setAttribute('data-motion','on')}})();`,
          }}
        />
      </head>
      {/* h-dvh, not h-full: height:100% chains off <html> and resolves against
          the large viewport (URL bar hidden), so with the bar showing the shell
          is ~56px taller than what you can see — and overflow-hidden means
          there is no way to scroll to it. Because the root scroller never
          scrolls, Chrome never auto-collapses the bar, so dvh is stable here.
          overscroll-none kills pull-to-refresh, which would otherwise reload
          the SPA and tear down every open SSE stream on a downward flick. */}
      <body className="h-dvh flex flex-col overflow-hidden overscroll-none">
        <Header />

        <div className="flex-1 flex min-h-0 overflow-hidden">
          <aside className="hidden md:flex w-52 bg-bg-card border-r border-border-dim flex-col shrink-0">
            <SideNav />
            <div className="p-3 border-t border-border-soft text-[11px] leading-relaxed text-terminal-muted-alt">
              Deus v2.0.0
              <br />
              Gemini · DeepSeek
            </div>
          </aside>

          <main className="flex-1 min-w-0 overflow-auto overscroll-contain bg-bg-main relative">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
