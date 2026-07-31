import type { Metadata } from "next";
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
            __html: `(function(){var d=document.documentElement;try{var t=localStorage.getItem('deus-theme');if(t==='light')d.classList.add('light');var m=localStorage.getItem('deus-motion');d.setAttribute('data-motion',m==='off'?'off':'on')}catch(e){d.setAttribute('data-motion','on')}})();`,
          }}
        />
      </head>
      <body className="h-full flex flex-col overflow-hidden">
        <Header />

        <div className="flex-1 flex overflow-hidden">
          <aside className="w-52 bg-bg-card border-r border-border-dim flex flex-col shrink-0">
            <SideNav />
            <div className="p-3 border-t border-border-soft text-[11px] leading-relaxed text-terminal-muted-alt">
              Deus v2.0.0
              <br />
              Gemini · DeepSeek
            </div>
          </aside>

          <main className="flex-1 overflow-auto bg-bg-main relative">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
