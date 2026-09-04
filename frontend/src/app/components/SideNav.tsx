"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  TrendingUp,
  Flame,
  Swords,
  MessageSquare,
  BarChart3,
  BookOpen,
  Newspaper,
  Globe2,
  CalendarDays,
  Waypoints,
} from "lucide-react";

const NAV_ITEMS: { href: string; label: string; icon: React.ElementType }[] = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/briefing", label: "Briefing", icon: Newspaper },
  { href: "/globe", label: "Globe", icon: Globe2 },
  { href: "/calendar", label: "Calendar", icon: CalendarDays },
  { href: "/watchlist", label: "Markets", icon: TrendingUp },
  { href: "/trending", label: "Trending", icon: Flame },
  { href: "/predict", label: "Debate", icon: Swords },
  { href: "/thesis", label: "Thesis", icon: Waypoints },
  { href: "/chat", label: "Analyst", icon: MessageSquare },
  { href: "/metrics", label: "Metrics", icon: BarChart3 },
  { href: "/reflections", label: "Reflections", icon: BookOpen },
];

/**
 * Rendered twice: once in the permanent desktop sidebar (no props) and once
 * inside MobileNav's drawer (`touch` for 44px rows, `onNavigate` to close it).
 */
export default function SideNav({
  onNavigate,
  touch = false,
}: {
  onNavigate?: () => void;
  touch?: boolean;
} = {}) {
  const pathname = usePathname();

  return (
    <nav className="flex-1 p-2.5 flex flex-col gap-0.5" aria-label="Main">
      <div className="label px-2.5 pt-1.5 pb-2">Terminal</div>

      {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
        const isActive =
          href === "/" ? pathname === "/" : pathname.startsWith(href);

        return (
          <Link
            key={href}
            href={href}
            aria-current={isActive ? "page" : undefined}
            onClick={onNavigate}
            className={`relative flex items-center gap-2.5 px-2.5 rounded text-[13px] transition-colors ${
              touch ? "py-3" : "py-1.5"
            } ${
              isActive
                ? "bg-bg-surface text-terminal-text font-medium before:absolute before:-left-2.5 before:top-1/2 before:-translate-y-1/2 before:w-0.5 before:h-4 before:rounded-r-sm before:bg-terminal-signal"
                : "text-terminal-muted hover:bg-bg-surface hover:text-terminal-text"
            }`}
          >
            <Icon
              size={16}
              strokeWidth={1.5}
              className={isActive ? "text-terminal-signal" : "opacity-85"}
            />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
