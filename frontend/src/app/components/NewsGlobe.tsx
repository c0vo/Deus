"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { geoOrthographic, geoPath, geoGraticule10, geoCentroid } from "d3-geo";
import { feature } from "topojson-client";
import worldAtlas from "world-atlas/countries-110m.json";

export interface GeoCountry {
  country: string;
  name?: string;
  count: number;
  avg_sentiment: number;
  max_importance: number;
}

interface NewsGlobeProps {
  countries: GeoCountry[];
  /** Rotation speed in degrees per second; 0 pauses the spin. */
  spinSpeed?: number;
}

/**
 * ISO 3166-1 alpha-2 -> numeric, for the countries the geo tagger can emit.
 * The atlas keys its geometries by the numeric code.
 */
const ALPHA2_TO_NUMERIC: Record<string, number> = {
  US: 840, CN: 156, JP: 392, KR: 410, IN: 356, GB: 826, DE: 276, FR: 250,
  IT: 380, ES: 724, NL: 528, CH: 756, RU: 643, UA: 804, CA: 124, MX: 484,
  BR: 76, AR: 32, AU: 36, NZ: 554, TW: 158, SG: 702, SA: 682, AE: 784,
  IL: 376, IR: 364, TR: 792, EG: 818, ZA: 710, NG: 566, ID: 360, TH: 764,
  VN: 704, MY: 458, PH: 608, PL: 616, SE: 752, NO: 578, DK: 208, FI: 246,
  IE: 372, PT: 620, GR: 300, QA: 634, CL: 152, CO: 170, PE: 604, PK: 586,
  BD: 50, VE: 862,
};

/** City-states too small to appear at 110m resolution still get a marker. */
const FALLBACK_CENTROIDS: Record<string, [number, number]> = {
  SG: [103.82, 1.35],
};

type Feature = { id?: string | number; geometry: any; properties: any };

const world = feature(
  worldAtlas as any,
  (worldAtlas as any).objects.countries
) as unknown as { features: Feature[] };

const GRATICULE = geoGraticule10();
const SPHERE = { type: "Sphere" } as any;

function readTheme() {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    ocean: v("--panel"),
    land: v("--surface"),
    line: v("--border"),
    signal: v("--signal"),
    up: v("--green"),
    down: v("--red"),
    muted: v("--muted-alt"),
    text: v("--foreground"),
  };
}

/** Mix two hex colours; t=0 returns a, t=1 returns b. */
function mix(a: string, b: string, t: number): string {
  const parse = (h: string) => {
    let s = h.replace("#", "");
    if (s.length === 3) s = s.split("").map((c) => c + c).join("");
    return [
      parseInt(s.slice(0, 2), 16),
      parseInt(s.slice(2, 4), 16),
      parseInt(s.slice(4, 6), 16),
    ];
  };
  try {
    const [r1, g1, b1] = parse(a);
    const [r2, g2, b2] = parse(b);
    const c = (x: number, y: number) => Math.round(x + (y - x) * t);
    return `rgb(${c(r1, r2)},${c(g1, g2)},${c(b1, b2)})`;
  } catch {
    return a;
  }
}

export default function NewsGlobe({ countries, spinSpeed = 6 }: NewsGlobeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rotationRef = useRef<[number, number]>([-20, -15]);
  const draggingRef = useRef<{ x: number; y: number } | null>(null);
  const [size, setSize] = useState(420);
  const [themeTick, setThemeTick] = useState(0);

  // Redraw when the theme class flips; the canvas has no CSS to cascade into.
  useEffect(() => {
    const observer = new MutationObserver(() => setThemeTick((t) => t + 1));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setSize(Math.max(240, el.clientWidth));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const byCode = useCallback(() => {
    const map = new Map<number, GeoCountry>();
    const extras: GeoCountry[] = [];
    for (const c of countries) {
      const numeric = ALPHA2_TO_NUMERIC[c.country];
      if (numeric !== undefined) map.set(numeric, c);
      if (FALLBACK_CENTROIDS[c.country]) extras.push(c);
    }
    return { map, extras };
  }, [countries]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(dpr, dpr);

    const theme = readTheme();
    const { map, extras } = byCode();
    const maxCount = Math.max(1, ...countries.map((c) => c.count));

    const projection = geoOrthographic()
      .fitExtent(
        [[6, 6], [size - 6, size - 6]],
        SPHERE
      )
      .rotate(rotationRef.current);
    const path = geoPath(projection, ctx as any);

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let raf = 0;
    let last = performance.now();
    let elapsed = 0;

    const centroidOf = (c: GeoCountry): [number, number] | null => {
      if (FALLBACK_CENTROIDS[c.country]) return FALLBACK_CENTROIDS[c.country];
      const numeric = ALPHA2_TO_NUMERIC[c.country];
      const f = world.features.find((x) => Number(x.id) === numeric);
      return f ? (geoCentroid(f as any) as [number, number]) : null;
    };

    const markers = countries
      .map((c) => ({ c, coords: centroidOf(c) }))
      .filter((m): m is { c: GeoCountry; coords: [number, number] } => !!m.coords);

    const draw = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      elapsed += dt;

      if (!draggingRef.current && spinSpeed > 0 && !reduceMotion) {
        rotationRef.current = [
          rotationRef.current[0] + spinSpeed * dt,
          rotationRef.current[1],
        ];
      }
      projection.rotate(rotationRef.current);

      ctx.clearRect(0, 0, size, size);

      // Ocean disc
      ctx.beginPath();
      path(SPHERE);
      ctx.fillStyle = theme.ocean;
      ctx.fill();

      // Graticule
      ctx.beginPath();
      path(GRATICULE);
      ctx.strokeStyle = theme.line;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Landmasses. Countries carrying news are tinted toward the accent in
      // proportion to volume; everything else stays inert.
      for (const f of world.features) {
        const entry = map.get(Number(f.id));
        ctx.beginPath();
        path(f as any);
        if (entry) {
          const t = 0.25 + 0.75 * (Math.log1p(entry.count) / Math.log1p(maxCount));
          ctx.fillStyle = mix(theme.land, theme.signal, t);
        } else {
          ctx.fillStyle = theme.land;
        }
        ctx.fill();
        ctx.strokeStyle = theme.ocean;
        ctx.lineWidth = 0.4;
        ctx.stroke();
      }

      // Rim
      ctx.beginPath();
      path(SPHERE);
      ctx.strokeStyle = theme.line;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Event markers. Colour is sentiment (a direction), size is volume.
      const pulse = reduceMotion ? 0 : (Math.sin(elapsed * 2) + 1) / 2;
      for (const { c, coords } of markers) {
        const projected = projection(coords);
        if (!projected) continue;

        // Cull points on the far side of the sphere.
        const centre: [number, number] = [-rotationRef.current[0], -rotationRef.current[1]];
        const toRad = Math.PI / 180;
        const cosAngle =
          Math.sin(coords[1] * toRad) * Math.sin(centre[1] * toRad) +
          Math.cos(coords[1] * toRad) *
            Math.cos(centre[1] * toRad) *
            Math.cos((coords[0] - centre[0]) * toRad);
        if (cosAngle < 0) continue;

        const [x, y] = projected;
        const scale = 0.4 + 0.6 * (Math.log1p(c.count) / Math.log1p(maxCount));
        const base = 2 + scale * 3;
        const colour =
          c.avg_sentiment > 0.15
            ? theme.up
            : c.avg_sentiment < -0.15
            ? theme.down
            : theme.muted;

        ctx.beginPath();
        ctx.arc(x, y, base + pulse * 6 * scale, 0, Math.PI * 2);
        ctx.fillStyle = colour;
        ctx.globalAlpha = 0.18 * (1 - pulse);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, base, 0, Math.PI * 2);
        ctx.fillStyle = colour;
        ctx.globalAlpha = 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [size, countries, byCode, spinSpeed, themeTick]);

  // Drag to rotate
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    draggingRef.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const start = draggingRef.current;
    if (!start) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    draggingRef.current = { x: e.clientX, y: e.clientY };
    rotationRef.current = [
      rotationRef.current[0] + dx * 0.4,
      Math.max(-90, Math.min(90, rotationRef.current[1] - dy * 0.4)),
    ];
  };
  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    draggingRef.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  return (
    <div ref={wrapRef} className="w-full flex justify-center">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Globe showing news volume across ${countries.length} countries`}
        className="cursor-grab active:cursor-grabbing touch-none select-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      />
    </div>
  );
}
