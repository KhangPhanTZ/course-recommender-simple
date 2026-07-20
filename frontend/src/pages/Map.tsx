import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ClusterMap, type MapPoint } from "../lib/api";
import { Compass } from "../components/icons";

// 20 distinct, theme-friendly cluster colors.
const PALETTE = [
  "#14b09a", "#e39a2f", "#6ea8fe", "#e5709b", "#8b7cf0", "#4fb477", "#e2725b",
  "#3fb6c9", "#c78a3b", "#9d6bd4", "#5b8def", "#d1567e", "#57a773", "#df8f4a",
  "#7c9cf5", "#b76ec4", "#48b0a0", "#c9605f", "#6bb14a", "#a08adb",
];
const color = (c: number) => PALETTE[((c % PALETTE.length) + PALETTE.length) % PALETTE.length];

const W = 960;
const H = 560;
const PAD = 26;

export default function MapPage() {
  const [data, setData] = useState<ClusterMap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [tip, setTip] = useState<{ p: MapPoint; left: number; top: number } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.map().then(setData).catch(() => setError("Could not reach the API."));
  }, []);

  const project = useMemo(() => {
    const pts = data?.points ?? [];
    if (!pts.length) return null;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;
    const sx = (x: number) => PAD + ((x - minX) / dx) * (W - 2 * PAD);
    const sy = (y: number) => PAD + ((maxY - y) / dy) * (H - 2 * PAD); // flip Y
    return { sx, sy };
  }, [data]);

  function onHover(p: MapPoint, e: React.MouseEvent) {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTip({ p, left: e.clientX - rect.left, top: e.clientY - rect.top });
  }

  const available = data?.available;

  return (
    <div className="container-page py-12">
      <div className="max-w-2xl">
        <span className="eyebrow flex items-center gap-2">
          <Compass width={15} height={15} /> Catalog map
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">UMAP cluster map</h1>
        <p className="mt-3 text-body">
          Every course embedding is projected from high-dimensional space into 2-D with UMAP, then colored by
          its K-Means cluster. Nearby points are semantically similar — the same signal the recommender uses.
        </p>
      </div>

      {error && (
        <div className="mt-8 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {data && !available && (
        <div className="card mt-8 p-8 text-center">
          <Compass width={40} height={40} className="mx-auto text-muted opacity-40" />
          <h2 className="mt-3 font-semibold">Projection not available</h2>
          <p className="mt-1 text-sm text-body">
            The UMAP embedding wasn’t built. Run the pipeline with{" "}
            <code className="rounded bg-[rgb(var(--surface-2))] px-1.5 py-0.5 font-mono text-xs">compute_viz: true</code>{" "}
            to generate it, then reload.
          </p>
          <Link to="/search" className="btn-primary mt-5">
            Go to search
          </Link>
        </div>
      )}

      {available && project && (
        <>
          <div className="mt-6 flex flex-wrap items-center gap-3 text-sm text-muted">
            <span>{data!.total.toLocaleString()} courses</span>
            <span>·</span>
            <span>{data!.n_clusters} clusters</span>
            {data!.count < data!.total && (
              <>
                <span>·</span>
                <span>showing {data!.count.toLocaleString()} sampled points</span>
              </>
            )}
          </div>

          {/* legend */}
          <div className="mt-4 flex flex-wrap gap-2">
            {Array.from({ length: data!.n_clusters }).map((_, c) => (
              <button
                key={c}
                onMouseEnter={() => setActive(c)}
                onMouseLeave={() => setActive(null)}
                onClick={() => setActive((a) => (a === c ? null : c))}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-xs transition ${
                  active === c
                    ? "border-brand-400 bg-brand-500/10 text-[rgb(var(--text))]"
                    : "border-[rgb(var(--border))] text-muted"
                }`}
              >
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: color(c) }} />
                {c}
              </button>
            ))}
          </div>

          {/* plot */}
          <div ref={wrapRef} className="card relative mt-4 overflow-hidden p-2 sm:p-4">
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ aspectRatio: `${W} / ${H}` }}>
              {data!.points.map((p, i) => {
                const dim = active !== null && p.cluster !== active;
                return (
                  <circle
                    key={`${p.id}-${i}`}
                    cx={project.sx(p.x)}
                    cy={project.sy(p.y)}
                    r={active === p.cluster ? 5 : 3.5}
                    fill={color(p.cluster)}
                    opacity={dim ? 0.12 : 0.85}
                    className="cursor-pointer transition-[r,opacity]"
                    onMouseEnter={(e) => onHover(p, e)}
                    onMouseMove={(e) => onHover(p, e)}
                    onMouseLeave={() => setTip(null)}
                  />
                );
              })}
            </svg>

            {tip && (
              <div
                className="pointer-events-none absolute z-10 max-w-xs -translate-x-1/2 -translate-y-full rounded-lg border border-[rgb(var(--border))] bg-[rgb(var(--surface))] px-3 py-2 text-xs shadow-card"
                style={{ left: tip.left, top: tip.top - 8 }}
              >
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ background: color(tip.p.cluster) }} />
                  <span className="font-mono text-muted">cluster {tip.p.cluster}</span>
                </div>
                <div className="mt-0.5 font-medium text-[rgb(var(--text))]">{tip.p.title}</div>
              </div>
            )}
          </div>
          <p className="mt-3 text-xs text-muted">
            Hover a point for its course · hover a legend chip to isolate a cluster.
          </p>
        </>
      )}

      {!data && !error && (
        <div className="mt-16 text-center text-muted">
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
          <p className="mt-3">Loading projection…</p>
        </div>
      )}
    </div>
  );
}
