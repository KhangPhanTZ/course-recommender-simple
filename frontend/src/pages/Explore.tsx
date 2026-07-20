import { useEffect, useState, type FormEvent } from "react";
import { api, ApiError, type CourseHit, type Health } from "../lib/api";
import CourseCard from "../components/CourseCard";
import { Compass, Layers } from "../components/icons";

export default function Explore() {
  const [health, setHealth] = useState<Health | null>(null);
  const [id, setId] = useState("0");
  const [seed, setSeed] = useState<string | null>(null);
  const [hits, setHits] = useState<CourseHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
  }, []);

  async function run(e?: FormEvent) {
    e?.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const [course, neighbours] = await Promise.all([
        api.course(id).catch(() => null),
        api.similar(id, 9),
      ]);
      setSeed((course?.title as string) ?? `#${id}`);
      setHits(neighbours);
    } catch (err) {
      setError(err instanceof ApiError ? `${err.message} (HTTP ${err.status})` : "Could not reach the API.");
      setHits(null);
    } finally {
      setLoading(false);
    }
  }

  const max = (health?.n_courses ?? 1) - 1;

  return (
    <div className="container-page py-12">
      <div className="max-w-2xl">
        <span className="eyebrow flex items-center gap-2">
          <Compass width={15} height={15} /> Explore
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">Course-to-course similarity</h1>
        <p className="mt-3 text-body">
          Every course is an embedding in the same vector space. Pick one and the index returns its nearest
          neighbours — the same signal that powers clustering and the 2-D map.
        </p>
      </div>

      {/* stats */}
      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        <Stat label="Courses indexed" value={health?.n_courses?.toLocaleString() ?? "—"} />
        <Stat label="Embedding backend" value={health?.backend ?? "—"} />
        <Stat label="API version" value={health?.version ? `v${health.version}` : "—"} />
      </div>

      {/* find similar */}
      <form onSubmit={run} className="card mt-8 flex flex-col gap-3 p-5 sm:flex-row sm:items-end">
        <label className="flex-1">
          <span className="mb-1.5 block text-sm text-muted">
            Course ID {health?.n_courses ? `(0 – ${max})` : ""}
          </span>
          <input
            value={id}
            onChange={(e) => setId(e.target.value)}
            inputMode="numeric"
            className="field"
            placeholder="e.g. 0"
          />
        </label>
        <button type="submit" className="btn-primary sm:w-44" disabled={loading}>
          {loading ? "Loading…" : "Find similar"}
        </button>
      </form>

      {error && (
        <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {hits && (
        <div className="mt-8 animate-fade-up">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <Layers width={18} height={18} className="text-brand-500" />
            Nearest neighbours of <span className="text-brand-700 dark:text-brand-300">“{seed}”</span>
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {hits.map((h, i) => (
              <CourseCard
                key={`${h.id}-${i}`}
                hit={h}
                rank={i + 1}
                onSimilar={(hit) => {
                  setId(String(hit.id));
                  setTimeout(() => run(), 0);
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card p-5">
      <div className="text-sm text-muted">{label}</div>
      <div className="mt-1 text-2xl font-bold tabular-nums">{value}</div>
    </div>
  );
}
