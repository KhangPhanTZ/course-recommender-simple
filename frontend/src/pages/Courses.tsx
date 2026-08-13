import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type CourseDetail } from "../lib/api";
import CourseModal from "../components/CourseModal";
import { Layers, SearchIcon } from "../components/icons";

const SOURCE_LABEL: Record<string, string> = { coursera: "Coursera", udemy: "Udemy", edx: "edX" };
const SOURCES = ["", "coursera", "udemy", "edx"];
const LEVELS = ["", "beginner", "intermediate", "advanced"];
const PAGE = 24;

export default function Courses() {
  const [q, setQ] = useState("");
  const [source, setSource] = useState("");
  const [level, setLevel] = useState("");
  const [items, setItems] = useState<CourseDetail[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CourseDetail | null>(null);
  const reqId = useRef(0);

  async function load(offset: number, replace: boolean) {
    const mine = ++reqId.current;
    setLoading(true);
    setError(null);
    try {
      const res = await api.catalog({ q, source, level, limit: PAGE, offset });
      if (mine !== reqId.current) return; // ignore stale responses
      setTotal(res.total);
      setItems((prev) => (replace ? res.items : [...prev, ...res.items]));
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : "Could not reach the API.");
    } finally {
      if (mine === reqId.current) setLoading(false);
    }
  }

  // initial load + reload when filters change (debounced for the text box)
  useEffect(() => {
    const t = setTimeout(() => load(0, true), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, source, level]);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    load(0, true);
  }

  return (
    <div className="container-page py-12">
      <div className="max-w-2xl">
        <span className="eyebrow flex items-center gap-2">
          <Layers width={15} height={15} /> Catalog
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">Browse courses</h1>
        <p className="mt-3 text-body">
          The full catalog powering the recommender. Search, filter, and click any course for its summary,
          skills, and link.
        </p>
      </div>

      {/* filters */}
      <form onSubmit={onSubmit} className="card mt-8 flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <SearchIcon width={18} height={18} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search title or skills…"
            className="field !pl-11"
          />
        </div>
        <select value={source} onChange={(e) => setSource(e.target.value)} className="field !w-auto !py-2">
          {SOURCES.map((s) => (
            <option key={s} value={s}>{s ? (SOURCE_LABEL[s] ?? s) : "all sources"}</option>
          ))}
        </select>
        <select value={level} onChange={(e) => setLevel(e.target.value)} className="field !w-auto !py-2">
          {LEVELS.map((l) => (
            <option key={l} value={l}>{l || "any level"}</option>
          ))}
        </select>
      </form>

      {error && (
        <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {/* count */}
      {!error && (
        <div className="mt-6 text-sm text-muted">
          {total.toLocaleString()} course{total === 1 ? "" : "s"}
          {(q || source || level) && " match your filters"}
        </div>
      )}

      {/* grid */}
      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((c, i) => (
          <button
            key={`${c.id}-${i}`}
            onClick={() => setSelected(c)}
            className="card group flex flex-col gap-2 p-5 text-left transition-transform duration-200 hover:-translate-y-0.5"
          >
            <div className="flex flex-wrap items-center gap-1.5">
              {c.source && (
                <span className="rounded-md bg-brand-500/12 px-2 py-0.5 text-xs font-medium text-brand-700 dark:text-brand-300">
                  {SOURCE_LABEL[String(c.source)] ?? c.source}
                </span>
              )}
              {c.level && <span className="rounded-md bg-[rgb(var(--surface-2))] px-2 py-0.5 text-xs text-muted">{c.level}</span>}
            </div>
            <h3 className="text-[15px] font-semibold leading-snug text-[rgb(var(--text))] group-hover:text-brand-600">
              {c.title}
            </h3>
            {c.provider && <div className="-mt-1 text-xs text-muted">{c.provider}</div>}
            {c.description && (
              <p className="line-clamp-3 text-sm text-body">{String(c.description)}</p>
            )}
          </button>
        ))}
      </div>

      {/* empty */}
      {!loading && items.length === 0 && !error && (
        <div className="mt-16 text-center text-muted">No courses match your filters.</div>
      )}

      {/* load more */}
      {items.length < total && (
        <div className="mt-8 flex justify-center">
          <button onClick={() => load(items.length, false)} disabled={loading} className="btn-ghost">
            {loading ? "Loading…" : `Load more (${items.length} / ${total.toLocaleString()})`}
          </button>
        </div>
      )}

      {selected && <CourseModal hit={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
