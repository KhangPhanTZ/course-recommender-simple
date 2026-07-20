import { useState, type FormEvent } from "react";
import { api, ApiError, type CourseHit, type RecommendResponse } from "../lib/api";
import CourseCard from "../components/CourseCard";
import { Sparkles, SearchIcon, Bolt } from "../components/icons";

const EXAMPLES = [
  "deep learning with pytorch for beginners",
  "become a data analyst with sql and python",
  "cloud computing and devops on aws",
  "nlp and transformers for advanced learners",
];

const LEVELS = ["", "beginner", "intermediate", "advanced"];

export default function Search() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(9);
  const [level, setLevel] = useState("");
  const [explain, setExplain] = useState(true);
  const [rerank, setRerank] = useState(false);

  const [data, setData] = useState<RecommendResponse | null>(null);
  const [similar, setSimilar] = useState<{ of: CourseHit; hits: CourseHit[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(q: string) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setSimilar(null);
    try {
      const res = await api.recommend({
        query: q,
        top_k: topK,
        level: level || null,
        explain,
        rerank,
        understand: true,
      });
      setData(res);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : "Could not reach the API. Is the backend running?");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  async function findSimilar(hit: CourseHit) {
    setLoading(true);
    setError(null);
    try {
      const hits = await api.similar(hit.id, 8);
      setSimilar({ of: hit, hits });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not load similar courses.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    run(query);
  }

  return (
    <div className="container-page py-12">
      <div className="max-w-2xl">
        <span className="eyebrow">Live demo</span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">Search courses</h1>
        <p className="mt-3 text-body">
          Describe what you want to learn. The system infers filters, retrieves the best matches, and can
          explain its picks.
        </p>
      </div>

      {/* search form */}
      <form onSubmit={onSubmit} className="card mt-8 p-5 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <SearchIcon
              width={18}
              height={18}
              className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted"
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. deep learning with pytorch for beginners"
              className="field !pl-11"
              autoFocus
            />
          </div>
          <button type="submit" className="btn-primary sm:w-36" disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        {/* examples */}
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuery(ex);
                run(ex);
              }}
              className="chip transition-colors hover:border-brand-400 hover:text-[rgb(var(--text))]"
            >
              {ex}
            </button>
          ))}
        </div>

        {/* controls */}
        <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-[rgb(var(--border))] pt-4">
          <label className="flex items-center gap-2 text-sm">
            <span className="text-muted">Level</span>
            <select value={level} onChange={(e) => setLevel(e.target.value)} className="field !w-auto !py-1.5">
              {LEVELS.map((l) => (
                <option key={l} value={l}>
                  {l || "any"}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <span className="text-muted">Results</span>
            <input
              type="range"
              min={3}
              max={18}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="accent-brand-600"
            />
            <span className="w-6 font-mono tabular-nums text-muted">{topK}</span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="accent-brand-600" />
            <span className="inline-flex items-center gap-1.5">
              <Sparkles width={14} height={14} className="text-amber-500" /> RAG explanation
            </span>
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} className="accent-brand-600" />
            <span className="inline-flex items-center gap-1.5">
              <Bolt width={14} height={14} className="text-brand-500" /> Cross-encoder rerank
            </span>
          </label>
        </div>
      </form>

      {/* error */}
      {error && (
        <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {/* similar banner */}
      {similar && (
        <div className="mt-8">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Similar to <span className="text-brand-700 dark:text-brand-300">“{similar.of.title}”</span>
            </h2>
            <button onClick={() => setSimilar(null)} className="text-sm text-muted hover:text-[rgb(var(--text))]">
              Back to results
            </button>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {similar.hits.map((h, i) => (
              <CourseCard key={`${h.id}-${i}`} hit={h} rank={i + 1} onSimilar={findSimilar} />
            ))}
          </div>
        </div>
      )}

      {/* results */}
      {!similar && data && (
        <div className="mt-8 animate-fade-up">
          {/* understanding summary */}
          <div className="mb-5 flex flex-wrap items-center gap-2 text-sm">
            {data.resolved_query !== data.query && (
              <span className="chip">
                understood: <span className="text-[rgb(var(--text))]">{data.resolved_query}</span>
              </span>
            )}
            {Object.entries(data.filters).map(([k, v]) => (
              <span key={k} className="chip !border-brand-400/40 text-brand-700 dark:text-brand-300">
                {k}: {v}
              </span>
            ))}
            <span className="text-muted">{data.results.length} results</span>
          </div>

          {/* RAG explanation */}
          {data.explanation && (
            <div className="mb-6 rounded-2xl border border-amber-500/25 bg-amber-500/[0.07] p-5">
              <div className="mb-1.5 flex items-center gap-2 text-sm font-semibold text-amber-600 dark:text-amber-400">
                <Sparkles width={16} height={16} />
                Why these courses
                {!data.llm_enabled && <span className="font-normal text-muted">(template fallback)</span>}
              </div>
              <p className="text-sm leading-relaxed text-body">{data.explanation}</p>
            </div>
          )}

          {data.results.length === 0 ? (
            <p className="text-body">No matches — try a broader query or clear the level filter.</p>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.results.map((h, i) => (
                <CourseCard key={`${h.id}-${i}`} hit={h} rank={i + 1} onSimilar={findSimilar} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* empty state */}
      {!data && !error && !loading && (
        <div className="mt-16 text-center text-muted">
          <SearchIcon width={40} height={40} className="mx-auto opacity-40" />
          <p className="mt-3">Enter a query or pick an example to get recommendations.</p>
        </div>
      )}
    </div>
  );
}
