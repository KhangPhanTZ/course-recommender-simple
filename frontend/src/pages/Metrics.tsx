import { useEffect, useState } from "react";
import { api, type Metrics } from "../lib/api";
import { Bolt, Check, Cpu } from "../components/icons";

function pct(v?: number) {
  return v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
}
function num(v?: number, d = 3) {
  return v === undefined ? "—" : v.toFixed(d);
}

export default function MetricsPage() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.metrics().then(setM).catch(() => setError("Could not reach the API."));
  }, []);

  const quality = m?.available
    ? [
        { label: "Precision@k", value: pct(m.precision_at_k), hint: "of the top-k, how many are relevant" },
        { label: "Recall@k", value: pct(m.recall_at_k), hint: "of all relevant, how many surfaced" },
        { label: "Hit-rate", value: pct(m.hit_rate), hint: "queries with ≥1 relevant in top-k" },
        { label: "MRR", value: num(m.mrr), hint: "mean reciprocal rank of first hit" },
        { label: "nDCG@k", value: num(m.ndcg_at_k), hint: "rank-weighted relevance" },
        { label: "Silhouette", value: num(m.silhouette), hint: "cluster separation (cosine)" },
      ]
    : [];

  return (
    <div className="container-page py-12">
      <div className="max-w-2xl">
        <span className="eyebrow flex items-center gap-2">
          <Cpu width={15} height={15} /> Offline evaluation
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">Metrics</h1>
        <p className="mt-3 text-body">
          Retrieval quality measured by leave-one-out with the catalog's own categories as a relevance proxy
          (no click logs). Numbers move with retrieval quality and are reproducible.
        </p>
      </div>

      {error && (
        <div className="mt-8 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {m && !m.available && (
        <div className="mt-8 rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--surface))]/50 p-6 text-body">
          Evaluation hasn't been run yet. Generate it with{" "}
          <code className="rounded bg-[rgb(var(--surface-2))] px-1.5 py-0.5 text-sm">python scripts/run_eval.py</code>{" "}
          (it writes <code>metrics.json</code> next to the artifacts).
        </div>
      )}

      {m?.available && (
        <>
          {/* context row */}
          <div className="mt-6 flex flex-wrap gap-2 text-sm">
            <span className="chip">backend: <span className="text-[rgb(var(--text))]">{m.backend}</span></span>
            <span className="chip">{m.n_courses?.toLocaleString()} courses</span>
            <span className="chip">k = {m.k}</span>
            <span className="chip">{m.n_eval} eval queries</span>
            {m.sources &&
              Object.entries(m.sources).map(([s, n]) => (
                <span key={s} className="chip !border-brand-400/40 text-brand-700 dark:text-brand-300">
                  {s}: {n}
                </span>
              ))}
          </div>

          {/* quality cards */}
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {quality.map((q) => (
              <div key={q.label} className="card p-5">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted">{q.label}</span>
                  <Check width={15} height={15} className="text-brand-500" />
                </div>
                <div className="mt-1 text-3xl font-bold tabular-nums">{q.value}</div>
                <p className="mt-1 text-xs text-muted">{q.hint}</p>
              </div>
            ))}
          </div>

          {/* latency */}
          {m.latency_ms && (
            <div className="mt-8">
              <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
                <Bolt width={18} height={18} className="text-amber-500" /> Query latency
                <span className="text-sm font-normal text-muted">(real /recommend path)</span>
              </h2>
              <div className="grid gap-4 sm:grid-cols-3">
                {([["p50", m.latency_ms.p50], ["p95", m.latency_ms.p95], ["mean", m.latency_ms.mean]] as const).map(
                  ([k, v]) => (
                    <div key={k} className="card p-5">
                      <div className="text-sm text-muted">{k}</div>
                      <div className="mt-1 text-2xl font-bold tabular-nums">
                        {v} <span className="text-base font-normal text-muted">ms</span>
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          )}

          <p className="mt-8 text-xs text-muted">
            Generated {m.generated_at ? new Date(m.generated_at).toLocaleString() : "—"} ·
            {" "}reproduce with <code>python scripts/run_eval.py</code>.
          </p>
        </>
      )}
    </div>
  );
}
