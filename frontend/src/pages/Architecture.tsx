import { Cloud, Cpu, Layers, Sparkles } from "../components/icons";

const layers = [
  {
    tag: "Serving",
    color: "brand",
    title: "FastAPI service",
    path: "src/api",
    points: ["/recommend, /similar, /courses, /health", "OpenAPI docs + request timing", "Models loaded once, reused"],
  },
  {
    tag: "GenAI / RAG",
    color: "amber",
    title: "LLM layer",
    path: "src/llm",
    points: [
      "Query understanding: NL → filters",
      "Grounded explanations (RAG)",
      "Claude API ⇄ AWS Bedrock, one env var",
    ],
  },
  {
    tag: "Engine",
    color: "brand",
    title: "Retrieval engine",
    path: "src/recsys",
    points: ["Encode (TF-IDF / SBERT)", "FAISS ANN, numpy fallback", "Cross-encoder rerank (optional)"],
  },
  {
    tag: "Storage",
    color: "brand",
    title: "Artifact store",
    path: "src/storage",
    points: ["Local filesystem ⇄ AWS S3", "Same code, one env var", "meta, embeddings, index, parquet"],
  },
];

const steps = [
  { n: "01", t: "Understand", d: "LLM rewrites the request and extracts level/category filters (heuristic fallback if no LLM)." },
  { n: "02", t: "Retrieve", d: "The query embedding is searched against the index with cosine similarity." },
  { n: "03", t: "Filter + rerank", d: "Filters mask the candidate set; a cross-encoder can rescore the shortlist." },
  { n: "04", t: "Explain", d: "Retrieved courses ground an LLM rationale — anti-hallucination by construction." },
];

const aws = [
  { t: "ALB", d: "Public HTTP entrypoint" },
  { t: "ECS Fargate", d: "FastAPI tasks, autoscaled" },
  { t: "ECR", d: "Container image registry" },
  { t: "S3", d: "Model artifacts (task IAM role)" },
  { t: "Bedrock", d: "Claude for the GenAI layer" },
  { t: "CloudWatch", d: "Logs + container insights" },
];

export default function Architecture() {
  return (
    <div className="container-page py-12">
      <div className="max-w-3xl">
        <span className="eyebrow flex items-center gap-2">
          <Cpu width={15} height={15} /> System design
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">Architecture</h1>
        <p className="mt-3 text-lg text-body">
          Three layers — retrieval engine, GenAI/RAG, and serving — separated by a storage abstraction, so the
          identical code runs on a laptop and on AWS.
        </p>
      </div>

      {/* layered stack */}
      <section className="mt-12">
        <div className="grid gap-4 lg:grid-cols-4">
          {layers.map((l) => (
            <div key={l.title} className="card p-5">
              <div className="flex items-center justify-between">
                <span
                  className={`rounded-md px-2 py-0.5 font-mono text-[11px] font-semibold uppercase tracking-wide ${
                    l.color === "amber"
                      ? "bg-amber-500/15 text-amber-600 dark:text-amber-400"
                      : "bg-brand-500/15 text-brand-700 dark:text-brand-300"
                  }`}
                >
                  {l.tag}
                </span>
                {l.color === "amber" ? (
                  <Sparkles width={16} height={16} className="text-amber-500" />
                ) : (
                  <Layers width={16} height={16} className="text-brand-500" />
                )}
              </div>
              <h3 className="mt-3 font-semibold">{l.title}</h3>
              <code className="text-xs text-muted">{l.path}</code>
              <ul className="mt-3 space-y-1.5">
                {l.points.map((p) => (
                  <li key={p} className="flex gap-2 text-sm text-body">
                    <span className="mt-2 h-1 w-1 flex-none rounded-full bg-brand-500" />
                    {p}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* request flow */}
      <section className="mt-16">
        <span className="eyebrow">Request flow — POST /recommend</span>
        <div className="mt-6 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((s, i) => (
            <div key={s.n} className="card relative p-5">
              <div className="font-mono text-sm text-brand-600 dark:text-brand-400">{s.n}</div>
              <h3 className="mt-1 font-semibold">{s.t}</h3>
              <p className="mt-1.5 text-sm text-body">{s.d}</p>
              {i < steps.length - 1 && (
                <div className="absolute -right-2.5 top-1/2 hidden -translate-y-1/2 text-brand-400 lg:block">→</div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* why two-stage */}
      <section className="mt-16 grid gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="text-lg font-semibold">Why two-stage retrieval?</h3>
          <p className="mt-2 text-sm text-body">
            The first stage (ANN) is cheap and high-recall — it pulls a broad candidate set fast. The second
            stage (cross-encoder) is expensive but high-precision — it rescoring only the shortlist. You get
            both speed and quality instead of trading one for the other.
          </p>
        </div>
        <div className="card p-6">
          <h3 className="text-lg font-semibold">Why RAG for a recommender?</h3>
          <p className="mt-2 text-sm text-body">
            Retrieval is the source of truth; the LLM only interprets it. Grounding the explanation in the
            retrieved courses means the model can’t invent a catalog that doesn’t exist — and the whole layer
            degrades to deterministic templates when no LLM is configured.
          </p>
        </div>
      </section>

      {/* AWS */}
      <section className="mt-16">
        <span className="eyebrow flex items-center gap-2">
          <Cloud width={15} height={15} /> Deployment on AWS
        </span>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {aws.map((a) => (
            <div key={a.t} className="card flex items-center gap-4 p-5">
              <span className="grid h-10 w-10 flex-none place-items-center rounded-xl bg-brand-500/12 text-brand-600 dark:text-brand-400">
                <Cloud width={20} height={20} />
              </span>
              <div>
                <div className="font-semibold">{a.t}</div>
                <div className="text-sm text-muted">{a.d}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-muted">
          Provisioned with Terraform (<code className="text-body">infra/aws</code>). Bedrock is the default
          GenAI provider on AWS — no API key, authorized by the task IAM role.
        </p>
      </section>
    </div>
  );
}
