import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Health } from "../lib/api";
import { ArrowRight, Bolt, Cloud, Cpu, Layers, Sparkles, SearchIcon } from "../components/icons";

const features = [
  {
    icon: SearchIcon,
    title: "Semantic search",
    body: "Describe a goal in plain English. Sentence-BERT embeddings match meaning, not just keywords.",
  },
  {
    icon: Layers,
    title: "Two-stage retrieval",
    body: "A fast FAISS nearest-neighbour recall, then an optional cross-encoder reranks for precision.",
  },
  {
    icon: Sparkles,
    title: "GenAI explanations",
    body: "A RAG layer grounds an LLM in the retrieved courses to explain why they fit — no hallucinated catalog.",
  },
  {
    icon: Cloud,
    title: "AWS-ready",
    body: "Pluggable S3 storage, Bedrock or Claude API, packaged for ECS Fargate with Terraform + CI/CD.",
  },
];

const steps = [
  { n: "01", t: "Understand", d: "The LLM rewrites your request into a clean query and infers filters (level, category)." },
  { n: "02", t: "Retrieve", d: "The query is embedded and searched against the course index with cosine similarity." },
  { n: "03", t: "Rerank", d: "A cross-encoder rescoring sharpens the shortlist for relevance (optional)." },
  { n: "04", t: "Explain", d: "The retrieved set grounds a short, per-user rationale and a suggested learning order." },
];

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setDown(true));
  }, []);

  return (
    <div>
      {/* ---------------- Hero ---------------- */}
      <section className="relative overflow-hidden border-b border-[rgb(var(--border))]">
        <div className="grid-dots pointer-events-none absolute inset-0 opacity-60" />
        <div className="pointer-events-none absolute -top-40 right-0 h-96 w-96 rounded-full bg-brand-500/20 blur-3xl" />
        <div className="container-page relative py-20 sm:py-28">
          <div className="max-w-3xl animate-fade-up">
            <span className="chip mb-6">
              <Sparkles width={14} height={14} className="text-brand-600 dark:text-brand-400" />
              Retrieval &nbsp;·&nbsp; RAG &nbsp;·&nbsp; AWS
            </span>
            <h1 className="text-balance text-4xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
              Find the right course with{" "}
              <span className="bg-gradient-to-r from-brand-600 to-amber-500 bg-clip-text text-transparent">
                AI that explains itself
              </span>
            </h1>
            <p className="mt-6 max-w-xl text-lg text-body">
              Pathfinder pairs a two-stage semantic retriever with a GenAI layer that understands messy
              requests and grounds its recommendations in real results.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link to="/search" className="btn-primary">
                Try the live demo <ArrowRight width={18} height={18} />
              </Link>
              <Link to="/about" className="btn-ghost">
                How it works
              </Link>
            </div>

            {/* live status */}
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-xs text-muted">
              <span className="inline-flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${
                    down ? "bg-rose-500" : health ? "bg-brand-500" : "bg-amber-500"
                  } ${!down && !health ? "animate-pulse" : ""}`}
                />
                API {down ? "offline" : health ? health.status : "checking…"}
              </span>
              {health?.backend && <span>backend: {health.backend}</span>}
              {typeof health?.n_courses === "number" && <span>{health.n_courses.toLocaleString()} courses</span>}
              {health?.llm_provider && <span>llm: {health.llm_provider}</span>}
            </div>
          </div>
        </div>
      </section>

      {/* ---------------- Features ---------------- */}
      <section className="container-page py-20">
        <div className="mb-12 max-w-2xl">
          <span className="eyebrow">What it does</span>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">More than nearest-neighbour lookup</h2>
        </div>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <div key={f.title} className="card p-6">
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-brand-500/12 text-brand-600 dark:text-brand-400">
                <f.icon width={22} height={22} />
              </span>
              <h3 className="mt-4 font-semibold">{f.title}</h3>
              <p className="mt-1.5 text-sm text-body">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------- How it works ---------------- */}
      <section className="border-y border-[rgb(var(--border))] bg-[rgb(var(--surface))]/40">
        <div className="container-page py-20">
          <div className="mb-12 max-w-2xl">
            <span className="eyebrow">How a request flows</span>
            <h2 className="mt-3 text-3xl font-bold tracking-tight">Four stages, one response</h2>
          </div>
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
            {steps.map((s) => (
              <div key={s.n} className="relative">
                <div className="font-mono text-sm text-brand-600 dark:text-brand-400">{s.n}</div>
                <h3 className="mt-2 text-lg font-semibold">{s.t}</h3>
                <p className="mt-1.5 text-sm text-body">{s.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---------------- Tech band ---------------- */}
      <section className="container-page py-20">
        <div className="card flex flex-col items-start gap-8 p-8 sm:flex-row sm:items-center sm:justify-between">
          <div className="max-w-lg">
            <span className="eyebrow flex items-center gap-2">
              <Cpu width={15} height={15} /> Engineered like production
            </span>
            <h2 className="mt-3 text-2xl font-bold tracking-tight">Built to demonstrate real AI-engineering</h2>
            <p className="mt-2 text-body">
              FastAPI service, FAISS ANN, cross-encoder reranking, multi-provider LLM, S3 storage, Terraform on
              AWS, and CI/CD — with 30 unit tests keeping it honest.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {["FastAPI", "FAISS", "Sentence-BERT", "Claude / Bedrock", "S3", "ECS Fargate", "Terraform", "pytest"].map(
              (t) => (
                <span key={t} className="chip">
                  <Bolt width={12} height={12} className="text-amber-500" />
                  {t}
                </span>
              ),
            )}
          </div>
        </div>
      </section>

      {/* ---------------- CTA ---------------- */}
      <section className="container-page pb-24">
        <div className="relative overflow-hidden rounded-3xl bg-brand-600 px-8 py-14 text-center text-white sm:py-20">
          <div className="pointer-events-none absolute inset-0 opacity-20 grid-dots" />
          <h2 className="relative text-3xl font-bold tracking-tight sm:text-4xl">Ready to search?</h2>
          <p className="relative mx-auto mt-3 max-w-md text-brand-50/90">
            Type what you want to learn and watch the recommender understand, retrieve, and explain.
          </p>
          <Link
            to="/search"
            className="btn relative mt-7 bg-white text-brand-700 hover:bg-brand-50"
          >
            Open the demo <ArrowRight width={18} height={18} />
          </Link>
        </div>
      </section>
    </div>
  );
}
