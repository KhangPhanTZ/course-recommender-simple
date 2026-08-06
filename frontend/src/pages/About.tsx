import { Cloud, Cpu, Github, Layers, Sparkles } from "../components/icons";

const REPO = "https://github.com/KhangPhanTZ/course-recommender-simple";

const stack = [
  ["Language", "Python 3.11 · TypeScript"],
  ["Retrieval", "scikit-learn · Sentence-BERT · FAISS · cross-encoder"],
  ["GenAI", "Anthropic Claude · AWS Bedrock (RAG)"],
  ["API", "FastAPI · Uvicorn · Pydantic"],
  ["Frontend", "React · Vite · Tailwind CSS"],
  ["Cloud", "AWS ECS Fargate · ALB · S3 · ECR · IAM"],
  ["IaC & CI/CD", "Terraform · GitHub Actions"],
  ["Quality", "pytest · ruff"],
];

const layers = [
  {
    tag: "Serving",
    color: "brand",
    title: "FastAPI service",
    path: "src/api",
    points: ["/recommend, /chat, /roadmap, /metrics, /health", "OpenAPI docs + request timing", "Models loaded once, reused"],
  },
  {
    tag: "GenAI / RAG",
    color: "amber",
    title: "LLM layer",
    path: "src/llm",
    points: ["Query understanding: NL → filters", "Grounded explanations, chat advisor & career-track roadmaps", "Claude API ⇄ AWS Bedrock, one env var"],
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
  { n: "01", t: "Understand", d: "The LLM rewrites the request and extracts level/category filters (heuristic fallback if no LLM)." },
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

export default function About() {
  return (
    <div className="container-page py-12">
      {/* ---------------- Intro ---------------- */}
      <div className="max-w-3xl">
        <span className="eyebrow">About the project</span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
          A recommender, engineered like a product
        </h1>
        <p className="mt-4 text-lg text-body">
          Pathfinder started as a content-based course recommender and was rebuilt into a production-shaped
          system: a two-stage retriever, a GenAI/RAG layer, a FastAPI service, and a full AWS deployment path.
          It’s designed to demonstrate the end-to-end skill set of an AI engineer — not just a model in a
          notebook.
        </p>
      </div>

      <section className="mt-12 grid gap-6 lg:grid-cols-3">
        <div className="card p-6 lg:col-span-2">
          <h2 className="text-xl font-semibold">What makes it more than a demo</h2>
          <ul className="mt-4 space-y-3 text-sm text-body">
            {[
              "Real ANN search (FAISS) with a numpy fallback so it runs anywhere.",
              "A GenAI layer that understands messy queries and grounds explanations in real results.",
              "A conversational advisor plus career-track roadmaps (PM, QA, ML, DevOps…) grounded in the catalog.",
              "A storage abstraction that switches from local files to S3 with one env var.",
              "Infrastructure as code: one terraform apply provisions the whole AWS stack.",
              "Graceful degradation everywhere — the API never fails because an LLM is missing.",
            ].map((t) => (
              <li key={t} className="flex gap-3">
                <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand-500" />
                {t}
              </li>
            ))}
          </ul>
          <div className="mt-6">
            <a href={REPO} target="_blank" rel="noreferrer" className="btn-ghost">
              <Github width={18} height={18} /> Source code
            </a>
          </div>
        </div>

        <div className="card p-6">
          <h2 className="text-xl font-semibold">Tech stack</h2>
          <dl className="mt-4 space-y-3">
            {stack.map(([k, v]) => (
              <div key={k}>
                <dt className="font-mono text-xs uppercase tracking-wide text-muted">{k}</dt>
                <dd className="text-sm text-body">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ---------------- Architecture ---------------- */}
      <section className="mt-20">
        <div className="max-w-3xl">
          <span className="eyebrow flex items-center gap-2">
            <Cpu width={15} height={15} /> System design
          </span>
          <h2 className="mt-3 text-2xl font-bold tracking-tight sm:text-3xl">Architecture</h2>
          <p className="mt-3 text-body">
            Three layers — retrieval engine, GenAI/RAG, and serving — separated by a storage abstraction, so the
            identical code runs on a laptop and on AWS.
          </p>
        </div>

        {/* layered stack */}
        <div className="mt-8 grid gap-4 lg:grid-cols-4">
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

      {/* why two-stage / why RAG */}
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
            Retrieval is the source of truth; the LLM only interprets it. Grounding explanations and the chat
            advisor in the retrieved courses means the model can’t invent a catalog that doesn’t exist — and the
            whole layer degrades to deterministic templates when no LLM is configured.
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

      {/* dataset */}
      <section className="mt-16 rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--surface))]/50 p-6">
        <h2 className="text-lg font-semibold">Dataset</h2>
        <p className="mt-2 text-sm text-body">
          A multi-platform catalog: public{" "}
          <a href="https://www.kaggle.com/datasets/siddharthm1698/coursera-course-dataset" target="_blank" rel="noreferrer" className="link-underline">Coursera</a>,{" "}
          <a href="https://www.kaggle.com/datasets/andrewmvd/udemy-courses" target="_blank" rel="noreferrer" className="link-underline">Udemy</a>, and{" "}
          <a href="https://www.kaggle.com/datasets/imuhammad/edx-courses" target="_blank" rel="noreferrer" className="link-underline">edX</a>{" "}
          datasets are normalized onto one schema (title, provider, skills, level, description, syllabus, …) and
          merged. Ingestion is by column signature, so dropping a new platform's CSV into <code>data/</code> is
          auto-detected. Content-based only — no user interaction history required.
        </p>
      </section>
    </div>
  );
}
