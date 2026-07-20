import { Link } from "react-router-dom";
import { ArrowRight, Github } from "../components/icons";

const REPO = "https://github.com/KhangPhanTZ/course-recommender-simple";

const stack = [
  ["Language", "Python 3.11 · TypeScript"],
  ["Retrieval", "scikit-learn · Sentence-BERT · FAISS · cross-encoder"],
  ["GenAI", "Anthropic Claude · AWS Bedrock (RAG)"],
  ["API", "FastAPI · Uvicorn · Pydantic"],
  ["Frontend", "React · Vite · Tailwind CSS"],
  ["Cloud", "AWS ECS Fargate · ALB · S3 · ECR · IAM"],
  ["IaC & CI/CD", "Terraform · GitHub Actions"],
  ["Quality", "pytest (30 tests) · ruff"],
];

export default function About() {
  return (
    <div className="container-page py-12">
      <div className="max-w-3xl">
        <span className="eyebrow">About the project</span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
          A recommender, engineered like a product
        </h1>
        <p className="mt-4 text-lg text-body">
          CourseIQ started as a content-based course recommender and was rebuilt into a production-shaped
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
              "A storage abstraction that switches from local files to S3 with one env var.",
              "Infrastructure as code: one terraform apply provisions the whole AWS stack.",
              "CI/CD that lints, tests, builds the image, and can deploy to ECS.",
              "Graceful degradation everywhere — the API never fails because an LLM is missing.",
            ].map((t) => (
              <li key={t} className="flex gap-3">
                <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand-500" />
                {t}
              </li>
            ))}
          </ul>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/architecture" className="btn-primary">
              Explore the architecture <ArrowRight width={18} height={18} />
            </Link>
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

      <section className="mt-8 rounded-2xl border border-[rgb(var(--border))] bg-[rgb(var(--surface))]/50 p-6">
        <h2 className="text-lg font-semibold">Dataset</h2>
        <p className="mt-2 text-sm text-body">
          Built on the Kaggle{" "}
          <a
            href="https://www.kaggle.com/datasets/everydaycodings/multi-platform-online-courses-dataset"
            target="_blank"
            rel="noreferrer"
            className="link-underline"
          >
            Multi-Platform Online Courses
          </a>{" "}
          dataset (Coursera slice). Content-based only — the system needs no user interaction history to make
          quality recommendations.
        </p>
      </section>
    </div>
  );
}
