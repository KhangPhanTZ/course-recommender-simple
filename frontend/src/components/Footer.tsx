import { Link } from "react-router-dom";
import { Github } from "./icons";

const REPO = "https://github.com/KhangPhanTZ/course-recommender-simple";

export default function Footer() {
  return (
    <footer className="mt-24 border-t border-[rgb(var(--border))]">
      <div className="container-page flex flex-col gap-6 py-10 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="font-semibold">
            Path<span className="text-brand-600 dark:text-brand-400">finder</span>
          </div>
          <p className="mt-1 max-w-md text-sm text-muted">
            Two-stage retrieval + a GenAI/RAG layer, served over FastAPI and deployable on AWS.
          </p>
        </div>
        <div className="flex items-center gap-5 text-sm text-body">
          <Link to="/architecture" className="link-underline">
            Architecture
          </Link>
          <Link to="/about" className="link-underline">
            About
          </Link>
          <a href={REPO} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 link-underline">
            <Github width={16} height={16} /> GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
