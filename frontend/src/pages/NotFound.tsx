import { Link } from "react-router-dom";
import { ArrowRight } from "../components/icons";

export default function NotFound() {
  return (
    <div className="container-page grid min-h-[60vh] place-items-center py-20 text-center">
      <div>
        <div className="font-mono text-6xl font-bold text-brand-600 dark:text-brand-400">404</div>
        <h1 className="mt-4 text-2xl font-bold tracking-tight">Page not found</h1>
        <p className="mt-2 text-body">The page you’re looking for doesn’t exist.</p>
        <Link to="/" className="btn-primary mt-6">
          Back home <ArrowRight width={18} height={18} />
        </Link>
      </div>
    </div>
  );
}
