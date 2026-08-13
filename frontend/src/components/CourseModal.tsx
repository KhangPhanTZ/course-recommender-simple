import { useEffect, useState } from "react";
import { api, type CourseDetail } from "../lib/api";
import { ArrowRight, Close, Layers, Star } from "./icons";

const SOURCE_LABEL: Record<string, string> = { coursera: "Coursera", udemy: "Udemy", edx: "edX" };

function splitList(value?: string | null): string[] {
  if (!value) return [];
  return value
    .split(/[;,|/]|\s·\s/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function syllabusItems(value?: string | null): string[] {
  if (!value) return [];
  return value
    .split(/\s*\|\s*|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Minimal shape needed to open the modal — anything with an id and title. */
export type CourseRef = {
  id: number | string;
  title: string;
  provider?: string | null;
  source?: string | null;
  category?: string | null;
  level?: string | null;
  rating?: number | string | null;
  url?: string | null;
  skills?: string | null;
  description?: string | null;
  syllabus?: string | null;
};

/** Detail overlay for a course: summary, tech stack (skills), syllabus, real URL. */
export default function CourseModal({ hit, onClose }: { hit: CourseRef; onClose: () => void }) {
  const [detail, setDetail] = useState<CourseDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .course(hit.id)
      .then((d) => alive && setDetail(d))
      .catch(() => alive && setDetail(null))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [hit.id]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Fall back to the search hit while the full record loads.
  const d: CourseDetail = detail ?? (hit as CourseDetail);
  const rating = typeof d.rating === "number" ? d.rating : parseFloat(String(d.rating ?? ""));
  const skills = splitList(d.skills);
  const summary = (d.description && String(d.description).trim()) || "";
  const syllabus = syllabusItems(d.syllabus);
  const url = d.url ? String(d.url) : "";

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-0 backdrop-blur-sm sm:items-center sm:p-4"
      onClick={onClose}
    >
      <div
        className="max-h-[88vh] w-full max-w-2xl overflow-y-auto rounded-t-2xl border border-[rgb(var(--border))] bg-[rgb(var(--bg))] p-6 shadow-xl sm:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              {d.source && (
                <span className="rounded-md bg-brand-500/12 px-2 py-0.5 text-xs font-medium text-brand-700 dark:text-brand-300">
                  {SOURCE_LABEL[String(d.source)] ?? d.source}
                </span>
              )}
              {d.level && <span className="rounded-md bg-[rgb(var(--surface-2))] px-2 py-0.5 text-xs text-muted">{d.level}</span>}
              {d.category && <span className="rounded-md bg-[rgb(var(--surface-2))] px-2 py-0.5 text-xs text-muted">{d.category}</span>}
              {!Number.isNaN(rating) && rating > 0 && (
                <span className="inline-flex items-center gap-1 font-mono text-xs text-muted">
                  <Star width={12} height={12} className="text-amber-500" /> {rating.toFixed(1)}
                </span>
              )}
            </div>
            <h2 className="mt-2 text-xl font-bold tracking-tight">{d.title}</h2>
            {d.provider && <div className="mt-0.5 text-sm text-muted">{d.provider}</div>}
          </div>
          <button onClick={onClose} className="btn-ghost !p-2" aria-label="Close">
            <Close width={18} height={18} />
          </button>
        </div>

        {loading && <div className="mt-6 text-sm text-muted">Loading course details…</div>}

        {/* summary */}
        {summary && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold text-muted">Summary</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-body">{summary}</p>
          </section>
        )}

        {/* tech stack / skills */}
        {skills.length > 0 && (
          <section className="mt-5">
            <h3 className="flex items-center gap-1.5 text-sm font-semibold text-muted">
              <Layers width={14} height={14} /> Tech stack &amp; skills
            </h3>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {skills.map((s) => (
                <span key={s} className="chip !px-2 !py-0.5">{s}</span>
              ))}
            </div>
          </section>
        )}

        {/* syllabus */}
        {syllabus.length > 0 && (
          <section className="mt-5">
            <h3 className="text-sm font-semibold text-muted">Syllabus</h3>
            <ul className="mt-2 space-y-1.5">
              {syllabus.map((s, i) => (
                <li key={i} className="flex gap-2 text-sm text-body">
                  <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand-500" />
                  {s}
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* real course link */}
        <div className="mt-6 border-t border-[rgb(var(--border))] pt-4">
          {url ? (
            <a href={url} target="_blank" rel="noreferrer" className="btn-primary">
              Open course <ArrowRight width={18} height={18} />
            </a>
          ) : (
            <p className="text-xs text-muted">
              No external link in this dataset — real course URLs appear when the catalog is built from the
              source platforms' data.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
