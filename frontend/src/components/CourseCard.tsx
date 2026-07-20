import type { CourseHit } from "../lib/api";
import { Star } from "./icons";

function levelTone(level?: string | null) {
  const l = (level ?? "").toLowerCase();
  if (l.includes("begin")) return "bg-brand-500/15 text-brand-700 dark:text-brand-300";
  if (l.includes("inter")) return "bg-amber-500/15 text-amber-600 dark:text-amber-400";
  if (l.includes("adv")) return "bg-rose-500/15 text-rose-600 dark:text-rose-400";
  return "bg-[rgb(var(--surface-2))] text-muted";
}

function skillList(skills?: string | null): string[] {
  if (!skills) return [];
  return skills
    .split(/[;,|/]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 4);
}

export default function CourseCard({
  hit,
  rank,
  onSimilar,
}: {
  hit: CourseHit;
  rank?: number;
  onSimilar?: (hit: CourseHit) => void;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(hit.score * 100)));
  const rating = typeof hit.rating === "number" ? hit.rating : parseFloat(String(hit.rating ?? ""));

  return (
    <article className="card group flex flex-col gap-3 p-5 transition-transform duration-200 hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          {rank !== undefined && (
            <span className="mt-0.5 grid h-6 w-6 flex-none place-items-center rounded-md bg-[rgb(var(--surface-2))] font-mono text-xs text-muted">
              {rank}
            </span>
          )}
          <h3 className="text-[15px] font-semibold leading-snug text-[rgb(var(--text))]">
            {hit.url ? (
              <a href={hit.url} target="_blank" rel="noreferrer" className="hover:text-brand-600">
                {hit.title}
              </a>
            ) : (
              hit.title
            )}
          </h3>
        </div>
        {!Number.isNaN(rating) && rating > 0 && (
          <span className="flex flex-none items-center gap-1 font-mono text-xs text-muted">
            <Star width={13} height={13} className="text-amber-500" />
            {rating.toFixed(1)}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {hit.level && (
          <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${levelTone(hit.level)}`}>{hit.level}</span>
        )}
        {hit.category && (
          <span className="rounded-md bg-[rgb(var(--surface-2))] px-2 py-0.5 text-xs text-muted">{hit.category}</span>
        )}
      </div>

      {skillList(hit.skills).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {skillList(hit.skills).map((s) => (
            <span key={s} className="chip !px-2 !py-0.5">
              {s}
            </span>
          ))}
        </div>
      )}

      <div className="mt-auto pt-1">
        <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-muted">
          <span>match</span>
          <span className="tabular-nums">{pct}%</span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-[rgb(var(--surface-2))]">
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-400"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {onSimilar && (
        <button
          onClick={() => onSimilar(hit)}
          className="mt-1 self-start text-xs font-medium text-brand-700 opacity-0 transition-opacity group-hover:opacity-100 dark:text-brand-300"
        >
          Find similar →
        </button>
      )}
    </article>
  );
}
