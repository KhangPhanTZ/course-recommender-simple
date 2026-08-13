import type { CourseHit, Roadmap } from "../lib/api";
import { ArrowRight } from "./icons";

const SOURCE_LABEL: Record<string, string> = { coursera: "Coursera", udemy: "Udemy", edx: "edX" };

/** A grounded career-track roadmap rendered as a vertical tier flow. */
export default function RoadmapGraph({
  roadmap,
  onOpenCourse,
}: {
  roadmap: Roadmap;
  onOpenCourse?: (course: CourseHit) => void;
}) {
  return (
    <div className="mt-3 w-full">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-sm font-semibold text-brand-700 dark:text-brand-300">{roadmap.label}</span>
        <span className="text-xs text-muted">roadmap · grounded in the catalog</span>
      </div>

      <ol className="relative space-y-3 border-l-2 border-dashed border-[rgb(var(--border))] pl-5">
        {roadmap.nodes.map((node, i) => (
          <li key={node.id} className="relative">
            {/* tier marker */}
            <span className="absolute -left-[27px] grid h-6 w-6 place-items-center rounded-full bg-brand-600 text-[11px] font-bold text-white ring-4 ring-[rgb(var(--bg))]">
              {i + 1}
            </span>

            <div className="surface rounded-xl p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-semibold text-[rgb(var(--text))]">{node.tier}</span>
                {node.skills.slice(0, 4).map((s) => (
                  <span key={s} className="chip !px-2 !py-0.5 !text-[11px]">{s}</span>
                ))}
              </div>

              {node.courses.length > 0 ? (
                <ul className="mt-2 space-y-1.5">
                  {node.courses.map((c, j) => (
                    <li key={`${c.id}-${j}`} className="flex items-start gap-2 text-sm">
                      <span className="mt-1 h-1.5 w-1.5 flex-none rounded-full bg-brand-500" />
                      <span className="text-body">
                        <button
                          onClick={() => onOpenCourse?.(c)}
                          className="text-left hover:text-brand-600"
                        >
                          {c.title}
                        </button>
                        {c.source && (
                          <span className="ml-1.5 rounded bg-brand-500/12 px-1.5 py-0.5 text-[10px] font-medium text-brand-700 dark:text-brand-300">
                            {SOURCE_LABEL[c.source] ?? c.source}
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-xs text-muted">No catalog course matched this tier yet — try a richer dataset.</p>
              )}
            </div>
          </li>
        ))}
      </ol>

      {roadmap.bridges.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted">
          <span className="inline-flex items-center gap-1 font-medium">
            <ArrowRight width={13} height={13} /> Where next:
          </span>
          {roadmap.bridges.map((b) => (
            <span key={b.track} className="chip !py-1" title={b.note}>{b.label}</span>
          ))}
        </div>
      )}
    </div>
  );
}
