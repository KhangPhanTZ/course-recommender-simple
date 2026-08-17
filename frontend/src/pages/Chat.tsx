import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type ChatMessage, type CourseHit, type Roadmap, type TrackInfo } from "../lib/api";
import { Chat as ChatIcon, Compass, Send, Sparkles } from "../components/icons";
import RoadmapGraph from "../components/RoadmapGraph";
import CourseModal, { type CourseRef } from "../components/CourseModal";

interface Turn {
  role: "user" | "assistant";
  content: string;
  courses?: CourseHit[];
  roadmap?: Roadmap;
}

const SUGGESTIONS = [
  "I want to move into data engineering — where do I start?",
  "Explain what a deep learning course covers and how to progress.",
  "Which courses cover SQL for analytics?",
];

const GROUP_ORDER = ["Data & AI", "Software & Cloud", "Business & Product", "Design"];

/** Bucket tracks by group, preserving a fixed group order. */
function groupTracks(tracks: TrackInfo[]): [string, TrackInfo[]][] {
  const byGroup = new Map<string, TrackInfo[]>();
  for (const t of tracks) {
    const g = t.group ?? "";
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g)!.push(t);
  }
  return [...byGroup.entries()].sort(
    (a, b) => (GROUP_ORDER.indexOf(a[0]) + 1 || 99) - (GROUP_ORDER.indexOf(b[0]) + 1 || 99),
  );
}

const GREETING: Turn = {
  role: "assistant",
  content:
    "Hi! I'm your learning advisor. Pick a career track for a grounded roadmap, or ask about any topic — I'll explain what the relevant courses cover and how to progress.",
};

const STORAGE_KEY = "pathfinder.chat.v1";
const MAX_STORED = 50;

/** Restore a persisted conversation, falling back to a fresh greeting. */
function loadTurns(): Turn[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed as Turn[];
    }
  } catch {
    /* corrupt or unavailable storage — start fresh */
  }
  return [GREETING];
}

export default function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>(loadTurns);
  const [tracks, setTracks] = useState<TrackInfo[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CourseRef | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  useEffect(() => {
    api.tracks().then(setTracks).catch(() => undefined);
  }, []);

  // Persist the conversation so it survives a reload (client-side memory).
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(turns.slice(-MAX_STORED)));
    } catch {
      /* quota exceeded or storage disabled — non-fatal */
    }
  }, [turns]);

  function clearChat() {
    if (loading) return;
    setTurns([GREETING]);
    setError(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
  }

  async function send(text: string) {
    const msg = text.trim();
    if (!msg || loading) return;
    setError(null);
    const next: Turn[] = [...turns, { role: "user", content: msg }];
    setTurns(next);
    setInput("");
    setLoading(true);
    try {
      const history: ChatMessage[] = next
        .filter((t) => t !== GREETING && !t.roadmap)
        .map((t) => ({ role: t.role, content: t.content }));
      const res = await api.chat(history);
      setTurns((t) => [...t, { role: "assistant", content: res.reply, courses: res.courses }]);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : "Could not reach the API.");
    } finally {
      setLoading(false);
    }
  }

  async function pickTrack(track: TrackInfo) {
    if (loading) return;
    setError(null);
    setTurns((prev) => [...prev, { role: "user", content: `Show me the ${track.label} roadmap` }]);
    setLoading(true);
    try {
      const rm = await api.roadmap(track.id);
      setTurns((t) => [...t, { role: "assistant", content: rm.intro || rm.summary, roadmap: rm }]);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.message} (HTTP ${e.status})` : "Could not build the roadmap.");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  return (
    <div className="container-page flex min-h-[calc(100vh-4rem)] flex-col py-8">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div className="max-w-2xl">
          <span className="eyebrow flex items-center gap-2">
            <ChatIcon width={15} height={15} /> Learning advisor
          </span>
          <h1 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Chat</h1>
          <p className="mt-1.5 text-sm text-body">
            Pick a career track for a grounded roadmap, or ask anything. Replies are grounded in the catalog (RAG).
          </p>
        </div>
        {turns.length > 1 && (
          <button
            onClick={clearChat}
            disabled={loading}
            className="chip shrink-0 transition-colors hover:border-brand-400 hover:text-[rgb(var(--text))] disabled:opacity-50"
            title="Clear this conversation and start over"
          >
            New chat
          </button>
        )}
      </div>

      {/* career-track picker */}
      {tracks.length > 0 && (
        <div className="mb-4 rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--surface))]/40 p-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-muted">
            <Compass width={14} height={14} /> Build a career roadmap
          </div>
          <div className="space-y-2.5">
            {groupTracks(tracks).map(([group, groupTracksList]) => (
              <div key={group}>
                {group && (
                  <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted/70">{group}</div>
                )}
                <div className="flex flex-wrap gap-2">
                  {groupTracksList.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => pickTrack(t)}
                      disabled={loading}
                      title={t.summary}
                      className="chip transition-colors hover:border-brand-400 hover:text-[rgb(var(--text))] disabled:opacity-50"
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* conversation */}
      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {turns.map((t, i) => (
          <div key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                t.role === "user"
                  ? "max-w-[85%] bg-brand-600 text-white"
                  : t.roadmap
                    ? "w-full max-w-2xl surface text-[rgb(var(--text))]"
                    : "max-w-[85%] surface text-[rgb(var(--text))]"
              }`}
            >
              {t.role === "assistant" && (
                <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400">
                  <Sparkles width={13} height={13} /> Advisor
                </div>
              )}
              <p className="whitespace-pre-wrap">{t.content}</p>

              {t.roadmap && <RoadmapGraph roadmap={t.roadmap} onOpenCourse={setSelected} />}

              {t.courses && t.courses.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5 border-t border-[rgb(var(--border))] pt-2.5">
                  {t.courses.slice(0, 6).map((c, j) => (
                    <button
                      key={`${c.id}-${j}`}
                      onClick={() => setSelected(c)}
                      className="chip !py-1 transition-colors hover:border-brand-400 hover:text-[rgb(var(--text))]"
                      title={c.skills ?? "View details"}
                    >
                      {c.title.length > 46 ? c.title.slice(0, 46) + "…" : c.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="surface rounded-2xl px-4 py-3">
              <span className="inline-flex gap-1">
                <span className="h-2 w-2 animate-bounce rounded-full bg-brand-500 [animation-delay:-0.2s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-brand-500 [animation-delay:-0.1s]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-brand-500" />
              </span>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {error && (
        <div className="mb-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-700 dark:text-rose-300">
          {error}
        </div>
      )}

      {/* suggestions (only before first user turn) */}
      {turns.length === 1 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button key={s} onClick={() => send(s)} className="chip transition-colors hover:border-brand-400 hover:text-[rgb(var(--text))]">
              {s}
            </button>
          ))}
        </div>
      )}

      {/* composer */}
      <form onSubmit={onSubmit} className="sticky bottom-0 flex gap-2 bg-[rgb(var(--bg))] pt-1">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a topic, a course, or your learning path…"
          className="field"
          autoFocus
        />
        <button type="submit" className="btn-primary !px-4" disabled={loading || !input.trim()} aria-label="Send">
          <Send width={18} height={18} />
        </button>
      </form>

      {selected && <CourseModal hit={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
