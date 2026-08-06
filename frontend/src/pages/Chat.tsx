import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type ChatMessage, type CourseHit, type Roadmap, type TrackInfo } from "../lib/api";
import { Chat as ChatIcon, Compass, Send, Sparkles } from "../components/icons";
import RoadmapGraph from "../components/RoadmapGraph";

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

const GREETING: Turn = {
  role: "assistant",
  content:
    "Hi! I'm your learning advisor. Pick a career track for a grounded roadmap, or ask about any topic — I'll explain what the relevant courses cover and how to progress.",
};

export default function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([GREETING]);
  const [tracks, setTracks] = useState<TrackInfo[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  useEffect(() => {
    api.tracks().then(setTracks).catch(() => undefined);
  }, []);

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
      <div className="mb-4 max-w-2xl">
        <span className="eyebrow flex items-center gap-2">
          <ChatIcon width={15} height={15} /> Learning advisor
        </span>
        <h1 className="mt-2 text-2xl font-bold tracking-tight sm:text-3xl">Chat</h1>
        <p className="mt-1.5 text-sm text-body">
          Pick a career track for a grounded roadmap, or ask anything. Replies are grounded in the catalog (RAG).
        </p>
      </div>

      {/* career-track picker */}
      {tracks.length > 0 && (
        <div className="mb-4 rounded-xl border border-[rgb(var(--border))] bg-[rgb(var(--surface))]/40 p-3">
          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-muted">
            <Compass width={14} height={14} /> Build a career roadmap
          </div>
          <div className="flex flex-wrap gap-2">
            {tracks.map((t) => (
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

              {t.roadmap && <RoadmapGraph roadmap={t.roadmap} />}

              {t.courses && t.courses.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5 border-t border-[rgb(var(--border))] pt-2.5">
                  {t.courses.slice(0, 6).map((c, j) => (
                    <span key={`${c.id}-${j}`} className="chip !py-1" title={c.skills ?? ""}>
                      {c.title.length > 46 ? c.title.slice(0, 46) + "…" : c.title}
                    </span>
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
    </div>
  );
}
