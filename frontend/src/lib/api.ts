// Typed client for the Course Recommender FastAPI backend.

const BASE = (import.meta.env.VITE_API_URL ?? "/api").replace(/\/$/, "");

export interface CourseHit {
  id: number | string;
  title: string;
  score: number;
  category?: string | null;
  level?: string | null;
  rating?: number | string | null;
  url?: string | null;
  skills?: string | null;
  provider?: string | null;
  source?: string | null;
}

export interface RecommendResponse {
  query: string;
  resolved_query: string;
  filters: Record<string, string>;
  results: CourseHit[];
  explanation?: string | null;
  llm_enabled: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  courses: CourseHit[];
  llm_enabled: boolean;
}

export interface Health {
  status: string;
  backend?: string | null;
  n_courses?: number | null;
  llm_provider?: string | null;
  llm_enabled: boolean;
  version: string;
}

export interface CourseDetail {
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
  [key: string]: unknown;
}

export interface Metrics {
  available: boolean;
  k?: number;
  backend?: string;
  n_courses?: number;
  generated_at?: string;
  n_eval?: number;
  precision_at_k?: number;
  recall_at_k?: number;
  hit_rate?: number;
  mrr?: number;
  ndcg_at_k?: number;
  latency_ms?: { p50: number; p95: number; mean: number };
  silhouette?: number;
  sources?: Record<string, number>;
}

export interface TrackInfo {
  id: string;
  label: string;
  summary: string;
}

export interface RoadmapNode {
  id: string;
  tier: string;
  skills: string[];
  courses: CourseHit[];
}

export interface RoadmapEdge {
  source: string;
  target: string;
  kind: string;
}

export interface RoadmapBridge {
  track: string;
  label: string;
  note: string;
}

export interface Roadmap {
  track: string;
  label: string;
  summary: string;
  intro?: string | null;
  nodes: RoadmapNode[];
  edges: RoadmapEdge[];
  bridges: RoadmapBridge[];
  llm_enabled: boolean;
}

export interface RecommendParams {
  query: string;
  top_k?: number;
  level?: string | null;
  category?: string | null;
  explain?: boolean;
  understand?: boolean;
  rerank?: boolean | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore non-JSON errors */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export const api = {
  health: () => request<Health>("/health"),
  recommend: (params: RecommendParams) =>
    request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify({ top_k: 10, explain: true, understand: true, ...params }),
    }),
  similar: (course_id: number | string, top_k = 8) =>
    request<CourseHit[]>("/similar", {
      method: "POST",
      body: JSON.stringify({ course_id, top_k }),
    }),
  course: (id: number | string) => request<CourseDetail>(`/courses/${id}`),
  chat: (messages: ChatMessage[], top_k = 6) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ messages, top_k }),
    }),
  metrics: () => request<Metrics>("/metrics"),
  tracks: () => request<TrackInfo[]>("/roadmap/tracks"),
  roadmap: (track: string, per_tier = 2) =>
    request<Roadmap>("/roadmap", {
      method: "POST",
      body: JSON.stringify({ track, per_tier }),
    }),
};

export { BASE as API_BASE };
