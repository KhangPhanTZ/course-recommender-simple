"""Career-track roadmaps for the learning advisor.

A small, curated registry of concrete IT career tracks (Data Analyst, ML
Engineer, …). Each track is a **tiered skill map** — Foundation → Core →
Advanced → Specialization — plus *bridges* to adjacent tracks ("where to go
next"). The skills are the seed queries used to pull **real catalog courses**
into each tier, so a roadmap is grounded in what the catalog actually offers
rather than invented by an LLM.

``assemble_roadmap`` is deliberately pure: it takes a ``retrieve(query, k)``
callable and returns a plain dict, so it works identically with any retriever
and needs no LLM. The LLM (when configured) only writes a short intro on top.
"""
from __future__ import annotations

from collections.abc import Callable

TIER_ORDER = ["Foundation", "Core", "Advanced", "Specialization"]

# retrieve(query, k) -> list of course dicts (CourseHit-shaped: id/title/…).
Retrieve = Callable[[str, int], list[dict]]


# --- Curated tracks --------------------------------------------------------
# skills double as retrieval seeds, so keep them close to catalog vocabulary.
TRACKS: dict[str, dict] = {
    "data-analyst": {
        "label": "Data Analyst",
        "summary": "Turn raw data into decisions. Start with spreadsheets and SQL, "
                   "grow into statistics and BI dashboards.",
        "tiers": [
            {"name": "Foundation", "skills": ["excel", "spreadsheets", "sql basics"]},
            {"name": "Core", "skills": ["sql", "data visualization", "tableau", "power bi"]},
            {"name": "Advanced", "skills": ["statistics", "python", "pandas", "analytics"]},
            {"name": "Specialization", "skills": ["business intelligence", "dashboards", "reporting"]},
        ],
        "bridges": ["data-engineer", "ml-engineer"],
    },
    "data-engineer": {
        "label": "Data Engineer",
        "summary": "Build the pipelines that move and shape data at scale — SQL and "
                   "Python first, then distributed processing and orchestration.",
        "tiers": [
            {"name": "Foundation", "skills": ["sql", "python", "databases"]},
            {"name": "Core", "skills": ["etl", "data pipelines", "data warehouse"]},
            {"name": "Advanced", "skills": ["spark", "big data", "kafka", "streaming"]},
            {"name": "Specialization", "skills": ["airflow", "dbt", "orchestration", "snowflake"]},
        ],
        "bridges": ["mlops", "ml-engineer"],
    },
    "ml-engineer": {
        "label": "ML Engineer",
        "summary": "Ship models, not just notebooks. Python and ML foundations, then "
                   "deep learning and deploying models to production.",
        "tiers": [
            {"name": "Foundation", "skills": ["python", "statistics", "scikit-learn"]},
            {"name": "Core", "skills": ["machine learning", "feature engineering", "model evaluation"]},
            {"name": "Advanced", "skills": ["deep learning", "pytorch", "tensorflow", "neural networks"]},
            {"name": "Specialization", "skills": ["nlp", "transformers", "model deployment", "llm"]},
        ],
        "bridges": ["mlops"],
    },
    "mlops": {
        "label": "MLOps / DevOps",
        "summary": "Automate the path from code to running service — containers, CI/CD "
                   "and infrastructure as code, then model lifecycle in production.",
        "tiers": [
            {"name": "Foundation", "skills": ["linux", "git", "python", "docker"]},
            {"name": "Core", "skills": ["ci/cd", "kubernetes", "containers"]},
            {"name": "Advanced", "skills": ["terraform", "aws", "cloud", "infrastructure"]},
            {"name": "Specialization", "skills": ["mlflow", "model monitoring", "mlops", "pipelines"]},
        ],
        "bridges": ["data-engineer", "ml-engineer"],
    },
    "product-manager": {
        "label": "Product / Project Manager",
        "summary": "Lead what gets built and why. Agile delivery and stakeholder work "
                   "first, then product strategy and data-informed decisions.",
        "tiers": [
            {"name": "Foundation", "skills": ["agile", "scrum", "project management"]},
            {"name": "Core", "skills": ["product management", "roadmap", "stakeholder management"]},
            {"name": "Advanced", "skills": ["product strategy", "prioritization", "metrics"]},
            {"name": "Specialization", "skills": ["analytics", "data-driven", "leadership"]},
        ],
        "bridges": ["data-analyst"],
    },
    "qa-engineer": {
        "label": "QA / Test Automation",
        "summary": "Guard quality with automation. Testing fundamentals first, then "
                   "UI and API automation, then CI-integrated end-to-end suites.",
        "tiers": [
            {"name": "Foundation", "skills": ["software testing", "test planning", "qa"]},
            {"name": "Core", "skills": ["test automation", "selenium", "pytest"]},
            {"name": "Advanced", "skills": ["api testing", "cypress", "playwright", "end-to-end testing"]},
            {"name": "Specialization", "skills": ["ci/cd", "performance testing", "test frameworks"]},
        ],
        "bridges": ["mlops"],
    },
}


def list_tracks() -> list[dict]:
    """Public track catalogue for the picker."""
    return [{"id": tid, "label": t["label"], "summary": t["summary"]} for tid, t in TRACKS.items()]


def get_track(track_id: str) -> dict | None:
    return TRACKS.get(track_id)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def assemble_roadmap(track_id: str, retrieve: Retrieve, per_tier: int = 2) -> dict | None:
    """Build a grounded roadmap: pull real courses into each tier + track bridges.

    Returns ``None`` for an unknown ``track_id``. Courses are de-duplicated
    across tiers so each course appears once, at the earliest tier that matches.
    """
    track = TRACKS.get(track_id)
    if track is None:
        return None

    seen_ids: set = set()
    seen_titles: set = set()
    nodes: list[dict] = []
    for tier in track["tiers"]:
        query = ", ".join(tier["skills"])
        picked: list[dict] = []
        for course in retrieve(query, per_tier + 8):
            cid = course.get("id")
            title_key = str(course.get("title", "")).strip().lower()
            if cid in seen_ids or (title_key and title_key in seen_titles):
                continue
            seen_ids.add(cid)
            if title_key:
                seen_titles.add(title_key)
            picked.append(course)
            if len(picked) >= per_tier:
                break
        nodes.append({
            "id": _slug(tier["name"]),
            "tier": tier["name"],
            "skills": tier["skills"],
            "courses": picked,
        })

    edges = [
        {"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "kind": "progress"}
        for i in range(len(nodes) - 1)
    ]
    bridges = [
        {"track": bid, "label": TRACKS[bid]["label"],
         "note": f"Branch into {TRACKS[bid]['label']} once you're solid here."}
        for bid in track.get("bridges", []) if bid in TRACKS
    ]

    return {
        "track": track_id,
        "label": track["label"],
        "summary": track["summary"],
        "nodes": nodes,
        "edges": edges,
        "bridges": bridges,
    }
