"""Career-track roadmaps for the learning advisor.

A curated registry of concrete career tracks grouped by domain (Data & AI,
Software & Cloud, Business & Product, Design). Each track is a **tiered skill
map** — Foundation → Core → Advanced → Specialization — plus *bridges* to
adjacent tracks ("where to go next"). The skills are the seed queries used to
pull **real catalog courses** into each tier, so a roadmap is grounded in what
the catalog actually offers rather than invented by an LLM.

``assemble_roadmap`` is deliberately pure: it takes a ``retrieve(query, k)``
callable and returns a plain dict, so it works identically with any retriever
and needs no LLM. ``covered_tracks`` uses the same retriever to surface only the
tracks the loaded catalog can actually support, so the picker reflects the
careers present in the data and adapts as the catalog grows.
"""
from __future__ import annotations

from collections.abc import Callable

TIER_ORDER = ["Foundation", "Core", "Advanced", "Specialization"]

# Display order for the grouped picker.
GROUP_ORDER = ["Data & AI", "Software & Cloud", "Business & Product", "Design"]

# retrieve(query, k) -> list of course dicts (CourseHit-shaped: id/title/…).
Retrieve = Callable[[str, int], list[dict]]


def _tiers(foundation, core, advanced, specialization) -> list[dict]:
    return [
        {"name": "Foundation", "skills": foundation},
        {"name": "Core", "skills": core},
        {"name": "Advanced", "skills": advanced},
        {"name": "Specialization", "skills": specialization},
    ]


# --- Curated tracks --------------------------------------------------------
# skills double as retrieval seeds, so keep them close to catalog vocabulary.
TRACKS: dict[str, dict] = {
    # ---------------------------------------------------------- Data & AI
    "data-analyst": {
        "group": "Data & AI",
        "label": "Data Analyst",
        "summary": "Turn raw data into decisions. Start with spreadsheets and SQL, "
                   "grow into statistics and BI dashboards.",
        "tiers": _tiers(
            ["excel", "spreadsheets", "sql basics"],
            ["sql", "data visualization", "tableau", "power bi"],
            ["statistics", "python", "data analysis"],
            ["business intelligence", "dashboards", "reporting"],
        ),
        "bridges": ["data-scientist", "business-analyst", "data-engineer"],
    },
    "data-scientist": {
        "group": "Data & AI",
        "label": "Data Scientist",
        "summary": "Find signal in data and model it. Python and statistics first, "
                   "then machine learning and predictive modelling.",
        "tiers": _tiers(
            ["python", "statistics", "probability & statistics"],
            ["data analysis", "machine learning", "exploratory data analysis"],
            ["regression", "machine learning algorithms", "statistical analysis"],
            ["deep learning", "predictive modeling", "data science"],
        ),
        "bridges": ["ml-engineer", "ai-engineer", "data-engineer"],
    },
    "data-engineer": {
        "group": "Data & AI",
        "label": "Data Engineer",
        "summary": "Build the pipelines that move and shape data at scale — SQL and "
                   "Python first, then distributed processing and orchestration.",
        "tiers": _tiers(
            ["sql", "python", "databases"],
            ["etl", "data pipelines", "data warehouse"],
            ["spark", "big data", "kafka", "streaming"],
            ["airflow", "dbt", "orchestration", "snowflake"],
        ),
        "bridges": ["mlops", "ml-engineer"],
    },
    "ml-engineer": {
        "group": "Data & AI",
        "label": "ML Engineer",
        "summary": "Ship models, not just notebooks. Python and ML foundations, then "
                   "deep learning and deploying models to production.",
        "tiers": _tiers(
            ["python", "statistics", "scikit-learn"],
            ["machine learning", "feature engineering", "model evaluation"],
            ["deep learning", "pytorch", "tensorflow", "neural networks"],
            ["nlp", "model deployment", "machine learning algorithms"],
        ),
        "bridges": ["ai-engineer", "mlops"],
    },
    "ai-engineer": {
        "group": "Data & AI",
        "label": "AI / GenAI Engineer",
        "summary": "Build with modern AI. Machine learning and deep learning first, "
                   "then NLP and generative / large language models.",
        "tiers": _tiers(
            ["python", "machine learning", "linear algebra"],
            ["deep learning", "neural networks", "tensorflow"],
            ["nlp", "computer vision", "transformers"],
            ["generative ai", "large language models", "artificial intelligence"],
        ),
        "bridges": ["ml-engineer", "mlops"],
    },
    # ---------------------------------------------------- Software & Cloud
    "web-developer": {
        "group": "Software & Cloud",
        "label": "Web Developer",
        "summary": "Build for the browser. HTML/CSS/JS foundations, then front-end "
                   "frameworks and full-stack applications.",
        "tiers": _tiers(
            ["html", "css", "computer programming"],
            ["javascript", "web development", "front-end"],
            ["react", "back-end", "web applications", "apis"],
            ["full stack", "databases", "software development"],
        ),
        "bridges": ["cloud-engineer", "ux-designer"],
    },
    "cloud-engineer": {
        "group": "Software & Cloud",
        "label": "Cloud Engineer",
        "summary": "Run software in the cloud. Cloud fundamentals and Linux first, "
                   "then AWS/Azure and infrastructure automation.",
        "tiers": _tiers(
            ["cloud computing", "linux", "networking"],
            ["aws", "azure", "virtualization"],
            ["infrastructure", "containers", "security"],
            ["devops", "terraform", "serverless"],
        ),
        "bridges": ["mlops", "cybersecurity"],
    },
    "mlops": {
        "group": "Software & Cloud",
        "label": "MLOps / DevOps",
        "summary": "Automate the path from code to running service — containers, CI/CD "
                   "and infrastructure as code, then model lifecycle in production.",
        "tiers": _tiers(
            ["linux", "git", "docker"],
            ["ci/cd", "kubernetes", "containers"],
            ["terraform", "aws", "cloud computing", "infrastructure"],
            ["mlflow", "model monitoring", "pipelines"],
        ),
        "bridges": ["cloud-engineer", "ml-engineer"],
    },
    "cybersecurity": {
        "group": "Software & Cloud",
        "label": "Cybersecurity",
        "summary": "Defend systems and data. Security and networking fundamentals, "
                   "then cryptography, threats, and security operations.",
        "tiers": _tiers(
            ["computer security", "networking", "security"],
            ["cyber security", "network security", "cryptography"],
            ["penetration testing", "ethical hacking", "risk management"],
            ["security operations", "incident response", "compliance"],
        ),
        "bridges": ["cloud-engineer"],
    },
    "qa-engineer": {
        "group": "Software & Cloud",
        "label": "QA / Test Automation",
        "summary": "Guard quality with automation. Testing fundamentals first, then "
                   "UI and API automation, then CI-integrated end-to-end suites.",
        "tiers": _tiers(
            ["software testing", "test planning", "quality assurance"],
            ["test automation", "selenium", "software development"],
            ["api testing", "automation", "end-to-end testing"],
            ["ci/cd", "performance testing", "test frameworks"],
        ),
        "bridges": ["mlops"],
    },
    # ---------------------------------------------------- Business & Product
    "business-analyst": {
        "group": "Business & Product",
        "label": "Business Analyst",
        "summary": "Bridge business and data. Requirements and process work first, "
                   "then analytics that inform decisions.",
        "tiers": _tiers(
            ["business analysis", "requirements", "excel"],
            ["data analysis", "process improvement", "stakeholder management"],
            ["analytics", "sql", "data visualization"],
            ["business intelligence", "decision making", "strategy"],
        ),
        "bridges": ["data-analyst", "product-manager"],
    },
    "product-manager": {
        "group": "Business & Product",
        "label": "Product / Project Manager",
        "summary": "Lead what gets built and why. Agile delivery and stakeholder work "
                   "first, then product strategy and data-informed decisions.",
        "tiers": _tiers(
            ["agile", "scrum", "project management"],
            ["product management", "planning", "stakeholder management"],
            ["product strategy", "strategy", "decision making"],
            ["analytics", "leadership", "market analysis"],
        ),
        "bridges": ["business-analyst", "management-leadership"],
    },
    "digital-marketing": {
        "group": "Business & Product",
        "label": "Digital Marketing",
        "summary": "Grow audiences and demand. Marketing and communication first, "
                   "then digital channels and marketing analytics.",
        "tiers": _tiers(
            ["marketing", "communication", "branding"],
            ["digital marketing", "social media", "market analysis"],
            ["advertising", "content", "sales"],
            ["marketing analytics", "strategy", "customer"],
        ),
        "bridges": ["business-analyst", "entrepreneurship"],
    },
    "financial-analyst": {
        "group": "Business & Product",
        "label": "Financial Analyst",
        "summary": "Read the numbers behind a business. Accounting and finance first, "
                   "then financial analysis, modelling, and markets.",
        "tiers": _tiers(
            ["finance", "financial accounting", "accounting"],
            ["financial analysis", "corporate finance", "valuation"],
            ["financial modeling", "investment", "risk management"],
            ["financial markets", "portfolio management", "forecasting"],
        ),
        "bridges": ["business-analyst", "data-analyst"],
    },
    "management-leadership": {
        "group": "Business & Product",
        "label": "Management & Leadership",
        "summary": "Lead people and organisations. Communication and teamwork first, "
                   "then strategy, operations, and executive leadership.",
        "tiers": _tiers(
            ["leadership", "communication", "collaboration"],
            ["leadership and management", "organizational development", "strategy"],
            ["strategy and operations", "decision making", "change management"],
            ["people management", "negotiation", "business"],
        ),
        "bridges": ["product-manager", "entrepreneurship"],
    },
    "entrepreneurship": {
        "group": "Business & Product",
        "label": "Entrepreneurship",
        "summary": "Turn an idea into a venture. Business fundamentals and innovation "
                   "first, then building, marketing, and funding a startup.",
        "tiers": _tiers(
            ["entrepreneurship", "business", "innovation"],
            ["business model", "product management", "marketing"],
            ["finance", "strategy", "growth"],
            ["fundraising", "leadership", "market analysis"],
        ),
        "bridges": ["product-manager", "digital-marketing"],
    },
    # ------------------------------------------------------------- Design
    "ux-designer": {
        "group": "Design",
        "label": "UX / UI Designer",
        "summary": "Design products people love to use. Design thinking and UX "
                   "fundamentals first, then research, UI, and design systems.",
        "tiers": _tiers(
            ["design", "user experience", "design thinking"],
            ["ui", "user interface", "prototyping"],
            ["user research", "usability", "interaction design"],
            ["product design", "design systems", "graphic design"],
        ),
        "bridges": ["web-developer", "product-manager"],
    },
}


def _track_info(tid: str, t: dict) -> dict:
    return {"id": tid, "label": t["label"], "summary": t["summary"], "group": t["group"]}


def list_tracks() -> list[dict]:
    """Full track catalogue (id, label, summary, group), in group order."""
    ordered = sorted(TRACKS.items(), key=lambda kv: GROUP_ORDER.index(kv[1]["group"]))
    return [_track_info(tid, t) for tid, t in ordered]


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


def covered_tracks(retrieve: Retrieve, min_tiers: int = 2) -> list[dict]:
    """Return only the tracks the loaded catalog can support.

    A track qualifies when its grounded roadmap fills at least ``min_tiers`` of
    its four tiers with real courses — i.e. the data actually covers that career
    (rather than a raw count, which wouldn't scale across catalog sizes).
    Preserves group order. Falls back to the full list if the retriever yields
    nothing at all (so an unbuilt/empty catalog still shows the menu).
    """
    survivors: list[dict] = []
    any_hit = False
    for info in list_tracks():
        roadmap = assemble_roadmap(info["id"], retrieve)
        filled = sum(1 for n in roadmap["nodes"] if n["courses"]) if roadmap else 0
        any_hit = any_hit or filled > 0
        if filled >= min_tiers:
            survivors.append(info)
    return survivors if any_hit else list_tracks()
