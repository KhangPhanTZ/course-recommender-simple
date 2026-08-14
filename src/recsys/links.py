"""Resolve a real, working URL for a course.

Datasets are inconsistent about course links: some ship a real per-course URL,
many (e.g. the deployed ``data/Coursera.csv``) ship none at all. Rather than
show a dead or missing link, fall back to a **provider search deep-link** built
from the course title — a genuine URL that lands on the provider's site for that
course. Callers get a flag telling them whether the link is the exact course
page (``is_direct=True``) or a search link, so the UI can label it honestly.
"""
from __future__ import annotations

from urllib.parse import quote_plus

# Provider search endpoints — real, stable URLs that resolve to the course.
_SEARCH = {
    "coursera": "https://www.coursera.org/search?query={q}",
    "udemy": "https://www.udemy.com/courses/search/?q={q}",
    "edx": "https://www.edx.org/search?q={q}",
}
_DEFAULT = _SEARCH["coursera"]


def resolve_course_url(source: str | None, title: str | None, url: str | None) -> tuple[str | None, bool]:
    """Return ``(url, is_direct)``.

    - A non-empty ``http(s)`` ``url`` is used as-is (``is_direct=True``).
    - Otherwise a provider search deep-link is built from ``title``.
    - Returns ``(None, False)`` only when there's neither a url nor a title.
    """
    if url and str(url).strip().lower().startswith(("http://", "https://")):
        return str(url).strip(), True

    q = quote_plus(str(title).strip()) if title and str(title).strip() else ""
    if not q:
        return None, False

    template = _SEARCH.get((source or "").strip().lower(), _DEFAULT)
    return template.format(q=q), False
