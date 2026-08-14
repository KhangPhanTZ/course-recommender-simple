"""Tests for course URL resolution (src/recsys/links.py)."""
from src.recsys.links import resolve_course_url


def test_direct_url_passthrough():
    url, direct = resolve_course_url("coursera", "Anything", "https://www.coursera.org/learn/ml")
    assert url == "https://www.coursera.org/learn/ml"
    assert direct is True


def test_search_link_per_source():
    for source, host in [
        ("coursera", "coursera.org/search?query="),
        ("udemy", "udemy.com/courses/search/?q="),
        ("edx", "edx.org/search?q="),
    ]:
        url, direct = resolve_course_url(source, "Deep Learning", None)
        assert host in url
        assert direct is False


def test_search_link_encodes_title():
    url, direct = resolve_course_url("coursera", "SQL & Python: A/B", None)
    assert direct is False
    assert " " not in url                 # spaces encoded
    assert "%26" in url or "+%26+" in url  # '&' encoded


def test_unknown_source_defaults_to_coursera_search():
    url, _ = resolve_course_url(None, "Data Analysis", None)
    assert "coursera.org/search" in url


def test_blank_url_and_title_returns_none():
    assert resolve_course_url("coursera", "", None) == (None, False)
    assert resolve_course_url("coursera", None, "") == (None, False)


def test_non_http_url_is_treated_as_missing():
    # a bare slug is not a usable link -> fall back to a real search link
    url, direct = resolve_course_url("udemy", "Course X", "not-a-url")
    assert direct is False
    assert url.startswith("https://www.udemy.com/")
