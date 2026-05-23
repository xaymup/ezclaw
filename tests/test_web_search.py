"""Tests for the web_search tool — multi-backend search with fallback."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import web_search


def test_empty_query_returns_error():
    out = web_search("")
    assert "Error" in out and "empty" in out.lower()
    out2 = web_search("   ")
    assert "Error" in out2


def _mock_response(data=None, text=None):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    if data is not None:
        r.json.return_value = data
    if text is not None:
        r.text = text
    return r


def test_wikipedia_backend_parses_opensearch():
    """Wikipedia opensearch returns [query, [titles], [snippets], [urls]].
    The parser should extract title/snippet/url for each entry."""
    fake_wiki = [
        "asyncio",
        ["Asyncio", "Coroutine"],
        ["Asyncio is a Python library…", "A coroutine is…"],
        ["https://en.wikipedia.org/wiki/Asyncio", "https://en.wikipedia.org/wiki/Coroutine"],
    ]
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.return_value = _mock_response(data=fake_wiki)

    # Ensure no brave key; DDG instant returns empty
    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("asyncio", limit=5)
    assert "Asyncio" in out
    assert "Coroutine" in out
    assert "en.wikipedia.org/wiki/Asyncio" in out
    assert "_wikipedia_" in out  # source tag


def test_brave_backend_used_when_key_set():
    """When BRAVE_SEARCH_API_KEY is set, the brave backend should run
    first and its results should appear."""
    fake_brave = {
        "web": {
            "results": [
                {"title": "Stack Overflow: ImportError fix",
                 "url": "https://stackoverflow.com/q/12345",
                 "description": "The error means the module path is wrong."},
                {"title": "GitHub issue: same error",
                 "url": "https://github.com/foo/bar/issues/42",
                 "description": "Reported here as well."},
            ],
        },
    }
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.return_value = _mock_response(data=fake_brave)

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "abc123"}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("ImportError no module named foo", limit=5)
    assert "Stack Overflow" in out
    assert "stackoverflow.com/q/12345" in out
    assert "_brave_" in out


def test_falls_back_through_backends_on_failure():
    """If brave raises (network down / 4xx), the wikipedia backend
    should still be tried."""
    fake_wiki = [
        "kubernetes",
        ["Kubernetes"],
        ["Container orchestration system."],
        ["https://en.wikipedia.org/wiki/Kubernetes"],
    ]

    def get_side_effect(url, **kwargs):
        if "brave" in url:
            raise RuntimeError("brave down")
        if "wikipedia" in url:
            return _mock_response(data=fake_wiki)
        return _mock_response(data={})

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.side_effect = get_side_effect

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "abc"}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("kubernetes")
    assert "Kubernetes" in out  # wikipedia rescued us


def test_zero_results_includes_tip_when_no_brave_key():
    """When no results come back AND there's no brave key, the message
    should tell the user how to set one up."""
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.return_value = _mock_response(data=[])  # both wiki and ddg empty

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("some obscure error xyz123")
    assert "No results found" in out
    assert "BRAVE_SEARCH_API_KEY" in out


def test_zero_results_no_tip_when_brave_key_set():
    """If the user already has a key and got nothing, don't nag them."""
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.return_value = _mock_response(data={"web": {"results": []}})

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "abc"}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("very specific phrase that nobody indexed")
    assert "No results found" in out
    assert "BRAVE_SEARCH_API_KEY" not in out  # no nag


def test_ddg_instant_answer_picked_up():
    """Test the DDG instant answer parser directly via the abstract field."""
    fake_ddg = {
        "Abstract": "Python is a programming language.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Heading": "Python (programming language)",
        "RelatedTopics": [],
    }

    def get_side_effect(url, **kwargs):
        if "wikipedia" in url:
            return _mock_response(data=[])
        if "duckduckgo" in url:
            return _mock_response(data=fake_ddg)
        return _mock_response(data={})

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.side_effect = get_side_effect

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("python language")
    assert "Python" in out
    assert "_ddg-instant_" in out
