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


def test_zero_results_includes_live_data_tip():
    """When no results come back, the message should point the agent at
    web_fetch on a specific live-data endpoint (wttr.in for weather, etc)."""
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    # All backends return empty / parse-empty
    empty = _mock_response(data=[], text="<html></html>")
    fake_client.get.return_value = empty

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("some obscure error xyz123")
    assert "No results found" in out
    assert "wttr.in" in out  # live-data hint surfaces


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


# ── DDG-lite tests ──────────────────────────────────────────────────────────

# A trimmed sample of a real /lite/ response. Two results, both wrapped
# in the standard //duckduckgo.com/l/?uddg=ENCODED redirect. Validates
# parsing, unwrapping, and HTML-entity decoding.
_DDG_LITE_SAMPLE = """
<html><body><table>
<tr><td>
  <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.accuweather.com%2Fen%2Feg%2Fgiza%2F127047&amp;rut=abc" class='result-link'>Giza, Giza, Egypt Weather | AccuWeather</a>
</td></tr>
<tr><td class='result-snippet'>Current weather in Giza, Egypt. Today&#x27;s forecast and conditions.</td></tr>
<tr><td>
  <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fweather.com%2Fweather%2Ftoday%2Fl%2FGiza&amp;rut=def" class='result-link'>Weather.com &amp; Giza Forecast</a>
</td></tr>
<tr><td class='result-snippet'>Doppler radar &amp; hourly forecast for Giza.</td></tr>
</table></body></html>
"""


def test_ddg_lite_parses_real_html_structure():
    """The /lite/ parser must extract the two results from a sample
    HTML page, unwrap the //duckduckgo.com/l/?uddg=… redirect, and
    decode &#x27; / &amp; entities."""
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.return_value = _mock_response(text=_DDG_LITE_SAMPLE)

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("weather in giza", limit=2)

    # Both titles surface, properly decoded
    assert "AccuWeather" in out
    assert "Weather.com & Giza" in out  # &amp; → &
    # Unwrapped destination URLs (not the //duckduckgo.com/l/ wrapper)
    assert "accuweather.com/en/eg/giza/127047" in out
    assert "weather.com/weather/today/l/Giza" in out
    # Snippets — apostrophe entity decoded
    assert "Today's forecast" in out
    # Source tag
    assert "_ddg-lite_" in out


def test_ddg_lite_used_first_when_no_brave_key():
    """Without a Brave key, ddg-lite is the workhorse — its results
    must appear ahead of wikipedia/ddg-instant fallbacks."""

    def get_side_effect(url, **kwargs):
        if "lite.duckduckgo.com" in url:
            return _mock_response(text=_DDG_LITE_SAMPLE)
        if "wikipedia" in url:
            return _mock_response(data=[
                "weather", ["Weather"], ["Weather is..."],
                ["https://en.wikipedia.org/wiki/Weather"],
            ])
        if "api.duckduckgo.com" in url:
            return _mock_response(data={"Abstract": "Weather encyclopedic"})
        return _mock_response(data={})

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.side_effect = get_side_effect

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("weather giza", limit=5)

    # DDG-lite hits surface
    assert "AccuWeather" in out
    # Wikipedia would still be there as backup
    assert "_ddg-lite_" in out


def test_ddg_lite_failure_falls_through_to_wikipedia():
    """If DDG-lite errors (network / parse), wikipedia still runs."""
    def get_side_effect(url, **kwargs):
        if "lite.duckduckgo.com" in url:
            raise RuntimeError("connection refused")
        if "wikipedia" in url:
            return _mock_response(data=[
                "kubernetes", ["Kubernetes"],
                ["Container orchestration."],
                ["https://en.wikipedia.org/wiki/Kubernetes"],
            ])
        return _mock_response(data={})

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.side_effect = get_side_effect

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("kubernetes")
    assert "Kubernetes" in out


def test_ddg_lite_unwrap_handles_plain_https_and_protocol_relative():
    """Sanity-check the unwrap helper: protocol-relative URLs become
    https://… ; URLs without uddg= are returned as-is (with the
    protocol-relative prefix promoted to https)."""
    from tools import _ddg_lite_unwrap

    assert _ddg_lite_unwrap("//example.com/x") == "https://example.com/x"
    assert _ddg_lite_unwrap("https://example.com/x") == "https://example.com/x"
    # uddg-wrapped URL gets unwrapped
    assert _ddg_lite_unwrap(
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fb&rut=z"
    ) == "https://a.com/b"


def test_ddg_lite_dedup_with_wikipedia_results():
    """When DDG-lite and Wikipedia both return the same URL, dedupe
    should keep only one — the higher-priority backend (DDG-lite)."""
    same_url = "https://en.wikipedia.org/wiki/Python"
    ddg_html = (
        f"""<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FPython&rut=x" """
        f"""class='result-link'>Python from DDG</a>"""
        f"""<td class='result-snippet'>Via DDG-lite.</td>"""
    )

    def get_side_effect(url, **kwargs):
        if "lite.duckduckgo.com" in url:
            return _mock_response(text=ddg_html)
        if "wikipedia" in url:
            return _mock_response(data=[
                "python", ["Python"], ["Programming language."],
                [same_url],
            ])
        return _mock_response(data={})

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.get.side_effect = get_side_effect

    with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
        with patch("tools.httpx.Client", return_value=fake_client):
            out = web_search("python", limit=5)

    # The dedupe means we should only see this URL once
    assert out.count(same_url) == 1
    # DDG-lite version wins (came first)
    assert "Python from DDG" in out
