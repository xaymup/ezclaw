"""Tests for the notifications module — must be best-effort and silent."""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notifications import (
    notify,
    URGENCY_LOW,
    URGENCY_NORMAL,
    URGENCY_CRITICAL,
    _enabled,
)


def test_notify_never_raises_even_when_disabled(monkeypatch):
    monkeypatch.setenv("EZCLAW_NOTIFY", "0")
    notify("title", "body")  # must not raise


def test_notify_never_raises_when_backend_missing(monkeypatch):
    """If notify-send / osascript aren't installed (or `which` returns None),
    notify must fall back silently to the bell — never raise."""
    monkeypatch.setenv("EZCLAW_NOTIFY", "1")
    with patch("notifications.shutil.which", return_value=None):
        notify("title", "body")


def test_notify_never_raises_when_subprocess_fails(monkeypatch):
    """Even if Popen blows up (permissions, ulimit, anything), notify
    swallows the exception."""
    monkeypatch.setenv("EZCLAW_NOTIFY", "1")
    with patch("notifications.shutil.which", return_value="/usr/bin/notify-send"):
        with patch("notifications.subprocess.Popen", side_effect=OSError("boom")):
            notify("title", "body")  # must not raise


def test_enabled_default_on(monkeypatch):
    monkeypatch.delenv("EZCLAW_NOTIFY", raising=False)
    assert _enabled() is True


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("EZCLAW_NOTIFY", "0")
    assert _enabled() is False


def test_notify_calls_notify_send_on_linux(monkeypatch):
    """On Linux with notify-send available, notify should invoke it
    with the title, message, urgency, and app name."""
    monkeypatch.setenv("EZCLAW_NOTIFY", "1")
    monkeypatch.setattr("notifications.sys.platform", "linux")
    fake_popen = MagicMock()
    with patch("notifications.shutil.which", return_value="/usr/bin/notify-send"):
        with patch("notifications.subprocess.Popen", fake_popen):
            notify("Hello", "World", urgency=URGENCY_CRITICAL)
    assert fake_popen.called
    args = fake_popen.call_args[0][0]
    assert args[0] == "notify-send"
    assert "--urgency" in args
    assert "critical" in args
    assert "Hello" in args
    assert "World" in args


def test_notify_omits_message_when_empty(monkeypatch):
    """One-line alerts pass only the title; notify-send treats the second
    positional arg as body, so passing an empty string would render an
    empty body line."""
    monkeypatch.setenv("EZCLAW_NOTIFY", "1")
    monkeypatch.setattr("notifications.sys.platform", "linux")
    fake_popen = MagicMock()
    with patch("notifications.shutil.which", return_value="/usr/bin/notify-send"):
        with patch("notifications.subprocess.Popen", fake_popen):
            notify("Title only", "")
    args = fake_popen.call_args[0][0]
    assert args[-1] == "Title only"  # no empty-string body trailing
