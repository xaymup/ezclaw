"""Native OS desktop notifications for attention-worthy events.

Best-effort and silent on failure — a notification system that ever
raises an exception would be worse than no notifications at all.

Backends:
  - Linux  → notify-send (libnotify, ships on every mainstream desktop)
  - macOS  → osascript "display notification"
  - Else   → terminal bell (BEL / \\a)

Set EZCLAW_NOTIFY=0 to disable entirely.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_APP_NAME = "ezclaw"
_ICON = "dialog-information"

URGENCY_LOW = "low"
URGENCY_NORMAL = "normal"
URGENCY_CRITICAL = "critical"


def _enabled() -> bool:
    return os.getenv("EZCLAW_NOTIFY", "1") != "0"


def notify(title: str, message: str = "", urgency: str = URGENCY_NORMAL) -> None:
    """Send a desktop notification. Never raises.

    Args:
        title: short headline (shown bold by most desktops).
        message: longer body. Empty is fine for one-line alerts.
        urgency: one of URGENCY_LOW / URGENCY_NORMAL / URGENCY_CRITICAL.
            Critical notifications stay on screen until dismissed on
            most Linux desktops.
    """
    if not _enabled():
        return

    try:
        if sys.platform.startswith("linux"):
            _notify_linux(title, message, urgency)
        elif sys.platform == "darwin":
            _notify_macos(title, message)
        else:
            _bell()
    except Exception:
        # Absolutely never let a notification crash anything upstream.
        # Best-effort means best-effort.
        try:
            _bell()
        except Exception:
            pass


def _notify_linux(title: str, message: str, urgency: str) -> None:
    if not shutil.which("notify-send"):
        _bell()
        return
    args = [
        "notify-send",
        "--app-name", _APP_NAME,
        "--urgency", urgency if urgency in (URGENCY_LOW, URGENCY_NORMAL, URGENCY_CRITICAL) else URGENCY_NORMAL,
        "--icon", _ICON,
        title,
    ]
    if message:
        args.append(message)
    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach so we don't wait on it
    )


def _notify_macos(title: str, message: str) -> None:
    if not shutil.which("osascript"):
        _bell()
        return
    # osascript's `display notification` builds the script inline; escape
    # the strings so quotes/backslashes don't break out.
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{safe_msg}" with title "{safe_title}"'
    subprocess.Popen(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _bell() -> None:
    """Terminal-bell fallback. Emits BEL on stderr so it doesn't disturb
    the TUI's stdout-rendered buffer."""
    try:
        sys.stderr.write("\a")
        sys.stderr.flush()
    except Exception:
        pass
