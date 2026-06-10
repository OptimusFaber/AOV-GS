"""Helpers for headless / SSH runs without X11."""

from __future__ import annotations

import os


def has_gui_display() -> bool:
    """True when an X11 or Wayland display is available."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def configure_headless_env() -> None:
    """Set safe defaults for matplotlib / Qt when no display is present."""
    if has_gui_display():
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
