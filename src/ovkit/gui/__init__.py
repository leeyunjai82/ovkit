"""A small desktop window for ovkit: pick a capability, point it at something.

    ovkit gui

The logic lives in :mod:`ovkit.gui.controller` (no Tk, so it is testable); the
window in :mod:`ovkit.gui.app`.
"""

from __future__ import annotations

from .controller import Choice, Controller, View, choices

__all__ = ["Choice", "Controller", "View", "choices", "main"]


def main(device: str = "AUTO", camera: int = 0) -> int:
    """Open the window (imports Tk only when actually asked to)."""
    from .app import main as _main

    return _main(device=device, camera=camera)
