"""Multi-object tracking = detect + associate. See :mod:`ovkit.pipelines.tracking`."""

from __future__ import annotations

from ..pipelines.tracking import Tracker, iou_matrix

__all__ = ["Tracker", "iou_matrix"]
