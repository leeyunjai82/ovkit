"""Task adapters: map a generic backend + image to task-specific Results.

The :func:`get_adapter` factory selects the adapter for a detected task.
``detect`` and ``classify`` are implemented; ``segment`` / ``pose`` are stubs.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .classify import ClassifyAdapter
from .detect import DetectAdapter
from .pose import PoseAdapter
from .segment import SegmentAdapter

_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "detect": DetectAdapter,
    "classify": ClassifyAdapter,
    "segment": SegmentAdapter,
    "pose": PoseAdapter,
}


def get_adapter(task: str, **kwargs: Any) -> BaseAdapter:
    """Instantiate the adapter registered for ``task``.

    Raises ``KeyError`` (wrapped) for unknown tasks. ``face`` is handled by the
    dedicated :mod:`ovkit.face` module, not here.
    """
    try:
        cls = _ADAPTERS[task]
    except KeyError as exc:
        raise KeyError(
            f"No recognition adapter for task '{task}'. Known: {sorted(_ADAPTERS)}."
        ) from exc
    return cls(**kwargs)


__all__ = [
    "BaseAdapter",
    "ClassifyAdapter",
    "DetectAdapter",
    "PoseAdapter",
    "SegmentAdapter",
    "get_adapter",
]
