"""Task adapters: map a generic backend + image to task-specific Results.

The :func:`get_adapter` factory selects the adapter for a detected task.
``detect`` (DETR / SSD / boxes+labels / YOLOv2), ``classify``, ``segment``, and
``pose`` have typed decoders; every other image task falls back to
:class:`GenericAdapter`, which returns the raw output tensors.
"""

from __future__ import annotations

from typing import Any

from .base import BaseAdapter
from .classify import ClassifyAdapter
from .detect import DetectAdapter
from .generic import GenericAdapter
from .pose import PoseAdapter
from .segment import SegmentAdapter

_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "detect": DetectAdapter,
    "classify": ClassifyAdapter,
    "segment": SegmentAdapter,
    "pose": PoseAdapter,
}

#: Tasks with a typed decoder (the rest use the generic raw-output adapter).
VISION_TASKS = frozenset(_ADAPTERS)


def get_adapter(task: str, **kwargs: Any) -> BaseAdapter:
    """Instantiate the adapter for ``task``.

    Vision tasks (detect/classify/segment/pose) get their typed decoder; any
    other task falls back to :class:`GenericAdapter`, which runs the model on the
    image and returns raw output tensors.
    """
    cls = _ADAPTERS.get(task)
    if cls is None:
        return GenericAdapter(task=task, **kwargs)
    return cls(**kwargs)


__all__ = [
    "BaseAdapter",
    "ClassifyAdapter",
    "DetectAdapter",
    "GenericAdapter",
    "PoseAdapter",
    "SegmentAdapter",
    "VISION_TASKS",
    "get_adapter",
]
