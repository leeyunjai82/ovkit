"""Task auto-detection.

Priority (spec §4):

1. The manifest ``task`` field, if the model came from the registry.
2. The IR ``rt_info`` metadata (``model_info/task`` or ``model_type``).
3. A heuristic over the output tensor signatures.
4. Otherwise raise :class:`TaskDetectionError` asking for an explicit ``task=``.
"""

from __future__ import annotations

from .backend import Backend
from .errors import TaskDetectionError

#: Tasks ovkit knows how to attach an adapter for.
KNOWN_TASKS = ("detect", "classify", "segment", "pose", "face")


def _from_rt_info(backend: Backend) -> str | None:
    for keys in (("model_info", "task"), ("model_type",), ("task",)):
        value = backend.rt_info(*keys)
        if value:
            v = value.strip().lower()
            for t in KNOWN_TASKS:
                if t in v:
                    return t
    return None


def _from_signature(backend: Backend) -> str | None:
    sigs = backend.output_signatures()
    shapes = [shape for _, shape in sigs]

    # DETR-style detection: two outputs, one ending in 4 (boxes) and one 3-D
    # logits tensor [N, queries, classes].
    if len(shapes) == 2:
        ndims = [len(s) for s in shapes]
        if set(ndims) == {3} or sorted(ndims) == [3, 3]:
            has_box = any(s[-1] == 4 for s in shapes)
            if has_box:
                return "detect"

    # SSD / DetectionOutput: a single tensor ending in 7
    # ([image_id, label, conf, x_min, y_min, x_max, y_max]).
    if len(shapes) == 1 and len(shapes[0]) >= 2 and shapes[0][-1] == 7:
        return "detect"

    # boxes [N, 5] + labels [N] (OMZ -0200/ATSS-style detectors).
    if len(shapes) == 2 and any(len(s) >= 2 and s[-1] == 5 for s in shapes):
        return "detect"

    # Single 2-D output [N, C] -> classification probabilities.
    if len(shapes) == 1 and len(shapes[0]) == 2:
        return "classify"

    # Single 4-D output: [N, C, 1, 1] -> classify; [N, C, H, W] -> segmentation.
    if len(shapes) == 1 and len(shapes[0]) == 4:
        s = shapes[0]
        if s[-1] == 1 and s[-2] == 1:
            return "classify"
        return "segment"

    # Single 3-D output [N, C, anchors] or [N, anchors, C] -> dense detector.
    if len(shapes) == 1 and len(shapes[0]) == 3:
        return "detect"

    return None


def detect_task(
    backend: Backend,
    manifest_task: str | None = None,
    override: str | None = None,
) -> str:
    """Determine the task for a loaded model following the documented priority.

    Parameters
    ----------
    backend:
        A compiled :class:`Backend` to introspect.
    manifest_task:
        ``task`` from the registry entry, if any (highest non-override priority).
    override:
        An explicit ``task=`` from the caller, which short-circuits everything.
    """
    if override:
        if override not in KNOWN_TASKS:
            raise TaskDetectionError(f"Unknown task '{override}'. Expected one of {KNOWN_TASKS}.")
        return override

    if manifest_task:
        return manifest_task

    rt = _from_rt_info(backend)
    if rt:
        return rt

    sig = _from_signature(backend)
    if sig:
        return sig

    raise TaskDetectionError(
        "Could not determine the model's task automatically. "
        "Pass an explicit task, e.g. Model(..., task='detect'). "
        f"Output signatures were: {backend.output_signatures()}"
    )
