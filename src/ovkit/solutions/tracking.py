"""Multi-object tracking = detector + a tracker, assembled over recognize/detect.

INTERFACE STUB for v0. Planned: feed detections into a lightweight tracker
(e.g. ByteTrack-style association) and emit track ids per box.
"""

from __future__ import annotations

from typing import Any


class Tracker:
    """Detection + association tracker (not yet implemented)."""

    def __init__(self, detector: Any = None, device: str = "AUTO", **kwargs: Any) -> None:
        self.detector = detector
        self.device = device

    def update(self, image: Any) -> Any:
        raise NotImplementedError(
            "Tracking is not implemented in v0. Planned: detector + association "
            "(track ids per box) over ovkit.recognize.detect. TODO."
        )
