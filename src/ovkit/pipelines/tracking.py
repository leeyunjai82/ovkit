"""Object tracking: give every detected object a stable id across frames.

    tracker = vis("track")
    for frame in video:
        r = tracker(frame)[0]
        print(r.summary())    # 2x person (#1, #4)

Detection alone answers "what is in this frame"; tracking answers "is that the
same car as a second ago", which is what counting, dwell time and trajectories
need. Association is IoU-based (SORT without the Kalman filter): simple,
dependency-free, and good enough while objects move less than their own size
between frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Results
from .base import Pipeline, detections


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` / ``(M, 4)`` sets of xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


class Tracker(Pipeline):
    """A detector plus IoU association, emitting a track id per box.

    Parameters
    ----------
    detector:
        Registered detector name (any ovkit detector works).
    iou:
        Minimum overlap for a detection to continue an existing track.
    max_age:
        How many frames a track survives without a match before it is dropped,
        so a brief occlusion does not restart the id.
    """

    name = "track"
    description = "Detect objects and keep a stable id for each across frames."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "detect",
        iou: float = 0.3,
        max_age: int = 30,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.iou = float(iou)
        self.max_age = int(max_age)
        self.reset()

    def reset(self) -> None:
        """Forget every track (call between videos)."""
        self._tracks: list[dict[str, Any]] = []
        self._next_id = 1

    def run(self, image: np.ndarray, *, conf: float = 0.25, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        ids = self.update(boxes.data)
        result = Results(image, task=self.name, names=found.names, boxes=boxes)
        result.track_ids = ids
        return result

    def update(self, detections_xyxycc: np.ndarray) -> list[int]:
        """Assign a track id to each detection row and age out stale tracks.

        Split out from :meth:`run` so the association can be used (and tested)
        with detections from anywhere.
        """
        dets = np.asarray(detections_xyxycc, np.float32).reshape(-1, 6)
        ids = [0] * len(dets)
        alive = [t for t in self._tracks if t["age"] <= self.max_age]

        scores = iou_matrix(dets[:, :4], np.array([t["box"] for t in alive], np.float32))
        # Greedy highest-overlap-first matching: one detection per track, and a
        # detection only continues a track of the same class.
        while scores.size and scores.max() >= self.iou:
            d, t = np.unravel_index(int(scores.argmax()), scores.shape)
            if int(dets[d, 5]) == alive[t]["cls"]:
                ids[d] = alive[t]["id"]
                alive[t].update(box=dets[d, :4].copy(), age=0)
                scores[d, :] = -1.0
            scores[:, t] = -1.0

        for t in alive:
            if t["age"] or not any(i == t["id"] for i in ids):
                t["age"] += 1
        for d, track_id in enumerate(ids):
            if track_id == 0:
                ids[d] = self._next_id
                alive.append(
                    {
                        "id": self._next_id,
                        "box": dets[d, :4].copy(),
                        "cls": int(dets[d, 5]),
                        "age": 0,
                    }
                )
                self._next_id += 1
        self._tracks = [t for t in alive if t["age"] <= self.max_age]
        return ids
