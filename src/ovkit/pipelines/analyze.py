"""Detect-then-describe pipelines: faces, people, vehicles.

Each one is the same shape — find the objects, crop each, ask a few small
models about it, and put the answers on the box:

    vis("face_analyze")(frame)   ->  "2 faces: male 31 · happy 0.92, ..."

Doing it by hand means a detector, a crop per object, one model per attribute
and the code to join it all up. That is what these classes hold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Keypoints, Results
from .base import Pipeline, detections


class _DetectAndDescribe(Pipeline):
    """Shared machinery: detect objects, run attribute models on each crop."""

    #: Registered name of the detector to find objects with.
    detector: str = "detect"
    #: Attribute models available, in the order their answers are joined.
    available: tuple[str, ...] = ()
    #: Which of them run by default (the rest cost another download).
    default_attributes: tuple[str, ...] = ()
    #: Grow each crop by this fraction — attribute models want some context.
    pad: float = 0.0
    #: What one detected object is called in the summary line.
    noun: str = "object"

    def __init__(
        self,
        device: str = "AUTO",
        attributes: tuple[str, ...] | list[str] | None = None,
        detector: str | None = None,
    ) -> None:
        super().__init__(device)
        if detector:
            self.detector = detector
        chosen = tuple(attributes) if attributes is not None else self.default_attributes
        unknown = [a for a in chosen if a not in self.available]
        if unknown:
            raise ValueError(
                f"{self.name}: unknown attribute(s) {unknown}. Available: {list(self.available)}."
            )
        self.attributes = chosen

    def run(self, image: np.ndarray, *, conf: float = 0.5, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        result = Results(image, task=self.name, names=found.names, boxes=boxes)
        result.labels = [self._describe(found.crop(i, self.pad)) for i in range(len(boxes))]
        result.text = self._headline(result)
        return result

    def _describe(self, crop: np.ndarray) -> str:
        """Join every attribute model's answer for one cropped object."""
        if crop.size == 0:
            return self.noun
        parts: list[str] = []
        for attribute in self.attributes:
            try:
                out = self.model(attribute)(crop)
            except Exception as exc:  # one bad attribute must not lose the object
                parts.append(f"{attribute}: unavailable ({type(exc).__name__})")
                continue
            if out:
                parts.append(out[0].summary())
        return " · ".join(parts) if parts else self.noun

    def _headline(self, result: Results) -> str:
        n = len(result.boxes) if result.boxes is not None else 0
        if not n:
            return f"no {self.noun} found"
        head = f"{n} {self.noun}{'s' if n > 1 else ''}"
        shown = ", ".join(result.labels[:3]) if result.labels else ""
        return f"{head}: {shown}" if shown else head


class FaceAnalyzer(_DetectAndDescribe):
    """Faces plus age, gender and emotion — the usual "who is in frame" answer.

        >>> from ovkit import vis
        >>> for r in vis("face_analyze")("group.jpg"):
        ...     print(r.summary())     # 2 faces: age 31 · male 98% · happy 92%, ...
        ...     r.save("faces.jpg")

    ``attributes`` picks what to run; ``head_pose`` and ``face_landmarks`` are
    off by default because each is another model to download.
    """

    name = "face_analyze"
    description = "Detect faces, then read age, gender and emotion from each."
    detector = "face_detection"
    available = ("age_gender", "emotion", "head_pose", "face_landmarks")
    default_attributes = ("age_gender", "emotion")
    pad = 0.15
    noun = "face"

    def run(self, image: np.ndarray, *, conf: float = 0.5, **kwargs: Any) -> Results:
        result = super().run(image, conf=conf, **kwargs)
        if "face_landmarks" in self.attributes:
            result.keypoints = self._landmarks(image, result)
        return result

    def _landmarks(self, image: np.ndarray, result: Results) -> Keypoints | None:
        """Landmark points, moved from crop coordinates back onto the image."""
        if result.boxes is None or not len(result.boxes):
            return None
        per_face = []
        for i in range(len(result.boxes)):
            crop = result.crop(i, self.pad)
            if crop.size == 0:
                continue
            out = self.model("face_landmarks")(crop)
            if not out or out[0].keypoints is None:
                continue
            x1, y1, x2, y2 = result.boxes.xyxy[i]
            dx, dy = (x2 - x1) * self.pad, (y2 - y1) * self.pad
            points = out[0].keypoints.data[0].copy()
            points[:, 0] += x1 - dx
            points[:, 1] += y1 - dy
            per_face.append(points)
        return Keypoints(np.stack(per_face)) if per_face else None


class PersonAnalyzer(_DetectAndDescribe):
    """People plus what they are wearing or carrying.

    >>> from ovkit import vis
    >>> vis("person_analyze")("street.jpg")[0].summary()
    '3 persons: male 0.98 · long pants 0.95 · bag 0.71, ...'
    """

    name = "person_analyze"
    description = "Detect people, then read their attributes (bag, hat, sleeves, ...)."
    detector = "person_detection"
    available = ("person_attributes",)
    default_attributes = ("person_attributes",)
    noun = "person"

    def _headline(self, result: Results) -> str:
        line = super()._headline(result)
        return line.replace("persons", "people").replace("no person found", "nobody found")


class VehicleAnalyzer(_DetectAndDescribe):
    """Vehicles plus their type and colour.

    >>> from ovkit import vis
    >>> vis("vehicle_analyze")("parking.jpg")[0].summary()
    '2 vehicles: type: car (0.98) · color: black (0.83), ...'
    """

    name = "vehicle_analyze"
    description = "Detect vehicles, then read type (car/bus/truck/van) and colour."
    detector = "vehicle_detection"
    available = ("vehicle_attributes",)
    default_attributes = ("vehicle_attributes",)
    noun = "vehicle"
