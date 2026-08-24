"""One sentence about a whole picture, from several models at once.

    Model("scene")("room.jpg")[0].summary()
    # 2 people (1 happy), a laptop and a cup · floor 47% · wall 31%

Detection lists objects, segmentation says what the space is made of, and the
face models say something about the people in it. Read together they are a
description; read separately they are three tensors. The pipeline is also the
cheapest way to see several models agree or disagree on the same frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Results
from .analyze import FaceAnalyzer
from .base import Pipeline, detections


class SceneReport(Pipeline):
    """Detection + segmentation + faces, summarised as one line.

    Each part can be switched off: ``Model("scene", segment=False)`` skips the
    segmentation model (and its download) entirely.
    """

    name = "scene"
    description = "Describe a whole picture: objects, what the space is made of, and the people."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "detect",
        segment: bool = True,
        faces: bool = True,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.use_segment = segment
        self.use_faces = faces
        self._faces = FaceAnalyzer(device=device) if faces else None

    def run(self, image: np.ndarray, *, conf: float = 0.3, **_: Any) -> Results:
        objects = detections(self.model(self.detector), image, conf)
        boxes = objects.boxes if objects.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        result = Results(image, task=self.name, names=objects.names, boxes=boxes)

        parts: list[str] = []
        people = self._people(image) if self.use_faces else ""
        objects_line = self._objects(objects, skip_people=bool(people))
        if people:
            parts.append(people)
        if objects_line:
            parts.append(objects_line)
        if self.use_segment:
            surface = self._surfaces(image)
            if surface:
                parts.append(surface)
        result.text = " · ".join(parts) if parts else "an empty-looking scene"
        # Label the boxes as the detector would; without this, summary() would
        # append its own class counts after the report and say everything twice.
        result.labels = [objects.label_for(i) for i in range(len(boxes))]
        return result

    def _objects(self, objects: Results, skip_people: bool) -> str:
        """ "a laptop and a cup", counted, with people left to the face part."""
        if objects.boxes is None or not len(objects.boxes):
            return ""
        counts: dict[str, int] = {}
        for cls in objects.boxes.cls.astype(int):
            label = objects.names.get(int(cls), str(int(cls)))
            if skip_people and label == "person":
                continue
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            return ""
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        return _join([f"{n} {label}s" if n > 1 else f"a {label}" for label, n in ranked])

    def _people(self, image: np.ndarray) -> str:
        """ "2 people (1 happy)" — the face pipeline, condensed."""
        if self._faces is None:
            return ""
        try:
            faces = self._faces.run(image, conf=0.5)
        except Exception:
            return ""
        n = len(faces.boxes) if faces.boxes is not None else 0
        if not n:
            return ""
        moods: dict[str, int] = {}
        for label in faces.labels or []:
            mood = _mood(label)
            if mood:
                moods[mood] = moods.get(mood, 0) + 1
        head = f"{n} {'people' if n > 1 else 'person'}"
        if not moods:
            return head
        top, count = max(moods.items(), key=lambda kv: kv[1])
        return f"{head} ({count} {top})"

    def _surfaces(self, image: np.ndarray) -> str:
        """What the frame is mostly made of, from the segmentation class map."""
        try:
            out = self.model("segment")(image)
        except Exception:
            return ""
        if not out or out[0].masks is None or not len(out[0].masks):
            return ""
        return out[0]._mask_summary(3)


def _mood(label: str) -> str:
    """Pull the emotion word out of "age 31 · male 98% · happy 92%"."""
    known = {"neutral", "happy", "sad", "surprise", "anger"}
    for chunk in label.split("·"):
        word = chunk.strip().split(" ")[0].lower()
        if word in known:
            return word
    return ""


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]
