"""What is the person actually looking at?

Gaze estimation returns a direction; object detection returns boxes. Neither
answers the question a shop, a museum or a UX study is asking — *which thing*
held their attention. Casting the gaze ray into the detected objects does::

    Model("attention")("shelf.jpg")[0].summary()
    # 1 person looking at: bottle (0.68 of the way across the frame)

The ray is 2-D: a gaze vector's depth cannot be recovered from one camera, so
this reports what lies along the line of sight in the image, which is what a
fixed camera can honestly say.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Results
from .base import Pipeline, detections
from .gaze import GazeEstimator


class AttentionAnalyzer(Pipeline):
    """Gaze plus object detection: name the object on the line of sight.

    >>> from ovkit import Model
    >>> r = Model("attention")("desk.jpg")[0]
    >>> r.text            # '1 person looking at: laptop'
    >>> r.arrows          # the ray drawn from each eye
    """

    name = "attention"
    description = "Work out which detected object a person is looking at (gaze + detection)."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "detect",
        reach: float = 2.0,
        steps: int = 120,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        #: How far to follow the ray, in multiples of the image diagonal.
        self.reach = float(reach)
        self.steps = int(steps)
        self.gaze = GazeEstimator(device=device)

    def run(self, image: np.ndarray, *, conf: float = 0.3, **_: Any) -> Results:
        looking = self.gaze.run(image, conf=0.5)
        objects = detections(self.model(self.detector), image, conf)
        boxes = objects.boxes if objects.boxes is not None else Boxes(np.zeros((0, 6), np.float32))

        result = Results(image, task=self.name, names=objects.names, boxes=boxes)
        result.arrows = looking.arrows
        if looking.keypoints is None or looking.tensors is None:
            result.text = looking.text or "no face found"
            return result

        vectors = looking.tensors["gaze"]
        eyes = looking.keypoints.data[..., :2]
        targets = []
        for face, vector in zip(eyes, vectors, strict=False):
            origin = face.mean(axis=0)  # between the eyes
            hit = self.first_hit(origin, vector, boxes, image.shape[:2])
            targets.append(objects.names.get(int(hit), str(hit)) if hit is not None else "nothing")

        result.labels = [""] * len(boxes)
        n = len(targets)
        named = ", ".join(targets[:3])
        result.text = (
            f"{n} person{'s' if n > 1 else ''} looking at: {named}" if n else "no face found"
        )
        return result

    def first_hit(
        self,
        origin: np.ndarray,
        vector: np.ndarray,
        boxes: Boxes,
        shape: tuple[int, int],
    ) -> int | None:
        """Class id of the first box the gaze ray enters, or ``None``.

        Walking the ray in small steps (rather than solving per box) keeps the
        nearest object first without any depth information.
        """
        if not len(boxes):
            return None
        height, width = shape
        length = float(np.hypot(width, height)) * self.reach
        # Image y grows downwards; gaze y grows upwards.
        direction = np.array([float(vector[0]), -float(vector[1])], np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction /= norm
        xyxy = boxes.xyxy
        for step in range(1, self.steps + 1):
            point = origin + direction * (length * step / self.steps)
            if not (-width <= point[0] <= 2 * width and -height <= point[1] <= 2 * height):
                break
            inside = (
                (xyxy[:, 0] <= point[0])
                & (point[0] <= xyxy[:, 2])
                & (xyxy[:, 1] <= point[1])
                & (point[1] <= xyxy[:, 3])
            )
            if inside.any():
                return int(boxes.cls[int(np.argmax(inside))])
        return None
