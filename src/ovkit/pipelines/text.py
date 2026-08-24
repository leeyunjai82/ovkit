"""OCR: find the text in a picture, then read it.

Two models, one call::

    vis("read_text")("sign.jpg")[0].text     # 'STOP AHEAD'

A text detector returns boxes but no words; a text recogniser reads one cropped
word but cannot find it. This joins them and puts each word on its own box.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Results
from .base import Pipeline, detections


class TextReader(Pipeline):
    """Text detection + text recognition.

    >>> from ovkit import vis
    >>> r = vis("read_text")("receipt.jpg")[0]
    >>> r.text                     # every word, reading order (top to bottom)
    >>> r.labels                   # the word on each box
    >>> r.save("read.jpg")         # boxes labelled with what they say
    """

    name = "read_text"
    description = "Find text regions and read them (detection + recognition)."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "text_detection",
        recognizer: str = "text_recognition",
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray, *, conf: float = 0.3, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        order = self._reading_order(boxes)

        words: list[str] = []
        for i in order:
            crop = found.crop(int(i))
            words.append(self._read(crop))

        result = Results(image, task=self.name, names={0: "text"}, boxes=Boxes(boxes.data[order]))
        result.labels = words
        result.text = " ".join(w for w in words if w)
        return result

    def _read(self, crop: np.ndarray) -> str:
        """Recognise one cropped word, returning '' when it cannot be read."""
        if crop.size == 0:
            return ""
        try:
            out = self.model(self.recognizer)(crop)
        except Exception:
            return ""
        return (out[0].text or "").strip() if out else ""

    @staticmethod
    def _reading_order(boxes: Boxes) -> np.ndarray:
        """Sort boxes top-to-bottom, then left-to-right within a line.

        Without this the words come back in the detector's confidence order,
        which reads as nonsense.
        """
        if not len(boxes):
            return np.zeros(0, dtype=int)
        xyxy = boxes.xyxy
        heights = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)
        line = np.round(xyxy[:, 1] / np.median(heights)).astype(int)  # group into rows
        return np.lexsort((xyxy[:, 0], line))
