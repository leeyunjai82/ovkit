"""OCR: find the text in a picture, then read it.

Two models, one call::

    Model("read_text")("sign.jpg")[0].text     # 'STOP AHEAD'

A text detector returns boxes but no words; a text recogniser reads one cropped
word but cannot find it. This joins them and puts each word on its own box.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..core.i18n import lang
from ..core.results import Boxes, Results
from .base import DEFAULT_CONF, Pipeline, detections

#: Recogniser per display language. The Latin model reads 0-9 a-z and nothing
#: else, so a Korean sign came back empty until the Korean one existed.
_RECOGNIZERS = {"ko": "text_recognition_ko"}
_FALLBACK = "text_recognition"


class TextReader(Pipeline):
    """Text detection + text recognition.

    >>> from ovkit import Model
    >>> r = Model("read_text")("receipt.jpg")[0]
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
        recognizer: str | None = None,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        #: ``None`` means "match the display language" — Korean text needs the
        #: Korean recogniser, and asking for one explicitly is the override.
        self.recognizer = recognizer or _RECOGNIZERS.get(lang(), _FALLBACK)
        #: Whether a crop has already failed — so the reason is said once, not
        #: once per word.
        self._complained = False

    def run(self, image: np.ndarray, *, conf: float = DEFAULT_CONF, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        order = self._reading_order(boxes)

        words: list[str] = []
        for i in order:
            crop = found.crop(int(i))
            words.append(self._read(crop))

        result = Results(image, task=self.name, names={0: "text"}, boxes=Boxes(boxes.data[order]))
        result.text = " ".join(w for w in words if w)
        if any(words):
            result.labels = words
        elif len(words):
            # Finding eight words and reading none of them is not "nothing
            # found" — that is what a blank wall looks like, and the two were
            # indistinguishable. Leave the boxes to speak ("8x text") and say
            # what happened once.
            warnings.warn(
                f"found {len(words)} text region(s) but '{self.recognizer}' read " f"none of them.",
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def _read(self, crop: np.ndarray) -> str:
        """Recognise one cropped word, returning '' when it cannot be read."""
        if crop.size == 0:
            return ""
        try:
            reader = self.model(self.recognizer)
        except Exception as exc:
            # A recogniser that cannot be loaded at all (not on the mirror yet,
            # offline) would otherwise turn every word into "" with no reason
            # given. Say it once and read what we can.
            if self.recognizer == _FALLBACK:
                raise
            warnings.warn(
                f"'{self.recognizer}' could not be loaded ({exc}); "
                f"reading with '{_FALLBACK}', which knows only 0-9 and a-z.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.recognizer = _FALLBACK
            reader = self.model(self.recognizer)
        try:
            out = reader(crop)
        except Exception as exc:  # noqa: BLE001 - one bad crop must not stop the page
            # Every crop failing the same way used to be indistinguishable from
            # a picture with no readable words: eight empty strings and the
            # cheerful summary "nothing found". Say it once, with the reason.
            if not self._complained:
                self._complained = True
                warnings.warn(
                    f"'{self.recognizer}' could not read a crop "
                    f"({type(exc).__name__}: {exc}). Further crops are read the "
                    f"same way and may fail too.",
                    RuntimeWarning,
                    stacklevel=2,
                )
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
