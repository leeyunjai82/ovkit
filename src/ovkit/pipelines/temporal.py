"""Pipelines that need more than one frame: drowsiness and gestures.

A single frame cannot say whether someone is falling asleep — closed eyes are a
blink until they last. And a gesture *is* motion: the sign-language model takes
eight frames, so handing it one frame eight times over asks it to recognise a
still photograph. Both keep state between calls, which is what makes them work
on a video or a webcam.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.constants import class_names
from ..core.results import Boxes, Probs, Results
from .base import Pipeline, detections
from .gaze import _square_crop

#: open-closed-eye-0001 emits [open, closed].
EYE_CLOSED = 1


class DrowsinessMonitor(Pipeline):
    """Driver monitoring: are the eyes closed, and for how long?

        >>> from ovkit import Model
        >>> monitor = Model("drowsiness")
        >>> for r in monitor.predict(0, stream=True):     # webcam
        ...     print(r.summary())     # 'awake' ... 'EYES CLOSED 1.4s — drowsy'

    Four models nobody can use alone: a face detector, the five-point landmark
    model for where the eyes are, ``open_closed_eye_0001`` for their state, and
    head pose for a nodding head. The pipeline adds what none of them has —
    *time*: a blink is a fifth of a second, so only a closure lasting longer
    than ``seconds`` is drowsiness.
    """

    name = "drowsiness"
    description = "Watch a face over time and warn when the eyes stay shut (driver monitoring)."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "face_detection",
        seconds: float = 1.0,
        nod_pitch: float = 20.0,
        eye_scale: float = 0.55,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.seconds = float(seconds)
        self.nod_pitch = float(nod_pitch)
        self.eye_scale = float(eye_scale)
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self) -> None:
        """Forget the closure currently being timed."""
        self._closed_since: float | None = None

    def run(self, image: np.ndarray, *, conf: float = 0.5, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        result = Results(image, task=self.name, names={0: "face"}, boxes=boxes)

        if not len(boxes):
            self.reset()
            result.text = "no face — cannot tell"
            return result

        # The largest face is the driver; passengers are not being monitored.
        face_index = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.xyxy]))
        face = found.crop(face_index)
        closed, score = self._eyes_closed(face)
        pitch = self._pitch(face)

        now = self._clock()
        if closed:
            self._closed_since = self._closed_since if self._closed_since is not None else now
        else:
            self._closed_since = None
        shut_for = (now - self._closed_since) if self._closed_since is not None else 0.0

        result.labels = [""] * len(boxes)
        result.labels[face_index] = self._verdict(closed, shut_for, pitch, score)
        result.text = result.labels[face_index]
        result.tensors = {"eyes_closed_seconds": np.array([shut_for], np.float32)}
        return result

    def _verdict(self, closed: bool, shut_for: float, pitch: float | None, score: float) -> str:
        if closed and shut_for >= self.seconds:
            return f"EYES CLOSED {shut_for:.1f}s — drowsy"
        if pitch is not None and pitch < -self.nod_pitch:
            return f"head nodding ({pitch:.0f}°) — drowsy"
        if closed:
            return f"blink ({shut_for:.1f}s)"
        return f"awake ({score:.2f})"

    def _eyes_closed(self, face: np.ndarray) -> tuple[bool, float]:
        """Both eyes classified shut, with the weaker of the two scores."""
        points = self._eye_points(face)
        if points is None:
            return False, 0.0
        size = int(round(float(np.linalg.norm(points[0] - points[1])) * self.eye_scale))
        if size < 4:
            return False, 0.0
        states = []
        for point in points:
            crop = _square_crop(face, point, size)
            if crop.size == 0:
                return False, 0.0
            out = self.model("open_closed_eye_0001")(crop)
            if not out or out[0].probs is None:
                return False, 0.0
            scores = np.asarray(out[0].probs.data, np.float32)
            states.append((int(np.argmax(scores)) == EYE_CLOSED, float(scores.max())))
        closed = all(state for state, _ in states)
        return closed, min(score for _, score in states)

    def _eye_points(self, face: np.ndarray) -> np.ndarray | None:
        out = self.model("face_landmarks")(face)
        if not out or out[0].keypoints is None or len(out[0].keypoints.data) == 0:
            return None
        points = out[0].keypoints.data[0]
        return points[:2, :2].astype(np.float32) if len(points) >= 2 else None

    def _pitch(self, face: np.ndarray) -> float | None:
        """Head pitch in degrees (negative = chin down), or ``None``."""
        try:
            out = self.model("head_pose")(face)
        except Exception:
            return None
        tensors = out[0].tensors if out else None
        if not tensors:
            return None
        for name, value in tensors.items():
            if name.lower().startswith("angle_p"):
                return float(np.asarray(value).reshape(-1)[0])
        return None


class GestureRecognizer(Pipeline):
    """Hand gestures from a moving picture, not a still one.

        >>> from ovkit import Model
        >>> gestures = Model("gesture")
        >>> for r in gestures.predict(0, stream=True):
        ...     print(r.summary())     # 'collecting frames (3/8)' ... 'thumb up 0.94'

    ``common_sign_language_0002`` takes a clip of eight frames. Fed a single
    image it is shown the same photograph eight times, which is not a gesture —
    so this keeps a rolling buffer of the last eight frames and classifies the
    motion across them.
    """

    name = "gesture"
    description = "Recognise hand gestures from the last few frames (needs motion, not a photo)."

    def __init__(
        self,
        device: str = "AUTO",
        model_name: str = "common_sign_language_0002",
        classes: str = "sign_language12",
    ) -> None:
        super().__init__(device)
        self.model_name = model_name
        self.classes = classes
        self._buffer: deque[np.ndarray] = deque(maxlen=1)
        self._clip_length = 0

    def reset(self) -> None:
        """Drop the buffered frames (call between videos)."""
        self._buffer.clear()

    def run(self, image: np.ndarray, **_: Any) -> Results:
        model = self.model(self.model_name)
        clip, height, width = _clip_shape(model.inputs[0][1])
        if clip != self._clip_length:
            self._clip_length = clip
            self._buffer = deque(self._buffer, maxlen=clip)

        frame = _resize_bgr(image, height, width)
        self._buffer.append(frame)

        result = Results(image, task=self.name)
        if len(self._buffer) < clip:
            result.text = f"collecting frames ({len(self._buffer)}/{clip})"
            return result

        # [T, C, H, W] -> [1, C, T, H, W], the layout the clip model declares.
        stacked = np.stack(list(self._buffer))
        feed = np.ascontiguousarray(np.transpose(stacked, (1, 0, 2, 3))[None], np.float32)
        outputs = model.infer({model.inputs[0][0]: feed})
        scores = _softmax(np.asarray(next(iter(outputs.values()))).reshape(-1).astype(np.float32))

        names = class_names(self.classes, scores.size)
        top = int(np.argmax(scores))
        result.names = names
        result.probs = Probs(scores)
        result.text = f"{names.get(top, f'class_{top}')} {float(scores[top]):.2f}"
        return result


def _clip_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """``(frames, height, width)`` from a ``[N, C, T, H, W]`` clip input."""
    if len(shape) == 5 and shape[2] > 0:
        return int(shape[2]), int(shape[3]), int(shape[4])
    return 8, int(shape[-2]) if shape[-2] > 0 else 224, int(shape[-1]) if shape[-1] > 0 else 224


def _resize_bgr(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(image, (width, height)).astype(np.float32)
    return np.transpose(resized, (2, 0, 1))  # HWC -> CHW


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
