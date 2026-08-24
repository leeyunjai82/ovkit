"""Gaze estimation — where the person is looking.

    vis("gaze")(frame)[0].summary()      # 'looking left and slightly up'

The gaze model alone cannot be run on a picture: it takes two eye crops and the
head pose angles, not an image. Producing those means a face detector, a
landmark model and a head-pose model first. This chains all four, so the model
is usable from a frame:

    face detection -> 5 landmarks -> eye crops
                   -> head pose  -> (yaw, pitch, roll)   ->  gaze vector

The eye crops follow the Open Model Zoo gaze demo: a square around each eye,
rotated to cancel head roll, with the roll then removed from the angles and
rotated back into the returned vector.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.results import Boxes, Keypoints, Results
from .base import Pipeline, detections

#: The gaze model's own input names (documented OMZ interface).
_LEFT, _RIGHT, _ANGLES = "left_eye_image", "right_eye_image", "head_pose_angles"


class GazeEstimator(Pipeline):
    """Face detection + landmarks + head pose + gaze, in one call.

        >>> from ovkit import vis
        >>> r = vis("gaze")("portrait.jpg")[0]
        >>> r.summary()          # '1 face: looking left and slightly up'
        >>> r.tensors["gaze"]    # (N, 3) unit vectors, one per face
        >>> r.save("gaze.jpg")   # an arrow drawn from each eye

    ``eye_scale`` sizes the eye crop as a fraction of the distance between the
    two eyes. The zoo demo sizes it from eye-corner landmarks, which the
    five-point landmark model does not provide; 0.55 is the equivalent width.
    """

    name = "gaze"
    description = "Estimate where a face is looking (detection + landmarks + head pose + gaze)."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "face_detection",
        eye_scale: float = 0.55,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.eye_scale = float(eye_scale)

    def run(self, image: np.ndarray, *, conf: float = 0.5, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))

        vectors, labels, eyes = [], [], []
        for i in range(len(boxes)):
            face = found.crop(i)
            if face.size == 0:
                continue
            points = self._eye_points(face)
            if points is None:
                continue
            angles = self._head_pose(face)
            vector = self._gaze(face, points, angles)
            if vector is None:
                continue
            x1, y1 = boxes.xyxy[i][:2]
            eyes.append(points + np.array([x1, y1], np.float32))
            vectors.append(vector)
            labels.append(describe_gaze(vector))

        result = Results(image, task=self.name, names={0: "face"}, boxes=boxes)
        if vectors:
            result.labels = labels
            result.tensors = {"gaze": np.stack(vectors)}
            pts = np.stack(eyes)  # (N, 2, 2) -> keypoints for plotting
            result.keypoints = Keypoints(
                np.concatenate([pts, np.ones((*pts.shape[:2], 1), np.float32)], axis=2)
            )
            result.arrows = gaze_arrows(pts, np.stack(vectors), boxes)
            result.text = f"{len(vectors)} face{'s' if len(vectors) > 1 else ''}: " + ", ".join(
                labels[:3]
            )
        else:
            result.text = "no face with usable eyes found"
        return result

    # -- stages -------------------------------------------------------------

    def _eye_points(self, face: np.ndarray) -> np.ndarray | None:
        """The two eye landmarks, in face-crop pixel coordinates."""
        out = self.model("face_landmarks")(face)
        if not out or out[0].keypoints is None or len(out[0].keypoints.data) == 0:
            return None
        points = out[0].keypoints.data[0]
        return points[:2, :2].astype(np.float32) if len(points) >= 2 else None

    def _head_pose(self, face: np.ndarray) -> np.ndarray:
        """``(yaw, pitch, roll)`` in degrees, or zeros if the model has nothing."""
        out = self.model("head_pose")(face)
        tensors = out[0].tensors if out else None
        if not tensors:
            return np.zeros(3, np.float32)
        angles = {}
        for name, value in tensors.items():
            key = name.lower()
            for axis, letter in (("yaw", "y"), ("pitch", "p"), ("roll", "r")):
                if key.startswith(f"angle_{letter}"):
                    angles[axis] = float(np.asarray(value).reshape(-1)[0])
        return np.array(
            [angles.get("yaw", 0.0), angles.get("pitch", 0.0), angles.get("roll", 0.0)], np.float32
        )

    def _gaze(self, face: np.ndarray, points: np.ndarray, angles: np.ndarray) -> np.ndarray | None:
        """Run the gaze model on both eye crops and return a unit vector."""
        roll = float(angles[2])
        size = int(round(np.linalg.norm(points[0] - points[1]) * self.eye_scale))
        if size < 4:
            return None
        crops = []
        for point in points:
            crop = _square_crop(face, point, size)
            if crop.size == 0:
                return None
            crops.append(_rotate(crop, roll) if roll else crop)

        model = self.model("gaze")
        shapes = {name: shape for name, shape, _dtype in model.inputs}
        if _LEFT not in shapes or _RIGHT not in shapes:
            return None
        feed = {
            _LEFT: _to_nchw(crops[0], shapes[_LEFT]),
            _RIGHT: _to_nchw(crops[1], shapes[_RIGHT]),
            # Roll is cancelled in the crops, so it is zeroed here and rotated
            # back into the answer below — exactly what the zoo demo does.
            _ANGLES: np.array([[angles[0], angles[1], 0.0]], np.float32),
        }
        raw = np.asarray(next(iter(model.infer(feed).values()))).reshape(-1)[:3]
        norm = float(np.linalg.norm(raw))
        if norm < 1e-6:
            return None
        vector = (raw / norm).astype(np.float32)
        if roll:
            cos, sin = math.cos(math.radians(roll)), math.sin(math.radians(roll))
            vector = np.array(
                [
                    vector[0] * cos + vector[1] * sin,
                    -vector[0] * sin + vector[1] * cos,
                    vector[2],
                ],
                np.float32,
            )
        return vector


def describe_gaze(vector: np.ndarray, deadzone: float = 0.15) -> str:
    """Turn a gaze vector into words — the point of the pipeline.

    Directions are as seen in the picture, matching the arrow :func:`gaze_arrows`
    draws: ``x`` positive is towards the right of the frame, ``y`` positive is up.
    """
    x, y = float(vector[0]), float(vector[1])
    horizontal = "right" if x > deadzone else "left" if x < -deadzone else ""
    vertical = "up" if y > deadzone else "down" if y < -deadzone else ""
    if not horizontal and not vertical:
        return "looking at the camera"
    if horizontal and vertical:
        strong = "slightly " if abs(y) < 2 * deadzone else ""
        return f"looking {horizontal} and {strong}{vertical}"
    return f"looking {horizontal or vertical}"


def _square_crop(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    """A ``size``x``size`` crop centred on ``center``, clipped to the image."""
    h, w = image.shape[:2]
    half = size // 2
    x1, y1 = int(round(center[0])) - half, int(round(center[1])) - half
    x2, y2 = x1 + size, y1 + size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((0, 0, 3), image.dtype)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre to cancel head roll (replicating the border)."""
    import cv2

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _to_nchw(crop: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Resize a BGR crop into the model's ``[1, 3, H, W]`` input."""
    import cv2

    h, w = (shape[2], shape[3]) if len(shape) == 4 and shape[2] > 0 else (60, 60)
    resized = cv2.resize(crop, (w, h)).astype(np.float32)
    return np.ascontiguousarray(np.transpose(resized, (2, 0, 1))[None])


def gaze_arrows(eyes: np.ndarray, vectors: np.ndarray, boxes: Boxes) -> np.ndarray:
    """One arrow per eye, pointing where that face is looking.

    Arrow length scales with the face so it stays visible on a small face and
    does not swamp a large one. Image ``y`` grows downwards while gaze ``y``
    grows upwards, hence the sign flip.
    """
    arrows = []
    for i, (pair, vector) in enumerate(zip(eyes, vectors, strict=False)):
        width = float(boxes.xyxy[i][2] - boxes.xyxy[i][0]) if len(boxes) > i else 100.0
        length = max(20.0, width * 0.6)
        for x, y in pair:
            arrows.append([x, y, x + vector[0] * length, y - vector[1] * length])
    return np.asarray(arrows, np.float32).reshape(-1, 4)
