"""Anonymise a picture: faces and number plates removed before it leaves.

    Model("anonymize")("street.jpg")[0].save("safe.jpg")

Publishing camera footage, sharing a dataset or filing a support screenshot all
need the people taken out of it first, and "run a face detector and then work
out how to blur the boxes" is exactly the plumbing ovkit should own. Plates go
too, because a number plate identifies a person as surely as a face does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Results
from .base import Pipeline, detections

#: Class id of the plate in the barrier detector's output.
PLATE = 2


class Anonymizer(Pipeline):
    """Blur or pixelate every face (and optionally every number plate).

    Parameters
    ----------
    plates:
        Also redact number plates. Costs one more model.
    method:
        ``"pixelate"`` (visibly redacted) or ``"blur"`` (softer).
    strength:
        Bigger is stronger: pixel block size and blur radius both scale with the
        region, so a small distant face is redacted as thoroughly as a close one.

    The result's image *is* the redacted picture, so ``r.plot()`` and
    ``r.save()`` never hand back the original by accident. The regions are kept
    in ``r.tensors["regions"]`` if you need to audit what was covered.
    """

    name = "anonymize"
    description = "Blur every face (and number plate) so a picture can be shared."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "face_detection",
        plates: bool = False,
        method: str = "pixelate",
        strength: float = 1.0,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.plates = plates
        if method not in {"pixelate", "blur"}:
            raise ValueError(f"method must be 'pixelate' or 'blur', got {method!r}.")
        self.method = method
        self.strength = float(strength)

    def run(self, image: np.ndarray, *, conf: float = 0.4, **_: Any) -> Results:
        regions: list[np.ndarray] = []
        faces = detections(self.model(self.detector), image, conf)
        if faces.boxes is not None and len(faces.boxes):
            regions.extend(faces.boxes.xyxy)
        n_faces = len(regions)

        n_plates = 0
        if self.plates:
            found = detections(self.model("license_plate"), image, conf)
            if found.boxes is not None and len(found.boxes):
                keep = found.boxes.cls.astype(int) == PLATE
                plate_boxes = found.boxes.xyxy[keep]
                regions.extend(plate_boxes)
                n_plates = len(plate_boxes)

        redacted = image.copy()
        for box in regions:
            redact(redacted, box, method=self.method, strength=self.strength)

        result = Results(redacted, task=self.name)
        result.tensors = {"regions": np.asarray(regions, np.float32).reshape(-1, 4)}
        result.text = _headline(n_faces, n_plates, self.method)
        return result


def redact(
    image: np.ndarray,
    box: np.ndarray,
    method: str = "pixelate",
    strength: float = 1.0,
) -> np.ndarray:
    """Destroy the detail inside ``box``, in place.

    Pixelating means downscaling and scaling back with nearest-neighbour, so the
    original detail is genuinely gone rather than smoothed — a blur can be
    partially undone, a 6x6 mosaic cannot.
    """
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image
    patch = image[y1:y2, x1:x2]
    if method == "blur":
        radius = max(3, int(min(patch.shape[:2]) * 0.35 * strength) | 1)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (radius, radius), 0)
    else:
        blocks = max(2, int(round(8 / max(strength, 0.1))))
        small = cv2.resize(patch, (blocks, blocks), interpolation=cv2.INTER_AREA)
        image[y1:y2, x1:x2] = cv2.resize(
            small, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return image


def _headline(faces: int, plates: int, method: str) -> str:
    if not faces and not plates:
        return "nothing to anonymise — no face or plate found"
    parts = []
    if faces:
        parts.append(f"{faces} face{'s' if faces > 1 else ''}")
    if plates:
        parts.append(f"{plates} plate{'s' if plates > 1 else ''}")
    return f"{method}d " + " and ".join(parts)
