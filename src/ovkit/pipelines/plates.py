"""Automatic number-plate reading — the vehicle, its colour, and its plate.

Three models that are useless apart::

    Model("read_plate")(frame)[0].summary()
    # 2 vehicles: black car — 12GA3456, white van — 34NA5678

`vehicle_license_plate_detection_barrier_0106` finds vehicles *and* plates but
cannot read a plate; `text_recognition_0014` reads a cropped word but cannot
find one; `vehicle_attributes_recognition_barrier_0042` describes a car it is
handed. Chained — and with each plate matched to the car it belongs to — they
are a number-plate reader.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.results import Boxes, Results
from .base import Pipeline, detections

#: Class ids of the barrier detector (documented OMZ interface).
VEHICLE, PLATE = 1, 2


class PlateReader(Pipeline):
    """Detect vehicles and plates, read the plates, describe the vehicles.

    >>> from ovkit import Model
    >>> r = Model("read_plate")("gate.jpg")[0]
    >>> r.text                    # 'black car — 12GA3456'
    >>> r.to_dict()["boxes"]      # each box with its own text
    """

    name = "read_plate"
    description = "Read number plates and describe the vehicle each belongs to."

    def __init__(
        self,
        device: str = "AUTO",
        detector: str = "license_plate",
        recognizer: str = "text_recognition",
        attributes: bool = True,
    ) -> None:
        super().__init__(device)
        self.detector = detector
        self.recognizer = recognizer
        self.attributes = attributes

    def run(self, image: np.ndarray, *, conf: float = 0.4, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        cls = boxes.cls.astype(int) if len(boxes) else np.zeros(0, int)

        plates = {i: self._read(found.crop(i)) for i in np.where(cls == PLATE)[0]}
        vehicles = {i: self._describe(found.crop(i)) for i in np.where(cls == VEHICLE)[0]}
        owner = pair_plates_to_vehicles(boxes, list(plates), list(vehicles))

        labels: list[str] = []
        for i in range(len(boxes)):
            if i in plates:
                labels.append(plates[i] or "plate")
            else:
                plate = plates.get(owner.get(i, -1), "")
                labels.append(
                    f"{vehicles.get(i, 'vehicle')} — {plate}"
                    if plate
                    else vehicles.get(i, "vehicle")
                )

        result = Results(image, task=self.name, names=found.names, boxes=boxes)
        result.labels = labels
        result.text = self._headline(labels, cls)
        return result

    def _read(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return ""
        try:
            out = self.model(self.recognizer)(crop)
        except Exception:
            return ""
        return (out[0].text or "").strip().upper() if out else ""

    def _describe(self, crop: np.ndarray) -> str:
        """ "black car" — the attribute model's two heads, said plainly."""
        if not self.attributes or crop.size == 0:
            return "vehicle"
        try:
            out = self.model("vehicle_attributes")(crop)
        except Exception:
            return "vehicle"
        if not out or not out[0].text:
            return "vehicle"
        # "type: car (0.98) · color: black (0.83)" -> "black car"
        parts = {}
        for chunk in out[0].text.split("·"):
            key, _, value = chunk.partition(":")
            parts[key.strip().lower()] = value.split("(")[0].strip()
        colour, kind = parts.get("color", ""), parts.get("type", "vehicle")
        return f"{colour} {kind}".strip() or "vehicle"

    @staticmethod
    def _headline(labels: list[str], cls: np.ndarray) -> str:
        vehicles = [labels[i] for i in np.where(cls == VEHICLE)[0]]
        if vehicles:
            n = len(vehicles)
            return f"{n} vehicle{'s' if n > 1 else ''}: " + ", ".join(vehicles[:3])
        plates = [labels[i] for i in np.where(cls == PLATE)[0] if labels[i]]
        return ", ".join(plates[:3]) if plates else "no vehicle or plate found"


def pair_plates_to_vehicles(boxes: Boxes, plates: list[int], vehicles: list[int]) -> dict[int, int]:
    """Map each vehicle index to the plate index sitting inside it.

    A plate belongs to the smallest vehicle box that contains its centre —
    smallest, because a car in front of a lorry is inside both.
    """
    owner: dict[int, int] = {}
    if not len(boxes):
        return owner
    xyxy = boxes.xyxy
    for p in plates:
        px = (xyxy[p][0] + xyxy[p][2]) / 2
        py = (xyxy[p][1] + xyxy[p][3]) / 2
        best, best_area = None, np.inf
        for v in vehicles:
            x1, y1, x2, y2 = xyxy[v]
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = float((x2 - x1) * (y2 - y1))
                if area < best_area:
                    best, best_area = v, area
        if best is not None:
            owner[best] = p
    return owner
