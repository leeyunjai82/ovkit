"""Detection adapter — end-to-end for DETR-family models (RT-DETR, D-FINE, ...).

DETR-style detectors emit a class-logits tensor ``[N, Q, C]`` and a box tensor
``[N, Q, 4]`` in normalized ``cxcywh`` form, and need **no NMS**: each of the
``Q`` queries is already one (de-duplicated) prediction. We sigmoid the logits,
take the best class per query, threshold by confidence, and map the normalized
boxes back to the original image size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.constants import class_names
from ..core.results import Boxes, Results
from .base import BaseAdapter


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


class DetectAdapter(BaseAdapter):
    """Adapter for object detection."""

    task = "detect"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.names:
            # Default to COCO-80 when the manifest doesn't name the classes —
            # the overwhelming majority of detection models are COCO-trained.
            self.names = class_names(self.post.get("classes", "coco80"))

    def run(
        self,
        backend: Backend,
        image: np.ndarray,
        *,
        conf: float = 0.25,
        max_det: int = 300,
        **_: Any,
    ) -> Results:
        feed = self.preprocess_square(image, rgb=True)
        outputs = backend.infer(feed)
        boxes = self._decode(outputs, image.shape[:2], conf=conf, max_det=max_det)
        return Results(image, task=self.task, names=self.names, boxes=Boxes(boxes))

    # -- decoding -----------------------------------------------------------

    def _decode(
        self,
        outputs: dict[str, np.ndarray],
        orig_hw: tuple[int, int],
        *,
        conf: float,
        max_det: int,
    ) -> np.ndarray:
        logits, boxes = self._split_outputs(outputs)
        # logits: [Q, C]  boxes: [Q, 4] (cxcywh)
        scores = _sigmoid(logits)
        cls = scores.argmax(axis=1)
        best = scores.max(axis=1)

        keep = best >= conf
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes = boxes[keep]
        cls = cls[keep]
        best = best[keep]

        xyxy = _cxcywh_to_xyxy(boxes)
        # Normalized boxes (values ~[0,1]) -> scale by original image size.
        # Some exports emit input-pixel coords instead; detect and rescale.
        h, w = orig_hw
        if xyxy.max(initial=0.0) <= 2.0:
            xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        else:
            xyxy = xyxy / self.imgsz * np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        data = np.concatenate([xyxy, best[:, None], cls[:, None].astype(np.float32)], axis=1)
        # Highest confidence first, capped at max_det.
        order = np.argsort(best)[::-1][:max_det]
        return data[order].astype(np.float32)

    @staticmethod
    def _split_outputs(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(logits[Q,C], boxes[Q,4])`` from the raw output dict."""
        arrays = [np.asarray(v) for v in outputs.values()]
        boxes_arr = None
        logits_arr = None
        for arr in arrays:
            a = arr[0] if arr.ndim == 3 else arr  # drop batch dim
            if a.ndim == 2 and a.shape[-1] == 4 and boxes_arr is None:
                boxes_arr = a
            elif a.ndim == 2:
                logits_arr = a
        # Prefer name hints when shapes are ambiguous.
        if boxes_arr is None or logits_arr is None:
            for name, arr in outputs.items():
                a = np.asarray(arr)
                a = a[0] if a.ndim == 3 else a
                low = name.lower()
                if ("box" in low or "bbox" in low) and a.shape[-1] == 4:
                    boxes_arr = a
                elif "logit" in low or "score" in low or "class" in low:
                    logits_arr = a
        if boxes_arr is None or logits_arr is None:
            raise ValueError(
                "Could not identify DETR logits/boxes among outputs "
                f"{[(k, np.asarray(v).shape) for k, v in outputs.items()]}."
            )
        return logits_arr, boxes_arr
