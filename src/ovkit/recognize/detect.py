"""Detection adapter — DETR-family and SSD/DetectionOutput models.

Two output families are auto-detected from the model's output signature:

* **DETR** (RT-DETR / D-FINE): a class-logits tensor ``[N, Q, C]`` and a box
  tensor ``[N, Q, 4]`` in normalized ``cxcywh`` form. No NMS — each query is one
  prediction. Sigmoid the logits, take the best class per query, threshold.
* **SSD / DetectionOutput** (most OMZ detectors: face/person/vehicle): a single
  ``[1, 1, N, 7]`` tensor of ``[image_id, label, conf, x_min, y_min, x_max,
  y_max]`` with normalized coordinates. Threshold by ``conf``.

Preprocessing follows the model's own input size (OMZ models are fixed-size) and
sensible per-family defaults (DETR: RGB ``[0,1]``; SSD: raw BGR ``[0,255]``),
both overridable via the manifest ``preprocess`` block.
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
    """Adapter for object detection (DETR and SSD output families)."""

    task = "detect"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._classes_key = self.post.get("classes")

    # -- pipeline -----------------------------------------------------------

    def run(
        self,
        backend: Backend,
        image: np.ndarray,
        *,
        conf: float = 0.25,
        max_det: int = 300,
        **_: Any,
    ) -> Results:
        fmt = self._format(backend)
        size = self.model_input_hw(backend)
        h, w = image.shape[:2]

        if fmt == "ssd":
            rgb = bool(self.pre.get("rgb", False))  # OMZ SSD: raw BGR
            feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
            outputs = backend.infer(feed)
            boxes = self._decode_ssd(outputs, (h, w), conf=conf, max_det=max_det)
            names = self.names or class_names(self._classes_key)
        else:
            rgb = bool(self.pre.get("rgb", True))  # DETR: RGB [0,1]
            feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 255.0))
            outputs = backend.infer(feed)
            boxes = self._decode_detr(outputs, (h, w), conf=conf, max_det=max_det)
            names = self.names or class_names(self._classes_key or "coco80")

        return Results(image, task=self.task, names=names, boxes=Boxes(boxes))

    def _format(self, backend: Backend) -> str:
        """Pick the decode format from the manifest or the output signature."""
        if self.post.get("format"):
            return str(self.post["format"])
        shapes = [s for _, s in backend.output_signatures()]
        # SSD DetectionOutput: a single [.., N, 7] tensor.
        if len(shapes) == 1 and len(shapes[0]) >= 2 and shapes[0][-1] == 7:
            return "ssd"
        return "detr"

    # -- SSD decode ---------------------------------------------------------

    @staticmethod
    def _decode_ssd(
        outputs: dict[str, np.ndarray], orig_hw: tuple[int, int], *, conf: float, max_det: int
    ) -> np.ndarray:
        arr = np.asarray(next(iter(outputs.values())))
        det = arr.reshape(-1, arr.shape[-1])  # [N, 7]
        if det.shape[1] != 7:
            return np.zeros((0, 6), dtype=np.float32)

        scores = det[:, 2]
        keep = scores >= conf
        det = det[keep]
        if det.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        labels = det[:, 1]
        scores = det[:, 2]
        h, w = orig_hw
        xyxy = det[:, 3:7] * np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        data = np.concatenate([xyxy, scores[:, None], labels[:, None]], axis=1)
        order = np.argsort(scores)[::-1][:max_det]
        return data[order].astype(np.float32)

    # -- DETR decode --------------------------------------------------------

    def _decode_detr(
        self,
        outputs: dict[str, np.ndarray],
        orig_hw: tuple[int, int],
        *,
        conf: float,
        max_det: int,
    ) -> np.ndarray:
        logits, boxes = self._split_detr(outputs)
        scores = _sigmoid(logits)
        cls = scores.argmax(axis=1)
        best = scores.max(axis=1)

        keep = best >= conf
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)
        boxes, cls, best = boxes[keep], cls[keep], best[keep]

        xyxy = _cxcywh_to_xyxy(boxes)
        h, w = orig_hw
        if xyxy.max(initial=0.0) <= 2.0:  # normalized -> scale to image
            xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        else:  # input-pixel coords
            size = self.imgsz or 640
            xyxy = xyxy / size * np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        data = np.concatenate([xyxy, best[:, None], cls[:, None].astype(np.float32)], axis=1)
        order = np.argsort(best)[::-1][:max_det]
        return data[order].astype(np.float32)

    @staticmethod
    def _split_detr(outputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(logits[Q,C], boxes[Q,4])`` from the raw output dict."""
        boxes_arr = logits_arr = None
        for arr in outputs.values():
            a = np.asarray(arr)
            a = a[0] if a.ndim == 3 else a
            if a.ndim == 2 and a.shape[-1] == 4 and boxes_arr is None:
                boxes_arr = a
            elif a.ndim == 2:
                logits_arr = a
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
