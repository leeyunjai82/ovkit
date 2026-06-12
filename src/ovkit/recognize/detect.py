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


def _softmax_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# Standard YOLOv2 (VOC) anchors in grid-cell units; override via manifest.
_YOLOV2_ANCHORS = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
]


def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45, max_det: int = 300) -> list:
    """Greedy non-max suppression; returns kept indices (highest score first)."""
    if len(scores) == 0:
        return []
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


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
        ih, iw = self.model_input_hw(backend)
        h, w = image.shape[:2]

        if fmt == "detr":
            rgb = bool(self.pre.get("rgb", True))  # DETR: RGB [0,1]
            feed = self.preprocess(image, (ih, iw), rgb=rgb, scale=self.pre.get("scale", 255.0))
            outputs = backend.infer(feed)
            boxes = self._decode_detr(outputs, (h, w), conf=conf, max_det=max_det)
            names = self.names or class_names(self._classes_key or "coco80")
        else:  # ssd / boxes_labels: OMZ raw BGR
            rgb = bool(self.pre.get("rgb", False))
            feed = self.preprocess(image, (ih, iw), rgb=rgb, scale=self.pre.get("scale", 1.0))
            outputs = backend.infer(feed)
            if fmt == "ssd":
                boxes = self._decode_ssd(outputs, (h, w), conf=conf, max_det=max_det)
            elif fmt == "yolo_v2":
                boxes = self._decode_yolo(outputs, (h, w), conf=conf, max_det=max_det)
            else:
                boxes = self._decode_boxes_labels(
                    outputs, (h, w), (ih, iw), conf=conf, max_det=max_det
                )
            names = self.names or class_names(self._classes_key)

        return Results(image, task=self.task, names=names, boxes=Boxes(boxes))

    def _format(self, backend: Backend) -> str:
        """Pick the decode format from the manifest or the output signature."""
        if self.post.get("format"):
            return str(self.post["format"])
        shapes = [s for _, s in backend.output_signatures()]
        # SSD DetectionOutput: a single [.., N, 7] tensor.
        if len(shapes) == 1 and len(shapes[0]) >= 2 and shapes[0][-1] == 7:
            return "ssd"
        # boxes [N, 5] (+ optional labels [N]): any output ending in 5.
        if any(len(s) >= 2 and s[-1] == 5 for s in shapes):
            return "boxes_labels"
        # A single 4-D [1, A*(5+C), H, W] tensor on a detect model -> YOLO region
        # (segmentation maps route to SegmentAdapter, so this is unambiguous here).
        if len(shapes) == 1 and len(shapes[0]) == 4:
            return "yolo_v2"
        return "detr"

    # -- boxes+labels decode (OMZ -0200/ATSS-style detectors) ---------------

    @staticmethod
    def _decode_boxes_labels(
        outputs: dict[str, np.ndarray],
        orig_hw: tuple[int, int],
        in_hw: tuple[int, int],
        *,
        conf: float,
        max_det: int,
    ) -> np.ndarray:
        boxes_arr = labels_arr = None
        for v in outputs.values():
            a = np.asarray(v)
            if a.ndim >= 2 and a.shape[-1] == 5:
                boxes_arr = a.reshape(-1, 5)
            elif a.ndim == 1:
                labels_arr = a.reshape(-1)
            elif a.ndim == 2 and a.shape[-1] == 1:
                labels_arr = a.reshape(-1)
        if boxes_arr is None:
            return np.zeros((0, 6), dtype=np.float32)

        keep = boxes_arr[:, 4] >= conf
        boxes_arr = boxes_arr[keep]
        if boxes_arr.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        if labels_arr is not None and labels_arr.shape[0] == keep.shape[0]:
            labels = labels_arr[keep].astype(np.float32)
        else:
            labels = np.zeros(boxes_arr.shape[0], dtype=np.float32)

        scores = boxes_arr[:, 4]
        xyxy = boxes_arr[:, :4].astype(np.float32)
        h, w = orig_hw
        ih, iw = in_hw
        if xyxy.max(initial=0.0) <= 2.0:  # normalized
            xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        else:  # input-pixel coords
            xyxy = xyxy / np.array([iw, ih, iw, ih], dtype=np.float32)
            xyxy = xyxy * np.array([w, h, w, h], dtype=np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)

        data = np.concatenate([xyxy, scores[:, None], labels[:, None]], axis=1)
        order = np.argsort(scores)[::-1][:max_det]
        return data[order].astype(np.float32)

    # -- YOLOv2 region decode ----------------------------------------------

    def _decode_yolo(
        self, outputs: dict[str, np.ndarray], orig_hw: tuple[int, int], *, conf: float, max_det: int
    ) -> np.ndarray:
        out = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        a = out[0] if out.ndim == 4 else out  # [Ch, H, W]
        if a.ndim != 3:
            return np.zeros((0, 6), dtype=np.float32)
        ch, gh, gw = a.shape
        anchors = self.post.get("anchors") or _YOLOV2_ANCHORS
        na = len(anchors)
        if na == 0 or ch % na != 0:
            return np.zeros((0, 6), dtype=np.float32)
        depth = ch // na
        ncls = depth - 5
        if ncls < 1:
            return np.zeros((0, 6), dtype=np.float32)

        a = a.reshape(na, depth, gh, gw)
        tx, ty, tw, th, to = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        cls = a[:, 5:]  # [na, ncls, gh, gw]

        gx = np.arange(gw, dtype=np.float32)[None, None, :]
        gy = np.arange(gh, dtype=np.float32)[None, :, None]
        cx = (_sigmoid(tx) + gx) / gw
        cy = (_sigmoid(ty) + gy) / gh
        aw = np.array([p[0] for p in anchors], dtype=np.float32)[:, None, None]
        ah = np.array([p[1] for p in anchors], dtype=np.float32)[:, None, None]
        bw = aw * np.exp(tw) / gw
        bh = ah * np.exp(th) / gh

        cls_sm = _softmax_axis(cls, axis=1)
        score = _sigmoid(to) * cls_sm.max(axis=1)  # [na, gh, gw]
        cls_id = cls_sm.argmax(axis=1)

        mask = score >= conf
        if not np.any(mask):
            return np.zeros((0, 6), dtype=np.float32)
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        sc, ids = score[mask], cls_id[mask].astype(np.float32)

        h, w = orig_hw
        xyxy = np.stack(
            [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1
        )
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)
        keep = _nms(xyxy, sc, max_det=max_det)
        data = np.concatenate([xyxy, sc[:, None], ids[:, None]], axis=1)
        return data[keep].astype(np.float32)

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
