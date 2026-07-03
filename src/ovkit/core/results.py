"""Result containers — the user-facing output of every prediction.

A :class:`Results` object bundles the original image, the task, and whichever of
``boxes`` / ``masks`` / ``keypoints`` / ``probs`` the task produced, plus
convenience :meth:`plot` and :meth:`save` helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Deterministic, visually distinct palette for class colors (BGR).
_PALETTE = np.array(
    [
        [56, 56, 255],
        [151, 157, 255],
        [31, 112, 255],
        [29, 178, 255],
        [49, 210, 207],
        [10, 249, 72],
        [23, 204, 146],
        [134, 219, 61],
        [52, 147, 26],
        [187, 212, 0],
        [168, 153, 44],
        [255, 194, 0],
        [147, 69, 52],
        [255, 115, 100],
        [236, 24, 0],
        [255, 56, 132],
        [133, 0, 82],
        [255, 56, 203],
        [200, 149, 255],
        [199, 55, 255],
    ],
    dtype=np.uint8,
)


def _color(idx: int) -> tuple[int, int, int]:
    c = _PALETTE[idx % len(_PALETTE)]
    return int(c[0]), int(c[1]), int(c[2])


class Boxes:
    """Detection boxes in ``xyxy`` pixel coordinates with scores and classes.

    ``data`` is an ``(N, 6)`` array of ``[x1, y1, x2, y2, conf, cls]``.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, 6)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @property
    def xyxy(self) -> np.ndarray:
        """``(N, 4)`` boxes as ``[x1, y1, x2, y2]``."""
        return self.data[:, :4]

    @property
    def conf(self) -> np.ndarray:
        """``(N,)`` confidence scores."""
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        """``(N,)`` integer class ids (as float; cast as needed)."""
        return self.data[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        """``(N, 4)`` boxes as ``[cx, cy, w, h]``."""
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


class Masks:
    """Instance segmentation masks: ``(N, H, W)`` boolean/float array."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)

    def __len__(self) -> int:
        return len(self.data)


class Keypoints:
    """Pose keypoints: ``(N, K, 3)`` array of ``[x, y, confidence]``."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def xy(self) -> np.ndarray:
        return self.data[..., :2]

    @property
    def conf(self) -> np.ndarray:
        return self.data[..., 2]


class Probs:
    """Classification probabilities over the class table."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=np.float32).ravel()

    @property
    def top1(self) -> int:
        """Index of the highest-probability class."""
        return int(np.argmax(self.data))

    @property
    def top5(self) -> np.ndarray:
        """Indices of the five highest-probability classes (descending)."""
        return np.argsort(self.data)[::-1][:5]


class Results:
    """Container for a single image's prediction.

    Attributes
    ----------
    orig_img:
        The original HWC BGR image.
    task:
        ``"detect"`` / ``"classify"`` / ``"segment"`` / ``"pose"``.
    names:
        ``{id: class_name}`` mapping.
    boxes, masks, keypoints, probs:
        Populated according to ``task``; the others are ``None``.
    path:
        Source path, when the prediction came from a file.
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        task: str,
        names: dict[int, str] | None = None,
        *,
        boxes: Boxes | None = None,
        masks: Masks | None = None,
        keypoints: Keypoints | None = None,
        probs: Probs | None = None,
        tensors: dict[str, np.ndarray] | None = None,
        path: str | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.task = task
        self.names = names or {}
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        #: Raw ``{output_name: ndarray}`` for tasks without a typed decoder
        #: (e.g. super-resolution, embeddings). ``None`` for vision tasks.
        self.tensors = tensors
        #: Decoded text for OCR/text-recognition tasks (otherwise ``None``).
        self.text: str | None = None
        self.path = path

    def __len__(self) -> int:
        for obj in (self.boxes, self.masks, self.keypoints):
            if obj is not None:
                return len(obj)
        if self.probs is not None:
            return 1
        return len(self.tensors) if self.tensors is not None else 0

    def __repr__(self) -> str:
        h, w = self.orig_img.shape[:2]
        n = len(self)
        return f"Results(task={self.task!r}, image={w}x{h}, items={n})"

    def name_for(self, cls_id: int) -> str:
        return self.names.get(int(cls_id), str(int(cls_id)))

    # -- visualization ------------------------------------------------------

    def plot(self, line_width: int = 2, font_scale: float = 0.5) -> np.ndarray:
        """Render detections/keypoints onto a copy of the image and return it."""
        import cv2

        img = self.orig_img.copy()

        if self.masks is not None and len(self.masks):
            data = self.masks.data
            if data.ndim == 3 and data.shape[0] > 1:  # (N, H, W) instance masks
                if data.shape[1:] == img.shape[:2]:
                    overlay = img.copy()
                    for i in range(data.shape[0]):
                        overlay[data[i] > 0] = _color(i)
                    img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            else:  # (1, H, W) / (H, W) semantic class map
                cmap = data[0] if data.ndim == 3 else data
                if cmap.shape[:2] == img.shape[:2]:
                    overlay = img.copy()
                    for cls_id in np.unique(cmap):
                        if int(cls_id) == 0:  # background
                            continue
                        overlay[cmap == cls_id] = _color(int(cls_id))
                    img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        if self.boxes is not None:
            for x1, y1, x2, y2, conf, cls in self.boxes.data:
                c = _color(int(cls))
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, c, line_width)
                label = f"{self.name_for(int(cls))} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), c, -1)
                cv2.putText(
                    img,
                    label,
                    (p1[0], p1[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self.keypoints is not None:
            for inst in self.keypoints.data:
                for x, y, kc in inst:
                    if kc > 0:
                        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

        if self.probs is not None:
            top = self.probs.top1
            label = f"{self.name_for(top)} {self.probs.data[top]:.2f}"
            cv2.putText(
                img,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Generic (raw-tensor) results: overlay output info so something shows.
        if self.text:
            cv2.putText(
                img,
                f'"{self.text}"',
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif self.tensors is not None and not any(
            (self.boxes, self.masks, self.keypoints, self.probs)
        ):
            # If an output IS an image (super-resolution, enhancement, ...),
            # show that image instead of tensor metadata.
            rendered = self._image_output()
            if rendered is not None:
                return rendered
            y = 30
            for name, arr in list(self.tensors.items())[:5]:
                a = np.asarray(arr)
                txt = f"{name}: {tuple(a.shape)}"
                if a.size and a.ndim <= 2:
                    flat = a.ravel()
                    txt += f"  argmax={int(flat.argmax())} ({float(flat.max()):.2f})"
                cv2.putText(
                    img,
                    txt[:60],
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28

        return img

    def _image_output(self) -> np.ndarray | None:
        """Return the first image-like output tensor as a displayable image.

        Matches ``[1, C, H, W]`` with ``C`` in {1, 3} and a real spatial size;
        float outputs in ``[0, 1]`` are scaled to ``[0, 255]``.
        """
        if not self.tensors:
            return None
        for arr in self.tensors.values():
            a = np.asarray(arr)
            if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] in (1, 3):
                if a.shape[2] < 32 or a.shape[3] < 32:
                    continue
                hwc = np.transpose(a[0], (1, 2, 0)).astype(np.float32)
                if hwc.shape[2] == 1:
                    hwc = np.repeat(hwc, 3, axis=2)
                if float(hwc.max(initial=0.0)) <= 2.0:  # [0,1]-range float output
                    hwc = hwc * 255.0
                return np.clip(hwc, 0, 255).astype(np.uint8)
        return None

    def save(self, path: str | Path) -> Path:
        """Render with :meth:`plot` and write the result to ``path``."""
        from ..image.ops import imwrite

        out = self.plot()
        imwrite(path, out)
        return Path(path)
