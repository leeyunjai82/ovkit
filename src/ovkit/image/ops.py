"""Image utilities — not models. Resize, letterbox, color, crop, zoom.

Everything here works on plain ``numpy`` HWC arrays (OpenCV's BGR convention by
default) so the rest of ovkit never has to depend on a particular image type.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def imread(path: str | Path) -> np.ndarray:
    """Read an image file into an HWC BGR ``uint8`` array.

    Reads the bytes with Python and decodes them with OpenCV, rather than
    handing OpenCV the path. ``cv2.imread`` passes the name to the C++ runtime,
    which on Windows uses the system code page — so ``철수.png`` arrived as
    ``泥좎닔.png`` and the read failed. A roster folder full of Korean names is
    exactly what ``Model("출석체크")`` is for, so the path never reaches OpenCV.
    """
    import cv2

    target = Path(path)
    try:
        data = np.fromfile(target, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read image: {path}") from exc
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    """Write an HWC BGR array to ``path``.

    Encoded by OpenCV, written by Python — the same reason as :func:`imread`.
    """
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, img)
    if not ok:
        raise OSError(f"Could not write image: {path}")
    try:
        buffer.tofile(target)
    except OSError as exc:
        raise OSError(f"Could not write image: {path}") from exc


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Swap the channel order between BGR and RGB (works both ways)."""
    return img[..., ::-1]


def resize(img: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Resize to ``size`` (a square edge length or ``(w, h)``)."""
    import cv2

    w, h = (size, size) if isinstance(size, int) else size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def letterbox(
    img: np.ndarray,
    size: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio and pad to ``size``.

    Returns ``(padded_image, scale, (pad_x, pad_y))`` so detections can be
    mapped back to the original image coordinates.
    """
    import cv2

    h0, w0 = img.shape[:2]
    tw, th = (size, size) if isinstance(size, int) else size
    scale = min(tw / w0, th / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, img.shape[2]), color, dtype=img.dtype)
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return out, scale, (pad_x, pad_y)


def crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop an ``xyxy`` box (floats are clamped to image bounds)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    return img[y1:y2, x1:x2]


def zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale an image by ``factor`` (``>1`` enlarges, ``<1`` shrinks)."""
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(
        img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_LINEAR
    )
