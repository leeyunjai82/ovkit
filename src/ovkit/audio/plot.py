"""Draw a waveform, so an audio result has something to show."""

from __future__ import annotations

import numpy as np


def waveform(
    audio: np.ndarray,
    sr: int = 16_000,
    width: int = 720,
    height: int = 200,
    color: tuple[int, int, int] = (200, 190, 120),
) -> np.ndarray:
    """Render a mono signal as a BGR image.

    Audio results carry this as their image so ``plot()``, ``save()`` and the
    web demo work the same way they do for a picture.
    """
    import cv2

    img = np.full((height, width, 3), 24, np.uint8)
    samples = np.asarray(audio, np.float32).reshape(-1)
    mid = height // 2
    cv2.line(img, (0, mid), (width, mid), (60, 60, 60), 1)
    if samples.size:
        # One vertical bar per column: the min/max of the samples it covers.
        edges = np.linspace(0, samples.size, width + 1).astype(int)
        peak = max(float(np.abs(samples).max()), 1e-6)
        for x in range(width):
            chunk = samples[edges[x] : max(edges[x + 1], edges[x] + 1)]
            if not chunk.size:
                continue
            lo, hi = float(chunk.min()) / peak, float(chunk.max()) / peak
            cv2.line(img, (x, mid - int(hi * mid * 0.9)), (x, mid - int(lo * mid * 0.9)), color, 1)
    seconds = samples.size / float(sr or 1)
    cv2.putText(
        img,
        f"{seconds:.1f}s @ {sr} Hz",
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    return img
