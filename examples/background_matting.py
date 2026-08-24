"""Background matting -> cut the subject out of a photo.

Needs TWO images: the frame with the subject, and a shot of the *same scene
without them* (same camera, same position). The model compares them, so a
single image cannot drive it.

    python examples/background_matting.py frame.jpg background.jpg [out.png]

Writes a cutout with a transparent background.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from ovkit import Model


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    frame_path, bg_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "cutout.png"

    frame = cv2.imread(frame_path)
    bg = cv2.imread(bg_path)
    if frame is None or bg is None:
        raise SystemExit(f"could not read {frame_path} or {bg_path}")
    if frame.shape != bg.shape:
        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

    model = Model("background_matting_mobilenetv2")
    names = [n for n, _, _ in model.inputs]  # ('src', 'bgr')

    def as_tensor(img: np.ndarray) -> np.ndarray:
        # The model takes NCHW RGB in [0, 1]; its spatial dims are dynamic, so
        # the image size is ours to choose — keep the original.
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None]

    out = model.infer({names[0]: as_tensor(frame), names[1]: as_tensor(bg)})

    # Outputs are pha (alpha, 1 channel) and fgr (foreground, 3 channels).
    alpha = next(np.asarray(v) for v in out.values() if np.asarray(v).shape[1] == 1)
    alpha = np.clip(alpha[0, 0], 0.0, 1.0)
    alpha = cv2.resize(alpha, (frame.shape[1], frame.shape[0]))

    rgba = np.dstack([frame, (alpha * 255).astype(np.uint8)])
    cv2.imwrite(out_path, rgba)
    covered = float((alpha > 0.5).mean()) * 100
    print(f"subject covers {covered:.1f}% of the frame -> {out_path}")


if __name__ == "__main__":
    main()
