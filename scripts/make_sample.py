#!/usr/bin/env python3
"""Write the sample image the tests read.

    python scripts/make_sample.py

Four flat quadrants in known BGR values, plus a thin border. Each quadrant is a
different colour so a test can assert on one pixel and a person can see a wrong
decode at a glance — swapped channels, a rotation, a half-read file.

Generated rather than photographed, deliberately: a picture of a real scene
would be someone's to licence, and megabytes of it would live in this
repository forever. Capability checks that need a real photograph get one at
run time (``scripts/verify_capabilities.py``).

Re-run this only if the sample needs to change; the tests assert on the values
below, so changing them means changing those tests too.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

H, W = 64, 96

#: ``(slice, slice) -> BGR``. The names are what a reader sees, not what OpenCV
#: stores: OpenCV is BGR, so "red" is ``(32, 32, 200)``.
QUADRANTS = {
    "red": ((slice(0, H // 2), slice(0, W // 2)), (32, 32, 200)),
    "green": ((slice(0, H // 2), slice(W // 2, W)), (32, 200, 32)),
    "blue": ((slice(H // 2, H), slice(0, W // 2)), (200, 32, 32)),
    "white": ((slice(H // 2, H), slice(W // 2, W)), (220, 220, 220)),
}


def build() -> np.ndarray:
    import cv2

    img = np.zeros((H, W, 3), np.uint8)
    for where, colour in QUADRANTS.values():
        img[where] = colour
    cv2.rectangle(img, (8, 8), (W - 9, H - 9), (0, 0, 0), 1)
    return img


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/assets/sample.png")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", build())
    if not ok:
        raise SystemExit("could not encode the sample")
    # Written the way ovkit writes images, so this script exercises the same
    # path the package does — including on a machine whose code page is cp949.
    buffer.tofile(out)
    print(f"{out}  {out.stat().st_size:,} bytes  {H}x{W}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
