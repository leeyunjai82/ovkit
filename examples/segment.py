"""Semantic segmentation -> colored class-map overlay.

python examples/segment.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("segment")(src)[0]
    print("class map:", r.masks.data.shape)
    r.save("segment_out.jpg")
    print("overlay -> segment_out.jpg")


if __name__ == "__main__":
    main()
