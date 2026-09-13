"""Super-resolution -> upscaled image.

python examples/super_resolution.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model
from ovkit.image.ops import imwrite


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("super_resolution", src)

    imwrite("sr_out.png", r.plot())  # plot() returns the upscaled image
    print("upscaled -> sr_out.png")


if __name__ == "__main__":
    main()
