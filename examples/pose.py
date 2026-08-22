"""Human pose -> keypoints drawn on the image.

python examples/pose.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("pose")(src)[0]
    print("keypoints:", r.keypoints.data.shape)  # (people, K, [x, y, conf])
    r.save("pose_out.jpg")
    print("skeleton -> pose_out.jpg")


if __name__ == "__main__":
    main()
