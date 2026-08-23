"""Face analysis -> age + gender from a face crop.

python examples/face_analysis.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("age_gender")(src)[0]
    print(r.text)  # e.g. "age 31 · male 98%"
    # Other one-liners: Model("emotion"), Model("head_pose"), Model("face_landmarks")


if __name__ == "__main__":
    main()
