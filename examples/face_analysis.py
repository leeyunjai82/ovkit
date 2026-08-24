"""Faces with age, gender and emotion — detection and three models, one call.

python examples/face_analysis.py path/to/group.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("face_analyze")(src)[0]
    print(r.summary())  # 2 faces: age 31 · male 98% · happy 92%, ...
    for i, label in enumerate(r.labels or []):
        x1, y1, x2, y2 = r.boxes.xyxy[i]
        print(f"  [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]  {label}")
    r.save("faces.jpg")
    print("-> faces.jpg")

    # Add head pose and landmarks (two more models, downloaded on first use):
    #   Model("face_analyze", attributes=("age_gender", "emotion", "head_pose"))
    # Or run one model on a cropped face: Model("emotion"), Model("age_gender").


if __name__ == "__main__":
    main()
