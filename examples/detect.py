"""Object detection -> boxes drawn on the image.

python examples/detect.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("detect")(src, conf=0.25)[0]
    for x1, y1, x2, y2, conf, cls in r.boxes.data:
        print(f"{r.name_for(int(cls)):12s} {conf:.2f}  [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    r.save("detect_out.jpg")
    print(f"{len(r.boxes)} objects -> detect_out.jpg")


if __name__ == "__main__":
    main()
