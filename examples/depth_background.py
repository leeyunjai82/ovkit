"""How far away things are, and cutting the subject out of the background.

python examples/depth_background.py depth room.jpg
python examples/depth_background.py cutout portrait.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "depth"
    src = sys.argv[2] if len(sys.argv) > 2 else "photo.jpg"

    if what == "depth":
        r = Model("depth", src)  # or Model("거리재기", src)
        print(r)  # nearest: bottom-left · 34% of the frame is close
        r.save("depth.jpg")  # the colourised map
        print("-> depth.jpg   (r.tensors['depth'] is the 0..1 map)")
    else:
        r = Model("remove_background", src)  # or Model("배경지우기", src)
        print(r)  # subject covers 41% of the frame
        r.save("cutout.png")  # .png keeps the transparency; .jpg would not
        print("-> cutout.png")


if __name__ == "__main__":
    main()
