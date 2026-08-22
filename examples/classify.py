"""Image classification -> top-5 classes.

python examples/classify.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("classify")(src)[0]
    for i in r.probs.top5:
        print(f"{r.name_for(int(i)):20s} {r.probs.data[int(i)]:.3f}")


if __name__ == "__main__":
    main()
