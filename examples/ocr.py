"""Text recognition -> read a cropped word/line image.

python examples/ocr.py path/to/image.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("text_recognition")(src)[0]
    print("text:", r.text)
    # Find text regions first with Model("text_detection"), then crop + read.


if __name__ == "__main__":
    main()
