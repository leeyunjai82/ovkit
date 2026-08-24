"""OCR — find the text in a picture and read it (detection + recognition).

python examples/ocr.py path/to/sign.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("read_text")(src)[0]
    print(r.text)  # every word, in reading order
    for i, word in enumerate(r.labels or []):
        x1, y1, x2, y2 = r.boxes.xyxy[i]
        print(f"  [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]  {word!r}")
    r.save("read_text.jpg")
    print("-> read_text.jpg")


if __name__ == "__main__":
    main()
