"""Face matching — build a small gallery, then name who is in a picture.

python examples/face_match.py yunjai.jpg dana.jpg query.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from ovkit import Model


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return
    *known, query = sys.argv[1:]

    matcher = Model("face_match")  # embedding model + cosine matching
    for path in known:
        matcher.add(Path(path).stem, path)  # the file name becomes the label
    print(f"gallery: {list(matcher.gallery)}")

    hit = matcher.who(query)
    print(f"{Path(query).name}: {hit[0]} ({hit[1]:.2f})" if hit else "nobody in the gallery")


if __name__ == "__main__":
    main()
