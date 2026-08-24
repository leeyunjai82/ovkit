"""Number-plate reading — the plate, and the car it belongs to.

python examples/read_plate.py gate.jpg
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
    r = Model("read_plate")(src)[0]
    print(r.summary())  # '2 vehicles: black car — 12GA3456, white van — 34NA5678'
    for box in r.to_dict().get("boxes", []):
        print(f"  {box['label']:14s} {box.get('text', '')}")
    r.save("plates.jpg")
    print("-> plates.jpg")


if __name__ == "__main__":
    main()
