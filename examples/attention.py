"""What is the person looking at? (gaze cast into detected objects)

python examples/attention.py desk.jpg      # or no argument for the webcam
"""

from __future__ import annotations

import sys

import cv2

from ovkit import Model


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    watcher = Model("attention")

    for r in watcher.predict(source, stream=True):
        print(r.summary())  # '1 person looking at: laptop'
        cv2.imshow("ovkit attention (q to quit)", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
