"""Driver monitoring — warn when the eyes stay shut.

python examples/drowsiness.py            # webcam
python examples/drowsiness.py drive.mp4
"""

from __future__ import annotations

import sys

import cv2

from ovkit import Model


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    monitor = Model("drowsiness", seconds=1.0)  # a blink is shorter than this

    for r in monitor.predict(source, stream=True):
        print(r.summary())  # 'awake (0.97)' ... 'EYES CLOSED 1.4s — drowsy'
        cv2.imshow("ovkit drowsiness (q to quit)", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
