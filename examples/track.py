"""Tracking — the same object keeps its id from frame to frame.

python examples/track.py path/to/video.mp4      # or no argument for the webcam
"""

from __future__ import annotations

import sys

import cv2

from ovkit import Model


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else 0  # 0 = webcam
    tracker = Model("track")  # detector + id association

    for r in tracker.predict(source, stream=True):
        print(r.summary())  # 2x person (#1, #4)
        cv2.imshow("ovkit track (q to quit)", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
