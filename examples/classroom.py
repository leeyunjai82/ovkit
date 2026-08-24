"""The classroom four, in one file — pick with the first argument.

python examples/classroom.py count desk.jpg
python examples/classroom.py posture                 # webcam
python examples/classroom.py exercise squat          # webcam
python examples/classroom.py attendance class_photos/
"""

from __future__ import annotations

import sys

import cv2

from ovkit import Model


def live(pipe) -> None:
    for r in pipe.predict(0, stream=True):
        print(r.summary())
        cv2.imshow("ovkit (q to quit)", r.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "count"
    if what == "count":
        print(Model("count", sys.argv[2] if len(sys.argv) > 2 else "desk.jpg").summary())
    elif what == "posture":
        live(Model("posture"))  # 목이 25° 넘게 5초 → 경고
    elif what == "exercise":
        live(Model("exercise", kind=sys.argv[2] if len(sys.argv) > 2 else "squat"))
    elif what == "attendance":
        roll = Model("attendance", roster=sys.argv[2])
        live(roll)
        print(roll.save_csv("roll.csv"), "->", roll.absent, "absent")
    else:
        print(__doc__.strip())


if __name__ == "__main__":
    main()
