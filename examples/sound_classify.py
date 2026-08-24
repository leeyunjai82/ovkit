"""Sound classification — what is making this noise?

python examples/sound_classify.py path/to/clip.wav
"""

from __future__ import annotations

import sys

from ovkit import Model


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "clip.wav"
    r = Model("sound_classification")(src)[0]
    print(r.summary())  # e.g. 'dog 0.82'

    for i in r.probs.top5:
        print(f"  {r.name_for(int(i)):20s} {r.probs.data[int(i)]:.3f}")
    r.save("sound.jpg")  # the waveform with the answer on it
    print("-> sound.jpg")


if __name__ == "__main__":
    main()
