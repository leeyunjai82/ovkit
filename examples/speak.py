"""Read text out loud — four networks chained into one WAV file.

python examples/speak.py                                 # a Korean greeting
python examples/speak.py "Good morning, everyone."       # your own sentence
python examples/speak.py 원고.txt M3                      # a file, in a voice
"""

from __future__ import annotations

import sys

from ovkit import Model

DEFAULT = "안녕하세요. 오늘은 기계 학습에 대해 배워 보겠습니다."


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    voice = sys.argv[2] if len(sys.argv) > 2 else "F1"

    r = Model("읽어주기", text, voice=voice)
    samples, rate = r.audio
    print(f"{len(samples) / rate:.1f}초 · {rate} Hz · 목소리 {voice}")

    out = r.save("speak.wav")
    print(f"-> {out}")

    # The waveform is a picture like any other, so the usual things work.
    r.save("speak.png")
    print("-> speak.png (파형)")


if __name__ == "__main__":
    main()
