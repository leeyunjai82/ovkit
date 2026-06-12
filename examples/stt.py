#!/usr/bin/env python3
"""Speech-to-text (Whisper) via openvino-genai.

    pip install "ovkit[genai]"
    python examples/stt.py audio.wav

Expects mono 16 kHz wav. The model is downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import argparse
import wave

import numpy as np

from ovkit.genai import pipeline


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != 16000:
            print(f"warning: {wf.getframerate()} Hz (Whisper expects 16 kHz mono)")
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="whisper_base")
    ap.add_argument("--device", default="AUTO")
    args = ap.parse_args()

    stt = pipeline(args.model, device=args.device)
    print(stt.generate(read_wav(args.audio)))


if __name__ == "__main__":
    main()
