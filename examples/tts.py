#!/usr/bin/env python3
"""Text-to-speech via openvino-genai.

    pip install "ovkit[genai]"
    # point at a local OpenVINO-converted Text2Speech model directory:
    python examples/tts.py "Hello from ovkit" /path/to/tts-ov-model out.wav

(No TTS model is registered by default; pass a local model directory. Once a
permissive TTS repo is added to manifests/genai.yaml you can use its name.)
"""

from __future__ import annotations

import argparse
import wave

import numpy as np

from ovkit.genai import pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("model", help="registered name, or local model directory")
    ap.add_argument("out", nargs="?", default="speech.wav")
    ap.add_argument("--device", default="AUTO")
    args = ap.parse_args()

    tts = pipeline(args.model, device=args.device, pipeline_type="text2speech")
    result = tts.generate(args.text)
    audio = np.asarray(result.speeches[0] if hasattr(result, "speeches") else result).reshape(-1)

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(args.out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
