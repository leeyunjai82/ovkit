"""Noise suppression — clean up a noisy recording.

    python examples/denoise_audio.py noisy.wav clean.wav

ovkit reads the file (any rate, mono or stereo), streams it through the model's
recurrent state frame by frame, and hands back the denoised audio.
"""

from __future__ import annotations

import argparse

from ovkit import Model


def main() -> None:
    ap = argparse.ArgumentParser(description="Denoise a .wav with an OMZ model.")
    ap.add_argument("in_wav")
    ap.add_argument("out_wav", nargs="?", default="denoised.wav")
    ap.add_argument("--model", default="noise_suppression")
    ap.add_argument("--device", default="AUTO")
    args = ap.parse_args()

    r = Model(args.model, device=args.device)(args.in_wav)[0]
    print(r.summary())  # 'denoised 4.2s of audio (16000 Hz)'
    r.save(args.out_wav)  # .wav saves the audio; .jpg would save the waveform
    print(f"saved -> {args.out_wav}")


if __name__ == "__main__":
    main()
