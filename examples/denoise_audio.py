#!/usr/bin/env python3
"""Run an OMZ noise-suppression model on a .wav (stateful streaming).

These models take an audio frame plus several recurrent **state** tensors and
emit a denoised frame plus updated states, processed frame by frame. This
example wires that loop generically:

    python examples/denoise_audio.py noise_suppression_poconetlike_0001 in.wav out.wav

Best-effort: input/state pairing follows the OMZ convention (audio input named
``input`` / output ``output``; states paired in sorted order). Mono 16 kHz wav.
"""

from __future__ import annotations

import argparse
import wave

import numpy as np

from ovkit import Model


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="Denoise a .wav with an OMZ model.")
    ap.add_argument("model")
    ap.add_argument("in_wav")
    ap.add_argument("out_wav", nargs="?", default="denoised.wav")
    ap.add_argument("--device", default="AUTO")
    args = ap.parse_args()

    model = Model(args.model, device=args.device)
    info = {name: tuple(shape) for name, shape, _ in model.inputs}

    # Audio input: the one named "input", else the largest input tensor.
    audio_name = "input" if "input" in info else max(info, key=lambda n: int(np.prod(info[n])))
    patch = int(info[audio_name][-1])
    state_in = sorted(n for n in info if n != audio_name)
    states = {n: np.zeros(info[n], dtype=np.float32) for n in state_in}

    audio, sr = read_wav(args.in_wav)
    print(f"loaded {len(audio)} samples @ {sr} Hz; frame={patch}, states={len(state_in)}")

    out_audio: list[np.ndarray] = []
    audio_out: str | None = None
    state_out: list[str] = []
    for i in range(0, len(audio), patch):
        chunk = audio[i : i + patch]
        if len(chunk) < patch:
            chunk = np.pad(chunk, (0, patch - len(chunk)))
        outputs = model.infer({audio_name: chunk[None].astype(np.float32), **states})

        if audio_out is None:  # discover output names once
            audio_out = "output" if "output" in outputs else _match_audio(outputs, patch)
            state_out = sorted(k for k in outputs if k != audio_out)
            print(f"audio out: {audio_out}; state outs: {len(state_out)}")
        out_audio.append(np.asarray(outputs[audio_out]).reshape(-1)[:patch])
        states = {n: np.asarray(outputs[o]) for n, o in zip(state_in, state_out, strict=False)}

    write_wav(args.out_wav, np.concatenate(out_audio)[: len(audio)], sr)
    print(f"saved -> {args.out_wav}")


def _match_audio(outputs: dict, patch: int) -> str:
    for name, arr in outputs.items():
        if int(np.asarray(arr).size) % patch == 0 and np.asarray(arr).size >= patch:
            return name
    return next(iter(outputs))


if __name__ == "__main__":
    main()
