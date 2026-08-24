"""Audio adapters — run a sound model straight from a file.

Audio models need framing, not an image: the file is whatever the recorder
produced, and the model wants a fixed number of float32 samples at a fixed rate.
These adapters close that gap, so::

    Model("sound_classification")("clip.wav")   ->  'dog 0.82'
    Model("noise_suppression")("noisy.wav")     ->  denoised audio, r.save("clean.wav")

is all a caller writes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import waveform
from ..core.backend import Backend
from ..core.constants import class_names
from ..core.results import Probs, Results

#: What OMZ audio models are trained at, when the IR does not say.
DEFAULT_SR = 16_000


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def window_length(backend: Backend, fallback: int = DEFAULT_SR) -> int:
    """Samples the model takes per inference (the last static input dim)."""
    length = backend.input_shape[-1] if backend.input_shape else -1
    return int(length) if length and length > 1 else fallback


def frames(audio: np.ndarray, length: int) -> list[np.ndarray]:
    """Split a signal into ``length``-sample windows, padding the last one.

    Classifying a ten-second clip means running the model on every window, not
    truncating to the first one and calling that the answer.
    """
    samples = np.asarray(audio, np.float32).reshape(-1)
    if samples.size == 0:
        return [np.zeros(length, np.float32)]
    out = []
    for start in range(0, samples.size, length):
        chunk = samples[start : start + length]
        if chunk.size < length:
            chunk = np.pad(chunk, (0, length - chunk.size))
        out.append(chunk.astype(np.float32))
    return out


class SoundClassifier:
    """Classify a whole clip by averaging the model over every window."""

    def __init__(self, classes: str | None = None, names: dict[int, str] | None = None) -> None:
        self.classes = classes
        self.names = names or {}

    def run(self, backend: Backend, audio: np.ndarray, sr: int) -> Results:
        length = window_length(backend)
        shape = backend.input_shape
        scores = None
        for chunk in frames(audio, length):
            feed = chunk.reshape([d if d and d > 0 else 1 for d in shape] or [1, 1, 1, length])
            raw = np.asarray(next(iter(backend.infer(feed).values()))).reshape(-1)
            probs = _softmax(raw.astype(np.float32))
            scores = probs if scores is None else scores + probs
        scores = (scores / max(1, len(frames(audio, length)))).astype(np.float32)

        names = self.names or class_names(self.classes, scores.size)
        result = Results(waveform(audio, sr), task="sound_classification", names=names)
        result.probs = Probs(scores)
        top = int(np.argmax(scores))
        result.text = f"{names.get(top, f'class_{top}')} {float(scores[top]):.2f}"
        result.audio = (audio, sr)
        return result


class Denoiser:
    """Speech noise suppression: stream the clip through the model's state."""

    def run(self, model: Any, audio: np.ndarray, sr: int) -> Results:
        shapes = {name: tuple(shape) for name, shape, _dtype in model.inputs}
        # The audio input is the big one; everything else is recurrent state.
        audio_name = "input" if "input" in shapes else max(shapes, key=lambda n: _size(shapes[n]))
        patch = int(shapes[audio_name][-1])
        state_inputs = sorted(n for n in shapes if n != audio_name)
        state = {n: np.zeros([max(d, 1) for d in shapes[n]], np.float32) for n in state_inputs}

        cleaned: list[np.ndarray] = []
        audio_out, state_outputs = None, []
        for start in range(0, max(len(audio), 1), patch):
            chunk = audio[start : start + patch]
            chunk = np.pad(chunk, (0, patch - len(chunk))) if len(chunk) < patch else chunk
            out = model.infer({audio_name: chunk[None].astype(np.float32), **state})
            if audio_out is None:
                audio_out = "output" if "output" in out else next(iter(out))
                state_outputs = sorted(k for k in out if k != audio_out)
            cleaned.append(np.asarray(out[audio_out]).reshape(-1)[:patch])
            state = {
                n: np.asarray(out[o]) for n, o in zip(state_inputs, state_outputs, strict=False)
            }

        signal = np.concatenate(cleaned)[: len(audio)] if cleaned else audio
        result = Results(waveform(signal, sr), task="noise_suppression")
        result.audio = (signal, sr)
        result.text = f"denoised {len(signal) / sr:.1f}s of audio ({sr} Hz)"
        return result


def _size(shape: tuple[int, ...]) -> int:
    return int(np.prod([max(d, 1) for d in shape]))
