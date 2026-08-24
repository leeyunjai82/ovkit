"""Read, convert and write audio as plain float32 numpy arrays."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OVKitError

#: Sample formats ``wave`` reports, mapped to (numpy dtype, full-scale value).
_PCM = {1: (np.int8, 128.0), 2: (np.int16, 32768.0), 4: (np.int32, 2147483648.0)}


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average an ``(N, channels)`` array down to a 1-D mono signal."""
    arr = np.asarray(audio, dtype=np.float32)
    return arr.mean(axis=1) if arr.ndim == 2 and arr.shape[1] > 1 else arr.reshape(-1)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal to ``target_sr`` by linear interpolation.

    Linear interpolation is not a studio-grade resampler, but it is exact when
    the rates match, needs no extra dependency, and is well within the tolerance
    of the classifiers and ASR models ovkit serves.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == target_sr or arr.size == 0:
        return arr
    n_out = int(round(arr.size * target_sr / float(sr)))
    if n_out <= 1:
        return arr[:1].copy()
    src = np.linspace(0.0, arr.size - 1, num=n_out, dtype=np.float32)
    return np.interp(src, np.arange(arr.size, dtype=np.float32), arr).astype(np.float32)


def read_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read an audio file as ``(float32 mono in [-1, 1], sample_rate)``.

    ``.wav`` is read with the standard library. Other formats are read through
    ``soundfile`` when it is installed; otherwise the error says what to do
    rather than failing deep inside a decoder. ``target_sr`` resamples the
    result (and is what most models need, e.g. 16 kHz for speech).
    """
    src = Path(path)
    if src.suffix.lower() == ".wav":
        audio, sr = read_wav(src)
    else:
        audio, sr = _read_soundfile(src)
    if target_sr:
        audio = resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def read_wav(source: Any, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read WAV from a path or an open/byte stream — any depth, any channels.

    Uploads arrive as bytes and recordings arrive as stereo; both go through
    here so callers never re-implement PCM decoding.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with wave.open(source if hasattr(source, "read") else str(source), "rb") as wf:
        sr, width, channels = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    if width not in _PCM:
        raise OVKitError(
            f"{width * 8}-bit WAV is not supported (8/16/32-bit PCM is). "
            f"Re-encode it, e.g. ffmpeg -i in.wav -acodec pcm_s16le out.wav"
        )
    dtype, full_scale = _PCM[width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / full_scale
    if channels > 1:
        samples = samples[: samples.size // channels * channels].reshape(-1, channels)
    audio = to_mono(samples)
    if target_sr:
        return resample(audio, sr, target_sr), target_sr
    return audio, sr


def _read_soundfile(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise OVKitError(
            f"Reading '{path.suffix}' needs an extra decoder. Either convert to WAV "
            f"(ffmpeg -i {path.name} -ar 16000 -ac 1 out.wav) or install one: "
            f"pip install soundfile"
        ) from exc
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return to_mono(audio), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write a float32 mono signal to a 16-bit PCM ``.wav`` and return the path."""
    arr = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return Path(path)
