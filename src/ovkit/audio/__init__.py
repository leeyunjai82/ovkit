"""Audio I/O helpers: read a file into the array a model actually wants.

Models want float32 mono at a fixed sample rate; files are int16 stereo at
whatever rate the recorder used. These helpers close that gap so callers never
have to touch :mod:`wave` or write their own resampler.
"""

from __future__ import annotations

from .ops import read_audio, read_wav, resample, to_mono, write_wav
from .plot import waveform

__all__ = ["read_audio", "read_wav", "resample", "to_mono", "waveform", "write_wav"]
