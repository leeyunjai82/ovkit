"""Text-recognition adapter — greedy CTC decoding.

OMZ text-recognition models (``text-recognition-*``, ``handwritten-*``,
``license-plate-recognition-*``) emit per-timestep symbol logits, typically
``[T, 1, C]`` (or ``[1, T, C]``). Greedy CTC: argmax per step, collapse repeats,
drop blanks. The decoded string is stored in :attr:`Results.text` and the raw
logits stay available in :attr:`Results.tensors`.

The symbol set defaults to OMZ ``text-recognition-0012``'s alphabet
(``0-9a-z#`` with ``#`` as blank); override per model via the manifest
``postprocess.symbols`` / ``postprocess.blank`` fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results
from .base import BaseAdapter

#: Default symbol table (OMZ text-recognition-0012): blank ('#') last.
_DEFAULT_SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyz#"


class OCRAdapter(BaseAdapter):
    """Adapter for text recognition (greedy CTC)."""

    task = "optical_character_recognition"

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)

        logits = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        text = self._ctc_greedy(logits)
        res = Results(image, task=self.task, names=self.names, tensors=outputs)
        res.text = text
        return res

    def _ctc_greedy(self, logits: np.ndarray) -> str:
        """Greedy CTC decode of ``[T, 1, C]`` / ``[1, T, C]`` / ``[T, C]`` logits."""
        a = logits
        if a.ndim == 3:  # [T,1,C] or [1,T,C] -> [T,C]
            a = a[:, 0, :] if a.shape[1] == 1 else a[0]
        if a.ndim != 2:
            return ""

        symbols = str(self.post.get("symbols", _DEFAULT_SYMBOLS))
        blank = self.post.get("blank")
        blank_idx = int(blank) if blank is not None else len(symbols) - 1

        ids = a.argmax(axis=1)
        chars: list[str] = []
        prev = -1
        for i in ids:
            i = int(i)
            if i != prev and i != blank_idx and i < len(symbols):
                chars.append(symbols[i])
            prev = i
        return "".join(chars)
