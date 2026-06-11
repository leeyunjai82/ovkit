"""OCR = text detection + text recognition, assembled on top of recognize/.

INTERFACE STUB for v0. The structure (detect text regions, then recognize each)
is fixed; implementations are TODO.
"""

from __future__ import annotations

from typing import Any


class OCR:
    """Two-stage OCR pipeline (not yet implemented)."""

    def __init__(self, detector: Any = None, recognizer: Any = None, device: str = "AUTO") -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.device = device

    def __call__(self, image: Any) -> Any:
        raise NotImplementedError(
            "OCR is not implemented in v0. Planned: compose a text detector + a text "
            "recognizer over ovkit.recognize. TODO."
        )
