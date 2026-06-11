"""Face analysis pipeline — INTERFACE STUB.

TODO(v0+): assemble detection + landmarks + recognition + head-pose + emotion +
age/gender + anti-spoofing from Apache-2.0 OMZ models on the ovkit mirror. If a
required model is absent from the mirror, raise a clear "needs upload to mirror"
error (see :class:`ovkit.core.errors.MirrorMissingError`).
"""

from __future__ import annotations

from typing import Any


class FaceAnalyzer:
    """Multi-task face analyzer (not yet implemented)."""

    def __init__(self, device: str = "AUTO", **kwargs: Any) -> None:
        self.device = device
        raise NotImplementedError(
            "FaceAnalyzer is not implemented in v0. Planned: Apache-2.0 OMZ models "
            "served from the ovkit HF mirror (leeyunjai/ovkit-models). "
            "InsightFace pretrained weights are intentionally NOT used (non-commercial)."
        )

    def __call__(self, image: Any) -> Any:  # pragma: no cover - stub
        raise NotImplementedError
