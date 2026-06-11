"""Re-identification = embed + match, assembled over an embedding model.

INTERFACE STUB for v0. Planned: compute embeddings for crops and match by cosine
similarity against a gallery.
"""

from __future__ import annotations

from typing import Any


class ReID:
    """Embedding + matching pipeline (not yet implemented)."""

    def __init__(self, embedder: Any = None, device: str = "AUTO", **kwargs: Any) -> None:
        self.embedder = embedder
        self.device = device

    def embed(self, image: Any) -> Any:
        raise NotImplementedError(
            "ReID is not implemented in v0. Planned: embed crops, cosine-match against "
            "a gallery. TODO."
        )

    def match(self, query: Any, gallery: Any) -> Any:
        raise NotImplementedError("ReID matching is not implemented in v0. TODO.")
