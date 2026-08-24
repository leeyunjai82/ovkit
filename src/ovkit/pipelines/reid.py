"""Re-identification: "is this the same person / face / car as before?"

An embedding model turns a crop into a vector; on its own that is 256 numbers
and no answer. Matching is what makes it useful::

    ids = vis("face_match")
    ids.add("yunjai", "photos/yunjai.jpg")
    ids.add("dana", "photos/dana.jpg")
    ids.who("frame.jpg")          # ('yunjai', 0.81)

The comparison is cosine similarity on L2-normalised vectors, so the score is
in ``[-1, 1]`` and 1.0 means identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import OVKitError
from ..image.ops import imread
from .base import Pipeline


class ReID(Pipeline):
    """Embed crops and match them against a named gallery.

    Parameters
    ----------
    embedder:
        Registered embedding model: ``face_reid`` (faces), ``person_reid`` or
        ``vehicle_reid_0001`` (whole bodies / cars), ``image_retrieval`` (scenes).
    threshold:
        Below this similarity :meth:`who` answers ``None`` instead of naming the
        closest gallery entry — without it every stranger gets somebody's name.
    """

    name = "face_match"
    description = "Match faces (or people, or vehicles) against a gallery you build."

    def __init__(
        self,
        device: str = "AUTO",
        embedder: str = "face_reid",
        threshold: float = 0.5,
    ) -> None:
        super().__init__(device)
        self.embedder = embedder
        self.threshold = float(threshold)
        self.gallery: dict[str, np.ndarray] = {}

    # -- embedding ----------------------------------------------------------

    def embed(self, image: Any) -> np.ndarray:
        """Return the L2-normalised descriptor for one image or crop."""
        arr = (
            imread(image)
            if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__")
            else image
        )
        arr = np.asarray(arr)
        if arr.size == 0:
            raise OVKitError("Cannot embed an empty image.")
        out = self.model(self.embedder)(arr)
        if not out or not out[0].tensors:
            raise OVKitError(
                f"'{self.embedder}' returned no descriptor. Use an embedding model "
                f"(face_reid, vehicle_reid_0001, image_retrieval)."
            )
        vector = np.asarray(next(iter(out[0].tensors.values()))).reshape(-1).astype(np.float32)
        return normalize(vector)

    # -- gallery ------------------------------------------------------------

    def add(self, label: str, image: Any) -> np.ndarray:
        """Put an image in the gallery under ``label``.

        Adding the same label twice averages the descriptors, so several photos
        of one person give a more robust match than any single one.
        """
        vector = self.embed(image)
        if label in self.gallery:
            vector = normalize(self.gallery[label] + vector)
        self.gallery[label] = vector
        return vector

    def remove(self, label: str) -> None:
        """Drop a gallery entry."""
        self.gallery.pop(label, None)

    # -- matching -----------------------------------------------------------

    def match(self, image: Any, top_k: int = 1) -> list[tuple[str, float]]:
        """Return the ``top_k`` closest gallery labels as ``(label, score)``."""
        if not self.gallery:
            raise OVKitError("The gallery is empty — call add(label, image) first.")
        query = self.embed(image)
        labels = list(self.gallery)
        scores = np.stack([self.gallery[k] for k in labels]) @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [(labels[int(i)], float(scores[int(i)])) for i in order]

    def who(self, image: Any) -> tuple[str, float] | None:
        """The best match, or ``None`` when nothing clears ``threshold``."""
        label, score = self.match(image, top_k=1)[0]
        return (label, score) if score >= self.threshold else None

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between two images, without touching the gallery."""
        return float(self.embed(a) @ self.embed(b))

    def run(self, image: np.ndarray, **_: Any):
        """Matching needs a gallery, so :meth:`who` is the entry point."""
        from ..core.results import Results

        result = Results(image, task=self.name)
        hit = self.who(image) if self.gallery else None
        result.text = f"{hit[0]} ({hit[1]:.2f})" if hit else "no match in the gallery"
        return result


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so a dot product is a cosine similarity."""
    arr = np.asarray(vector, np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-9 else arr
