"""Base class for composed pipelines — several models chained into one answer.

A :class:`~ovkit.Model` runs one network. A :class:`Pipeline` runs several and
returns a single :class:`~ovkit.Results`, so "analyse the faces in this frame"
is one call instead of a detector, three crops, three more models and the code
to stitch them together.

Pipelines take the same sources as ``Model`` (path, ndarray, folder, video,
camera index) and return the same ``list[Results]``, so anything that already
works with a model works with a pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..core.model import Model, _iter_sources
from ..core.results import Results


class Pipeline:
    """Several models composed into one capability.

    Subclasses implement :meth:`run` for a single image; source handling,
    model caching and lazy loading come from here.
    """

    #: Name this pipeline is registered under (``vis("face_analyze")``).
    name: str = "pipeline"
    #: One-line description, shown by :func:`ovkit.list_pipelines`.
    description: str = ""

    def __init__(self, device: str = "AUTO") -> None:
        self.device = device
        #: Mirrors ``Model.task`` so a pipeline is a drop-in for a model.
        self.task = self.name
        self._models: dict[str, Model] = {}

    # -- models -------------------------------------------------------------

    def model(self, name: str) -> Model:
        """Return a sub-model, loading (and downloading) it on first use.

        Loading is lazy so a pipeline configured without, say, head pose never
        downloads that model.
        """
        if name not in self._models:
            self._models[name] = Model(name, device=self.device)
        return self._models[name]

    # -- running ------------------------------------------------------------

    def run(self, image: np.ndarray, **kwargs: Any) -> Results:
        """Analyse one image. Implemented by each pipeline."""
        raise NotImplementedError

    def predict(
        self, source: Any, *, stream: bool = False, **kwargs: Any
    ) -> list[Results] | Iterator[Results]:
        """Run on any source ``Model`` accepts and return ``Results`` per image."""
        gen = self._stream(source, **kwargs)
        return gen if stream else list(gen)

    def _stream(self, source: Any, **kwargs: Any) -> Iterator[Results]:
        for image, path in _iter_sources(source):
            result = self.run(image, **kwargs)
            result.path = path
            yield result

    def __call__(self, source: Any, **kwargs: Any) -> list[Results] | Iterator[Results]:
        """Alias for :meth:`predict` (a pipeline is callable, like a model)."""
        return self.predict(source, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, device={self.device!r})"


def first_line(result: Results) -> str:
    """The one-line answer a sub-model gave, for building a per-object label."""
    return result.summary()


def detections(model: Model, image: np.ndarray, conf: float) -> Results:
    """Run a detector and return its ``Results`` (empty boxes are fine)."""
    out = model(image, conf=conf)
    return out[0] if out else Results(image, task="detect")
