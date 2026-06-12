"""Generic adapter — run any image-input model and return its raw outputs.

Used for tasks that have no typed decoder (super-resolution, OCR, embeddings,
face attributes, action recognition, ...). The image is resized to the model's
input and run; the raw output tensors are returned in :attr:`Results.tensors`
so the caller can post-process them.

Models that take **non-image** input (NLP / audio / time series) cannot use this
image pipeline — call :meth:`ovkit.Model.infer` with your own input tensors
instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results
from .base import BaseAdapter


class GenericAdapter(BaseAdapter):
    """Run an image through a model and return raw output tensors."""

    task = "generic"

    def __init__(self, task: str = "generic", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task = task

    def run(self, backend: Backend, image: np.ndarray, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)
        return Results(image, task=self.task, names=self.names, tensors=outputs)
