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
        rgb = bool(self.pre.get("rgb", False))
        scale = self.pre.get("scale", 1.0)
        if len(backend.inputs) > 1:
            # All-image multi-input models (e.g. super-resolution takes the image
            # plus a pre-upscaled copy): feed the same image resized per input.
            feed = self._multi_image_feed(backend, image, rgb=rgb, scale=scale)
            outputs = backend.infer(feed)
            return Results(image, task=self.task, names=self.names, tensors=outputs)
        size = self.model_input_hw(backend)
        chw = self.preprocess(image, size, rgb=rgb, scale=scale)[0]
        feed = _fit_to_shape(chw, backend.input_shape)
        outputs = backend.infer(feed)
        return Results(image, task=self.task, names=self.names, tensors=outputs)

    def _multi_image_feed(
        self, backend: Backend, image: np.ndarray, *, rgb: bool, scale: float
    ) -> dict[str, np.ndarray]:
        """Build a feed for a model whose inputs are all 4-D images.

        Raises ``ValueError`` when any input is not image-like (those models
        need :meth:`ovkit.Model.infer` with hand-built tensors).
        """
        feeds: dict[str, np.ndarray] = {}
        for inp in backend.inputs:
            ps = inp.get_partial_shape()
            dims = [int(d.get_length()) if d.is_static else -1 for d in ps]
            if len(dims) != 4 or dims[1] not in (1, 3) or dims[2] <= 0 or dims[3] <= 0:
                names = ", ".join(i.get_any_name() for i in backend.inputs)
                raise ValueError(
                    f"This model takes {len(backend.inputs)} inputs ({names}) and not all "
                    f"of them are images; build the tensors and call model.infer({{...}})."
                )
            arr = self.preprocess(image, (dims[2], dims[3]), rgb=rgb, scale=scale)
            if dims[1] == 1:  # grayscale input
                arr = arr.mean(axis=1, keepdims=True).astype(np.float32)
            feeds[inp.get_any_name()] = arr
        return feeds


def _fit_to_shape(chw: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Place a ``[C, H, W]`` image into the model's full input shape.

    Handles 4-D ``[N,C,H,W]`` and 5-D video clips ``[N,C,T,H,W]`` (the frame is
    repeated across ``T``); dynamic dims (``-1``) become 1.
    """
    c, h, w = chw.shape
    dims = [d if (d and d > 0) else 1 for d in shape]
    if len(dims) == 5:  # [N, C, T, H, W] — tile the frame across time
        arr = np.repeat(chw[:, None, :, :], dims[2], axis=1)[None]
    elif len(dims) <= 4:
        arr = chw[None]
    else:
        arr = chw.reshape((1,) * (len(dims) - 3) + (c, h, w))
    return np.ascontiguousarray(arr, dtype=np.float32)
