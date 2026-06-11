"""Classification adapter — INTERFACE STUB.

TODO(v0+): implement softmax decoding over a single ``[N, C]`` output and wire a
permissive (Apache-2.0/MIT) backbone into the manifest. The shape of the public
API is fixed here so ``Model`` can route classify tasks once filled in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Results
from .base import BaseAdapter


class ClassifyAdapter(BaseAdapter):
    """Adapter for image classification (not yet implemented)."""

    task = "classify"

    def run(self, backend: Backend, image: np.ndarray, **kwargs: Any) -> Results:
        raise NotImplementedError(
            "Classification is not implemented in v0. "
            "Planned: softmax over a single [N, C] output. Contributions welcome."
        )
