"""Small numeric helpers shared across adapters and pipelines.

These are three lines each, which is exactly why they had drifted into five
near-identical private copies. One home means one place to be right about
overflow, empty inputs and axis handling.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Softmax over ``axis`` (the whole array when ``axis`` is ``None``).

    Shifted by the max before exponentiating, so large logits do not overflow.
    """
    arr = np.asarray(x, dtype=np.float32)
    shifted = arr - np.max(arr, axis=axis, keepdims=axis is not None)
    exp = np.exp(shifted)
    total = np.sum(exp, axis=axis, keepdims=axis is not None)
    return exp / total


def unit(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length, so a dot product is a cosine similarity.

    A zero vector is returned unchanged rather than becoming ``nan``.
    """
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-9 else arr
