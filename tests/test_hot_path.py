"""What every frame pays for.

Preprocessing was two thirds of the time a frame cost — not because resizing
is slow, but because the normalisation was written as five statements and each
one allocated another full-size float32 array. On a 640x640 input that is five
4.9 MB allocations per frame to do arithmetic that fits in one.

The same shape: a compiled model's input shape and output signatures never
change, and both were being re-read from the OpenVINO runtime on every single
frame — `input_shape` once per infer, `output_signatures()` twice per frame in
the detect adapter, which asks it to pick a decode format.

These tests do not time anything (a timing assertion on a shared CI runner is a
flake waiting to happen). They pin the two properties that made it fast: the
arithmetic is still exactly right, and the description is read once.
"""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.core.backend import Backend
from ovkit.image import ops
from ovkit.recognize.base import BaseAdapter


class _Adapter(BaseAdapter):
    """BaseAdapter is abstract only in `run`; preprocessing is all here."""

    task = "test"

    def run(self, backend, image, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError


def _reference(image, size, *, rgb, scale, mean, std):
    """Normalisation written the slow, obvious way, as it used to be.

    Kept verbatim so the fast path has something to be wrong against.
    """
    h, w = size
    img = ops.resize(image, (w, h))
    if rgb:
        img = ops.bgr_to_rgb(img)
    arr = img.astype(np.float32)
    if scale != 1.0:
        arr = arr / scale
    if np.any(mean != 0.0) or np.any(std != 1.0):
        arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None]
    return np.ascontiguousarray(arr, dtype=np.float32)


@pytest.mark.parametrize(
    "pre",
    [
        {},  # the default: /255, no mean/std
        {"scale": 1.0},  # raw 0-255 input
        {"scale": 255.0, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        {"scale": 1.0, "mean": [123.675, 116.28, 103.53]},
        {"rgb": False},
    ],
)
def test_preprocess_still_computes_the_same_numbers(pre):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)  # odd size, on purpose

    adapter = _Adapter(preprocess=dict(pre))
    got = adapter.preprocess(image, (64, 64), rgb=bool(pre.get("rgb", True)))

    scale, mean, std = adapter._scale_mean_std()
    want = _reference(
        image, (64, 64), rgb=bool(pre.get("rgb", True)), scale=scale, mean=mean, std=std
    )

    # Same order of operations, so this is exact — not "close enough".
    assert np.array_equal(got, want), "the fused path changed the arithmetic"


def test_preprocess_returns_a_contiguous_nchw_float32_tensor():
    adapter = _Adapter()
    arr = adapter.preprocess(np.zeros((40, 50, 3), np.uint8), (32, 32))
    assert arr.shape == (1, 3, 32, 32)
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"], "OpenVINO copies again if this is not contiguous"


def test_a_grayscale_crop_still_preprocesses():
    """`Results.crop` on a single-channel image hands back a 2-D array."""
    arr = _Adapter().preprocess(np.zeros((20, 20), np.uint8), (16, 16), rgb=False)
    assert arr.shape == (1, 1, 16, 16)


def test_the_compiled_models_description_is_read_once(synthetic_detr_ir):
    """Not re-read per frame: the same object comes back every time."""
    backend = Backend(str(synthetic_detr_ir), "CPU")

    assert backend.output_signatures() is backend.output_signatures()
    assert backend.input_shape is backend.input_shape
    assert backend.input_shape == (1, 3, backend.input_shape[2], backend.input_shape[3])


def test_every_output_gets_its_own_key_even_with_no_name(synthetic_detr_ir):
    """`_named` keys a dict by these — two blanks would lose an output."""
    backend = Backend(str(synthetic_detr_ir), "CPU")
    names = backend._output_names
    assert len(names) == len(backend.outputs)
    assert len(set(names)) == len(names), f"colliding output keys: {names}"
    assert all(n for n in names), "an unnamed output must still get a placeholder"
