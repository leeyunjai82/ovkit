"""Backend should reconcile a 3-channel image tensor with a 1-channel model."""

from __future__ import annotations

import numpy as np
import pytest

ov = pytest.importorskip("openvino")


def _gray_model(tmp_path):
    import openvino.opset13 as op

    x = op.parameter([1, 1, 8, 8], np.float32, name="im")
    g = op.reduce_mean(x, np.array([2, 3]), keep_dims=False)  # [1, 1]
    w = op.constant(np.ones((1, 4), np.float32))
    model = ov.Model([op.matmul(g, w, False, False)], [x], "gray")  # [1, 4]
    p = tmp_path / "gray.xml"
    ov.save_model(model, str(p), compress_to_fp16=False)
    return str(p)


def test_three_channel_input_adapted_to_one_channel(tmp_path):
    from ovkit.core.backend import Backend

    backend = Backend(_gray_model(tmp_path), "CPU")
    assert backend.input_shape[1] == 1

    three = np.ones((1, 3, 8, 8), np.float32)  # preprocessor output (3 ch)
    out = backend.infer(three)  # must not raise on the channel mismatch
    assert next(iter(out.values())).shape[-1] == 4


def test_matching_channels_unchanged(tmp_path):
    from ovkit.core.backend import Backend

    backend = Backend(_gray_model(tmp_path), "CPU")
    one = np.ones((1, 1, 8, 8), np.float32)
    out = backend.infer(one)
    assert next(iter(out.values())).shape[-1] == 4


def test_missing_weights_gives_an_actionable_error(tmp_path):
    """IR without its .bin must say what is wrong, not leak OpenVINO's nesting."""
    import openvino.opset13 as op

    from ovkit.core.backend import Backend
    from ovkit.core.errors import OVKitError

    x = op.parameter([1, 4], np.float32, name="x")
    w = op.constant(np.random.randn(4, 3).astype(np.float32))
    model = ov.Model([op.matmul(x, w, False, False)], [x], "needs_weights")
    xml = tmp_path / "model.xml"
    ov.save_model(model, str(xml), compress_to_fp16=False)
    (tmp_path / "model.bin").write_bytes(b"")  # mirror dropped the weights

    with pytest.raises(OVKitError) as err:
        Backend(str(xml), "CPU")
    assert "weights file" in str(err.value)
    assert "model.bin" in str(err.value)
