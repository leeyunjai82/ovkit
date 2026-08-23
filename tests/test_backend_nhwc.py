"""NHWC (channels-last) models — TF-converted OMZ style — must run end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from ovkit import Model  # noqa: E402


def _nhwc_classify(tmp_path):
    import openvino.opset13 as op

    x = op.parameter([1, 64, 96, 3], np.float32, name="data")  # NHWC, H!=W
    nchw = op.transpose(x, op.constant(np.array([0, 3, 1, 2])))
    g = op.reduce_mean(nchw, np.array([2, 3]), keep_dims=False)  # [1, 3]
    w = op.constant(np.random.rand(3, 7).astype(np.float32))
    out = op.softmax(op.matmul(g, w, False, False), 1)
    m = ov.Model([out.output(0)], [x], "nhwc_cls")
    p = tmp_path / "nhwc.xml"
    ov.save_model(m, str(p), compress_to_fp16=False)
    return str(p)


def test_nhwc_model_runs_via_predict(tmp_path):
    img = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    r = Model(_nhwc_classify(tmp_path), task="classify")(img)[0]
    assert r.probs is not None and r.probs.data.shape == (7,)


def test_nhwc_input_hw_detected(tmp_path):
    from ovkit.core.backend import Backend
    from ovkit.recognize.base import BaseAdapter

    class A(BaseAdapter):
        task = "x"

        def run(self, backend, image, **kw):  # pragma: no cover
            raise NotImplementedError

    backend = Backend(_nhwc_classify(tmp_path), "CPU")
    assert A().model_input_hw(backend) == (64, 96)  # H, W — not (96, 3)
