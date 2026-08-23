"""PixelLink-style text detectors (segm+link logits) decode to text boxes."""

from __future__ import annotations

import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from ovkit import Model  # noqa: E402
from ovkit.recognize.detect import DetectAdapter  # noqa: E402


def test_decode_segm_boxes_finds_the_blob():
    # 32x32 grid, one 8x10 high-logit text region.
    segm = np.zeros((1, 32, 32, 2), np.float32)
    segm[..., 0] = 5.0  # background logit
    segm[0, 10:18, 4:14, 1] = 10.0  # text logit wins inside the blob
    boxes = DetectAdapter._decode_segm_boxes(
        {"model/segm_logits/add": segm}, (320, 320), conf=0.25, max_det=10
    )
    assert boxes.shape[0] == 1
    x1, y1, x2, y2, score, cls = boxes[0]
    assert score > 0.9 and cls == 0
    assert 30 <= x1 <= 50 and 90 <= y1 <= 110  # scaled 10x from grid coords
    assert 130 <= x2 <= 150 and 170 <= y2 <= 190


def test_pixel_link_style_model_end_to_end(tmp_path):
    import openvino.opset13 as op

    x = op.parameter([1, 64, 64, 3], np.float32, name="data")  # NHWC like the real one
    nchw = op.transpose(x, op.constant(np.array([0, 3, 1, 2])))
    g = op.reduce_mean(nchw, np.array([2, 3]), keep_dims=True)  # [1,3,1,1]
    segw = op.constant(np.random.rand(2, 3, 1, 1).astype(np.float32))
    segm = op.convolution(
        op.multiply(nchw, op.add(g, op.constant(np.float32(1.0)))),
        segw,
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
    )  # [1,2,64,64]
    linkw = op.constant(np.random.rand(16, 3, 1, 1).astype(np.float32))
    link = op.convolution(nchw, linkw, [1, 1], [0, 0], [0, 0], [1, 1])  # [1,16,64,64]
    m = ov.Model([segm.output(0), link.output(0)], [x], "pl")
    p = tmp_path / "pl.xml"
    ov.save_model(m, str(p), compress_to_fp16=False)

    img = np.random.randint(0, 255, (120, 160, 3), np.uint8)
    r = Model(str(p), task="detect")(img)[0]  # must not raise; boxes may be 0..N
    assert r.boxes is not None
