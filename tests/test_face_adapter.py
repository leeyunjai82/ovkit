"""Typed decodes for the face model family + classify/generic display fixes."""

from __future__ import annotations

import numpy as np
import pytest

ov = pytest.importorskip("openvino")

from ovkit import Model  # noqa: E402


@pytest.fixture()
def img():
    return np.random.randint(0, 255, (96, 96, 3), np.uint8)


def _save(model, tmp_path, name):
    p = tmp_path / f"{name}.xml"
    ov.save_model(model, str(p), compress_to_fp16=False)
    return str(p)


def _head(x, n):
    import openvino.opset13 as op

    g = op.reduce_mean(x, np.array([2, 3]), keep_dims=False)
    return op.matmul(g, op.constant(np.random.rand(3, n).astype(np.float32)), False, False)


def _named(node, name):
    node.output(0).set_names({name})
    return node.output(0)


def test_age_gender_decodes_to_text(tmp_path, img):
    import openvino.opset13 as op

    x = op.parameter([1, 3, 62, 62], np.float32, name="data")
    age = op.multiply(_head(x, 1), op.constant(np.float32(0.003)))
    prob = op.softmax(_head(x, 2), 1)
    m = ov.Model([_named(age, "age_conv3"), _named(prob, "prob")], [x], "ag")
    r = Model(_save(m, tmp_path, "ag"), task="face")(img)[0]
    assert r.text is not None and "age" in r.text
    assert r.probs is not None and r.name_for(r.probs.top1) in ("female", "male")


def test_emotions_decode(tmp_path, img):
    import openvino.opset13 as op

    x = op.parameter([1, 3, 64, 64], np.float32, name="data")
    m = ov.Model([_named(op.softmax(_head(x, 5), 1), "prob_emotion")], [x], "emo")
    r = Model(_save(m, tmp_path, "emo"), task="face")(img)[0]
    assert r.name_for(r.probs.top1) in ("neutral", "happy", "sad", "surprise", "anger")
    assert r.text


def test_face_landmarks_become_keypoints(tmp_path, img):
    import openvino.opset13 as op

    x = op.parameter([1, 3, 48, 48], np.float32, name="data")
    m = ov.Model([_named(op.sigmoid(_head(x, 10)), "align_fc3")], [x], "lm")
    r = Model(_save(m, tmp_path, "lm"), task="face")(img)[0]
    assert r.keypoints is not None and r.keypoints.data.shape == (1, 5, 3)


def test_classify_landmark_regressor_yields_keypoints_not_top1(tmp_path, img):
    import openvino.opset13 as op

    x = op.parameter([1, 3, 60, 60], np.float32, name="data")
    m = ov.Model([_named(op.sigmoid(_head(x, 70)), "fc70")], [x], "lm70")
    r = Model(_save(m, tmp_path, "lm70"), task="classify")(img)[0]
    assert r.keypoints is not None and r.keypoints.data.shape == (1, 35, 3)
    assert r.probs is None


def test_classify_multi_head_summary(tmp_path, img):
    import openvino.opset13 as op

    x = op.parameter([1, 3, 72, 72], np.float32, name="data")
    t = op.softmax(_head(x, 4), 1)
    c = op.softmax(_head(x, 7), 1)
    m = ov.Model([_named(t, "type"), _named(c, "color")], [x], "veh")
    r = Model(_save(m, tmp_path, "veh"), task="classify")(img)[0]
    assert r.text is not None and "type:" in r.text and "color:" in r.text


def test_multi_image_input_super_resolution_plots_output_image(tmp_path, img):
    import openvino.opset13 as op

    x1 = op.parameter([1, 3, 24, 24], np.float32, name="0")
    x2 = op.parameter([1, 3, 96, 96], np.float32, name="1")
    up = op.interpolate(
        x1,
        op.constant(np.array([96, 96], np.int64)),
        "linear",
        "sizes",
        axes=op.constant(np.array([2, 3], np.int64)),
    )
    out = op.multiply(op.add(up, x2), op.constant(np.float32(0.002)))
    m = ov.Model([_named(out, "sr_out")], [x1, x2], "sr")
    r = Model(_save(m, tmp_path, "sr"), task="image_processing")(img)[0]
    plot = r.plot()
    assert plot.shape == (96, 96, 3)  # the model's output image, not the input
