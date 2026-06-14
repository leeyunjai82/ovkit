"""plot() works for every result type (boxes / masks / keypoints / text / raw)."""

from __future__ import annotations

import numpy as np

from ovkit import Model


def _check_plot(r, image):
    out = r.plot()
    assert out.shape == image.shape and out.dtype == np.uint8


def test_plot_detect(synthetic_detr_ir, synthetic_image, imgsz):
    _check_plot(
        Model(str(synthetic_detr_ir), device="CPU")(synthetic_image, imgsz=imgsz)[0],
        synthetic_image,
    )


def test_plot_semantic_seg(synthetic_seg_ir, synthetic_image):
    _check_plot(Model(str(synthetic_seg_ir), device="CPU")(synthetic_image)[0], synthetic_image)


def test_plot_instance_seg(synthetic_instance_seg_ir, synthetic_image):
    r = Model(str(synthetic_instance_seg_ir), task="segment", device="CPU")(synthetic_image)[0]
    _check_plot(r, synthetic_image)


def test_plot_pose(synthetic_pose_ir, synthetic_image):
    r = Model(str(synthetic_pose_ir), task="pose", device="CPU")(synthetic_image)[0]
    _check_plot(r, synthetic_image)


def test_plot_ocr_text(synthetic_ocr_ir, synthetic_image):
    r = Model(str(synthetic_ocr_ir), task="ocr", device="CPU")(synthetic_image)[0]
    assert r.text  # text overlay path
    _check_plot(r, synthetic_image)


def test_plot_generic_tensors(synthetic_seg_ir, synthetic_image):
    # An unsupported task -> generic raw -> plot() overlays output info.
    r = Model(str(synthetic_seg_ir), task="image_processing", device="CPU")(synthetic_image)[0]
    assert r.tensors is not None
    _check_plot(r, synthetic_image)
