"""Smoke tests: instance segmentation and OCR (CTC) decoding."""

from __future__ import annotations

from ovkit import Model


def test_instance_segmentation(synthetic_instance_seg_ir, synthetic_image, imgsz):
    model = Model(str(synthetic_instance_seg_ir), task="segment", device="CPU")
    r = model(synthetic_image, conf=0.25)[0]

    assert model.task == "segment"
    # Instance results carry both boxes and per-instance masks.
    assert r.boxes is not None and len(r.boxes) == 2
    assert {int(c) for c in r.boxes.cls} == {0, 1}
    h, w = synthetic_image.shape[:2]
    assert r.masks is not None and r.masks.data.shape == (2, h, w)
    # The pasted mask covers its box area.
    assert r.masks.data[0].sum() > 0

    # Threshold filters the 0.7 instance.
    r2 = model(synthetic_image, conf=0.8)[0]
    assert len(r2.boxes) == 1


def test_ocr_ctc_decode(synthetic_ocr_ir, synthetic_image):
    model = Model(str(synthetic_ocr_ir), task="ocr", device="CPU")
    r = model(synthetic_image)[0]

    assert r.text == "ab1"  # repeats collapsed, blanks dropped
    assert r.tensors is not None  # raw logits still available


def test_pose_multi_instance(synthetic_pose_ir, synthetic_image):
    # The synthetic pose IR has one peak per channel -> one instance, but the
    # multi-peak decoder must still return well-formed (N, K, 3).
    model = Model(str(synthetic_pose_ir), task="pose", device="CPU")
    r = model(synthetic_image)[0]
    assert r.keypoints is not None
    n, k, three = r.keypoints.data.shape
    assert n >= 1 and k == 2 and three == 3
    assert (r.keypoints.conf > 0).any()
