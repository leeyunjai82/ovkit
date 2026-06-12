"""Smoke tests for YOLOv2 decode, the generic raw adapter, and low-level infer."""

from __future__ import annotations

import numpy as np

from ovkit import Model


def test_yolo_end_to_end(synthetic_yolo_ir, synthetic_image):
    # Single 4-D output is ambiguous with segmentation, so request detect.
    model = Model(str(synthetic_yolo_ir), task="detect", device="CPU")
    r = model(synthetic_image, conf=0.25)[0]

    assert model.task == "detect"
    assert len(r.boxes) == 1  # one confident anchor/cell after NMS
    assert int(r.boxes.cls[0]) == 0
    assert r.boxes.conf[0] > 0.5

    h, w = synthetic_image.shape[:2]
    xyxy = r.boxes.xyxy[0]
    assert 0 <= xyxy[0] <= w and 0 <= xyxy[3] <= h


def test_generic_raw_outputs(synthetic_seg_ir, synthetic_image):
    # An unsupported task falls back to the generic adapter (raw tensors).
    model = Model(str(synthetic_seg_ir), task="image_processing", device="CPU")
    r = model(synthetic_image)[0]

    assert model.task == "image_processing"
    assert r.boxes is None and r.masks is None
    assert r.tensors is not None and len(r.tensors) >= 1
    arr = next(iter(r.tensors.values()))
    assert arr.shape[1] == 3  # [1, 3, 4, 4]


def test_low_level_infer(synthetic_classify_ir, imgsz):
    model = Model(str(synthetic_classify_ir), device="CPU")

    info = model.inputs
    assert info and info[0][1] == (1, 3, imgsz, imgsz)

    out = model.infer(np.zeros((1, 3, imgsz, imgsz), dtype=np.float32))
    assert isinstance(out, dict) and len(out) == 1


def test_auto_raw_ndarray(synthetic_classify_ir, imgsz):
    # A non-image ndarray is auto-routed to raw inference (returns a dict).
    model = Model(str(synthetic_classify_ir), device="CPU")
    out = model(np.zeros((1, 3, imgsz, imgsz), dtype=np.float32))
    assert isinstance(out, dict) and len(out) == 1


def test_auto_raw_npy(synthetic_classify_ir, imgsz, tmp_path):
    arr = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    p = tmp_path / "x.npy"
    np.save(p, arr)
    model = Model(str(synthetic_classify_ir), device="CPU")
    out = model(str(p))
    assert isinstance(out, dict) and len(out) == 1
