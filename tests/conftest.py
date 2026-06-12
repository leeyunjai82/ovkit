"""Shared pytest fixtures.

Builds a tiny synthetic DETR-style OpenVINO IR model so the detection slice can
be tested end-to-end without any network access or large model download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ov = pytest.importorskip("openvino")


def _opset():
    # opset location moved across OpenVINO versions; try the common spellings.
    try:
        from openvino.runtime import opset13 as op
    except Exception:  # pragma: no cover - version variety
        from openvino import opset13 as op  # type: ignore
    return op


@pytest.fixture(scope="session")
def imgsz() -> int:
    return 64


@pytest.fixture(scope="session")
def synthetic_detr_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create and serialize a deterministic DETR-style detector to IR.

    Output ``logits`` is shaped ``[1, Q, C]`` and ``boxes`` ``[1, Q, 4]``
    (cxcywh, normalized). Query 0 is a confident "car" (class 2); the rest are
    below threshold. Outputs are constants tied to the input by a zero term so
    OpenVINO keeps the parameter live.
    """
    import openvino as ovino

    op = _opset()
    q, c = 5, 80

    logits = np.full((1, q, c), -5.0, dtype=np.float32)
    logits[0, 0, 2] = 3.0  # query 0 -> class 2 (car), sigmoid(3)=~0.95
    logits[0, 1, 15] = 2.0  # query 1 -> class 15 (cat), sigmoid(2)=~0.88
    boxes = np.zeros((1, q, 4), dtype=np.float32)
    boxes[0, 0] = [0.50, 0.50, 0.20, 0.20]
    boxes[0, 1] = [0.25, 0.75, 0.10, 0.10]

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="images")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(
        op.reduce_sum(x, axes, keep_dims=False),
        op.constant(np.float32(0.0)),
    )
    logits_node = op.add(op.constant(logits), zero)
    boxes_node = op.add(op.constant(boxes), zero)
    logits_node.set_friendly_name("logits")
    boxes_node.set_friendly_name("pred_boxes")

    model = ovino.Model([logits_node, boxes_node], [x], "synthetic_detr")

    out = tmp_path_factory.mktemp("ir") / "synthetic_detr.xml"
    ovino.save_model(model, str(out))
    return out


@pytest.fixture
def synthetic_image(imgsz: int) -> np.ndarray:
    """A small random BGR uint8 image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(imgsz * 2, imgsz * 3, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def synthetic_classify_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic single-output ``[1, C]`` classifier IR.

    Class 3 has the highest logit; outputs are tied to the input by a zero term
    so OpenVINO keeps the parameter live.
    """
    import openvino as ovino

    op = _opset()
    c = 10
    logits = np.full((1, c), -5.0, dtype=np.float32)
    logits[0, 3] = 4.0  # argmax -> class 3

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    out = op.add(op.constant(logits), zero)
    out.set_friendly_name("probs")

    model = ovino.Model([out], [x], "synthetic_classify")
    path = tmp_path_factory.mktemp("ir") / "synthetic_classify.xml"
    ovino.save_model(model, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_ssd_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic SSD/DetectionOutput IR: ``[1, 1, N, 7]``.

    Rows are ``[image_id, label, conf, x_min, y_min, x_max, y_max]`` (normalized).
    Two detections clear conf=0.25 (labels 1 and 2); one is below threshold.
    """
    import openvino as ovino

    op = _opset()
    det = np.zeros((1, 1, 3, 7), dtype=np.float32)
    det[0, 0, 0] = [0, 1, 0.90, 0.10, 0.10, 0.50, 0.50]
    det[0, 0, 1] = [0, 2, 0.80, 0.50, 0.50, 0.90, 0.90]
    det[0, 0, 2] = [0, 1, 0.10, 0.00, 0.00, 0.20, 0.20]  # below threshold

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    out = op.add(op.constant(det), zero)
    out.set_friendly_name("detection_out")

    model = ovino.Model([out], [x], "synthetic_ssd")
    path = tmp_path_factory.mktemp("ir") / "synthetic_ssd.xml"
    ovino.save_model(model, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_boxes_labels_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic two-output detector IR: boxes ``[N,5]`` + labels ``[N]``.

    Boxes are input-pixel coords (model input is ``imgsz`` square). Two clear the
    conf threshold (labels 1 and 2); one is below it.
    """
    import openvino as ovino

    op = _opset()
    boxes = np.array(
        [[8, 8, 40, 40, 0.90], [40, 40, 60, 60, 0.80], [0, 0, 4, 4, 0.10]],
        dtype=np.float32,
    )
    labels = np.array([1, 2, 1], dtype=np.float32)

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    boxes_out = op.add(op.constant(boxes), zero)
    boxes_out.set_friendly_name("boxes")
    labels_out = op.add(op.constant(labels), zero)
    labels_out.set_friendly_name("labels")

    model = ovino.Model([boxes_out, labels_out], [x], "synthetic_boxes_labels")
    path = tmp_path_factory.mktemp("ir") / "synthetic_boxes_labels.xml"
    ovino.save_model(model, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_yolo_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic YOLOv2 region IR: ``[1, A*(5+C), H, W]``.

    5 anchors, 1 class, 2x2 grid -> 30 channels. Only anchor 0 at cell (0,0) is
    confident (objectness + class logit high); everything else is suppressed.
    """
    import openvino as ovino

    op = _opset()
    grid = np.full((1, 30, 2, 2), -10.0, dtype=np.float32)
    grid[0, 0, 0, 0] = 0.0  # tx
    grid[0, 1, 0, 0] = 0.0  # ty
    grid[0, 2, 0, 0] = 0.0  # tw
    grid[0, 3, 0, 0] = 0.0  # th
    grid[0, 4, 0, 0] = 10.0  # objectness
    grid[0, 5, 0, 0] = 10.0  # class 0 logit

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    out = op.add(op.constant(grid), zero)
    out.set_friendly_name("region")

    model = ovino.Model([out], [x], "synthetic_yolo")
    path = tmp_path_factory.mktemp("ir") / "synthetic_yolo.xml"
    ovino.save_model(model, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_seg_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic ``[1, C, H, W]`` segmentation IR (argmax -> class 2)."""
    import openvino as ovino

    op = _opset()
    seg = np.zeros((1, 3, 4, 4), dtype=np.float32)
    seg[0, 2, :, :] = 5.0  # channel 2 dominates -> argmax == 2 everywhere

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    out = op.add(op.constant(seg), zero)
    out.set_friendly_name("segmentation")

    model = ovino.Model([out], [x], "synthetic_seg")
    path = tmp_path_factory.mktemp("ir") / "synthetic_seg.xml"
    ovino.save_model(model, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_pose_ir(tmp_path_factory: pytest.TempPathFactory, imgsz: int) -> Path:
    """Create a deterministic ``[1, K, H, W]`` keypoint-heatmap IR.

    Two keypoints peak at grid cells (row, col) = (1, 1) and (2, 3) in a 4x4 map.
    """
    import openvino as ovino

    op = _opset()
    heat = np.zeros((1, 2, 4, 4), dtype=np.float32)
    heat[0, 0, 1, 1] = 1.0
    heat[0, 1, 2, 3] = 1.0

    x = op.parameter([1, 3, imgsz, imgsz], ovino.Type.f32, name="data")
    axes = op.constant(np.array([0, 1, 2, 3], dtype=np.int64))
    zero = op.multiply(op.reduce_sum(x, axes, keep_dims=False), op.constant(np.float32(0.0)))
    out = op.add(op.constant(heat), zero)
    out.set_friendly_name("heatmaps")

    model = ovino.Model([out], [x], "synthetic_pose")
    path = tmp_path_factory.mktemp("ir") / "synthetic_pose.xml"
    ovino.save_model(model, str(path))
    return path
