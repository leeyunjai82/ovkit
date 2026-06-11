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
