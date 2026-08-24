"""Anchor detectors (scores [N,2] + boxes [N,4]) must filter, NMS and scale."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openvino")

from ovkit.recognize.detect import DetectAdapter  # noqa: E402


def _outputs(scores_fg, boxes):
    scores = np.stack([1 - np.asarray(scores_fg), np.asarray(scores_fg)], axis=1)[None]
    return {"scores": scores.astype(np.float32), "boxes": np.asarray(boxes, np.float32)[None]}


def test_low_scores_are_dropped_and_duplicates_suppressed():
    # Three anchors on the same face (should collapse to one) + one background.
    box = [0.25, 0.25, 0.75, 0.75]
    near = [0.26, 0.26, 0.74, 0.74]
    out = _outputs(
        [0.95, 0.90, 0.01],
        [box, near, [0.0, 0.0, 0.1, 0.1]],
    )
    boxes = DetectAdapter._decode_scores_boxes(out, (200, 400), conf=0.25, max_det=300)
    assert boxes.shape[0] == 1, "overlapping anchors must be merged by NMS"
    x1, y1, x2, y2, score, cls = boxes[0]
    assert score == pytest.approx(0.95, abs=1e-5) and cls == 0
    # normalized corners scaled to pixels: x by width 400, y by height 200
    assert (x1, y1, x2, y2) == pytest.approx((100.0, 50.0, 300.0, 150.0), abs=0.5)


def test_centre_form_boxes_are_detected_and_converted():
    # cx, cy, w, h — x2 < x1 for these rows, so the decoder must convert.
    out = _outputs([0.9], [[0.5, 0.5, 0.4, 0.2]])
    boxes = DetectAdapter._decode_scores_boxes(out, (100, 100), conf=0.25, max_det=10)
    assert boxes.shape[0] == 1
    assert tuple(boxes[0][:4]) == pytest.approx((30.0, 40.0, 70.0, 60.0), abs=0.5)


def test_nothing_above_threshold_returns_empty():
    out = _outputs([0.05, 0.01], [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
    assert DetectAdapter._decode_scores_boxes(out, (100, 100), conf=0.25, max_det=10).shape == (
        0,
        6,
    )
