"""End-to-end SSD detection smoke test on a synthetic DetectionOutput IR."""

from __future__ import annotations

from ovkit import Model
from ovkit.core.results import Results


def test_ssd_detect_end_to_end(synthetic_ssd_ir, synthetic_image):
    model = Model(str(synthetic_ssd_ir), device="CPU")
    results = model(synthetic_image, conf=0.25)

    assert model.task == "detect"
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert isinstance(r, Results)

    # Two detections clear the threshold (labels 1 and 2); the 0.10 one is cut.
    assert len(r.boxes) == 2
    assert {int(c) for c in r.boxes.cls} == {1, 2}
    assert (r.boxes.conf >= 0.25).all()
    assert r.boxes.conf[0] >= r.boxes.conf[1]  # sorted by confidence

    # Normalized SSD coords mapped to original image pixels.
    h, w = synthetic_image.shape[:2]
    xyxy = r.boxes.xyxy
    assert (xyxy[:, 0::2] <= w + 1).all()
    assert (xyxy[:, 1::2] <= h + 1).all()
    # First box is [0.1, 0.1, 0.5, 0.5] * (w, h)
    assert abs(xyxy[0, 0] - 0.10 * w) < 1.0
    assert abs(xyxy[0, 2] - 0.50 * w) < 1.0


def test_ssd_conf_threshold(synthetic_ssd_ir, synthetic_image):
    model = Model(str(synthetic_ssd_ir), device="CPU")
    r = model(synthetic_image, conf=0.85)[0]
    # Only the 0.90 detection (label 1) survives.
    assert len(r.boxes) == 1
    assert int(r.boxes.cls[0]) == 1
