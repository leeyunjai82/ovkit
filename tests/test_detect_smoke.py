"""End-to-end detection smoke test on the synthetic DETR IR.

Exercises the full vertical slice: load IR -> auto-detect task -> preprocess ->
infer -> DETR decode -> Results -> plot/save -> stream.
"""

from __future__ import annotations

import numpy as np

from ovkit import Model
from ovkit.core.results import Results


def test_detect_end_to_end(synthetic_detr_ir, synthetic_image, imgsz):
    model = Model(str(synthetic_detr_ir), device="CPU")
    results = model(synthetic_image, imgsz=imgsz, conf=0.25)

    assert model.task == "detect"
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert isinstance(r, Results)

    # Two queries are above threshold (class 2 car @0.95, class 15 cat @0.88).
    assert len(r.boxes) == 2
    classes = set(int(c) for c in r.boxes.cls)
    assert classes == {2, 15}
    assert (r.boxes.conf >= 0.25).all()

    # Highest-confidence first.
    assert r.boxes.conf[0] >= r.boxes.conf[1]

    # Boxes are within image bounds and in pixel coords.
    h, w = synthetic_image.shape[:2]
    xyxy = r.boxes.xyxy
    assert (xyxy[:, 0::2] <= w + 1).all()
    assert (xyxy[:, 1::2] <= h + 1).all()

    # COCO names resolved.
    assert r.name_for(2) == "car"


def test_plot_and_save(synthetic_detr_ir, synthetic_image, imgsz, tmp_path):
    model = Model(str(synthetic_detr_ir), device="CPU")
    r = model(synthetic_image, imgsz=imgsz)[0]

    canvas = r.plot()
    assert canvas.shape == synthetic_image.shape
    assert canvas.dtype == np.uint8

    out = tmp_path / "out.jpg"
    r.save(out)
    assert out.is_file() and out.stat().st_size > 0


def test_stream_returns_generator(synthetic_detr_ir, synthetic_image, imgsz):
    model = Model(str(synthetic_detr_ir), device="CPU")
    stream = model.predict([synthetic_image, synthetic_image], imgsz=imgsz, stream=True)
    assert not isinstance(stream, list)
    out = list(stream)
    assert len(out) == 2


def test_conf_threshold_filters(synthetic_detr_ir, synthetic_image, imgsz):
    model = Model(str(synthetic_detr_ir), device="CPU")
    # Raise threshold above the cat (0.88) but below... both are high; use 0.9
    r = model(synthetic_image, imgsz=imgsz, conf=0.9)[0]
    # Only the car (sigmoid(3)=~0.95) survives.
    assert len(r.boxes) == 1
    assert int(r.boxes.cls[0]) == 2
