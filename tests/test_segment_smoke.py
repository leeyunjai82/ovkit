"""End-to-end semantic segmentation smoke test on a synthetic IR."""

from __future__ import annotations

import numpy as np

from ovkit import Model
from ovkit.core.results import Results


def test_segment_end_to_end(synthetic_seg_ir, synthetic_image):
    model = Model(str(synthetic_seg_ir), device="CPU")
    results = model(synthetic_image)

    assert model.task == "segment"
    r = results[0]
    assert isinstance(r, Results)

    assert r.masks is not None
    h, w = synthetic_image.shape[:2]
    # One class map, resized back to the original image size.
    assert r.masks.data.shape == (1, h, w)
    # argmax picked channel 2 everywhere.
    assert set(np.unique(r.masks.data).tolist()) == {2}
    assert r.boxes is None
