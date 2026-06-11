"""End-to-end classification smoke test on a synthetic single-output IR."""

from __future__ import annotations

import numpy as np

from ovkit import Model
from ovkit.core.results import Results


def test_classify_end_to_end(synthetic_classify_ir, synthetic_image, imgsz):
    model = Model(str(synthetic_classify_ir), device="CPU")
    results = model(synthetic_image, imgsz=imgsz)

    assert model.task == "classify"
    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert isinstance(r, Results)

    assert r.probs is not None
    assert r.probs.top1 == 3
    assert 3 in list(r.probs.top5)
    # softmax output is a normalized distribution
    assert abs(float(np.sum(r.probs.data)) - 1.0) < 1e-4
    assert r.boxes is None
