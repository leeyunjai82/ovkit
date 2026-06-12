"""End-to-end pose (heatmap peak) smoke test on a synthetic IR."""

from __future__ import annotations

from ovkit import Model
from ovkit.core.results import Results


def test_pose_end_to_end(synthetic_pose_ir, synthetic_image):
    # [1, K, H, W] is ambiguous with segmentation, so request the task explicitly
    # (real OMZ pose models carry task=pose in the manifest).
    model = Model(str(synthetic_pose_ir), task="pose", device="CPU")
    r = model(synthetic_image)[0]

    assert model.task == "pose"
    assert isinstance(r, Results)
    assert r.keypoints is not None
    assert r.keypoints.data.shape == (1, 2, 3)  # one instance, 2 keypoints, xyc

    h, w = synthetic_image.shape[:2]
    xy = r.keypoints.xy[0]
    # peaks at grid (col,row) = (1,1) and (3,2) on a 4x4 map
    assert abs(xy[0, 0] - 1 / 4 * w) < 1.0 and abs(xy[0, 1] - 1 / 4 * h) < 1.0
    assert abs(xy[1, 0] - 3 / 4 * w) < 1.0 and abs(xy[1, 1] - 2 / 4 * h) < 1.0
    assert (r.keypoints.conf[0] > 0).all()
