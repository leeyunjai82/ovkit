"""The trainable RT-DETR facade: names, torch gating, decode compatibility."""

from __future__ import annotations

import numpy as np
import pytest

import ovkit
from ovkit.core.errors import ModelNotFoundError
from ovkit.recognize.detect import DetectAdapter
from ovkit.rtdetr import RTDETR

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_rtdetr_is_importable_from_the_package_root_without_torch():
    assert ovkit.RTDETR is RTDETR  # lazy attribute, no torch at import time


def test_hyphen_and_underscore_names_reach_the_same_mirror_entry():
    assert RTDETR("rtdetr-r50")._registry_name() == "rtdetr_r50"
    assert RTDETR("rtdetr_r50")._registry_name() == "rtdetr_r50"


def test_a_variant_not_on_the_mirror_says_what_to_do_instead():
    with pytest.raises(ModelNotFoundError, match="train your own"):
        RTDETR("rtdetr-r18")._registry_name()


def test_a_missing_checkpoint_is_a_file_error_not_a_torch_error():
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        RTDETR("no_such.pt")


@pytest.mark.skipif(HAS_TORCH, reason="only meaningful without torch")
def test_training_without_torch_names_the_extra_to_install():
    model = RTDETR("rtdetr-r18")
    with pytest.raises(ImportError, match=r"ovkit\[train\]"):
        model.train(data="data.yaml")


def test_detr_decode_does_not_sigmoid_twice():
    """An IR exported by the trainer emits probabilities, not logits.

    sigmoid(0.95) = 0.72 — double-squashing silently broke conf thresholds.
    """
    adapter = DetectAdapter(postprocess={"format": "detr"})
    boxes = np.full((1, 4, 4), 0.5, np.float32)

    probs = np.zeros((1, 4, 3), np.float32)
    probs[0, 0, 1] = 0.95  # already a probability
    out = adapter._decode_detr(
        {"scores": probs[0], "boxes": boxes[0]}, (100, 100), conf=0.9, max_det=10
    )
    assert len(out) == 1 and out[0, 4] == pytest.approx(0.95)

    logits = np.full((1, 4, 3), -8.0, np.float32)
    logits[0, 0, 1] = 3.0  # raw logits still get the sigmoid
    out = adapter._decode_detr(
        {"scores": logits[0], "boxes": boxes[0]}, (100, 100), conf=0.9, max_det=10
    )
    assert len(out) == 1 and out[0, 4] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-4)


def test_cxcywh_numpy_matches_the_adapter_math():
    from ovkit.rtdetr.utils.ops import cxcywh2xyxy_np

    box = np.array([[0.5, 0.5, 0.2, 0.4]], np.float32)
    assert cxcywh2xyxy_np(box) == pytest.approx(np.array([[0.4, 0.3, 0.6, 0.7]]))


def test_exporter_writes_labels_txt_for_ovkits_model(tmp_path):
    """The bridge: exported IR + labels.txt means Model() knows the names."""
    if not HAS_TORCH:
        pytest.skip("needs torch")
    from ovkit.rtdetr.exporter import export_openvino
    from ovkit.rtdetr.nn.rtdetr_net import RTDETRNet

    net = RTDETRNet("r18", num_classes=2, pretrained_backbone=False)
    export_openvino(net, {0: "can", 1: "bottle"}, imgsz=64, out_dir=tmp_path, fname="m")
    assert (tmp_path / "labels.txt").read_text().splitlines() == ["can", "bottle"]
