"""Results.summary(): every task must answer in words, not tensor shapes."""

from __future__ import annotations

import numpy as np

from ovkit.core.results import Boxes, Keypoints, Masks, Probs, Results

IMG = np.zeros((100, 200, 3), np.uint8)


def test_detection_summary_counts_by_label():
    boxes = Boxes(
        np.array(
            [
                [0, 0, 10, 10, 0.9, 0],
                [20, 20, 30, 30, 0.8, 0],
                [40, 40, 50, 50, 0.7, 1],
            ],
            np.float32,
        )
    )
    r = Results(IMG, task="detect", names={0: "person", 1: "car"}, boxes=boxes)
    assert r.summary() == "2x person, car"


def test_detection_with_nothing_found_says_so():
    r = Results(IMG, task="detect", names={0: "person"}, boxes=Boxes(np.zeros((0, 6), np.float32)))
    assert "nothing found" in r.summary()


def test_classification_summary_names_the_class_with_its_score():
    probs = Probs(np.array([0.1, 0.7, 0.2], np.float32))
    r = Results(IMG, task="classify", names={0: "cat", 1: "dog", 2: "bird"}, probs=probs)
    assert r.summary() == "dog 0.70"


def test_segmentation_summary_reports_class_coverage():
    # class map: 75% class 0, 25% class 1
    cmap = np.zeros((1, 20, 20), np.int32)
    cmap[0, :, 15:] = 1
    r = Results(IMG, task="segment", names={0: "road", 1: "car"}, masks=Masks(cmap))
    summary = r.summary()
    assert "road 75%" in summary and "car 25%" in summary


def test_pose_summary_counts_instances_and_keypoints():
    r = Results(IMG, task="pose", keypoints=Keypoints(np.zeros((2, 17, 3), np.float32)))
    assert "2 instance(s), 17 keypoints" in r.summary()


def test_image_output_is_described_as_an_image():
    out = {"sr": np.zeros((1, 3, 64, 128), np.float32)}
    r = Results(IMG, task="image_processing", tensors=out)
    assert "produced a 128x64 image" in r.summary()


def test_repr_uses_the_summary():
    r = Results(IMG, task="classify", names={0: "cat"}, probs=Probs(np.array([1.0], np.float32)))
    assert "cat 1.00" in repr(r)
