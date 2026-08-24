"""Class names: every mirrored model must answer in words, not indices."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.core import registry
from ovkit.core.constants import CLASS_TABLES, class_names
from ovkit.recognize.generic import GenericAdapter

#: Models whose class ids come from a table rather than a shipped labels.txt.
LABELLED = [
    ("rtdetr_r50", "coco80"),
    ("open_closed_eye_0001", "open_closed_eye"),
    ("pspnet_pytorch", "voc21"),
    ("road_segmentation_adas_0001", "road4"),
    ("person_vehicle_bike_detection_2000", "person_vehicle_bike"),
    ("vehicle_license_plate_detection_barrier_0106", "vehicle_plate"),
    ("weld_porosity_detection_0001", "weld3"),
    ("common_sign_language_0002", "sign_language12"),
    ("nfnet_f0", "imagenet1000"),
]


@pytest.mark.parametrize("name,table", LABELLED)
def test_model_is_wired_to_its_class_table(name, table):
    entry = registry.resolve(name)
    assert entry is not None, f"{name} missing from the manifest"
    assert entry.postprocess.get("classes") == table


def test_generated_manifest_cannot_clobber_hand_written_metadata():
    """omz.yaml is regenerated and carries no postprocess; merging is key-wise."""
    entry = registry.resolve("rtdetr_r50")
    assert entry.postprocess == {"format": "detr", "classes": "coco80"}
    assert entry.filename.endswith("detect/rtdetr_r50/model.xml")  # source still from omz.yaml


def test_class_tables_are_named_not_numbered():
    for key, table in CLASS_TABLES.items():
        assert all(n and not n.startswith("class_") for n in table), key


def test_imagenet_table_loads_from_the_data_file():
    names = class_names("imagenet1000", 1000)
    assert len(names) == 1000
    assert names[0] == "tench" and names[281] == "tabby"


def test_imagenet_table_shifts_for_the_background_variant():
    names = class_names("imagenet1000", 1001)
    assert names[0] == "background" and names[1] == "tench"


class _FakeInput:
    def __init__(self, shape, name="input"):
        self._shape = shape
        self._name = name

    def get_partial_shape(self):
        return self._shape

    def get_any_name(self):
        return self._name


class _FakeBackend:
    """Minimal stand-in: a clip-input model with one small labelled output."""

    def __init__(self):
        self.inputs = [_FakeInput((1, 3, 8, 224, 224))]
        self.input_shape = (1, 3, 8, 224, 224)

    def infer(self, feed):
        assert feed.shape == (1, 3, 8, 224, 224)
        return {"features": np.array([[0.1, 4.0, 0.2]], np.float32)}


def test_generic_adapter_decodes_a_labelled_score_vector():
    adapter = GenericAdapter(task="action_recognition", postprocess={"classes": "weld3"})
    r = adapter.run(_FakeBackend(), np.zeros((240, 320, 3), np.uint8))
    assert r.summary().startswith("normal weld ")
    assert "raw output" not in r.summary()
