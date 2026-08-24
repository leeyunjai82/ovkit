"""The 0.2.0 API contract: Model("기능", 입력), Korean names, found, timing.

These are the promises student code relies on; breaking any of them is a
breaking change from here on.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from ovkit import Model
from ovkit.core.errors import ModelNotFoundError
from ovkit.core.i18n import canonical, display_name, name_key, position
from ovkit.core.results import Boxes, Probs, Results
from ovkit.pipelines import PIPELINES, is_pipeline
from ovkit.pipelines.base import Pipeline

IMG = np.zeros((120, 200, 3), np.uint8)


class _EchoPipe(Pipeline):
    """A capability stand-in that records how it was called."""

    name = "_echo"
    description = "test double"

    def __init__(self, device="AUTO", **opts):
        super().__init__(device)
        self.opts = opts
        self.calls: list[dict] = []

    def run(self, image, *, conf=0.25, **_):
        self.calls.append({"conf": conf})
        return Results(
            image,
            task=self.name,
            names={0: "person"},
            boxes=Boxes(np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)),
        )


@pytest.fixture
def echo(monkeypatch):
    monkeypatch.setitem(PIPELINES, "_echo", _EchoPipe)
    yield


# -- Korean names -----------------------------------------------------------


def test_korean_capability_names_reach_the_same_pipeline():
    assert canonical("얼굴분석") == "face_analyze"
    assert type(Model("얼굴분석")) is type(Model("face_analyze"))
    assert type(Model("졸음감지")).name == "drowsiness"


def test_korean_model_aliases_map_to_registry_names():
    assert canonical("자세") == "pose"
    assert canonical("받아쓰기") == "stt"
    assert canonical("detect") == "detect"  # canonical names pass through


def test_every_korean_alias_points_at_something_real():
    from ovkit.core.i18n import KO_CAPS
    from ovkit.core.registry import resolve

    for ko, target in KO_CAPS.items():
        assert is_pipeline(target) or resolve(target) is not None, f"{ko} -> {target}"


def test_a_typo_gets_a_suggestion_in_korean(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    with pytest.raises(ModelNotFoundError, match="얼굴분석"):
        Model("얼국분석")


def test_the_same_typo_in_english_mode(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    with pytest.raises(ModelNotFoundError, match="Did you mean"):
        Model("face_analize")


# -- Model("기능", 입력): the immediate call --------------------------------


def test_a_photo_answers_with_one_result_not_a_list(echo):
    r = Model("_echo", IMG)
    assert isinstance(r, Results)
    assert r.summary()  # readable


def test_a_folder_answers_with_a_list(echo, tmp_path):
    import cv2

    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), IMG)
    out = Model("_echo", str(tmp_path))
    assert isinstance(out, list) and len(out) == 2


def test_a_webcam_index_answers_with_a_lazy_stream(echo):
    stream = Model("_echo", 0)
    assert isinstance(stream, Iterator)  # nothing opened until iteration


def test_a_video_path_answers_with_a_lazy_stream(echo):
    assert isinstance(Model("_echo", "clip.mp4"), Iterator)


def test_run_options_reach_the_call_not_the_constructor(echo):
    Model("_echo", IMG, conf=0.7)
    pipe = Model("_echo")
    assert "conf" not in pipe.opts  # conf is a run option...
    r = Model("_echo", IMG, conf=0.7, detector="x")
    assert isinstance(r, Results)


def test_constructor_options_still_reach_the_pipeline(echo):
    pipe = Model("_echo", detector="face_detection")
    assert pipe.opts == {"detector": "face_detection"}


def test_building_first_and_calling_later_still_works(echo):
    pipe = Model("_echo")
    out = pipe(IMG)
    assert isinstance(out, list)  # the object form keeps its list contract


# -- Results: found / name_en / pos / timing --------------------------------


def _detection() -> Results:
    boxes = Boxes(np.array([[150, 5, 195, 35, 0.91, 67]], np.float32))
    return Results(IMG, task="detect", names={67: "cell phone"}, boxes=boxes)


def test_found_gives_name_in_both_languages(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    row = _detection().found[0]
    assert row["name"] == "휴대폰"
    assert row["name_en"] == "cell-phone"  # hyphenated, language-independent
    assert row["score"] == pytest.approx(0.91)
    assert row["box"] == [150, 5, 195, 35]


def test_found_name_follows_the_display_language(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    row = _detection().found[0]
    assert row["name"] == "cell phone"
    assert row["name_en"] == "cell-phone"


def test_found_pos_is_a_nine_grid_label(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    assert _detection().found[0]["pos"] == "오른쪽 위"
    monkeypatch.setenv("OVKIT_LANG", "en")
    assert _detection().found[0]["pos"] == "top-right"


def test_classification_found_is_one_row_without_a_box():
    r = Results(IMG, task="classify", names={1: "dog"}, probs=Probs([0.1, 0.9]))
    (row,) = r.found
    assert row["name_en"] == "dog" and row["box"] is None


def test_position_center_is_a_single_word(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    assert position(100, 60, 200, 120) == "가운데"
    monkeypatch.setenv("OVKIT_LANG", "en")
    assert position(100, 60, 200, 120) == "center"


def test_untranslated_names_fall_back_to_english(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    assert display_name("tench") == "tench"  # ImageNet stays English
    assert name_key("cell phone") == "cell-phone"


def test_pipeline_results_carry_elapsed_and_device(echo):
    r = Model("_echo", IMG)
    assert r.elapsed_ms is not None and r.elapsed_ms >= 0.0
    assert r.device  # "AUTO" here; a real backend reports the resolved device
    assert r.to_dict()["device"] == r.device
