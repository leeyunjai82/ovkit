"""teach: learn from examples, answer by nearest neighbours, save and reload."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit import Model
from ovkit.core.errors import OVKitError
from ovkit.core.results import Results
from ovkit.pipelines.teach import MODES, Score, Teach, _normalize_points, _pose_feature

IMG = np.zeros((100, 100, 3), np.uint8)


def _fake_embedder(teach: Teach, table: dict[int, np.ndarray]) -> Teach:
    """Replace the model-backed embedding with a lookup on the pixel value."""

    def embed(image):
        vec = table[int(np.asarray(image).ravel()[0])]
        return vec / np.linalg.norm(vec)

    teach._embed = embed
    return teach


def _img(value: int) -> np.ndarray:
    return np.full((10, 10, 3), value, np.uint8)


@pytest.fixture
def taught():
    table = {
        10: np.array([1.0, 0.05, 0.0]),
        11: np.array([0.95, 0.1, 0.0]),
        20: np.array([0.0, 1.0, 0.05]),
        21: np.array([0.05, 0.9, 0.0]),
        15: np.array([0.9, 0.2, 0.0]),  # clearly a "can"
        25: np.array([0.1, 0.95, 0.0]),  # clearly a "bottle"
    }
    ai = _fake_embedder(Teach(), table)
    ai.learn("can", [_img(10), _img(11)])
    ai.learn("bottle", [_img(20), _img(21)])
    return ai


# -- modes and names --------------------------------------------------------


def test_korean_mode_values_map_to_english_modes():
    assert Teach(mode="사진").mode == "photo"
    assert Teach(mode="전신").mode == "body"
    assert Teach(mode="상반신").mode == "upper"


def test_an_unknown_mode_lists_the_real_ones():
    with pytest.raises(OVKitError, match="사진"):
        Teach(mode="swirl")


def test_model_builds_teach_by_korean_name():
    ai = Model("가르치기", mode="표정")
    assert isinstance(ai, Teach) and ai.mode == "face"


def test_korean_method_names_are_the_same_functions():
    assert Teach.배우기 is Teach.learn
    assert Teach.맞혀봐 is Teach.guess
    assert Teach.점수 is Teach.score
    assert Teach.저장 is Teach.save


# -- learning and guessing --------------------------------------------------


def test_guess_names_the_closest_taught_thing(taught):
    label, confidence = taught.guess(_img(15))
    assert label == "can" and confidence > 0.5
    assert taught.guess(_img(25))[0] == "bottle"


def test_run_answers_with_a_normal_results(taught):
    r = taught.run(_img(15))
    assert isinstance(r, Results)
    assert r.summary().startswith("can ")
    assert set(r.names.values()) == {"can", "bottle"}


def test_guessing_before_learning_explains_what_to_do(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    ai = _fake_embedder(Teach(), {5: np.array([1.0, 0.0, 0.0])})
    with pytest.raises(OVKitError, match="learn"):
        ai.guess(_img(5))


def test_learning_nothing_is_an_error():
    ai = Teach()
    ai._embed = lambda img: (_ for _ in ()).throw(OVKitError("no face"))
    with pytest.raises(OVKitError, match="배울 수 있는 예시|nothing to learn"):
        ai.learn("x", [_img(1)])


def test_more_examples_of_the_same_label_accumulate(taught):
    before = len(taught._vectors)
    taught.learn("can", _img(11))
    assert len(taught._vectors) == before + 1
    assert taught.labels == ["can", "bottle"]  # no duplicate label


def test_forget_removes_a_label(taught):
    taught.forget("can")
    assert taught.labels == ["bottle"]


# -- scoring ----------------------------------------------------------------


def test_score_reports_accuracy_and_confusion(taught, tmp_path, monkeypatch):
    import cv2

    for label, value in (("can", 15), ("bottle", 25), ("bottle", 21)):
        sub = tmp_path / label
        sub.mkdir(exist_ok=True)
        cv2.imwrite(str(sub / f"{value}.png"), _img(value))
    monkeypatch.setenv("OVKIT_LANG", "ko")
    score = taught.score(tmp_path)
    assert isinstance(score, Score)
    assert score.total == 3 and score.accuracy == 1.0
    assert "정확도 100%" in str(score)


def test_score_needs_the_label_folder_layout(taught, tmp_path):
    with pytest.raises(OVKitError, match="라벨|subfolder"):
        taught.score(tmp_path)


# -- save / load ------------------------------------------------------------


def test_save_and_load_round_trip(taught, tmp_path):
    path = taught.save("재활용 분류", dir=tmp_path)
    assert path.name == "재활용-분류.json"

    table = {15: np.array([0.9, 0.2, 0.0]), 25: np.array([0.1, 0.95, 0.0])}
    loaded = _fake_embedder(Teach(load=path), table)
    assert loaded.labels == ["can", "bottle"]
    assert loaded.guess(_img(15))[0] == "can"


def test_loading_a_missing_ai_says_where_it_looked(tmp_path):
    with pytest.raises(OVKitError, match="못 찾았어요|no saved"):
        Teach(load=tmp_path / "nope.json")


# -- feature math -----------------------------------------------------------


def test_pose_feature_is_position_and_size_invariant():
    person = np.zeros((17, 3), np.float32)
    person[:, 2] = 1.0
    person[:, 0] = np.linspace(0, 1, 17)
    person[:, 1] = np.linspace(0, 2, 17)
    a = _pose_feature(person, upper=False)
    moved = person.copy()
    moved[:, 0] += 50  # translate
    moved[:, :2] *= 3  # ...after scaling
    moved[:, 0] += 50
    b = _pose_feature(moved, upper=False)
    assert a == pytest.approx(b, abs=1e-5)
    assert _pose_feature(person, upper=True).shape == (22,)


def test_pose_feature_zeroes_invisible_joints():
    person = np.random.rand(17, 3).astype(np.float32)
    person[:, 2] = 1.0
    person[3, 2] = 0.0  # joint 3 unseen
    feat = _pose_feature(person, upper=False).reshape(17, 2)
    assert feat[3, 0] == 0.0 and feat[3, 1] == 0.0


def test_hand_points_normalization_is_wrist_relative():
    points = np.array([[0.5, 0.5], [0.6, 0.5], [0.5, 0.7]], np.float32)
    flat = _normalize_points(points)
    assert flat[:2] == pytest.approx([0.0, 0.0])  # wrist at the origin
    scaled = _normalize_points(points * 4.0 + 10.0)
    # scaling is removed; translation shifts everything relative to the wrist
    assert flat[2:] == pytest.approx(scaled[2:], abs=1e-5)


def test_hand_mode_without_mediapipe_names_the_extra():
    try:
        import mediapipe  # noqa: F401

        pytest.skip("mediapipe installed here")
    except ImportError:
        pass
    ai = Teach(mode="손모양")
    with pytest.raises(ImportError, match=r"ovkit\[hand\]"):
        ai._embed_hand(IMG)


def test_every_mode_value_resolves():
    assert set(MODES.values()) == {"photo", "face", "hand", "upper", "body"}
