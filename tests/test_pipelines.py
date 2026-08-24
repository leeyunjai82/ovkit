"""Composed capabilities: several models chained into one readable answer."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit import Model, list_pipelines
from ovkit.core.results import Boxes, Keypoints, Results
from ovkit.pipelines import PIPELINES, build_pipeline, is_pipeline
from ovkit.pipelines.analyze import FaceAnalyzer
from ovkit.pipelines.gaze import describe_gaze
from ovkit.pipelines.reid import ReID, normalize
from ovkit.pipelines.text import TextReader
from ovkit.pipelines.tracking import Tracker, iou_matrix

IMG = np.zeros((240, 320, 3), np.uint8)


class _Fake:
    """A stand-in sub-model: returns whatever Results it was given."""

    def __init__(self, result_for):
        self.result_for = result_for
        self.calls = 0

    def __call__(self, image, **_kwargs):
        self.calls += 1
        return [self.result_for(image)]


def _boxes(rows):
    return Boxes(np.array(rows, np.float32))


def _install(pipeline, **fakes):
    pipeline._models.update(fakes)
    return pipeline


# -- discovery --------------------------------------------------------------


def test_every_pipeline_has_a_description():
    listed = list_pipelines()
    assert set(listed) == set(PIPELINES)
    assert all(desc.strip() for desc in listed.values())


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_model_builds_each_capability_by_name(name):
    assert is_pipeline(name)
    assert isinstance(Model(name), PIPELINES[name])


def test_model_still_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown pipeline"):
        build_pipeline("not_a_capability")


def test_aliases_reach_the_same_pipeline():
    assert type(Model("ocr")) is type(Model("read_text"))
    assert type(Model("faces")) is type(Model("face_analyze"))


def test_pipeline_reports_its_task_like_a_model():
    assert Model("face_analyze").task == "face_analyze"


# -- face analysis ----------------------------------------------------------


def _face_detector(rows):
    return _Fake(lambda img: Results(img, task="detect", names={0: "face"}, boxes=_boxes(rows)))


def _attribute(text):
    def build(img):
        r = Results(img, task="face")
        r.text = text
        return r

    return _Fake(build)


def test_face_analyze_joins_every_attribute_onto_each_face():
    pipe = FaceAnalyzer(attributes=("age_gender", "emotion"))
    _install(
        pipe,
        face_detection=_face_detector([[10, 10, 90, 110, 0.99, 0], [150, 20, 230, 120, 0.9, 0]]),
        age_gender=_attribute("age 31 · male 98%"),
        emotion=_attribute("happy 92%"),
    )
    r = pipe.run(IMG)
    assert len(r.boxes) == 2
    assert r.labels == ["age 31 · male 98% · happy 92%"] * 2
    assert r.summary().startswith("2 faces: age 31 · male 98% · happy 92%")


def test_face_analyze_says_so_when_there_is_no_face():
    pipe = FaceAnalyzer(attributes=())
    _install(pipe, face_detection=_face_detector(np.zeros((0, 6), np.float32)))
    assert pipe.run(IMG).summary() == "no face found"


def test_one_broken_attribute_does_not_lose_the_face():
    class _Broken:
        def __call__(self, image, **_kwargs):
            raise RuntimeError("model exploded")

    pipe = FaceAnalyzer(attributes=("age_gender", "emotion"))
    _install(
        pipe,
        face_detection=_face_detector([[10, 10, 90, 110, 0.99, 0]]),
        age_gender=_Broken(),
        emotion=_attribute("happy 92%"),
    )
    r = pipe.run(IMG)
    assert len(r.boxes) == 1
    assert "unavailable" in r.labels[0] and "happy 92%" in r.labels[0]


def test_face_landmarks_are_mapped_back_onto_the_full_image():
    def landmarks(_img):
        r = Results(_img, task="face")
        r.keypoints = Keypoints(np.array([[[5.0, 6.0, 1.0]]], np.float32))
        return r

    pipe = FaceAnalyzer(attributes=("face_landmarks",))
    _install(
        pipe,
        face_detection=_face_detector([[100, 100, 200, 200, 0.99, 0]]),
        face_landmarks=_Fake(landmarks),
    )
    r = pipe.run(IMG)
    # crop starts at x1 - 15% of the box, so the point lands well right of 5,6
    assert r.keypoints is not None
    assert r.keypoints.data[0, 0, 0] == pytest.approx(100 - 15 + 5)
    assert r.keypoints.data[0, 0, 1] == pytest.approx(100 - 15 + 6)


# -- OCR --------------------------------------------------------------------


def test_read_text_reads_boxes_in_reading_order():
    """Detector order is confidence order; a reader has to work top-left down."""
    # Each region is painted a distinct grey so the fake recogniser can tell
    # which crop it was handed.
    image = np.zeros((240, 320, 3), np.uint8)
    regions = {
        (10, 5, 60, 25): (10, "HELLO"),
        (100, 5, 150, 25): (20, "WORLD"),
        (10, 60, 60, 80): (30, "AGAIN"),
    }
    for (x1, y1, x2, y2), (value, _word) in regions.items():
        image[y1:y2, x1:x2] = value
    by_value = {value: word for value, word in regions.values()}

    # detections deliberately out of reading order
    rows = [[100, 5, 150, 25, 0.9, 0], [10, 60, 60, 80, 0.8, 0], [10, 5, 60, 25, 0.7, 0]]

    def recognize(crop):
        r = Results(crop, task="ocr")
        r.text = by_value[int(round(float(crop.mean())))]
        return r

    pipe = TextReader()
    detector = _Fake(lambda img: Results(img, task="detect", names={0: "text"}, boxes=_boxes(rows)))
    _install(pipe, text_detection=detector, text_recognition=_Fake(recognize))
    r = pipe.run(image)
    assert [list(map(int, b)) for b in r.boxes.xyxy] == [
        [10, 5, 60, 25],
        [100, 5, 150, 25],
        [10, 60, 60, 80],
    ]
    assert r.text == "HELLO WORLD AGAIN"
    assert r.labels == ["HELLO", "WORLD", "AGAIN"]


def test_read_text_survives_an_unreadable_crop():
    class _Broken:
        def __call__(self, image, **_kwargs):
            raise RuntimeError("nope")

    pipe = TextReader()
    _install(
        pipe,
        text_detection=_Fake(
            lambda img: Results(img, task="detect", boxes=_boxes([[0, 0, 10, 10, 0.9, 0]]))
        ),
        text_recognition=_Broken(),
    )
    assert pipe.run(IMG).text == ""


# -- tracking ---------------------------------------------------------------


def test_iou_matrix_matches_known_overlaps():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    scores = iou_matrix(a, b)[0]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1 / 3)
    assert scores[2] == 0.0


def test_track_ids_survive_movement_and_a_brief_disappearance():
    tracker = Tracker(max_age=2)
    frame1 = np.array([[10, 10, 50, 50, 0.9, 0], [100, 100, 150, 150, 0.8, 0]], np.float32)
    assert tracker.update(frame1) == [1, 2]

    moved = frame1.copy()
    moved[:, :4] += 6
    assert tracker.update(moved) == [1, 2]  # same objects, same ids

    assert tracker.update(np.zeros((0, 6), np.float32)) == []  # nothing this frame
    assert tracker.update(moved) == [1, 2]  # back within max_age


def test_a_different_class_never_inherits_a_track_id():
    tracker = Tracker()
    car = np.array([[10, 10, 50, 50, 0.9, 2]], np.float32)
    person = np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)
    assert tracker.update(car) == [1]
    assert tracker.update(person) == [2]


def test_tracker_reset_forgets_everything():
    tracker = Tracker()
    box = np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)
    tracker.update(box)
    tracker.reset()
    assert tracker.update(box) == [1]


# -- re-identification ------------------------------------------------------


def test_normalize_makes_a_dot_product_a_cosine():
    v = normalize(np.array([3.0, 4.0]))
    assert float(v @ v) == pytest.approx(1.0)


def test_gallery_matching_names_the_closest_entry():
    reid = ReID(threshold=0.9)
    reid.gallery = {
        "yunjai": normalize(np.array([1.0, 0.0, 0.0])),
        "dana": normalize(np.array([0.0, 1.0, 0.0])),
    }
    reid.embed = lambda _image: normalize(np.array([0.95, 0.31, 0.0]))
    assert reid.match(IMG)[0][0] == "yunjai"
    assert reid.who(IMG)[0] == "yunjai"


def test_a_stranger_is_not_given_somebody_elses_name():
    reid = ReID(threshold=0.9)
    reid.gallery = {"yunjai": normalize(np.array([1.0, 0.0, 0.0]))}
    reid.embed = lambda _image: normalize(np.array([0.0, 1.0, 0.0]))
    assert reid.who(IMG) is None


def test_adding_a_label_twice_averages_the_descriptors():
    reid = ReID()
    vectors = iter([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    reid.embed = lambda _image: normalize(next(vectors))
    reid.add("yunjai", IMG)
    merged = reid.add("yunjai", IMG)
    assert merged == pytest.approx(normalize(np.array([1.0, 1.0])))


def test_matching_without_a_gallery_explains_itself():
    reid = ReID()
    reid.embed = lambda _image: np.array([1.0, 0.0], np.float32)
    with pytest.raises(Exception, match="gallery is empty"):
        reid.match(IMG)


# -- gaze -------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector,expected",
    [
        ([0.5, 0.02, -0.8], "looking left"),
        ([-0.5, 0.02, -0.8], "looking right"),
        ([0.02, 0.5, -0.8], "looking up"),
        ([0.02, -0.5, -0.8], "looking down"),
        ([0.01, 0.01, -1.0], "looking at the camera"),
    ],
)
def test_gaze_vectors_are_described_in_words(vector, expected):
    assert describe_gaze(np.array(vector, np.float32)) == expected
