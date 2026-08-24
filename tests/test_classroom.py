"""Classroom pipelines: counting, posture over time, reps, attendance."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit import Model
from ovkit.core.errors import OVKitError
from ovkit.core.results import Boxes, Keypoints, Results
from ovkit.pipelines.classroom import (
    Attendance,
    Counter,
    PostureCoach,
    RepCounter,
    joint_angle,
    neck_angle,
)
from ovkit.pipelines.reid import normalize

IMG = np.zeros((200, 200, 3), np.uint8)


class _Fake:
    def __init__(self, build):
        self.build = build

    def __call__(self, image, **_kw):
        return [self.build(image)]


def _boxes(rows):
    return Boxes(np.array(rows, np.float32))


def _detector(rows, names):
    return _Fake(lambda img: Results(img, task="detect", names=names, boxes=_boxes(rows)))


def _pose(person):
    def build(img):
        r = Results(img, task="pose")
        r.keypoints = Keypoints(np.asarray(person, np.float32)[None])
        return r

    return _Fake(build)


def _person(joints):
    """A 17x3 pose with the given joints visible at (x, y)."""
    p = np.zeros((17, 3), np.float32)
    for idx, (x, y) in joints.items():
        p[idx] = [x, y, 1.0]
    return p


# -- count ------------------------------------------------------------------


def test_count_tallies_by_kind(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    pipe = Counter()
    pipe._models["detect"] = _detector(
        [[0, 0, 10, 10, 0.9, 0], [20, 0, 30, 10, 0.9, 0], [40, 0, 50, 10, 0.9, 41]],
        {0: "pencil", 41: "cup"},
    )
    assert pipe.run(IMG).summary() == "pencil 2 · cup 1"


def test_count_can_watch_one_kind_only():
    pipe = Counter(what="person")
    pipe._models["detect"] = _detector(
        [[0, 0, 10, 10, 0.9, 0], [20, 0, 30, 10, 0.9, 2]], {0: "person", 2: "car"}
    )
    r = pipe.run(IMG)
    assert len(r.boxes) == 1  # the car is filtered out of the picture too
    assert "person 1" in r.summary() or "사람 1" in r.summary()


def test_count_of_nothing_says_so(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    pipe = Counter()
    pipe._models["detect"] = _detector(np.zeros((0, 6), np.float32), {})
    assert pipe.run(IMG).summary() == "nothing counted"


# -- posture ----------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _upright():
    return _person({0: (100, 40), 5: (90, 100), 6: (110, 100)})


def _slouched():
    return _person({0: (145, 70), 5: (90, 100), 6: (110, 100)})


def test_neck_angle_is_zero_when_straight_and_grows_leaning():
    assert neck_angle(_upright()) == pytest.approx(0.0, abs=6.0)
    assert neck_angle(_slouched()) > 40.0
    assert neck_angle(_person({5: (90, 100), 6: (110, 100)})) is None  # no nose


def test_a_glance_down_is_not_bad_posture(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    clock = _Clock()
    pipe = PostureCoach(seconds=5.0, clock=clock)
    pipe._models["pose"] = _pose(_slouched())
    assert "자세 고치세요" not in pipe.run(IMG).summary()


def test_staying_slouched_past_the_timer_warns(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    clock = _Clock()
    pipe = PostureCoach(seconds=5.0, clock=clock)
    pipe._models["pose"] = _pose(_slouched())
    pipe.run(IMG)
    clock.now = 6.0
    assert "자세 고치세요" in pipe.run(IMG).summary()


def test_sitting_up_resets_the_timer(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    clock = _Clock()
    pipe = PostureCoach(seconds=5.0, clock=clock)
    pipe._models["pose"] = _pose(_slouched())
    pipe.run(IMG)
    clock.now = 3.0
    pipe._models["pose"] = _pose(_upright())
    assert "자세 좋아요" in pipe.run(IMG).summary()
    clock.now = 8.0
    pipe._models["pose"] = _pose(_slouched())
    assert "자세 고치세요" not in pipe.run(IMG).summary()  # timer restarted


# -- exercise ---------------------------------------------------------------


def _legs(knee_deg: float):
    """Hips-knees-ankles laid out so the knee angle is ``knee_deg``."""
    import math

    person = np.zeros((17, 3), np.float32)
    for hip, knee, ankle, x in ((11, 13, 15, 80.0), (12, 14, 16, 120.0)):
        person[hip] = [x, 100.0, 1.0]
        person[knee] = [x, 150.0, 1.0]
        rad = math.radians(180.0 - knee_deg)
        person[ankle] = [x + 50.0 * math.sin(rad), 150.0 + 50.0 * math.cos(rad), 1.0]
    return person


def test_joint_angle_matches_the_geometry():
    assert joint_angle(_legs(170.0), (11, 13, 15)) == pytest.approx(170.0, abs=0.5)
    assert joint_angle(_legs(90.0), (11, 13, 15)) == pytest.approx(90.0, abs=0.5)
    assert joint_angle(np.zeros((17, 3), np.float32), (11, 13, 15)) is None


def test_a_full_squat_cycle_counts_one_rep():
    counter = RepCounter(kind="squat")
    for deg in (170, 150, 100, 95, 120, 165):
        counter.update(_legs(deg))
    assert counter.reps == 1


def test_a_shallow_bounce_never_counts():
    counter = RepCounter(kind="스쿼트")
    for deg in (170, 140, 125, 150, 168):  # never below the down threshold
        counter.update(_legs(deg))
    assert counter.reps == 0


def test_three_squats_count_three(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    counter = RepCounter(kind="squat")
    for _ in range(3):
        for deg in (170, 100, 170):
            counter.update(_legs(deg))
    assert counter.reps == 3
    counter._models["pose"] = _pose(_legs(170))
    assert counter.run(IMG).summary() == "squat x 3 (up)"


def test_an_unknown_exercise_lists_the_real_ones():
    with pytest.raises(OVKitError, match="squat"):
        RepCounter(kind="curl")


# -- attendance -------------------------------------------------------------


def _face_detector(rows):
    return _Fake(lambda img: Results(img, task="detect", names={0: "face"}, boxes=_boxes(rows)))


@pytest.fixture
def roll(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    att = Attendance()
    att.matcher.gallery = {
        "철수": normalize(np.array([1.0, 0.0, 0.0])),
        "영희": normalize(np.array([0.0, 1.0, 0.0])),
        "민수": normalize(np.array([0.0, 0.0, 1.0])),
    }
    vectors = {60: [0.95, 0.1, 0.0], 120: [0.05, 0.9, 0.1], 30: [0.5, 0.5, 0.5]}

    def embed(image):
        return normalize(np.array(vectors[int(np.asarray(image).ravel()[0])], np.float32))

    att.matcher.embed = embed
    return att


def _frame(value: int) -> np.ndarray:
    return np.full((200, 200, 3), value, np.uint8)


def test_attendance_marks_matched_faces_present(roll):
    roll._models["face_detection"] = _face_detector([[0, 0, 100, 100, 0.9, 0]])
    r = roll.run(_frame(60), conf=0.5)
    assert "철수" in r.summary()
    assert roll.present.keys() == {"철수"}
    assert set(roll.absent) == {"영희", "민수"}


def test_attendance_accumulates_across_frames(roll):
    roll._models["face_detection"] = _face_detector([[0, 0, 100, 100, 0.9, 0]])
    roll.run(_frame(60))
    roll.run(_frame(120))
    assert set(roll.present) == {"철수", "영희"}
    assert roll.absent == ["민수"]


def test_a_stranger_is_a_question_mark_not_a_student(roll):
    roll.matcher.threshold = 0.9
    roll._models["face_detection"] = _face_detector([[0, 0, 100, 100, 0.9, 0]])
    r = roll.run(_frame(30))  # equally far from everyone
    assert r.labels == ["?"]
    assert not roll.present


def test_attendance_without_a_roster_explains_itself(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    with pytest.raises(OVKitError, match="roster"):
        Attendance().run(IMG)


def test_attendance_csv_lists_the_whole_roster(roll, tmp_path):
    roll._models["face_detection"] = _face_detector([[0, 0, 100, 100, 0.9, 0]])
    roll.run(_frame(60))
    path = roll.save_csv(str(tmp_path / "roll.csv"))
    rows = open(path, encoding="utf-8-sig").read().splitlines()
    assert rows[0] == "name,status,score"
    assert len(rows) == 4  # header + 3 students
    assert any("철수,출석" in row for row in rows)
    assert any("민수,결석" in row for row in rows)


def test_roster_folder_layouts(monkeypatch, tmp_path):
    import cv2

    (tmp_path / "철수.png").parent.mkdir(exist_ok=True)
    cv2.imwrite(str(tmp_path / "철수.png"), _frame(60))
    (tmp_path / "영희").mkdir()
    cv2.imwrite(str(tmp_path / "영희" / "a.png"), _frame(120))

    att = Attendance()
    seen = []
    att.matcher.add = lambda name, path: seen.append(name)
    loaded = att.load_roster(str(tmp_path))
    assert loaded == ["영희", "철수"] or loaded == ["철수", "영희"]
    assert set(seen) == {"철수", "영희"}


def test_korean_names_build_the_classroom_pipelines():
    assert isinstance(Model("개수세기"), Counter)
    assert isinstance(Model("거북목"), PostureCoach)
    assert isinstance(Model("운동횟수", kind="스쿼트"), RepCounter)
    assert isinstance(Model("출석체크"), Attendance)
