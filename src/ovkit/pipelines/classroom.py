"""Classroom capabilities: count things, watch posture, count reps, take roll.

Four small pipelines that turn the models already on the mirror into the
things a lesson actually asks for::

    Model("count", "desk.jpg")                  # pencil 3 · cup 1
    Model("posture", 0)                         # neck 32° — sit up!
    Model("exercise", 0, kind="squat")          # squat x 12
    Model("attendance", roster="class_photos/") # present 24 / absent 2
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.errors import OVKitError
from ..core.i18n import display_name, lang
from ..core.results import Boxes, Results
from .base import Pipeline, detections
from .reid import ReID

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _msg(ko: str, en: str) -> str:
    return ko if lang() == "ko" else en


# ---------------------------------------------------------------- count


class Counter(Pipeline):
    """Count what a detector sees, optionally only one kind of thing.

    >>> Model("count", "desk.jpg").summary()        # 'pencil 3 · cup 1'
    >>> Model("count", 0, what="person")            # live head-count
    """

    name = "count"
    description = "Count the objects in view, by kind (optionally just one kind)."

    def __init__(self, device: str = "AUTO", detector: str = "detect", what: str | None = None):
        super().__init__(device)
        self.detector = detector
        #: English class key to count exclusively (``person``, ``cell-phone`` ...).
        self.what = str(what).strip().lower().replace(" ", "-") if what else None

    def run(self, image: np.ndarray, *, conf: float = 0.4, **_: Any) -> Results:
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        result = Results(image, task=self.name, names=found.names, boxes=boxes)

        counts: dict[str, int] = {}
        keep_rows = []
        for i, cls in enumerate(boxes.cls.astype(int)):
            name_en = found.names.get(int(cls), str(int(cls))).replace(" ", "-")
            if self.what and name_en != self.what:
                continue
            keep_rows.append(i)
            counts[found.names.get(int(cls), str(int(cls)))] = (
                counts.get(found.names.get(int(cls), str(int(cls))), 0) + 1
            )
        if self.what:
            result.boxes = (
                Boxes(boxes.data[keep_rows]) if keep_rows else Boxes(np.zeros((0, 6), np.float32))
            )
        result.labels = [result.label_for(i) for i in range(len(result.boxes))]
        if not counts:
            result.text = _msg("아무것도 못 찾았어요", "nothing counted")
        else:
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            result.text = " · ".join(f"{display_name(n)} {c}" for n, c in ranked)
        result.tensors = {"counts": np.array(sorted(counts.values()), np.int32)}
        return result


# ---------------------------------------------------------------- posture

#: COCO-17 indices used for the neck estimate.
_NOSE, _L_SHOULDER, _R_SHOULDER = 0, 5, 6


def neck_angle(person: np.ndarray) -> float | None:
    """Degrees between vertical and the shoulder-midpoint→nose vector.

    0° is a straight neck; a head drifting forward or down grows the angle.
    ``None`` when the nose or both shoulders are unseen.
    """
    keypoints = np.asarray(person, np.float32)
    if keypoints[_NOSE, 2] <= 0:
        return None
    shoulders = [i for i in (_L_SHOULDER, _R_SHOULDER) if keypoints[i, 2] > 0]
    if not shoulders:
        return None
    mid = keypoints[shoulders, :2].mean(axis=0)
    dx, dy = keypoints[_NOSE, :2] - mid  # image y grows downward
    up = np.array([0.0, -1.0])
    vec = np.array([dx, dy], np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return 0.0
    cos = float(np.clip(vec @ up / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class PostureCoach(Pipeline):
    """Warn when the neck stays bent past a threshold — the turtle-neck timer.

        >>> for r in Model("posture", 0):
        ...     print(r)          # '자세 좋아요 (12°)' ... '목이 34° — 자세 고치세요'

    A single glance down is not bad posture, so the warning needs the angle to
    *stay* past ``degrees`` for ``seconds`` — the same time-not-frames logic as
    drowsiness. Works best with the camera at screen height, face-on or side-on.
    """

    name = "posture"
    description = "Watch the neck angle over time and warn when posture slips."

    def __init__(
        self,
        device: str = "AUTO",
        degrees: float = 25.0,
        seconds: float = 5.0,
        clock: Callable[[], float] | None = None,
    ):
        super().__init__(device)
        self.degrees = float(degrees)
        self.seconds = float(seconds)
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self) -> None:
        self._bad_since: float | None = None

    def run(self, image: np.ndarray, **_: Any) -> Results:
        out = self.model("pose")(image)
        keypoints = out[0].keypoints if out else None
        result = Results(image, task=self.name)
        if keypoints is None or len(keypoints.data) == 0:
            self.reset()
            result.text = _msg("사람이 안 보여요", "nobody in view")
            return result
        result.keypoints = keypoints

        angle = neck_angle(keypoints.data[0])
        if angle is None:
            self.reset()
            result.text = _msg("목이 잘 안 보여요", "cannot see the neck")
            return result

        now = self._clock()
        if angle >= self.degrees:
            self._bad_since = self._bad_since if self._bad_since is not None else now
        else:
            self._bad_since = None
        bad_for = (now - self._bad_since) if self._bad_since is not None else 0.0

        if bad_for >= self.seconds:
            result.text = _msg(
                f"목이 {angle:.0f}° — 자세 고치세요 ({bad_for:.0f}초째)",
                f"neck at {angle:.0f}° — sit up ({bad_for:.0f}s)",
            )
        else:
            result.text = _msg(f"자세 좋아요 ({angle:.0f}°)", f"posture ok ({angle:.0f}°)")
        result.tensors = {"neck_deg": np.array([angle], np.float32)}
        return result


# ---------------------------------------------------------------- exercise

#: exercise -> (three keypoint indices per side, down-threshold°, up-threshold°)
EXERCISES: dict[str, tuple[tuple[tuple[int, int, int], ...], float, float]] = {
    # knee angle: hip-knee-ankle
    "squat": (((11, 13, 15), (12, 14, 16)), 110.0, 160.0),
    # elbow angle: shoulder-elbow-wrist
    "pushup": (((5, 7, 9), (6, 8, 10)), 95.0, 150.0),
}
_EXERCISE_KO = {"스쿼트": "squat", "팔굽혀펴기": "pushup", "푸시업": "pushup"}


def joint_angle(person: np.ndarray, triple: tuple[int, int, int]) -> float | None:
    """Angle in degrees at the middle joint of ``triple`` (None if unseen)."""
    keypoints = np.asarray(person, np.float32)
    a, b, c = (keypoints[i] for i in triple)
    if min(a[2], b[2], c[2]) <= 0:
        return None
    v1, v2 = a[:2] - b[:2], c[:2] - b[:2]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class RepCounter(Pipeline):
    """Count exercise repetitions from the pose stream.

        >>> for r in Model("exercise", 0, kind="squat"):
        ...     print(r)          # 'squat x 12 (down)'

    A rep is one full cycle through hysteresis: the joint angle must drop
    below the exercise's *down* threshold and come back above *up* — so a
    half-hearted bounce in the middle never counts.
    """

    name = "exercise"
    description = "Count squats / push-ups by tracking joint angles over time."

    def __init__(self, device: str = "AUTO", kind: str = "squat"):
        super().__init__(device)
        key = _EXERCISE_KO.get(str(kind).strip(), str(kind).strip().lower())
        if key not in EXERCISES:
            known = ", ".join(sorted(EXERCISES) + sorted(_EXERCISE_KO))
            raise OVKitError(
                _msg(
                    f"'{kind}'는 모르는 운동이에요. 가능: {known}",
                    f"unknown exercise '{kind}'. Known: {known}",
                )
            )
        self.kind = key
        self.reset()

    def reset(self) -> None:
        self.reps = 0
        self._down = False

    def run(self, image: np.ndarray, **_: Any) -> Results:
        out = self.model("pose")(image)
        keypoints = out[0].keypoints if out else None
        result = Results(image, task=self.name)
        if keypoints is None or len(keypoints.data) == 0:
            result.text = _msg("사람이 안 보여요", "nobody in view")
            return result
        result.keypoints = keypoints
        self.update(keypoints.data[0])

        phase = _msg("내려감", "down") if self._down else _msg("올라옴", "up")
        result.text = f"{self.kind} x {self.reps} ({phase})"
        result.tensors = {"reps": np.array([self.reps], np.int32)}
        return result

    def update(self, person: np.ndarray) -> int:
        """Feed one pose; returns the rep count (exposed for testing)."""
        triples, down_deg, up_deg = EXERCISES[self.kind]
        angles = [angle for t in triples if (angle := joint_angle(person, t)) is not None]
        if not angles:
            return self.reps
        angle = float(np.mean(angles))
        if not self._down and angle <= down_deg:
            self._down = True
        elif self._down and angle >= up_deg:
            self._down = False
            self.reps += 1
        return self.reps


# ---------------------------------------------------------------- attendance


class Attendance(Pipeline):
    """Take the roll with face matching against a roster you provide.

        >>> att = Model("attendance", roster="class_photos/")
        >>> for r in att.predict(0, stream=True):
        ...     print(r)                      # 출석 3: 철수, 영희, 민수
        >>> att.save_csv("roll.csv")          # name,status,score

    The roster folder holds either one photo per student (``철수.jpg``) or a
    folder per student with several photos (better). Matching is the same
    cosine gallery as ``face_match``, run on every detected face crop.
    """

    name = "attendance"
    description = "Face-match against a roster folder and keep who is present."

    def __init__(
        self,
        device: str = "AUTO",
        roster: str | None = None,
        threshold: float = 0.5,
        detector: str = "face_detection",
    ):
        super().__init__(device)
        self.detector = detector
        self.matcher = ReID(device=device, threshold=threshold)
        self.present: dict[str, float] = {}
        if roster:
            self.load_roster(roster)

    # -- roster -------------------------------------------------------------

    def load_roster(self, folder: str) -> list[str]:
        """Fill the gallery from a roster folder; returns the names loaded."""
        from pathlib import Path

        root = Path(folder)
        loaded: list[str] = []
        for entry in sorted(root.iterdir()) if root.is_dir() else []:
            if entry.is_file() and entry.suffix.lower() in _IMAGE_EXT:
                self.matcher.add(entry.stem, str(entry))
                loaded.append(entry.stem)
            elif entry.is_dir():
                images = [p for p in sorted(entry.iterdir()) if p.suffix.lower() in _IMAGE_EXT]
                for img in images:
                    self.matcher.add(entry.name, str(img))
                if images:
                    loaded.append(entry.name)
        if not loaded:
            raise OVKitError(
                _msg(
                    f"{folder} 에서 명단을 못 만들었어요 — 학생당 사진 한 장(철수.jpg) "
                    f"또는 폴더(철수/)가 필요해요.",
                    f"no roster in {folder} — one photo (or one folder) per person.",
                )
            )
        return loaded

    @property
    def roster(self) -> list[str]:
        return sorted(self.matcher.gallery)

    @property
    def absent(self) -> list[str]:
        return [name for name in self.roster if name not in self.present]

    def reset(self) -> None:
        self.present.clear()

    # -- running ------------------------------------------------------------

    def run(self, image: np.ndarray, *, conf: float = 0.5, **_: Any) -> Results:
        if not self.matcher.gallery:
            raise OVKitError(
                _msg(
                    "명단이 비어 있어요 — Model('attendance', roster='반사진/') 처럼 "
                    "명단 폴더를 먼저 주세요.",
                    "the roster is empty — pass roster='folder/' first.",
                )
            )
        found = detections(self.model(self.detector), image, conf)
        boxes = found.boxes if found.boxes is not None else Boxes(np.zeros((0, 6), np.float32))
        result = Results(image, task=self.name, names={0: "face"}, boxes=boxes)

        labels: list[str] = []
        for i in range(len(boxes)):
            crop = found.crop(i, pad=0.15)
            hit = self.matcher.who(crop) if crop.size else None
            if hit is None:
                labels.append("?")
                continue
            name, score = hit
            labels.append(f"{name} {score:.2f}")
            self.present[name] = max(score, self.present.get(name, 0.0))
        result.labels = labels

        names = ", ".join(sorted(self.present)) or "-"
        result.text = _msg(
            f"출석 {len(self.present)}/{len(self.roster)}: {names}",
            f"present {len(self.present)}/{len(self.roster)}: {names}",
        )
        return result

    def save_csv(self, path: str) -> str:
        """Write ``name,status,score`` for the whole roster."""
        import csv

        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "status", "score"])
            for name in self.roster:
                if name in self.present:
                    writer.writerow([name, _msg("출석", "present"), f"{self.present[name]:.2f}"])
                else:
                    writer.writerow([name, _msg("결석", "absent"), ""])
        return path
