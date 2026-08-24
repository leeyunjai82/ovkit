"""Teach your own AI — a Teachable Machine in five lines of Python.

    ai = Model("teach")                     # or Model("가르치기")
    ai.learn("can", "photos/cans/")         # a folder per thing to recognise
    ai.learn("bottle", "photos/bottles/")
    print(ai.guess("new_photo.jpg"))        # ('can', 0.93)
    print(ai.score("test_photos/"))         # accuracy + what it confuses
    ai.save("recycling")                    # Documents/ovkit/recycling.json

No GPU and no training loop: an embedding model turns each example into a
vector once, and new inputs are matched to the nearest stored examples
(cosine k-NN). Five modes decide what the vector describes — the same five
The Maker validated in classrooms:

    photo (사진)   whole images            image embedding
    face  (표정)   facial expressions      face crop embedding + emotion
    hand  (손모양)  hand shapes            MediaPipe 21 landmarks  [ovkit[hand]]
    upper (상반신)  upper-body posture     pose keypoints, upper subset
    body  (전신)   whole-body poses        pose keypoints, all 17

A ``Teach`` object is also a pipeline: ``ai(frame)`` / ``ai.predict(0)``
answer with normal ovkit ``Results``, so a webcam loop over your own
classes looks exactly like every other capability.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OVKitError
from ..core.i18n import lang
from ..core.results import Probs, Results
from .base import Pipeline

#: mode -> canonical english key (Korean values accepted, arguments stay english)
MODES = {
    "photo": "photo",
    "사진": "photo",
    "face": "face",
    "표정": "face",
    "hand": "hand",
    "손모양": "hand",
    "손": "hand",
    "upper": "upper",
    "상반신": "upper",
    "body": "body",
    "전신": "body",
}

#: COCO-17 indices that belong to the upper body (nose..wrists).
_UPPER = tuple(range(11))

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _msg(ko: str, en: str) -> str:
    return ko if lang() == "ko" else en


@dataclass
class Score:
    """What :meth:`Teach.score` hands back — prints as one readable line."""

    accuracy: float
    total: int
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def __str__(self) -> str:
        worst = [
            (t, p)
            for (t, p), n in sorted(self.confusion.items(), key=lambda kv: -kv[1])
            if t != p and n
        ]
        head = (
            f"정확도 {self.accuracy * 100:.0f}% ({self.total}장)"
            if lang() == "ko"
            else f"accuracy {self.accuracy * 100:.0f}% ({self.total} images)"
        )
        if worst:
            t, p = worst[0]
            head += f" · {'헷갈린 것' if lang() == 'ko' else 'confuses'}: {t}↔{p}"
        return head


class Teach(Pipeline):
    """Learn categories from examples; recognise them like any other model."""

    name = "teach"
    description = "Teach your own AI from a few examples — no GPU training (5 modes)."

    def __init__(
        self,
        device: str = "AUTO",
        mode: str = "photo",
        k: int = 5,
        load: str | Path | None = None,
    ) -> None:
        super().__init__(device)
        key = MODES.get(str(mode).strip().lower(), MODES.get(str(mode).strip()))
        if key is None:
            raise OVKitError(
                _msg(
                    f"'{mode}'는 모르는 모드예요. 가능한 모드: 사진, 표정, 손모양, 상반신, 전신",
                    f"unknown mode '{mode}'. Modes: photo, face, hand, upper, body",
                )
            )
        self.mode = key
        self.k = int(k)
        self._vectors: list[np.ndarray] = []
        self._labels: list[str] = []
        if load is not None:
            self._load_into(load)

    # -- teaching -----------------------------------------------------------

    def learn(self, label: str, source: Any) -> int:
        """Add examples of ``label``: a folder, one image path, or an array.

        Returns how many examples were stored. Call again with the same label
        to add more — more examples make a steadier answer.
        """
        label = str(label).strip()
        if not label:
            raise OVKitError(_msg("이름을 적어 주세요.", "give the label a name."))
        added = 0
        for image in self._iter_examples(source):
            try:
                vector = self._embed(image)
            except OVKitError:
                continue  # e.g. no face in this particular photo — skip it
            self._vectors.append(vector)
            self._labels.append(label)
            added += 1
        if not added:
            raise OVKitError(
                _msg(
                    f"'{label}'에서 배울 수 있는 예시를 하나도 못 찾았어요.",
                    f"found nothing to learn for '{label}'.",
                )
            )
        return added

    def forget(self, label: str) -> None:
        """Drop everything learned under ``label``."""
        keep = [i for i, name in enumerate(self._labels) if name != label]
        self._vectors = [self._vectors[i] for i in keep]
        self._labels = [self._labels[i] for i in keep]

    @property
    def labels(self) -> list[str]:
        """The labels taught so far (in first-taught order)."""
        seen: list[str] = []
        for name in self._labels:
            if name not in seen:
                seen.append(name)
        return seen

    # -- answering ----------------------------------------------------------

    def guess(self, source: Any) -> tuple[str, float]:
        """The best label and its confidence for one image."""
        weights = self._weights(self._embed(self._one_image(source)))
        label = max(weights, key=weights.get)
        return label, round(weights[label], 3)

    def run(self, image: np.ndarray, **_: Any) -> Results:
        weights = self._weights(self._embed(image))
        names = {i: name for i, name in enumerate(self.labels)}
        scores = np.array([weights.get(names[i], 0.0) for i in sorted(names)], np.float32)
        result = Results(image, task=self.name, names=names, probs=Probs(scores))
        top = int(np.argmax(scores))
        result.text = f"{names[top]} {float(scores[top]):.2f}"
        return result

    def score(self, source: Any) -> Score:
        """Grade the AI on a folder of subfolders (one per true label)."""
        root = Path(str(source))
        cases: list[tuple[str, Path]] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXT):
                cases.append((sub.name, img))
        if not cases:
            raise OVKitError(
                _msg(
                    f"{root} 안에 '라벨/사진들' 구조의 폴더가 필요해요.",
                    f"{root} needs one subfolder per label, with images inside.",
                )
            )
        confusion: dict[tuple[str, str], int] = {}
        correct = 0
        for truth, path in cases:
            predicted, _ = self.guess(path)
            confusion[(truth, predicted)] = confusion.get((truth, predicted), 0) + 1
            correct += truth == predicted
        return Score(accuracy=correct / len(cases), total=len(cases), confusion=confusion)

    # -- persistence --------------------------------------------------------

    @staticmethod
    def _store_dir() -> Path:
        docs = Path.home() / "Documents"
        base = docs / "ovkit" if docs.is_dir() else Path.home() / ".cache" / "ovkit" / "user"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def save(self, name: str, dir: str | Path | None = None) -> Path:
        """Write everything learned to ``<Documents>/ovkit/<name>.json``."""
        slug = re.sub(r"\s+", "-", str(name).strip()) or "my-ai"
        path = Path(dir) if dir is not None else self._store_dir()
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{slug}.json"
        out.write_text(
            json.dumps(
                {
                    "format": "ovkit-teach-1",
                    "mode": self.mode,
                    "labels": self._labels,
                    "vectors": [v.tolist() for v in self._vectors],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return out

    def _load_into(self, name: str | Path) -> None:
        path = Path(str(name))
        if path.suffix != ".json":
            slug = re.sub(r"\s+", "-", str(name).strip())
            path = self._store_dir() / f"{slug}.json"
        if not path.is_file():
            raise OVKitError(_msg(f"저장된 AI를 못 찾았어요: {path}", f"no saved AI at {path}"))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "ovkit-teach-1":
            raise OVKitError(_msg(f"모르는 파일 형식이에요: {path}", f"unknown format: {path}"))
        self.mode = data["mode"]
        self._labels = list(data["labels"])
        self._vectors = [np.asarray(v, np.float32) for v in data["vectors"]]

    # -- Korean method names (the docs teach both) --------------------------

    배우기 = learn
    맞혀봐 = guess
    점수 = score
    저장 = save
    잊어버려 = forget

    # -- features -----------------------------------------------------------

    def _weights(self, vector: np.ndarray) -> dict[str, float]:
        """Cosine k-NN vote: label -> share of the top-k similarity mass."""
        if not self._vectors:
            raise OVKitError(
                _msg(
                    "아직 배운 게 없어요 — 먼저 learn(이름, 사진폴더) 을 불러 주세요.",
                    "nothing learned yet — call learn(label, folder) first.",
                )
            )
        sims = np.stack(self._vectors) @ vector
        order = np.argsort(sims)[::-1][: max(1, min(self.k, len(sims)))]
        mass: dict[str, float] = {name: 0.0 for name in self.labels}
        total = 0.0
        for i in order:
            weight = float(max(sims[int(i)], 0.0))
            mass[self._labels[int(i)]] += weight
            total += weight
        if total <= 0:
            return {name: 1.0 / len(mass) for name in mass}
        return {name: weight / total for name, weight in mass.items()}

    def _embed(self, image: np.ndarray) -> np.ndarray:
        image = self._one_image(image)
        feature = {
            "photo": self._embed_photo,
            "face": self._embed_face,
            "hand": self._embed_hand,
            "upper": lambda img: self._embed_pose(img, upper=True),
            "body": lambda img: self._embed_pose(img, upper=False),
        }[self.mode](image)
        return _unit(feature)

    def _embed_photo(self, image: np.ndarray) -> np.ndarray:
        return self._image_vector(image)

    def _image_vector(self, image: np.ndarray) -> np.ndarray:
        out = self.model("image_retrieval")(image)
        tensors = out[0].tensors if out else None
        if not tensors:
            raise OVKitError("embedding model returned nothing")
        return np.asarray(next(iter(tensors.values())), np.float32).reshape(-1)

    def _embed_face(self, image: np.ndarray) -> np.ndarray:
        found = self.model("face_detection")(image, conf=0.5)
        boxes = found[0].boxes if found else None
        if boxes is None or not len(boxes):
            raise OVKitError(_msg("사진에서 얼굴을 못 찾았어요.", "no face in this image."))
        largest = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.xyxy]))
        crop = found[0].crop(largest, pad=0.15)
        vector = _unit(self._image_vector(crop))
        emotion = self.model("emotion")(crop)
        probs = (
            np.asarray(emotion[0].probs.data, np.float32)
            if emotion and emotion[0].probs is not None
            else np.zeros(5, np.float32)
        )
        # The expression carries little weight in a generic image embedding, so
        # the 5 emotion probabilities ride along with equal overall influence.
        return np.concatenate([vector, probs])

    def _embed_hand(self, image: np.ndarray) -> np.ndarray:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                _msg(
                    '손모양 모드는 추가 설치가 필요해요:  pip install "ovkit[hand]"',
                    "hand mode needs the extra:  pip install 'ovkit[hand]'",
                )
            ) from exc
        import cv2

        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            found = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not found.multi_hand_landmarks:
            raise OVKitError(_msg("사진에서 손을 못 찾았어요.", "no hand in this image."))
        marks = found.multi_hand_landmarks[0].landmark
        points = np.array([[m.x, m.y] for m in marks], np.float32)
        return _normalize_points(points)

    def _embed_pose(self, image: np.ndarray, upper: bool) -> np.ndarray:
        out = self.model("pose")(image)
        keypoints = out[0].keypoints if out else None
        if keypoints is None or len(keypoints.data) == 0:
            raise OVKitError(_msg("사진에서 사람을 못 찾았어요.", "no person in this image."))
        person = keypoints.data[0]  # the most confident instance
        return _pose_feature(person, upper=upper)

    def _iter_examples(self, source: Any):
        if isinstance(source, np.ndarray):
            yield source
            return
        if isinstance(source, (list, tuple)):
            for item in source:
                yield from self._iter_examples(item)
            return
        path = Path(str(source))
        if path.is_dir():
            for img in sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXT):
                yield self._read(img)
            return
        if path.is_file():
            yield self._read(path)
            return
        raise OVKitError(_msg(f"예시를 못 찾았어요: {source}", f"no examples found at: {source}"))

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        from ..image.ops import imread

        return imread(path)

    @staticmethod
    def _one_image(source: Any) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        from ..image.ops import imread

        return imread(str(source))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Center on the first point and scale by spread — translation/size free."""
    coords = points - points[0]
    scale = float(np.abs(coords).max())
    return (coords / scale if scale > 1e-9 else coords).reshape(-1)


def _pose_feature(person: np.ndarray, upper: bool) -> np.ndarray:
    """A 17-keypoint pose as a position/size-free vector; unseen joints zero."""
    keypoints = np.asarray(person, np.float32)
    visible = keypoints[:, 2] > 0
    if not visible.any():
        raise OVKitError("no visible keypoints")
    xy = keypoints[:, :2].copy()
    center = xy[visible].mean(axis=0)
    spread = float(np.abs(xy[visible] - center).max())
    xy = (xy - center) / (spread if spread > 1e-9 else 1.0)
    xy[~visible] = 0.0
    if upper:
        xy = xy[list(_UPPER)]
    return xy.reshape(-1)


def collect(
    label: str,
    count: int = 30,
    camera: int = 0,
    out: str | Path | None = None,
    every: float = 0.4,
) -> Path:
    """Capture ``count`` webcam frames into a folder ready for :meth:`learn`.

        collect("가위", 30)      ->  Documents/ovkit/examples/가위/0001.jpg ...

    A preview window opens when a display is available (q quits early).
    """
    import cv2

    folder = Path(out) if out is not None else Teach._store_dir() / "examples" / str(label)
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise OVKitError(
            _msg(f"웹캠({camera})을 열 수 없어요.", f"could not open camera {camera}.")
        )
    saved, last = 0, 0.0
    try:
        while saved < count:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.monotonic()
            if now - last >= every:
                cv2.imwrite(str(folder / f"{saved + 1:04d}.jpg"), frame)
                saved += 1
                last = now
            try:
                cv2.imshow(f"collect: {label} ({saved}/{count}) — q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass  # headless: keep capturing without a preview
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return folder
