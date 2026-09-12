"""The GUI's brain — everything except the widgets.

Keeping the state machine, the worker thread and the model calls out of the Tk
layer means this can be tested without a display, and a different front end
(Qt, a web page) could drive the same logic.
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Choice:
    """One entry in the GUI's list of things ovkit can do."""

    name: str  #: what Model() is called with
    label: str  #: what the button says, in the display language
    hint: str  #: one line under the picture
    korean_name: str = ""  #: the Korean name Model() also accepts, if there is one


#: ``name -> (한국어 버튼, 한국어 설명, English button, English hint)``.
#:
#: The buttons used to be English only, in a package whose display language
#: defaults to Korean and whose whole point is that a student can type
#: ``Model("얼굴분석")``. Someone who opens the window first met "Describe /
#: Faces / Objects" — the one place where the Korean was missing was the place
#: a beginner starts.
_CHOICES: tuple[tuple[str, str, str, str, str], ...] = (
    # Capabilities first: they answer a question rather than emit a tensor.
    (
        "scene",
        "장면 설명",
        "사진 한 장을 한 문장으로",
        "Describe",
        "One sentence about the whole picture",
    ),
    (
        "face_analyze",
        "얼굴 분석",
        "얼굴마다 나이·성별·표정",
        "Faces",
        "Age, gender and emotion for every face",
    ),
    (
        "detect",
        "물체 찾기",
        "사진 속 물건 찾기 (COCO 80종)",
        "Objects",
        "Find the objects in the picture (COCO-80)",
    ),
    ("count", "개수 세기", "무엇이 몇 개인지 세기", "Count", "How many of each kind"),
    ("read_text", "글자 읽기", "글자를 찾아서 읽기", "Read text", "Find text and read it"),
    (
        "read_plate",
        "번호판 읽기",
        "번호판을 읽고 차를 설명",
        "Plates",
        "Read number plates, describe the car",
    ),
    (
        "track",
        "따라가기",
        "프레임이 바뀌어도 같은 번호",
        "Track",
        "Objects keep an id from frame to frame",
    ),
    (
        "drowsiness",
        "졸음 감지",
        "눈이 감긴 채로 있으면 알림 (웹캠)",
        "Drowsy?",
        "Warn when the eyes stay shut (webcam)",
    ),
    (
        "posture",
        "거북목 알림",
        "목 각도를 보고 알림 (웹캠)",
        "Posture",
        "Watch the neck angle (webcam)",
    ),
    (
        "exercise",
        "운동 횟수",
        "스쿼트·팔굽혀펴기 세기 (웹캠)",
        "Exercise",
        "Count squats and push-ups (webcam)",
    ),
    (
        "gesture",
        "동작 알아보기",
        "움직임으로 손동작 알아보기 (웹캠)",
        "Gesture",
        "Hand gestures from motion (webcam)",
    ),
    (
        "attention",
        "뭘 보나",
        "사람이 무엇을 보고 있는지",
        "Attention",
        "Which object the person is looking at",
    ),
    ("gaze", "시선", "얼굴이 어디를 보는지", "Gaze", "Where a face is looking"),
    (
        "anonymize",
        "모자이크",
        "얼굴을 가려서 공유할 수 있게",
        "Blur faces",
        "Redact faces so the picture can be shared",
    ),
    (
        "remove_background",
        "배경 지우기",
        "피사체만 남기기",
        "Cut out",
        "Keep the subject, drop the background",
    ),
    ("depth", "거리 재기", "무엇이 얼마나 가까운지", "Distance", "How far away everything is"),
    (
        "person_analyze",
        "사람 분석",
        "무엇을 입고 들었는지",
        "People",
        "What each person wears or carries",
    ),
    ("vehicle_analyze", "차량 분석", "차 종류와 색", "Vehicles", "Type and colour of each vehicle"),
    # A few single models, for when you want exactly one.
    ("pose", "자세", "사람마다 관절 위치", "Pose", "Body keypoints for every person"),
    ("segment", "영역 나누기", "픽셀마다 이름 붙이기", "Segment", "Label every pixel"),
    ("classify", "이건 뭐야", "이 사진은 무엇인가", "Classify", "What is this a picture of?"),
)


def choices() -> list[Choice]:
    """The short, opinionated list a beginner should start from.

    Not every registered model — the point of the GUI is to answer a question,
    so it offers capabilities first and a handful of single models after. The
    labels follow the display language.
    """
    from ..core.i18n import KO_CAPS, lang

    korean = {target: name for name, target in KO_CAPS.items()}
    in_korean = lang() == "ko"
    return [
        Choice(
            name=name,
            label=ko_label if in_korean else en_label,
            hint=ko_hint if in_korean else en_hint,
            korean_name=korean.get(name, ""),
        )
        for name, ko_label, ko_hint, en_label, en_hint in _CHOICES
    ]


@dataclass
class View:
    """A snapshot of what the window should show right now."""

    frame: np.ndarray | None = None
    answer: str = ""
    status: str = "Pick something on the left."
    busy: bool = False
    live: bool = False
    error: str = ""
    choice: str = ""
    counter: int = 0


@dataclass
class _Job:
    kind: str
    payload: Any = None
    extra: dict = field(default_factory=dict)


class Controller:
    """Loads models, runs them, and publishes frames for the window to draw.

    Everything slow happens on one worker thread; the window only ever reads
    :meth:`view`, so it never blocks while a model downloads.
    """

    def __init__(
        self,
        device: str = "AUTO",
        model_factory: Callable[..., Any] | None = None,
        capture_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.device = device
        self._model_factory = model_factory or _default_model_factory
        self._capture_factory = capture_factory or _default_capture_factory
        self._models: dict[str, Any] = {}
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._lock = threading.Lock()
        self._view = View()
        self._stop_live = threading.Event()
        self._closing = threading.Event()
        self._current: str = ""
        self._image: np.ndarray | None = None
        self._conf = 0.25
        self._worker = threading.Thread(target=self._run, name="ovkit-gui", daemon=True)
        self._worker.start()

    # -- what the window calls ---------------------------------------------

    def view(self) -> View:
        """The current snapshot (cheap; safe to poll every frame)."""
        with self._lock:
            return View(**vars(self._view))

    def select(self, name: str) -> None:
        """Switch to a capability, loading it in the background."""
        self._jobs.put(_Job("select", name))

    def open_image(self, path: str | Path) -> None:
        """Load a picture and run the current capability on it."""
        self._jobs.put(_Job("image", str(path)))

    def start_webcam(self, index: int = 0) -> None:
        """Run continuously on the camera until :meth:`stop`."""
        self._jobs.put(_Job("webcam", index))

    def stop(self) -> None:
        """Stop the live loop (the loaded model stays loaded)."""
        self._stop_live.set()

    def set_conf(self, value: float) -> None:
        """Change the confidence threshold and re-run a still image.

        A live stream picks the new value up on its next frame, so only a
        still picture needs re-running.
        """
        self._conf = float(value)
        if self._image is not None and not self.view().live:
            self._jobs.put(_Job("rerun"))

    def set_device(self, device: str) -> None:
        """Switch device — every model is reloaded on the next run."""
        self.stop()
        self.device = device
        self._models.clear()
        self._jobs.put(_Job("select", self._current or ""))

    def close(self) -> None:
        """Stop everything and let the worker exit."""
        self._stop_live.set()
        self._closing.set()
        self._jobs.put(_Job("quit"))

    def save(self, path: str | Path) -> Path | None:
        """Write the frame currently on screen."""
        frame = self.view().frame
        if frame is None:
            return None
        from ..image.ops import imwrite

        imwrite(path, frame)
        return Path(path)

    # -- worker -------------------------------------------------------------

    def _publish(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._view, key, value)
            self._view.counter += 1

    def _run(self) -> None:
        while not self._closing.is_set():
            job = self._jobs.get()
            try:
                self._handle(job)
            except Exception as exc:  # a GUI must never die of a bad model
                self._publish(
                    busy=False,
                    live=False,
                    error=_readable(exc),
                    status="Something went wrong.",
                )
            if job.kind == "quit":
                return

    def _handle(self, job: _Job) -> None:
        if job.kind == "quit":
            return
        if job.kind == "select":
            self._select(job.payload)
        elif job.kind == "image":
            self._open_image(job.payload)
        elif job.kind == "rerun":
            self._rerun()
        elif job.kind == "webcam":
            self._webcam(int(job.payload))

    def _model(self, name: str) -> Any:
        if name not in self._models:
            self._publish(
                busy=True,
                error="",
                status=f"Loading {name}... (the first run downloads it)",
            )
            self._models[name] = self._model_factory(name, device=self.device)
        return self._models[name]

    def _select(self, name: str) -> None:
        if not name:
            return
        self._stop_live.set()
        self._current = name
        self._model(name)
        self._publish(busy=False, choice=name, error="", status=f"{name} ready.")
        if self._image is not None:
            self._rerun()

    def _open_image(self, path: str) -> None:
        from ..image.ops import imread

        self._stop_live.set()
        self._image = imread(path)
        self._publish(status=f"Loaded {Path(path).name}")
        self._rerun()

    def _rerun(self) -> None:
        if self._image is None or not self._current:
            return
        self._publish(busy=True, error="")
        frame, answer = self._infer(self._image)
        self._publish(busy=False, live=False, frame=frame, answer=answer, status="Done.")

    def _webcam(self, index: int) -> None:
        if not self._current:
            self._publish(status="Pick something on the left first.")
            return
        self._model(self._current)  # load before opening the camera
        capture = self._capture_factory(index)
        if capture is None or not capture.isOpened():
            self._publish(
                busy=False,
                error=f"Could not open camera {index}. Is another program using it?",
                status="No camera.",
            )
            return
        self._stop_live.clear()
        self._publish(busy=False, live=True, error="", status="Live. Press Stop to end.")
        try:
            while not self._stop_live.is_set() and not self._closing.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, answer = self._infer(frame)
                self._publish(frame=annotated, answer=answer)
        finally:
            capture.release()
            self._publish(live=False, status="Stopped.")

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Run the current model and return ``(annotated frame, one-line answer)``."""
        try:
            results = self._model(self._current)(image, conf=self._conf)
        except TypeError:  # a model that takes no conf argument
            results = self._model(self._current)(image)
        if not results:
            return image, "no result"
        result = results[0]
        return result.plot(), result.summary()


def _readable(exc: Exception) -> str:
    """One short, useful line out of an OpenVINO/HF exception."""
    text = " ".join(str(exc).split())
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    # OpenVINO nests errors; the last line is the only informative one.
    detail = lines[-1] if lines else text
    if not detail:
        detail = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return f"{type(exc).__name__}: {detail}"


def _default_model_factory(name: str, device: str = "AUTO") -> Any:
    from ..core.model import Model

    return Model(name, device=device)


def _default_capture_factory(index: int) -> Any:
    import cv2

    return cv2.VideoCapture(index)
