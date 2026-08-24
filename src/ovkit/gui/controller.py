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
    label: str  #: what the button says
    hint: str  #: one line under the picture


def choices() -> list[Choice]:
    """The short, opinionated list a beginner should start from.

    Not every registered model — the point of the GUI is to answer a question,
    so it offers capabilities first and a handful of single models after.
    """
    return [
        # Capabilities first: they answer a question rather than emit a tensor.
        Choice("scene", "Describe", "One sentence about the whole picture"),
        Choice("face_analyze", "Faces", "Age, gender and emotion for every face"),
        Choice("detect", "Objects", "Find the objects in the picture (COCO-80)"),
        Choice("read_text", "Read text", "Find text and read it"),
        Choice("read_plate", "Plates", "Read number plates, describe the car"),
        Choice("track", "Track", "Objects keep an id from frame to frame"),
        Choice("drowsiness", "Drowsy?", "Warn when the eyes stay shut (needs a webcam)"),
        Choice("gesture", "Gesture", "Hand gestures from motion (needs a webcam)"),
        Choice("attention", "Attention", "Which object the person is looking at"),
        Choice("gaze", "Gaze", "Where a face is looking"),
        Choice("anonymize", "Blur faces", "Redact faces so the picture can be shared"),
        Choice("person_analyze", "People", "What each person wears or carries"),
        Choice("vehicle_analyze", "Vehicles", "Type and colour of each vehicle"),
        # A few single models, for when you want exactly one.
        Choice("pose", "Pose", "Body keypoints for every person"),
        Choice("segment", "Segment", "Label every pixel"),
        Choice("classify", "Classify", "What is this a picture of?"),
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
