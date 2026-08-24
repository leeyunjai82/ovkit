"""The GUI's logic, tested without a display."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ovkit.core.results import Boxes, Results
from ovkit.gui.controller import Controller, choices

FRAME = np.zeros((60, 80, 3), np.uint8)


def _wait(predicate, timeout: float = 3.0) -> bool:
    """Wait for the worker thread to reach a state (no sleeps in the assertions)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class _FakeModel:
    def __init__(self, name, device="AUTO"):
        self.name = name
        self.device = device
        self.calls = 0

    def __call__(self, image, conf=0.25):
        self.calls += 1
        r = Results(
            image,
            task="detect",
            names={0: "person"},
            boxes=Boxes(np.array([[1, 1, 10, 10, 0.9, 0]], np.float32)),
        )
        return [r]


class _FakeCamera:
    def __init__(self, frames=5):
        self.frames = frames
        self.released = False
        self._n = 0

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        if self._n > self.frames:
            time.sleep(0.005)  # keep yielding so stop() can win the race
            return True, FRAME.copy()
        return True, FRAME.copy()

    def release(self):
        self.released = True


@pytest.fixture
def controller():
    made: list[_FakeModel] = []

    def factory(name, device="AUTO"):
        model = _FakeModel(name, device)
        made.append(model)
        return model

    ctl = Controller(model_factory=factory, capture_factory=lambda _i: _FakeCamera())
    ctl.made = made
    yield ctl
    ctl.close()


def test_the_button_list_is_short_and_described():
    entries = choices()
    assert 6 <= len(entries) <= 20  # a beginner should not face 43 models
    assert all(c.label and c.hint for c in entries)
    assert entries[0].name == "scene"  # the friendliest thing first


def test_selecting_a_capability_loads_it_in_the_background(controller):
    controller.select("face_analyze")
    assert _wait(lambda: controller.view().choice == "face_analyze")
    view = controller.view()
    assert not view.busy and "ready" in view.status
    assert controller.made[0].name == "face_analyze"


def test_a_model_that_fails_to_load_shows_a_message_instead_of_crashing():
    def factory(name, device="AUTO"):
        raise RuntimeError("Exception from core.cpp:135:\nEmpty weights data in bin file")

    ctl = Controller(model_factory=factory)
    try:
        ctl.select("detect")
        assert _wait(lambda: ctl.view().error != "")
        error = ctl.view().error
        assert "Empty weights data" in error  # the useful last line, not the header
        assert not ctl.view().busy
    finally:
        ctl.close()


def test_opening_an_image_runs_the_model_and_publishes_a_frame(controller, tmp_path):
    import cv2

    path = tmp_path / "in.png"
    cv2.imwrite(str(path), FRAME)

    controller.select("detect")
    assert _wait(lambda: controller.view().choice == "detect")
    controller.open_image(path)
    assert _wait(lambda: controller.view().frame is not None)
    view = controller.view()
    assert view.answer == "person"
    assert not view.live and not view.busy


def test_changing_the_threshold_reruns_a_still_image(controller, tmp_path):
    import cv2

    path = tmp_path / "in.png"
    cv2.imwrite(str(path), FRAME)
    controller.select("detect")
    controller.open_image(path)
    assert _wait(lambda: controller.view().frame is not None)
    before = controller.made[0].calls

    controller.set_conf(0.8)
    assert _wait(lambda: controller.made[0].calls > before)


def test_the_webcam_streams_until_it_is_stopped(controller):
    controller.select("detect")
    assert _wait(lambda: controller.view().choice == "detect")
    controller.start_webcam(0)
    assert _wait(lambda: controller.view().live)
    assert _wait(lambda: controller.made[0].calls >= 3)

    controller.stop()
    assert _wait(lambda: not controller.view().live)
    assert "Stopped" in controller.view().status


def test_a_camera_that_will_not_open_says_so_rather_than_hanging():
    class _Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    ctl = Controller(model_factory=_FakeModel, capture_factory=lambda _i: _Closed())
    try:
        ctl.select("detect")
        ctl.start_webcam(3)
        assert _wait(lambda: "camera 3" in ctl.view().error)
        assert not ctl.view().live
    finally:
        ctl.close()


def test_the_webcam_needs_a_capability_chosen_first(controller):
    controller.start_webcam(0)
    assert _wait(lambda: "Pick something" in controller.view().status)
    assert not controller.view().live


def test_saving_writes_the_frame_on_screen(controller, tmp_path):
    import cv2

    source = tmp_path / "in.png"
    cv2.imwrite(str(source), FRAME)
    controller.select("detect")
    controller.open_image(source)
    assert _wait(lambda: controller.view().frame is not None)

    out = controller.save(tmp_path / "out.jpg")
    assert out is not None and out.exists()


def test_saving_before_anything_ran_does_nothing(controller, tmp_path):
    assert controller.save(tmp_path / "out.jpg") is None


def test_switching_device_reloads_the_models(controller):
    controller.select("detect")
    assert _wait(lambda: controller.view().choice == "detect")
    controller.set_device("CPU")
    assert _wait(lambda: len(controller.made) >= 2)
    assert controller.made[-1].device == "CPU"


def test_closing_ends_the_worker_thread():
    ctl = Controller(model_factory=_FakeModel)
    ctl.close()
    assert _wait(
        lambda: not any(t.name == "ovkit-gui" and t.is_alive() for t in threading.enumerate())
    )
