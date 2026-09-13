"""The GUI's logic, tested without a display."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ovkit.core.results import Boxes, Results
from ovkit.gui.controller import Controller, choices
from ovkit.image.ops import imwrite

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
    assert 6 <= len(entries) <= 24  # a beginner should not face 68 models
    assert all(c.label and c.hint for c in entries)
    assert entries[0].name == "scene"  # the friendliest thing first


def test_the_buttons_are_in_the_display_language(monkeypatch):
    """The window is where a beginner starts, and it greeted them in English.

    In a package whose display language defaults to Korean and whose point is
    that ``Model("얼굴분석")`` works, the one place the Korean was missing was
    the first screen.
    """
    monkeypatch.setenv("OVKIT_LANG", "ko")
    korean = {c.name: c for c in choices()}
    assert korean["face_analyze"].label == "얼굴 분석"
    assert (
        korean["face_analyze"].korean_name == "얼굴분석"
    ), "the name to type, learned by seeing it"

    monkeypatch.setenv("OVKIT_LANG", "en")
    english = {c.name: c for c in choices()}
    assert english["face_analyze"].label == "Faces"


def test_every_button_names_something_ovkit_can_load():
    from ovkit import list_pipelines
    from ovkit.core import registry

    caps = set(list_pipelines())
    unknown = [c.name for c in choices() if c.name not in caps and registry.resolve(c.name) is None]
    assert not unknown, f"the window offers what ovkit cannot load: {unknown}"


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

    path = tmp_path / "in.png"
    imwrite(path, FRAME)

    controller.select("detect")
    assert _wait(lambda: controller.view().choice == "detect")
    controller.open_image(path)
    assert _wait(lambda: controller.view().frame is not None)
    view = controller.view()
    assert view.answer == "person"
    assert not view.live and not view.busy


def test_changing_the_threshold_reruns_a_still_image(controller, tmp_path):

    path = tmp_path / "in.png"
    imwrite(path, FRAME)
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

    source = tmp_path / "in.png"
    imwrite(source, FRAME)
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


# -- reading text aloud ------------------------------------------------------


class _FakeSpeaker:
    """Stands in for the speak pipeline: takes a sentence, answers with sound."""

    def __init__(self, name, device="AUTO"):
        self.name = name
        self.device = device
        self.said: list[tuple[str, str]] = []

    def __call__(self, text, voice="F1", **kwargs):
        self.said.append((text, voice))
        samples = np.full(4410, 0.1, np.float32)
        r = Results(np.zeros((40, 200, 3), np.uint8), task="speak")
        r.audio = (samples, 44_100)
        r.text = text
        return [r]


@pytest.fixture
def speaker():
    made: list[_FakeSpeaker] = []

    def factory(name, device="AUTO"):
        model = _FakeSpeaker(name, device)
        made.append(model)
        return model

    ctl = Controller(model_factory=factory, capture_factory=lambda _i: _FakeCamera())
    ctl.made = made
    ctl.select("speak")
    yield ctl
    ctl.close()


def test_a_text_capability_is_marked_as_one():
    """The window swaps its webcam and file buttons on this flag alone."""
    by_name = {c.name: c for c in choices()}
    assert by_name["speak"].takes_text
    assert not by_name["detect"].takes_text


def test_speaking_publishes_a_waveform_and_an_answer(speaker):
    speaker.speak("안녕하세요", voice="M3")
    assert _wait(lambda: speaker.view().frame is not None)
    # The sound and the frame arrive together: Save must never be offered a
    # .wav for a result that is not on screen yet.
    assert speaker.has_audio()
    view = speaker.view()
    assert view.answer == "안녕하세요"
    assert "0.1초" in view.status and "M3" in view.status
    assert speaker.made[-1].said == [("안녕하세요", "M3")]


def test_empty_text_asks_for_some_instead_of_running(speaker):
    speaker.speak("   ")
    assert _wait(lambda: "먼저" in speaker.view().status)
    assert speaker.made[-1].said == []


def test_saving_a_spoken_result_writes_the_sound(speaker, tmp_path):
    import wave

    speaker.speak("안녕하세요")
    assert _wait(lambda: speaker.view().frame is not None)
    out = tmp_path / "hello.wav"
    assert speaker.save(out) is not None
    with wave.open(str(out)) as fh:
        assert fh.getframerate() == 44_100
        assert fh.getnframes() == 4410


def test_saving_a_spoken_result_as_a_picture_writes_the_waveform(speaker, tmp_path):
    """`.wav` gets the sound; anything else gets the picture of it."""
    speaker.speak("안녕하세요")
    assert _wait(lambda: speaker.view().frame is not None)
    out = tmp_path / "wave.png"
    assert speaker.save(out) is not None
    assert out.is_file() and out.stat().st_size > 0


def test_opening_a_picture_clears_the_sound(controller, tmp_path):
    """Otherwise Save would still be offering the last sentence's audio."""
    path = tmp_path / "x.png"
    imwrite(path, FRAME)
    controller.select("detect")
    controller.open_image(path)
    assert _wait(lambda: controller.view().frame is not None)
    assert not controller.has_audio()
