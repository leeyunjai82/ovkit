"""`Results.show()` — the window every webcam demo needs, and its fallback.

Every example wrote `cv2.imshow`, which **raises** on the OpenCV ovkit depends
on (`opencv-python-headless` has no window support) — so the README's demos
crashed for anyone who installed exactly what the README said. Worse, full
OpenCV on a machine with no display does not raise at all: it takes the
process down with a Qt error. A demo must never do either.
"""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.core import results as results_mod
from ovkit.core.results import Results


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(results_mod, "_WINDOWS_UNAVAILABLE", False)
    monkeypatch.setattr(Results, "_shown", 0)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OVKIT_QUIET", raising=False)


def _result() -> Results:
    return Results(np.zeros((16, 16, 3), np.uint8), task="detect")


def test_no_display_saves_a_file_instead_of_dying(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")

    assert _result().show("demo") is True, "a stream must keep going"
    assert (tmp_path / "demo_0000.jpg").is_file()
    assert "pip install opencv-python" in capsys.readouterr().err


def test_the_advice_is_given_once_not_per_frame(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    r = _result()
    r.show("demo")
    capsys.readouterr()
    for _ in range(3):
        r.show("demo")
    assert capsys.readouterr().err == "", "a message per frame is noise, not help"
    assert (tmp_path / "demo_0003.jpg").is_file(), "frames still land one per call"


def test_a_headless_build_that_raises_is_caught(monkeypatch, tmp_path, capsys):
    import cv2

    monkeypatch.setenv("DISPLAY", ":0")

    def _raise(*_args, **_kwargs):
        raise cv2.error("The function is not implemented")

    monkeypatch.setattr(cv2, "imshow", _raise)
    assert _result().show("demo") is True
    assert (tmp_path / "demo_0000.jpg").is_file()


def test_q_stops_the_loop(monkeypatch):
    import cv2

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _wait: ord("q"))
    assert _result().show("demo") is False, "q must end the loop"

    monkeypatch.setattr(cv2, "waitKey", lambda _wait: ord("x"))
    assert _result().show("demo") is True
