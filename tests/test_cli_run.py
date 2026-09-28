"""``ovkit run`` answers in the shape of its input, like ``Model(name, source)``.

It used to go through ``predict`` and read a whole video into a list before
printing anything, and ``ovkit run detect 0`` looked for a file called ``0``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from ovkit.__main__ import _shell_source, main
from ovkit.core.results import Boxes, Results
from ovkit.pipelines import PIPELINES
from ovkit.pipelines.base import Pipeline

SAMPLE = Path(__file__).parent / "assets" / "sample.png"


class _EchoPipe(Pipeline):
    name = "_cli_echo"
    description = "test double"

    def run(self, image, *, conf=0.25, **_):
        return Results(
            image,
            task="detect",
            names={0: "person"},
            boxes=Boxes(np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)),
        )


@pytest.fixture
def echo(monkeypatch):
    monkeypatch.setitem(PIPELINES, "_cli_echo", _EchoPipe)


def test_a_digit_is_a_camera_and_a_path_is_a_path():
    assert _shell_source("0") == 0
    assert _shell_source("1") == 1
    assert _shell_source("photo.png") == "photo.png"
    assert _shell_source("0.png") == "0.png", "a filename that starts with a digit is still a file"


def test_a_photo_prints_once_and_saves_next_to_the_shell(echo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    shutil.copy(SAMPLE, tmp_path / "교실.png")

    assert main(["run", "_cli_echo", "교실.png"]) == 0

    out = capsys.readouterr().out
    assert out.count("person") == 2, "the summary line and one box line"
    assert (tmp_path / "교실_out.jpg").is_file(), "the default save, named after the photo"


def test_a_folder_prints_one_line_per_file_and_saves_only_on_request(
    echo, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "우리반"
    folder.mkdir()
    for name in ("철수.png", "영희.png"):
        shutil.copy(SAMPLE, folder / name)

    assert main(["run", "_cli_echo", "우리반"]) == 0
    out = capsys.readouterr().out
    assert "철수.png" in out and "영희.png" in out
    assert not list(tmp_path.glob("*_out.jpg")), "a folder has no single photo to save by default"

    assert main(["run", "_cli_echo", "우리반", "--save", "first.jpg"]) == 0
    assert (tmp_path / "first.jpg").is_file()


def test_a_stream_without_a_window_does_not_write_a_file_per_frame(
    echo, tmp_path, monkeypatch, capsys
):
    """A camera at 30 fps would have filled the working directory.

    `Results.show()` writes the frame to a file when it cannot open a window,
    which is right for a loop someone wrote themselves and wrong for
    `ovkit run`. And `has_display()` does not catch the common case: ovkit
    depends on `opencv-python-headless`, so a Linux desktop with DISPLAY set
    passes that check and `cv2.imshow` raises anyway. The window has to be
    dropped on the first failure, not merely never asked for.
    """
    import cv2

    import ovkit.core.results as results_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_mod, "_WINDOWS_UNAVAILABLE", False)
    monkeypatch.setattr("ovkit.core.results.has_display", lambda: True)

    def no_windows(*_args, **_kwargs):
        raise cv2.error("The function is not implemented.")

    monkeypatch.setattr(cv2, "imshow", no_windows)

    frames = [np.zeros((16, 16, 3), np.uint8) for _ in range(12)]
    monkeypatch.setattr(
        "ovkit.pipelines.base._iter_sources", lambda src: ((f, "cam") for f in frames)
    )

    assert main(["run", "_cli_echo", "clip.mp4"]) == 0

    written = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert len(written) <= 1, f"one frame per file again: {written}"
    assert "person" in capsys.readouterr().out, "it must still print every frame"


def test_windows_available_goes_false_for_good_once_one_fails(monkeypatch):
    import ovkit.core.results as results_mod
    from ovkit.core.results import windows_available

    monkeypatch.setattr("ovkit.core.results.has_display", lambda: True)
    monkeypatch.setattr(results_mod, "_WINDOWS_UNAVAILABLE", False)
    assert windows_available() is True

    monkeypatch.setattr(results_mod, "_WINDOWS_UNAVAILABLE", True)
    assert windows_available() is False
