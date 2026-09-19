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
