"""Reading and writing image files, including ones a Korean teacher would name.

Cheap tests with an expensive history. ``Model("출석체크")`` reads a roster folder
where every file is a student's name — ``철수.png`` — and on a Korean Windows
machine that failed outright: ``cv2.imread`` hands the path to the C++ runtime,
the runtime reads it in the system code page, the name arrived as ``泥좎닔.png``
and the file "did not exist".

These read ``assets/sample.png``, which this repository ships, so the bytes
under test are the bytes anyone gets with a clone. On Linux they pass either
way; the Windows job in CI is where they earn their keep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ovkit.image.ops import imread, imwrite

SAMPLE = Path(__file__).parent / "assets" / "sample.png"

#: What ``scripts/make_sample.py`` put in each quadrant, in OpenCV's BGR order.
#: Asserting on these catches a decode that swapped channels or flipped the
#: image — failures that a shape check sails straight past.
CORNERS = {
    "top-left": ((4, 4), (32, 32, 200)),
    "top-right": ((4, -5), (32, 200, 32)),
    "bottom-left": ((-5, 4), (200, 32, 32)),
    "bottom-right": ((-5, -5), (220, 220, 220)),
}


def test_the_sample_is_there():
    """A missing asset should say so once, not fail every test below it."""
    assert SAMPLE.is_file(), f"{SAMPLE} is missing — run scripts/make_sample.py"


def test_the_sample_decodes_to_what_it_should_be():
    img = imread(SAMPLE)
    assert img.shape == (64, 96, 3)
    for corner, ((y, x), expected) in CORNERS.items():
        assert tuple(int(v) for v in img[y, x]) == expected, f"{corner} is wrong"


@pytest.mark.parametrize(
    "name",
    [
        "ascii.png",
        "철수.png",  # a name straight off a school roster
        "학생 이름.png",  # and one with a space in it
        "日本語.png",
        "Ünicode.png",
    ],
)
def test_a_name_in_any_script_survives_the_round_trip(tmp_path, name):
    original = imread(SAMPLE)
    imwrite(tmp_path / name, original)
    assert (tmp_path / name).is_file(), "written under the name it was given"
    back = imread(tmp_path / name)
    assert np.array_equal(back, original), "PNG is lossless, so this is exact"


def test_jpeg_round_trips_too(tmp_path):
    """The extension picks the encoder, so it still has to reach OpenCV."""
    original = imread(SAMPLE)
    imwrite(tmp_path / "사진.jpg", original)
    back = imread(tmp_path / "사진.jpg")
    assert back.shape == original.shape  # lossy, so shape and no more


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "없는파일.png")


def test_an_empty_file_is_not_an_image(tmp_path):
    """Zero bytes decode to nothing; that is a failed read, not a blank image."""
    (tmp_path / "empty.png").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "empty.png")


def test_a_file_that_is_not_an_image_says_so(tmp_path):
    (tmp_path / "notes.png").write_text("this is text", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "notes.png")


def test_writing_creates_the_folder(tmp_path):
    imwrite(tmp_path / "새폴더" / "깊이" / "철수.png", imread(SAMPLE))
    assert (tmp_path / "새폴더" / "깊이" / "철수.png").is_file()


def test_nothing_calls_cv2_imread_or_imwrite_directly():
    """Fixing ``ops.py`` is not enough while call sites go around it.

    That is what happened: ``imread``/``imwrite`` were taught to handle a
    Korean filename, and the roster test still failed — because it called
    ``cv2.imwrite`` itself and wrote ``泥좎닔.png``. The same bypass sat in
    ``teach.py``, where ``collect("가위", 30)`` names the folder.

    ``cv2.imdecode`` and ``cv2.imencode`` are fine: those take bytes, and bytes
    have no encoding problem. It is only the calls that take a *path*.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    here = Path(__file__).resolve()
    allowed = {root / "src" / "ovkit" / "image" / "ops.py", here}

    call = re.compile(r"\bcv2\.(imread|imwrite)\s*\(")
    offenders = []
    for folder in ("src", "tests", "scripts", "examples"):
        for path in (root / folder).rglob("*.py"):
            if path.resolve() in allowed:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if call.search(line):
                    offenders.append(f"{path.relative_to(root)}:{n}")

    assert not offenders, (
        "use ovkit.image.ops.imread/imwrite — cv2's take a path and cannot see "
        f"a non-ASCII one on Windows: {offenders}"
    )
