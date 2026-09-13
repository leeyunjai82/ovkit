"""Depth and background: a map of floats made into a picture and a sentence."""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.core.registry import resolve, tier_of
from ovkit.image.ops import imread
from ovkit.recognize import get_adapter
from ovkit.recognize.depth import BackgroundAdapter, DepthAdapter, _normalize01, _single_map

IMG = np.full((120, 200, 3), 128, np.uint8)


class _Backend:
    """A model that returns a prepared map, whatever it is fed."""

    def __init__(self, map_2d, shape=(1, 3, 64, 64)):
        self.map_2d = np.asarray(map_2d, np.float32)
        self.input_shape = shape
        self.inputs = [object()]

    def infer(self, _feed):
        return {"out": self.map_2d[None, None]}


# -- shared helpers ---------------------------------------------------------


def test_a_map_is_squeezed_to_two_dimensions():
    assert _single_map({"x": np.zeros((1, 1, 4, 4), np.float32)}).shape == (4, 4)
    assert _single_map({"x": np.zeros((1, 8, 8), np.float32)}).shape == (8, 8)


def test_normalising_a_flat_map_does_not_divide_by_zero():
    flat = np.full((4, 4), 7.0, np.float32)
    assert _normalize01(flat).max() == 0.0  # no nan
    assert _normalize01(np.array([[0.0, 4.0]])).tolist() == [[0.0, 1.0]]


# -- depth ------------------------------------------------------------------


def _depth_map(bright_at):
    """A 64x64 inverse-depth map with one bright (near) spot."""
    m = np.zeros((64, 64), np.float32)
    m[bright_at] = 1.0
    return m


def test_depth_says_where_the_nearest_thing_is(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    adapter = DepthAdapter()
    r = adapter.run(_Backend(_depth_map((2, 2))), IMG)  # bright = near, top-left
    assert "왼쪽 위" in r.summary()


def test_depth_reports_the_far_corner_too(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    r = DepthAdapter().run(_Backend(_depth_map((60, 60))), IMG)
    assert "bottom-right" in r.summary()


def test_depth_returns_a_colour_picture_the_size_of_the_input():
    r = DepthAdapter().run(_Backend(_depth_map((2, 2))), IMG)
    assert r.display is not None
    assert r.display.shape == IMG.shape  # colourised, resized back
    assert r.plot().shape == IMG.shape
    assert r.tensors["depth"].shape == IMG.shape[:2]


def test_depth_is_not_drawn_over_the_original_photo():
    """A depth map means nothing pasted on top of the input."""
    r = DepthAdapter().run(_Backend(_depth_map((2, 2))), IMG)
    assert not np.array_equal(r.plot(), IMG)


# -- background -------------------------------------------------------------


def _subject_mask(fraction: float) -> np.ndarray:
    m = np.zeros((64, 64), np.float32)
    rows = int(64 * fraction)
    m[:rows] = 1.0
    return m


def test_background_reports_how_much_the_subject_covers(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    r = BackgroundAdapter().run(_Backend(_subject_mask(0.5)), IMG)
    assert "50%" in r.summary()


def test_an_empty_mask_says_no_subject(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    r = BackgroundAdapter().run(_Backend(np.zeros((64, 64), np.float32)), IMG)
    assert "못 찾았" in r.summary()


def test_the_background_is_actually_dropped_from_the_picture():
    r = BackgroundAdapter().run(_Backend(_subject_mask(0.5)), IMG)
    top, bottom = r.display[:50], r.display[70:]
    assert top.mean() > 100  # subject kept
    assert bottom.mean() < 10  # background gone


def test_saving_a_png_keeps_the_transparency(tmp_path):
    import cv2

    r = BackgroundAdapter().run(_Backend(_subject_mask(0.5)), IMG)
    out = r.save(tmp_path / "cut.png")
    saved = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED)
    assert saved.shape[2] == 4, "a cut-out must carry an alpha channel"
    assert saved[0, 0, 3] == 255 and saved[-1, -1, 3] == 0


def test_saving_a_jpg_falls_back_to_the_flattened_picture(tmp_path):

    r = BackgroundAdapter().run(_Backend(_subject_mask(0.5)), IMG)
    saved = imread(r.save(tmp_path / "cut.jpg"))
    assert saved.shape[2] == 3


# -- registration -----------------------------------------------------------


@pytest.mark.parametrize(
    "alias,target,task",
    [
        ("depth", "depth_anything_v2_small", "depth"),
        ("remove_background", "u2net", "background"),
    ],
)
def test_the_new_models_are_registered_and_permissive(alias, target, task):
    entry = resolve(alias)
    assert entry is not None and entry.name == target
    assert entry.task == task
    assert entry.license == "apache-2.0", "nothing copyleft ships with ovkit"
    assert tier_of(alias) == "core"


def test_they_are_mirrored_not_read_from_the_agpl_repo():
    """edge-lab is tagged agpl-3.0; ovkit serves its own copies."""
    for name in ("depth_anything_v2_small", "u2net"):
        entry = resolve(name)
        source = f"{entry.repo or ''} {entry.url or ''}"
        assert "edge-lab" not in source
        assert "ovkit-models" in source


def test_korean_names_reach_them():
    from ovkit.core.i18n import canonical

    assert canonical("거리재기") == "depth"
    assert canonical("배경지우기") == "remove_background"
    assert isinstance(get_adapter("depth"), DepthAdapter)


def test_only_the_small_depth_checkpoint_is_mirrored():
    """Base/Large/Giant are CC-BY-NC and must never appear here."""
    entry = resolve("depth_anything_v2_small")
    assert "small" in entry.filename.lower()
    assert not any(v in entry.name.lower() for v in ("base", "large", "giant"))
